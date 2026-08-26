"""Focused offline tests for direct Schwab routing and option validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
import inspect
import sys
import types
from pathlib import Path
from decimal import Decimal

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
# This extracted patch fixture lacks the host project's money helper.  Supply
# the tiny compatible surface before importing the unchanged PaperBroker model.
money_module = types.ModuleType("engine.utils.money")
money_module.D = Decimal
money_module.money = lambda value: Decimal(str(value)).quantize(Decimal("0.01"))
money_module.to_float = float
money_module.safe_div = lambda numerator, denominator, default=0: numerator / denominator if denominator else default
sys.modules.setdefault("engine.utils", types.ModuleType("engine.utils"))
sys.modules.setdefault("engine.utils.money", money_module)

from engine.data.schwab_data import SecureTokenStore, SchwabMarketData
from engine.execution.l7_unified_execution_surface import (
    L7Order, L7RiskEngine, L7UnifiedExecutionSurface, ProductType, TransactionCostAnalyzer,
)
from engine.execution.paper_broker import Order, OrderSide, OrderStatus, SignalType
from engine.execution.schwab_broker import SchwabAPIError, SchwabBroker
from engine.execution.options_engine import (
    OptionsEngine, OptionsSizer, OptionOverlayAllocator, monte_carlo_option_price,
)
from engine.allocation.allocation_engine import AllocationEngine, AllocationRules, BucketType, PositionAllocation


@dataclass
class Response:
    status_code: int
    payload: object
    headers: dict | None = None
    def json(self): return self.payload


def test_schwab_retries_once_after_401_and_normalizes_chain(tmp_path):
    store = SecureTokenStore(tmp_path / "token.json")
    store.save({"access_token": "old", "refresh_token": "refresh", "expires_at": time.time() + 600})
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/oauth/token"):
            return Response(200, {"access_token": "new", "refresh_token": "refresh", "expires_in": 1800})
        if len([item for item in calls if item[1].endswith("/chains")]) == 1:
            return Response(401, {})
        return Response(200, {
            "underlyingPrice": 100,
            "callExpDateMap": {"2026-09-18:28": {"100.0": [{
                "symbol": "ABC  260918C00100000", "strikePrice": 100, "bid": 2, "ask": 2.2,
                "mark": 2.1, "volatility": 25, "delta": .5, "gamma": .02, "theta": -.03,
                "vega": .1, "openInterest": 100, "totalVolume": 50,
            }]}},
            "putExpDateMap": {},
        })

    client = SchwabMarketData("id", "secret", "https://127.0.0.1", token_store=store, request_func=request)
    chain = client.option_chain("ABC")
    assert len(chain) == 1
    assert chain[0]["option_type"] == "CALL"
    assert chain[0]["iv"] == .25
    assert len([item for item in calls if item[1].endswith("/chains")]) == 2


def test_schwab_broker_never_submits_futures(tmp_path):
    broker = SchwabBroker(market_data=SchwabMarketData(token_store=SecureTokenStore(tmp_path / "token.json"),
                                                       request_func=lambda *a, **k: Response(500, {})))
    order = broker.place_order("ES", OrderSide.BUY, 1)
    assert order.status == OrderStatus.REJECTED
    assert "never sent" in order.reason


def test_options_engine_creates_order_ready_validation_metadata(monkeypatch):
    class MarketData:
        def option_chain(self, ticker):
            return [{
                "symbol": "ABC  260918C00100000", "underlying": ticker, "underlying_price": 100,
                "option_type": "CALL", "expiry": "2026-09-18", "dte": 30, "strike": 100,
                "bid": 1.9, "ask": 2.1, "mark": 2.0, "last": 2.0, "iv": .25,
                "delta": .5, "gamma": .02, "theta": -.03, "vega": .1,
            }]
    monkeypatch.setattr(OptionsSizer, "size_option", lambda *a, **k: {
        "contracts": 1, "rejected": False, "edge_bps": 250, "market_price": 2.0,
    })
    result = OptionsEngine(nav=100_000, market_data=MarketData()).scan_opportunities(
        "ABC", allocation={
            "sector": "Technology", "current_beta": 0.10, "target_beta": 0.35,
        }, regime="NORMAL",
    )
    assert result[0]["option_symbol"] == "ABC  260918C00100000"
    validation = result[0]["options_validation"]
    assert validation["source"] == "OptionsEngine"
    assert validation["strategy_classification"]["kind"] == "SINGLE_LEG_LONG_CALL"
    assert validation["strategy_classification"]["leg_count"] == 1
    assert validation["sector"] == "Technology"
    assert validation["signal_source"] == "BETA_CORRIDOR"
    assert validation["beta_corridor"]["beta_gap"] == pytest.approx(.25)
    assert "target_delta_dollars" not in validation["beta_corridor"]
    assert "target_delta_dollars" not in validation["sizing"]
    assert validation["bsm"]["quoted_iv"] == pytest.approx(.25)
    assert validation["bsm"]["model_iv"] == pytest.approx(.25)
    assert validation["predictive_signals"]
    assert validation["bsm"]["greeks"]["gamma"] > 0


def test_option_sizer_has_no_delta_exposure_target():
    parameters = inspect.signature(OptionsSizer.size_option).parameters
    assert "target_delta_dollars" not in parameters


def test_options_engine_fails_closed_without_beta_corridor_state():
    class MarketData:
        def option_chain(self, ticker):
            return [{
                "symbol": "ABC  260918C00100000", "underlying": ticker,
                "underlying_price": 100, "option_type": "CALL", "dte": 30,
                "strike": 100, "bid": 1.9, "ask": 2.1, "mark": 2.0,
                "last": 2.0, "iv": .25,
            }]

    assert OptionsEngine(nav=100_000, market_data=MarketData()).scan_opportunities(
        "ABC", allocation={"sector": "Technology"}, regime="NORMAL",
    ) == []


def test_options_engine_beta_corridor_sign_selects_option_direction(monkeypatch):
    class MarketData:
        def option_chain(self, ticker):
            common = {
                "underlying": ticker, "underlying_price": 100, "dte": 30,
                "strike": 100, "bid": 1.9, "ask": 2.1, "mark": 2.0,
                "last": 2.0, "iv": .25, "open_interest": 100, "volume": 10,
            }
            return [
                {**common, "symbol": "ABC  260918C00100000", "option_type": "CALL"},
                {**common, "symbol": "ABC  260918P00100000", "option_type": "PUT"},
            ]

    monkeypatch.setattr(OptionsSizer, "size_option", lambda *a, **k: {
        "contracts": 1, "rejected": False, "edge_bps": 0,
        "market_price": 2.1, "fair_value": 2.0,
    })
    bullish = OptionsEngine(nav=100_000, market_data=MarketData()).scan_opportunities(
        "ABC", allocation={
            "sector": "Technology", "current_beta": 0.10, "target_beta": 0.35,
        }, regime="NORMAL",
    )
    bearish = OptionsEngine(nav=100_000, market_data=MarketData()).scan_opportunities(
        "ABC", allocation={
            "sector": "Technology", "current_beta": 0.35, "target_beta": 0.10,
        }, regime="NORMAL",
    )
    assert {candidate["option_type"] for candidate in bullish} == {"CALL"}
    assert {candidate["option_type"] for candidate in bearish} == {"PUT"}


def test_monte_carlo_pricer_does_not_reset_global_numpy_rng():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    monte_carlo_option_price(
        S=100, K=100, T=30 / 365, sigma=.25,
        n_sims=100, n_steps=5, seed=42,
    )
    actual = np.random.random(3)
    assert actual == pytest.approx(expected)


class ConfirmingBroker:
    """Offline broker that confirms only the narrow calls used in this test."""
    is_connected = True
    def __init__(self):
        self.state = type("State", (), {"nav": 100_000.0, "cash": 100_000.0, "positions": {}})()
        self.market_data = None
        self.option_calls = []
    def get_quote(self, ticker): return 100.0
    def compute_exposures(self): return {"gross": 0.0, "net": 0.0}
    def place_order(self, ticker, side, quantity, **kwargs):
        return Order(ticker=ticker, side=side, quantity=quantity, fill_price=100.0, status=OrderStatus.FILLED)
    def place_option_order(self, **kwargs):
        self.option_calls.append(kwargs)
        return Order(ticker=kwargs["option_symbol"], side=OrderSide.BUY, quantity=kwargs["quantity"],
                     fill_price=2.0, status=OrderStatus.FILLED)


def test_l7_rejects_raw_option_and_accepts_options_engine_validation(tmp_path):
    broker = ConfirmingBroker()
    options_engine = object()
    l7 = L7UnifiedExecutionSurface(
        initial_cash=100_000, log_dir=str(tmp_path), broker=broker,
        options_engine=options_engine, test_stage=True,
    )
    raw = l7.submit_order("ABC", "BUY", 1, product_type="OPTION", limit_price=2.0,
                          option_symbol="ABC  260918C00100000", instruction="BUY_TO_OPEN")
    assert raw.status == "REJECTED"
    assert "OptionsEngine validation" in raw.reason

    metadata = {
        "source": "OptionsEngine", "sector": "Technology",
        "strategy_classification": {"kind": "SINGLE_LEG_LONG_CALL", "leg_count": 1},
        "bsm": {"spot": 100.0, "greeks": {"delta": .5}},
        "sizing": {"contracts": 1}, "estimated_initial_margin": 70.0,
    }
    approved = l7.submit_order(
        "ABC", "BUY", 1, product_type="OPTION", limit_price=2.0,
        option_symbol="ABC  260918C00100000", instruction="BUY_TO_OPEN",
        legs=[{"instruction": "BUY_TO_OPEN", "quantity": 1,
               "instrument": {"symbol": "ABC  260918C00100000", "assetType": "OPTION"}}],
        options_validation=metadata, sector="Technology",
    )
    assert approved.status == "FILLED"
    assert broker.option_calls[0]["instruction"] == "BUY_TO_OPEN"
    assert l7._learning is None


def test_l7_keeps_accepted_option_order_pending_until_broker_fill(tmp_path):
    class PendingOptionBroker(ConfirmingBroker):
        def place_option_order(self, **kwargs):
            self.option_calls.append(kwargs)
            return Order(
                ticker=kwargs["option_symbol"], side=OrderSide.BUY, quantity=kwargs["quantity"],
                status=OrderStatus.PENDING, reason="Accepted by Schwab",
            )

    metadata = {
        "source": "OptionsEngine", "sector": "Technology",
        "strategy_classification": {"kind": "SINGLE_LEG_LONG_CALL", "leg_count": 1},
        "bsm": {"spot": 100.0, "greeks": {"delta": .5}},
        "sizing": {"contracts": 1}, "estimated_initial_margin": 70.0,
    }
    l7 = L7UnifiedExecutionSurface(
        initial_cash=100_000, log_dir=str(tmp_path), broker=PendingOptionBroker(),
        options_engine=object(), test_stage=True,
    )
    order = l7.submit_order(
        "ABC", "BUY", 1, product_type="OPTION", limit_price=2.0,
        option_symbol="ABC  260918C00100000", instruction="BUY_TO_OPEN",
        legs=[{"instruction": "BUY_TO_OPEN", "quantity": 1,
               "instrument": {"symbol": "ABC  260918C00100000", "assetType": "OPTION"}}],
        options_validation=metadata, sector="Technology",
    )
    assert order.status == "PENDING"
    assert not l7.get_filled_orders()


class RoutedMarketData:
    """Offline Schwab transport with three approved accounts and captured posts."""
    def __init__(self, accounts, snapshots):
        self.accounts = accounts
        self.snapshots = snapshots
        self.posts = []
        self.token_store = type("Tokens", (), {"load": lambda self: {"access_token": "test"}})()

    def request(self, method, path, **kwargs):
        if path.endswith("accountNumbers"):
            return Response(200, self.accounts)
        if method == "GET":
            account_hash = path.split("/")[4]
            return Response(200, self.snapshots[account_hash])
        self.posts.append((path, kwargs))
        return Response(201, {}, headers={"Location": f"{path}/123"})

    def get_quote(self, ticker):
        return {"mark": 10.0, "last": 10.0}


def _snapshot(nav=100_000, cash=50_000, option_market_value=0, positions=None):
    positions = list(positions or [])
    if option_market_value:
        positions.append({
            "longQuantity": 2, "shortQuantity": 0, "marketValue": option_market_value,
            "averagePrice": option_market_value / 200,
            "instrument": {"assetType": "OPTION", "symbol": "ABC  260918C00100000"},
        })
    return {
        "securitiesAccount": {
            "currentBalances": {"liquidationValue": nav, "cashAvailableForTrading": cash},
            "positions": positions,
        }
    }


def _routed_broker(option_account="1114806", non_option_accounts=("2229565",)):
    hashes = {option_account: "hash-option"}
    hashes.update({number: f"hash-{number[-4:]}" for number in non_option_accounts})
    accounts = [{"accountNumber": number, "hashValue": value} for number, value in hashes.items()]
    snapshots = {value: _snapshot() for value in hashes.values()}
    market_data = RoutedMarketData(accounts, snapshots)
    return (
        SchwabBroker(
            market_data=market_data, option_account_number=option_account,
            non_option_account_numbers=non_option_accounts, account_rate_limit=1_000_000,
        ),
        market_data,
    )


def test_schwab_option_routes_only_to_configured_4806_and_stays_pending():
    broker, market_data = _routed_broker()
    order = broker.place_option_order("ABC  260918C00100000", "BUY_TO_OPEN", 1, limit_price=2.0)
    assert order.status == OrderStatus.PENDING
    assert market_data.posts[0][0] == "/trader/v1/accounts/hash-option/orders"


def test_schwab_rejects_unapproved_or_ambiguous_live_account_routing():
    bad_option, _ = _routed_broker(option_account="1119999")
    assert bad_option.place_option_order("ABC  260918C00100000", "BUY_TO_OPEN", 1, limit_price=2).status == OrderStatus.REJECTED

    ambiguous, market_data = _routed_broker(non_option_accounts=("2229565", "3330514"))
    assert ambiguous.place_order("ABC", OrderSide.BUY, 1, limit_price=10).status == OrderStatus.REJECTED
    assert not market_data.posts
    explicit = ambiguous.place_order("ABC", OrderSide.BUY, 1, limit_price=10, account_number="3330514")
    assert explicit.status == OrderStatus.PENDING
    assert market_data.posts[0][0] == "/trader/v1/accounts/hash-0514/orders"


def test_schwab_account_hash_discovery_fails_closed_on_duplicate_mapping():
    broker, market_data = _routed_broker()
    market_data.accounts.append({"accountNumber": "1114806", "hashValue": "duplicate-hash"})
    with pytest.raises(SchwabAPIError, match="Duplicate"):
        broker.discover_account_hash("1114806", "OPTION")
    with pytest.raises(SchwabAPIError, match="not explicitly configured"):
        broker.discover_account_hash("9999565", "EQUITY")


def test_schwab_does_not_apply_money_market_reserve_to_4806():
    broker, market_data = _routed_broker()
    market_data.snapshots["hash-option"] = _snapshot(nav=100_000, cash=2_100)
    market_data.snapshots["hash-9565"] = _snapshot(nav=300_000, cash=50_000)
    order = broker.place_option_order("ABC  260918C00100000", "BUY_TO_OPEN", 1, limit_price=2.0)
    # The 2% money-market reserve belongs only to non-option cash accounts.
    # 4806 may use its option sleeve cash without being assigned a reserve.
    assert order.status == OrderStatus.PENDING
    assert market_data.posts


def test_schwab_cash_floor_uses_only_non_option_accounts():
    broker, market_data = _routed_broker(non_option_accounts=("2229565", "3330514"))
    market_data.snapshots["hash-option"] = _snapshot(nav=900_000, cash=0)
    market_data.snapshots["hash-9565"] = _snapshot(nav=100_000, cash=2_050)
    market_data.snapshots["hash-0514"] = _snapshot(nav=100_000, cash=50_000)
    order = broker.place_order("ABC", OrderSide.BUY, 1, limit_price=200, account_number="2229565")
    # The two cash accounts have $200,000 NAV.  The 9565 floor is 2% of its
    # $100,000 NAV ($2,000), independent of 4806's $900,000 NAV.
    assert order.status == OrderStatus.REJECTED
    assert not market_data.posts


def test_sync_account_consolidates_all_configured_accounts_without_fallback():
    broker, market_data = _routed_broker(non_option_accounts=("2229565", "3330514"))
    market_data.snapshots["hash-option"] = _snapshot(
        nav=100_000, cash=1_000,
        positions=[{
            "longQuantity": 2, "shortQuantity": 0, "marketValue": 400,
            "averagePrice": 2, "instrument": {"assetType": "OPTION", "symbol": "ABC_OPT"},
        }],
    )
    market_data.snapshots["hash-9565"] = _snapshot(
        nav=200_000, cash=2_000,
        positions=[{
            "longQuantity": 5, "shortQuantity": 0, "marketValue": 500,
            "averagePrice": 90, "instrument": {"assetType": "EQUITY", "symbol": "ABC"},
        }],
    )
    market_data.snapshots["hash-0514"] = _snapshot(
        nav=300_000, cash=3_000,
        positions=[{
            "longQuantity": 3, "shortQuantity": 0, "marketValue": 330,
            "averagePrice": 105, "instrument": {"assetType": "EQUITY", "symbol": "ABC"},
        }],
    )

    summary = broker.sync_account()
    assert summary["nav"] == 600_000
    assert summary["cash"] == 6_000
    assert summary["account_count"] == 3
    assert summary["account_numbers"] == ("1114806", "2229565", "3330514")
    assert broker.state.positions["ABC"].quantity == 8
    assert broker.state.positions["ABC"].current_price == pytest.approx(103.75)

    explicit = broker.sync_account(account_number="3330514")
    assert explicit["nav"] == 300_000
    assert explicit["cash"] == 3_000
    assert explicit["account_count"] == 1
    assert set(broker.state.positions) == {"ABC"}


def test_sync_account_fails_closed_on_missing_or_duplicate_mapping():
    duplicate, duplicate_data = _routed_broker(non_option_accounts=("2229565", "3330514"))
    duplicate_data.accounts.append({"accountNumber": "3330514", "hashValue": "duplicate-hash"})
    with pytest.raises(SchwabAPIError, match="Duplicate"):
        duplicate.sync_account()

    missing, missing_data = _routed_broker(non_option_accounts=("2229565", "3330514"))
    missing_data.accounts = [
        account for account in missing_data.accounts if account["accountNumber"] != "3330514"
    ]
    with pytest.raises(SchwabAPIError, match="was not returned"):
        missing.sync_account()


def test_l7_option_multiplier_applies_to_tca_risk_and_reserve():
    order = L7Order(
        ticker="ABC", side="BUY", quantity=2, limit_price=5.0,
        fill_quantity=2, product_type=ProductType.OPTION,
    )
    tca = TransactionCostAnalyzer()
    snapshot = tca.analyze(order, arrival_price=4.0, fill_price=5.0)
    assert snapshot.implementation_shortfall_usd == 200.0
    assert tca.get_aggregate().total_volume_usd == 1_000.0

    risk = L7RiskEngine(initial_nav=10_000)
    passed, violations = risk.pre_trade_check(
        L7Order(ticker="ABC", side="BUY", quantity=99, limit_price=100.0),
        nav=10_000, cash=10_000, positions={}, daily_pnl=0, gross_exposure=0, net_exposure=0,
    )
    assert not passed
    assert any("G8_CASH" in violation for violation in violations)


def test_l7_option_overlay_and_margin_caps_fail_closed():
    risk = L7RiskEngine(initial_nav=1_000_000)
    order = L7Order(
        ticker="ABC", side="BUY", quantity=2_300, limit_price=1.0,
        product_type=ProductType.OPTION, sector="Technology",
        options_validation={
            "bsm": {"spot": 100.0, "greeks": {"delta": .5}},
            "estimated_initial_margin": 80_500,
        },
    )
    passed, violations = risk.pre_trade_check(
        order, nav=1_000_000, cash=1_000_000, positions={},
        daily_pnl=0, gross_exposure=0, net_exposure=0,
    )
    assert not passed
    assert any("G11_OPTIONS_OVERLAY" in violation for violation in violations)
    assert any("G12_OPTIONS_INITIAL_MARGIN" in violation for violation in violations)


def _validated_option_metadata():
    return {
        "source": "OptionsEngine", "sector": "Technology",
        "strategy_classification": {"kind": "SINGLE_LEG_LONG_CALL", "leg_count": 1},
        "bsm": {"spot": 100.0, "greeks": {"delta": .5}},
        "sizing": {"contracts": 1}, "estimated_initial_margin": 70.0,
    }


def test_g9_uses_actual_delta_dollar_exposure_and_pending_reservations():
    risk = L7RiskEngine(initial_nav=100_000)
    first = L7Order(
        ticker="ABC", side="BUY", quantity=1, limit_price=2.0,
        product_type=ProductType.OPTION, sector="Technology",
        options_validation=_validated_option_metadata(),
    )
    assert risk.reserve_option_order(first)
    assert risk._reserved_option_metrics()["delta_dollars"] == 5_000.0

    # 4 more contracts would bring actual delta-dollar exposure to $25,000,
    # beyond the $20,000 G9 cap.  The old premium*0.5 approximation would not.
    second = L7Order(
        ticker="ABC", side="BUY", quantity=4, limit_price=2.0,
        product_type=ProductType.OPTION, sector="Technology",
        options_validation=_validated_option_metadata(),
    )
    passed, violations = risk.pre_trade_check(
        second, nav=100_000, cash=100_000, positions={}, daily_pnl=0,
        gross_exposure=0, net_exposure=0,
    )
    assert not passed
    assert any("G9_OPTIONS_DELTA" in violation for violation in violations)

    risk.release_option_reservation(first.order_id)
    assert risk._reserved_option_metrics()["delta_dollars"] == 0.0


def test_option_reservation_reconciles_fill_without_double_count():
    risk = L7RiskEngine(initial_nav=100_000)
    order = L7Order(
        ticker="ABC", side="BUY", quantity=1, limit_price=2.0,
        fill_quantity=1, fill_price=2.0, product_type=ProductType.OPTION,
        sector="Technology", options_validation=_validated_option_metadata(),
    )
    assert risk.reserve_option_order(order)
    risk.post_trade_update(
        order, nav=100_000, cash=100_000, positions={}, daily_pnl=0,
        gross_exposure=0, net_exposure=0,
    )
    assert not risk._option_reservations
    assert risk._options_delta_exposure == 5_000.0
    assert risk._options_notional_exposure == 200.0
    assert risk._options_initial_margin == 70.0


def test_sector_metadata_fails_closed_and_unpaired_sell_to_open_is_blocked(tmp_path):
    risk = L7RiskEngine(initial_nav=100_000)
    missing_sector = L7Order(ticker="ABC", side="BUY", quantity=1, limit_price=1)
    passed, violations = risk.pre_trade_check(
        missing_sector, nav=100_000, cash=100_000, positions={}, daily_pnl=0,
        gross_exposure=0, net_exposure=0,
    )
    assert not passed
    assert any("G2_SECTOR" in violation for violation in violations)

    broker = ConfirmingBroker()
    l7 = L7UnifiedExecutionSurface(
        initial_cash=100_000, log_dir=str(tmp_path), broker=broker,
        options_engine=object(), test_stage=True,
    )
    rejected = l7.submit_order(
        "ABC", "SELL", 1, product_type="OPTION", limit_price=2.0,
        option_symbol="ABC  260918C00100000", instruction="SELL_TO_OPEN",
        legs=[{"instruction": "SELL_TO_OPEN", "quantity": 1,
               "instrument": {"symbol": "ABC  260918C00100000", "assetType": "OPTION"}}],
        options_validation=_validated_option_metadata(), sector="Technology",
    )
    assert rejected.status == "REJECTED"
    assert "Unpaired SELL_TO_OPEN" in rejected.reason


def test_allocation_consolidates_option_overlay_cap():
    engine = AllocationEngine(rules=AllocationRules(), nav=1_000_000)
    first = PositionAllocation("AAA", BucketType.OPTIONS_IG, "OPTION", .10, 100_000, 1, 1, "BUY", "NORMAL")
    second = PositionAllocation("BBB", BucketType.OPTIONS_HY, "OPTION", .10, 100_000, 1, 1, "BUY", "HY")
    third = PositionAllocation("CCC", BucketType.OPTIONS_DISTRESSED, "OPTION", .03, 30_000, 1, 1, "BUY", "DISTRESSED")
    assert engine._fits_in_bucket(first)
    engine._apply_utilization(first, 1_000_000)
    assert engine._fits_in_bucket(second)
    engine._apply_utilization(second, 1_000_000)
    assert not engine._fits_in_bucket(third)


def test_option_sizer_does_not_override_candidate_cap_with_minimum(monkeypatch):
    import engine.execution.options_engine as options_module

    sizer = OptionsSizer()
    monkeypatch.setattr(sizer.bs, "call_price", lambda *args: 100.0)
    monkeypatch.setattr(options_module, "monte_carlo_option_price", lambda **kwargs: (100.0, 0.0))
    sizer.MC_SIMS = 10
    sizer.MC_STEPS = 2
    result = sizer.size_option(
        spot=100, strike=100, expiry_days=30, vol=.2, is_call=True,
        nav=1_000, budget_dollars=1_000, market_price=1.0,
    )
    assert result["rejected"]
    assert result["contracts"] == 0
    assert "per-candidate cap" in result["reject_reason"]


def test_portfolio_option_allocator_shares_caps_and_allows_one_contract():
    allocator = OptionOverlayAllocator(
        nav=100_000,
        existing_notional=22_700,  # $157 overlay room remains
        existing_initial_margin=7_900,  # $100 margin room remains
        existing_delta_dollars=19_000,  # $1,000 delta room remains
    )
    assert allocator.allocate(5, contract_cost=150, delta_dollars_per_contract=500, initial_margin_per_contract=90) == 1
    # The first reservation consumes the remaining overlay room; independent
    # candidate budgeting must not allow the next name to reuse it.
    assert allocator.allocate(1, contract_cost=150, delta_dollars_per_contract=500, initial_margin_per_contract=90) == 0
