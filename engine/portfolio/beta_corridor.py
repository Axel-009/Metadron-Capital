"""Beta Corridor Engine — Dataset 1 integration.
Manages portfolio beta within the 7%-12% return corridor.
Alpha extracted through IG/Fallen Angel names or RV mispricing.
Uses direct Schwab market data. Futures execution remains on PaperBroker.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("metadron.decision.beta_corridor")

from ..data.openbb_data import get_market_stats, get_adj_close

try:
    from ..utils.money import D, money, to_float, safe_div
except ImportError:
    from decimal import Decimal as D  # type: ignore
    def money(v): return round(float(v), 2)  # type: ignore
    def to_float(v): return float(v)  # type: ignore
    def safe_div(n, d, default=0): return n / d if d != 0 else default  # type: ignore

# ---- Configuration --------------------------------------------------------
ALPHA = 0.02                  # 2% secular alpha headstart
R_LOW = 0.07                  # Gamma corridor lower bound
R_HIGH = 0.12                 # Gamma corridor upper bound
BETA_MAX = 2.0                # Full throttle cap
BETA_INV = -0.136             # Strategic hedge floor
EXECUTION_MULTIPLIER = 4.7    # Thesis scaling factor
MIN_TRADE_THRESHOLD = 0.05    # Gamma throttle
VOL_STANDARD = 0.15           # Thesis standard vol (15%)
VOL_PERCENTILE_LOW = 25
VOL_PERCENTILE_ELEVATED = 75
VOL_PERCENTILE_CRISIS = 95


@dataclass
class BetaState:
    """Current beta corridor state."""
    current_beta: float = 0.0
    target_beta: float = 0.0
    Rm: float = 0.0
    sigma_m: float = 0.15
    vol_adjustment: float = 1.0
    base_beta: float = 0.0
    regime_beta_cap: float = BETA_MAX
    corridor_position: str = "NEUTRAL"  # BELOW / WITHIN / ABOVE
    last_price: float = 0.0
    timestamp: str = ""


@dataclass
class BetaAction:
    """Rebalance action from the beta engine."""
    action: str = "HOLD"      # BUY / SELL / HOLD
    quantity: int = 0
    instrument: str = "SPY"   # trade log uses SPY as proxy
    target_beta: float = 0.0
    current_beta: float = 0.0
    reason: str = ""


# ---- Vol Regime Classifier ------------------------------------------------
class VolRegime(Enum):
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"


class VolRegimeClassifier:
    """Classify vol regime using realized vol percentile (20d vs 252d).
    Adjusts beta targets to scale exposure inversely with vol stress."""

    REGIME_MULTIPLIERS = {
        VolRegime.LOW_VOL: 1.20, VolRegime.NORMAL: 1.00,
        VolRegime.ELEVATED: 0.65, VolRegime.CRISIS: 0.30,
    }
    REGIME_CAPS = {
        VolRegime.LOW_VOL: BETA_MAX, VolRegime.NORMAL: BETA_MAX,
        VolRegime.ELEVATED: 1.0, VolRegime.CRISIS: 0.3,
    }

    def __init__(self, short_window: int = 20, long_window: int = 252):
        self.short_window = short_window
        self.long_window = long_window
        self._regime_history: list[tuple[str, VolRegime, float]] = []

    def classify(self, returns: np.ndarray) -> tuple[VolRegime, float]:
        """Return (VolRegime, vol_percentile) from daily log returns."""
        if len(returns) < self.short_window:
            return VolRegime.NORMAL, 50.0
        short_vol = np.std(returns[-self.short_window:]) * np.sqrt(252)
        if len(returns) >= self.long_window:
            rolling_vols = np.array([
                np.std(returns[i - self.short_window:i]) * np.sqrt(252)
                for i in range(self.short_window, min(len(returns), self.long_window) + 1)
            ])
            percentile = float(
                np.searchsorted(np.sort(rolling_vols), short_vol) / len(rolling_vols) * 100.0
            )
        else:
            percentile = 50.0
        regime = self._percentile_to_regime(percentile)
        self._regime_history.append((datetime.now().isoformat(), regime, percentile))
        return regime, percentile

    @staticmethod
    def _percentile_to_regime(pct: float) -> VolRegime:
        if pct < VOL_PERCENTILE_LOW:
            return VolRegime.LOW_VOL
        if pct < VOL_PERCENTILE_ELEVATED:
            return VolRegime.NORMAL
        if pct < VOL_PERCENTILE_CRISIS:
            return VolRegime.ELEVATED
        return VolRegime.CRISIS

    def get_beta_multiplier(self, regime: VolRegime) -> float:
        return self.REGIME_MULTIPLIERS.get(regime, 1.0)

    def get_beta_cap(self, regime: VolRegime) -> float:
        return self.REGIME_CAPS.get(regime, BETA_MAX)

    def get_history(self) -> list[tuple[str, VolRegime, float]]:
        return list(self._regime_history)


# ---- Beta Smoothing -------------------------------------------------------
class BetaSmoothing:
    """EMA smoothing, anti-whipsaw filter, rate limiter, and Kalman estimation."""

    def __init__(self, ema_alpha: float = 0.3, whipsaw_threshold: float = 0.02,
                 max_shift_per_cycle: float = 0.25, kalman_process_noise: float = 0.01,
                 kalman_measurement_noise: float = 0.05):
        self.ema_alpha = ema_alpha
        self.whipsaw_threshold = whipsaw_threshold
        self.max_shift_per_cycle = max_shift_per_cycle
        self._ema_state: Optional[float] = None
        self._kalman_estimate: Optional[float] = None
        self._kalman_variance: float = 1.0
        self._process_noise = kalman_process_noise
        self._measurement_noise = kalman_measurement_noise
        self._prev_target: Optional[float] = None

    def smooth(self, raw_target: float, current_beta: float) -> float:
        """Full pipeline: EMA -> anti-whipsaw -> rate limiter -> Kalman."""
        # EMA
        if self._ema_state is None:
            self._ema_state = raw_target
        else:
            self._ema_state = self.ema_alpha * raw_target + (1 - self.ema_alpha) * self._ema_state
        smoothed = self._ema_state
        # Anti-whipsaw
        if self._prev_target is not None:
            prev_dir = np.sign(self._prev_target - current_beta)
            new_dir = np.sign(smoothed - current_beta)
            if prev_dir != 0 and new_dir != 0 and prev_dir != new_dir:
                if abs(smoothed - current_beta) < self.whipsaw_threshold:
                    smoothed = current_beta
        self._prev_target = smoothed
        # Rate limiter
        delta = smoothed - current_beta
        if abs(delta) > self.max_shift_per_cycle:
            smoothed = current_beta + np.sign(delta) * self.max_shift_per_cycle
        # Kalman
        return self._kalman_update(smoothed)

    def _kalman_update(self, measurement: float) -> float:
        if self._kalman_estimate is None:
            self._kalman_estimate = measurement
            self._kalman_variance = self._measurement_noise
            return measurement
        pred_var = self._kalman_variance + self._process_noise
        gain = pred_var / (pred_var + self._measurement_noise)
        self._kalman_estimate += gain * (measurement - self._kalman_estimate)
        self._kalman_variance = (1.0 - gain) * pred_var
        return self._kalman_estimate

    def reset(self):
        self._ema_state = None
        self._kalman_estimate = None
        self._kalman_variance = 1.0
        self._prev_target = None


# ---- Corridor Analytics ---------------------------------------------------
class CorridorAnalytics:
    """Track corridor zone stats, breach frequency, mean-reversion, entry/exit."""

    def __init__(self, r_low: float = R_LOW, r_high: float = R_HIGH):
        self.r_low, self.r_high = r_low, r_high
        self._zone_ticks: dict[str, int] = {"BELOW": 0, "WITHIN": 0, "ABOVE": 0}
        self._total_ticks: int = 0
        self._breach_events: list[dict] = []
        self._rm_history: list[float] = []
        self._zone_history: list[str] = []

    def record(self, Rm: float, zone: str):
        self._rm_history.append(Rm)
        self._zone_history.append(zone)
        self._zone_ticks[zone] = self._zone_ticks.get(zone, 0) + 1
        self._total_ticks += 1
        if len(self._zone_history) >= 2:
            if self._zone_history[-2] == "WITHIN" and zone != "WITHIN":
                self._breach_events.append({
                    "tick": self._total_ticks, "direction": zone,
                    "Rm": Rm, "timestamp": datetime.now().isoformat(),
                })

    def zone_time_fractions(self) -> dict[str, float]:
        if self._total_ticks == 0:
            return {"BELOW": 0.0, "WITHIN": 0.0, "ABOVE": 0.0}
        return {z: c / self._total_ticks for z, c in self._zone_ticks.items()}

    def breach_frequency(self, lookback: Optional[int] = None) -> float:
        if self._total_ticks == 0:
            return 0.0
        if lookback is None:
            return len(self._breach_events) / self._total_ticks
        recent = [b for b in self._breach_events if b["tick"] > self._total_ticks - lookback]
        return len(recent) / min(lookback, self._total_ticks)

    def mean_reversion_probability(self, window: int = 20) -> float:
        """Empirical prob that Rm reverts to WITHIN after breaching."""
        if not self._breach_events or len(self._zone_history) < 2:
            return 0.5
        reversions, checked = 0, 0
        for ev in self._breach_events:
            tick = ev["tick"] - 1
            if tick + window > len(self._zone_history):
                continue
            checked += 1
            if "WITHIN" in self._zone_history[tick:tick + window]:
                reversions += 1
        return reversions / checked if checked else 0.5

    def optimal_entry_exit(self) -> dict:
        """Mean Rm at BELOW->WITHIN (entry) and WITHIN->ABOVE (exit) transitions."""
        entries, exits = [], []
        for i in range(1, len(self._zone_history)):
            prev, curr = self._zone_history[i - 1], self._zone_history[i]
            if prev == "BELOW" and curr == "WITHIN":
                entries.append(self._rm_history[i])
            elif prev == "WITHIN" and curr == "ABOVE":
                exits.append(self._rm_history[i])
        return {
            "optimal_entry_Rm": float(np.mean(entries)) if entries else self.r_low,
            "optimal_exit_Rm": float(np.mean(exits)) if exits else self.r_high,
            "entry_observations": len(entries), "exit_observations": len(exits),
        }

    def dynamic_corridor_width(self, base_width: Optional[float] = None) -> tuple[float, float]:
        """Widen corridor when Rm is noisy, tighten when stable."""
        if base_width is None:
            base_width = self.r_high - self.r_low
        if len(self._rm_history) < 10:
            return self.r_low, self.r_high
        recent = np.array(self._rm_history[-60:])
        vol_ratio = np.std(recent) / max(base_width, 1e-6)
        width_factor = np.clip(1.0 + (vol_ratio - 0.3) * 0.5, 0.8, 1.5)
        mid = (self.r_low + self.r_high) / 2.0
        hw = base_width * width_factor / 2.0
        return mid - hw, mid + hw

    def summary(self) -> dict:
        return {
            "zone_fractions": self.zone_time_fractions(),
            "breach_frequency": self.breach_frequency(),
            "mean_reversion_prob": self.mean_reversion_probability(),
            "optimal_levels": self.optimal_entry_exit(),
            "dynamic_bounds": self.dynamic_corridor_width(),
            "total_observations": self._total_ticks,
            "total_breaches": len(self._breach_events),
        }


# ---- Beta Corridor Engine -------------------------------------------------
class BetaCorridor:
    """Manages portfolio beta within the 7-12% return corridor.
    trade log mode: uses SPY as the beta instrument proxy."""

    def __init__(self, nav: float = 1_000_000, alpha: float = ALPHA,
                 r_low: float = R_LOW, r_high: float = R_HIGH,
                 beta_max: float = BETA_MAX, beta_inv: float = BETA_INV,
                 execution_multiplier: float = EXECUTION_MULTIPLIER):
        self.nav = nav
        self.alpha = alpha
        self.r_low = r_low
        self.r_high = r_high
        self.beta_max = beta_max
        self.beta_inv = beta_inv
        self.execution_multiplier = execution_multiplier
        self.current_beta = 0.0
        self._history: list[BetaState] = []
        # Enhanced components
        self._vol_classifier = VolRegimeClassifier()
        self._smoother = BetaSmoothing()
        self._analytics = CorridorAnalytics(r_low=r_low, r_high=r_high)
        self._current_vol_regime: VolRegime = VolRegime.NORMAL
        self._current_vol_percentile: float = 50.0

    def update_market_stats(self) -> dict:
        """Refresh market drift and vol from OpenBB."""
        stats = get_market_stats(benchmark="^GSPC", lookback_years=1)
        return stats

    def _vol_regime_adjustment(self, returns: Optional[np.ndarray] = None) -> tuple[float, float]:
        """Return (beta_multiplier, regime_cap) based on vol regime classification."""
        if returns is None:
            try:
                _2y_ago = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
                prices = get_adj_close("^GSPC", start=_2y_ago)
                if prices is not None and len(prices) > 30:
                    returns = np.diff(np.log(prices.values.flatten()))
                else:
                    return 1.0, self.beta_max
            except Exception:
                return 1.0, self.beta_max
        regime, pct = self._vol_classifier.classify(returns)
        self._current_vol_regime = regime
        self._current_vol_percentile = pct
        return self._vol_classifier.get_beta_multiplier(regime), self._vol_classifier.get_beta_cap(regime)

    def _smooth_beta_transition(self, raw_target: float) -> float:
        """Apply smoothing pipeline to raw beta target."""
        return self._smoother.smooth(raw_target, self.current_beta)

    def calculate_target_beta(self, Rm: float, sigma_m: float,
                              regime_beta_cap: Optional[float] = None,
                              market_returns: Optional[np.ndarray] = None,
                              apply_smoothing: bool = False) -> BetaState:
        """Convert market drift into target exposure.
        Includes vol-adjustment (Section VI). Enhanced with vol-regime integration."""
        state = BetaState()
        state.Rm = Rm
        state.sigma_m = sigma_m
        state.last_price = 0.0
        state.timestamp = datetime.now().isoformat()

        # 1. Base linear interpolation
        if Rm <= self.r_low:
            base_beta = -0.029
        elif Rm >= self.r_high:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (self.r_high - self.r_low)
            base_beta = -0.029 + slope * (Rm - self.r_low)
        state.base_beta = base_beta

        # 2. Vol normalization (thesis standard: 15% vol)
        vol_adj = VOL_STANDARD / max(sigma_m, 0.05)
        state.vol_adjustment = vol_adj
        target = base_beta * self.execution_multiplier * vol_adj

        # 2b. Vol regime adjustment
        vol_mult, vol_cap = self._vol_regime_adjustment(returns=market_returns)
        target *= vol_mult

        # 3. Apply caps
        effective_cap = regime_beta_cap if regime_beta_cap is not None else self.beta_max
        effective_cap = min(effective_cap, vol_cap)
        state.regime_beta_cap = effective_cap
        target = max(self.beta_inv, min(effective_cap, target))

        # 3b. Optional smoothing (default off for backward compat)
        if apply_smoothing:
            target = self._smooth_beta_transition(target)
            target = max(self.beta_inv, min(effective_cap, target))

        state.target_beta = target
        state.current_beta = self.current_beta

        # Corridor position
        if Rm < self.r_low:
            state.corridor_position = "BELOW"
        elif Rm > self.r_high:
            state.corridor_position = "ABOVE"
        else:
            state.corridor_position = "WITHIN"

        self._analytics.record(Rm, state.corridor_position)
        self._history.append(state)
        return state

    def compute_rebalance(self, state: BetaState, instrument_price: float = 500.0,
                          contract_multiplier: float = 1.0) -> BetaAction:
        """Translate target beta into trade log action (SPY proxy)."""
        action = BetaAction()
        action.target_beta = state.target_beta
        action.current_beta = self.current_beta

        target_notional = state.target_beta * self.nav
        current_notional = self.current_beta * self.nav
        delta_notional = target_notional - current_notional

        contract_value = instrument_price * contract_multiplier
        if contract_value <= 0:
            action.action = "HOLD"
            return action

        qty_diff = int(delta_notional / contract_value)

        # Gamma throttle: only rebalance if shift > threshold
        if abs(state.target_beta - self.current_beta) > MIN_TRADE_THRESHOLD and qty_diff != 0:
            if qty_diff > 0:
                action.action = "BUY"
                action.quantity = abs(qty_diff)
            else:
                action.action = "SELL"
                action.quantity = abs(qty_diff)
            action.instrument = "SPY"
            action.reason = (
                f"Beta shift {self.current_beta:.3f} → {state.target_beta:.3f} "
                f"| Rm={state.Rm:.2%} σ={state.sigma_m:.2%} "
                f"| corridor={state.corridor_position}"
            )
            self.current_beta = state.target_beta
        else:
            action.action = "HOLD"
            action.reason = f"Delta {abs(state.target_beta - self.current_beta):.3f} < threshold {MIN_TRADE_THRESHOLD}"

        return action

    def run_cycle(self, regime_beta_cap: Optional[float] = None) -> tuple[BetaState, BetaAction]:
        """Full cycle: fetch data -> compute beta -> generate action."""
        stats = self.update_market_stats()
        if stats.get("error") or stats.get("Rm") is None:
            # No market data available — return safe HOLD state
            state = BetaState(
                current_beta=self.current_beta,
                target_beta=self.current_beta,
                Rm=0.0, sigma_m=0.15,
                vol_adjustment=1.0,
                base_beta=self.current_beta,
                regime_beta_cap=regime_beta_cap,
                corridor_position="UNKNOWN",
                last_price=0.0,
            )
            action = BetaAction(action="HOLD", quantity=0, instrument="SPY",
                                target_beta=self.current_beta, current_beta=self.current_beta,
                                reason=f"No market data: {stats.get('error', 'unknown')}")
            return state, action
        state = self.calculate_target_beta(
            Rm=stats["Rm"], sigma_m=stats["sigma_m"], regime_beta_cap=regime_beta_cap,
        )
        state.last_price = stats["last_price"]
        action = self.compute_rebalance(state, instrument_price=stats["last_price"])
        return state, action

    def get_history(self) -> list[BetaState]:
        return list(self._history)

    def update_nav(self, new_nav: float):
        """Update NAV for dynamic sizing."""
        self.nav = new_nav

    # ---- Enhanced methods ----

    def get_corridor_analytics(self) -> dict:
        """Return full corridor analytics dict with vol regime info."""
        analytics = self._analytics.summary()
        analytics["vol_regime"] = self._current_vol_regime.value
        analytics["vol_percentile"] = self._current_vol_percentile
        analytics["current_beta"] = self.current_beta
        analytics["history_length"] = len(self._history)
        return analytics

    def get_beta_history_df(self) -> pd.DataFrame:
        """Return beta history as a DataFrame indexed by timestamp."""
        if not self._history:
            return pd.DataFrame()
        records = [{
            "timestamp": s.timestamp, "current_beta": s.current_beta,
            "target_beta": s.target_beta, "Rm": s.Rm, "sigma_m": s.sigma_m,
            "vol_adjustment": s.vol_adjustment, "base_beta": s.base_beta,
            "regime_beta_cap": s.regime_beta_cap,
            "corridor_position": s.corridor_position, "last_price": s.last_price,
        } for s in self._history]
        df = pd.DataFrame(records)
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        except Exception:
            pass
        return df

    def calculate_hedge_ratio(self, portfolio_beta: Optional[float] = None,
                              hedge_instrument_beta: float = 1.0,
                              target_beta: float = 0.0) -> dict:
        """Calculate hedge ratio to move portfolio beta to target."""
        if portfolio_beta is None:
            portfolio_beta = self.current_beta
        if abs(hedge_instrument_beta) < 1e-8:
            return {"hedge_ratio": 0.0, "notional_hedge": 0.0, "hedge_shares": 0}
        beta_gap = portfolio_beta - target_beta
        hedge_ratio = beta_gap / hedge_instrument_beta
        notional_hedge = hedge_ratio * self.nav
        last_price = self._history[-1].last_price if self._history else 500.0
        hedge_shares = int(notional_hedge / max(last_price, 1.0))
        return {
            "hedge_ratio": round(hedge_ratio, 4),
            "notional_hedge": round(notional_hedge, 2),
            "hedge_shares": hedge_shares,
            "portfolio_beta": portfolio_beta,
            "target_beta": target_beta,
            "instrument_beta": hedge_instrument_beta,
        }

    def stress_test_beta(self, scenarios: Optional[dict[str, dict]] = None) -> pd.DataFrame:
        """Test beta under stress scenarios. Returns DataFrame indexed by scenario name."""
        if scenarios is None:
            scenarios = {
                "crash_2008":     {"Rm": -0.38, "sigma_m": 0.60},
                "covid_2020":     {"Rm": -0.15, "sigma_m": 0.55},
                "high_vol_flat":  {"Rm":  0.05, "sigma_m": 0.35},
                "low_vol_rally":  {"Rm":  0.18, "sigma_m": 0.09},
                "stagflation":    {"Rm":  0.02, "sigma_m": 0.22},
                "goldilocks":     {"Rm":  0.10, "sigma_m": 0.12},
                "normal":         {"Rm":  0.09, "sigma_m": 0.15},
                "deep_recession": {"Rm": -0.05, "sigma_m": 0.40},
            }
        results = []
        for name, params in scenarios.items():
            Rm = params.get("Rm", 0.09)
            sigma_m = params.get("sigma_m", 0.15)
            # Compute without mutating real state
            if Rm <= self.r_low:
                base_beta = -0.029
            elif Rm >= self.r_high:
                base_beta = 0.425
            else:
                slope = (0.425 - (-0.029)) / (self.r_high - self.r_low)
                base_beta = -0.029 + slope * (Rm - self.r_low)
            vol_adj = VOL_STANDARD / max(sigma_m, 0.05)
            target = base_beta * self.execution_multiplier * vol_adj
            target = max(self.beta_inv, min(self.beta_max, target))
            position = "BELOW" if Rm < self.r_low else ("ABOVE" if Rm > self.r_high else "WITHIN")
            results.append({
                "scenario": name, "Rm": Rm, "sigma_m": sigma_m,
                "base_beta": round(base_beta, 4), "vol_adjustment": round(vol_adj, 4),
                "target_beta": round(target, 4), "corridor_position": position,
                "notional_exposure": round(target * self.nav, 2),
            })
        return pd.DataFrame(results).set_index("scenario")
