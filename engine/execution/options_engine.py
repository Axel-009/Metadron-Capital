"""
Metadron Capital — Options & Sophisticated Securities Engine
=============================================================

Handles the predictive/options allocation layer:
  - Black-Scholes pricing with full Greeks
  - Volatility surface construction and anomaly detection
  - Strategy construction (spreads, condors, butterflies, etc.)
  - Convexity / tail-hedge management
  - Options portfolio analytics with aggregate Greeks and P&L attribution
  - Predictive signals derived from options data

The engine does not submit directly. It emits validated, order-ready candidates
for the mandatory AllocationEngine -> OptionsEngine -> L7 -> Schwab path.

All math is pure numpy (no scipy). Normal CDF is implemented via an
Abramowitz-and-Stegun rational approximation of the error function.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# External imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    from ..data.universe_engine import SECTOR_ETFS
    from ..signals.metadron_cube import CubeOutput, REGIME_PARAMS
    from ..signals.macro_engine import CubeRegime
except ImportError:
    SECTOR_ETFS: List[str] = []

    @dataclass
    class CubeOutput:
        regime: str = "NORMAL"
        confidence: float = 0.5
        signals: dict = field(default_factory=dict)

    REGIME_PARAMS: dict = {
        "RISK_ON": {"vol_target": 0.16},
        "NORMAL": {"vol_target": 0.12},
        "CAUTIOUS": {"vol_target": 0.08},
        "STRESS": {"vol_target": 0.05},
        "CRASH": {"vol_target": 0.03},
    }

    class CubeRegime(str, Enum):
        RISK_ON = "RISK_ON"
        NORMAL = "NORMAL"
        CAUTIOUS = "CAUTIOUS"
        STRESS = "STRESS"
        CRASH = "CRASH"

logger = logging.getLogger("metadron.execution.options")

# ===================================================================
#  Mathematical primitives (no scipy)
# ===================================================================

def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Cumulative distribution function of the standard normal.

    Uses the Abramowitz & Stegun approximation (formula 7.1.26) of the
    complementary error function, accurate to ~1.5e-7.
    """
    # erf approximation constants
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = np.where(x < 0, -1.0, 1.0)
    x_abs = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x_abs)
    erf_approx = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
        -x_abs * x_abs
    )
    return 0.5 * (1.0 + sign * erf_approx)


def _norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal probability density function."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


# ===================================================================
#  1. Black-Scholes Model  (~100 lines)
# ===================================================================

class BlackScholesModel:
    """
    Closed-form Black-Scholes-Merton pricing and Greeks.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years  (must be > 0)
    r : risk-free rate (annualised, continuous compounding)
    sigma : volatility (annualised)

    All methods are static so callers need not instantiate.
    """

    # -- helper d1 / d2 ---------------------------------------------------

    @staticmethod
    def _d1d2(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> Tuple[float, float]:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # -- pricing -----------------------------------------------------------

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """European call price via Black-Scholes formula."""
        if T <= 0:
            return max(S - K, 0.0)
        d1, d2 = BlackScholesModel._d1d2(S, K, T, r, sigma)
        return float(S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2))

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """European put price via Black-Scholes formula (put-call parity)."""
        if T <= 0:
            return max(K - S, 0.0)
        d1, d2 = BlackScholesModel._d1d2(S, K, T, r, sigma)
        return float(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))

    # -- Greeks ------------------------------------------------------------

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
        """Option delta: dV/dS."""
        if T <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1, _ = BlackScholesModel._d1d2(S, K, T, r, sigma)
        if is_call:
            return float(_norm_cdf(d1))
        return float(_norm_cdf(d1) - 1.0)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Option gamma: d^2V/dS^2.  Same for calls and puts."""
        if T <= 0:
            return 0.0
        d1, _ = BlackScholesModel._d1d2(S, K, T, r, sigma)
        return float(_norm_pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def theta(
        S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True
    ) -> float:
        """Option theta: dV/dT  (per calendar day, negative convention)."""
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholesModel._d1d2(S, K, T, r, sigma)
        common = -(S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
        if is_call:
            val = common - r * K * np.exp(-r * T) * _norm_cdf(d2)
        else:
            val = common + r * K * np.exp(-r * T) * _norm_cdf(-d2)
        return float(val / 365.0)  # per calendar day

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Option vega: dV/d(sigma).  Same for calls and puts.  Per 1% vol move."""
        if T <= 0:
            return 0.0
        d1, _ = BlackScholesModel._d1d2(S, K, T, r, sigma)
        return float(S * _norm_pdf(d1) * np.sqrt(T) / 100.0)

    @staticmethod
    def rho(
        S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True
    ) -> float:
        """Option rho: dV/dr.  Per 1% rate move."""
        if T <= 0:
            return 0.0
        _, d2 = BlackScholesModel._d1d2(S, K, T, r, sigma)
        if is_call:
            return float(K * T * np.exp(-r * T) * _norm_cdf(d2) / 100.0)
        return float(-K * T * np.exp(-r * T) * _norm_cdf(-d2) / 100.0)

    # -- implied vol -------------------------------------------------------

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        is_call: bool = True,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """
        Newton-Raphson implied volatility solver.

        Returns annualised implied vol or NaN if convergence fails.
        """
        if T <= 0:
            return np.nan
        sigma = 0.25  # initial guess
        for _ in range(max_iter):
            if is_call:
                price = BlackScholesModel.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholesModel.put_price(S, K, T, r, sigma)
            diff = price - market_price
            v = BlackScholesModel.vega(S, K, T, r, sigma) * 100.0  # undo per-pct
            if abs(v) < 1e-12:
                return np.nan
            sigma -= diff / v
            if sigma <= 0:
                sigma = 1e-4
            if abs(diff) < tol:
                return float(sigma)
        return np.nan


# ===================================================================
#  2. Volatility Surface  (~120 lines)
# ===================================================================

@dataclass
class VolPoint:
    """A single point on the vol surface."""
    tenor_days: int
    delta_strike: float   # expressed as delta (0-1) or moneyness
    implied_vol: float


class VolatilitySurface:
    """
    Constructs an implied-volatility surface from VIX level, historical
    realised vol, and simple skew / term-structure heuristics.

    The model surface is used as an independent fair-value surface alongside
    quoted implied volatility from the direct Schwab option chain.
    """

    # Standard tenors (calendar days)
    TENORS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}

    def __init__(self, vix: float, hist_vol_30d: float, hist_vol_90d: float):
        """
        Parameters
        ----------
        vix : current VIX level (annualised %)
        hist_vol_30d : 30-day realised vol (annualised decimal, e.g. 0.15)
        hist_vol_90d : 90-day realised vol (annualised decimal)
        """
        self.vix = vix
        self.vix_decimal = vix / 100.0 if vix > 1.0 else vix
        self.hv30 = hist_vol_30d
        self.hv90 = hist_vol_90d
        self.surface: Dict[str, Dict[str, float]] = {}  # tenor -> {delta_label: iv}
        self._build()

    # -- construction -------------------------------------------------------

    def _build(self) -> None:
        """Build the vol surface grid."""
        base_atm = self.vix_decimal
        # Term structure: short-dated slightly higher if VIX elevated
        term_factors = self._term_structure_factors()
        for label, days in self.TENORS.items():
            atm = base_atm * term_factors[label]
            skew = self._skew_for_tenor(days)
            self.surface[label] = {
                "25d_put": atm + skew["put_25d"],
                "10d_put": atm + skew["put_10d"],
                "ATM": atm,
                "25d_call": atm + skew["call_25d"],
                "10d_call": atm + skew["call_10d"],
            }

    def _term_structure_factors(self) -> Dict[str, float]:
        """
        Ratio of each tenor's ATM vol to VIX.

        Normal: upward-sloping (short < long).
        Stressed: inverted (short > long).
        """
        vrp = self.vix_decimal - self.hv30  # vol risk premium
        if vrp > 0.08:
            # inverted / stressed
            return {"1M": 1.10, "3M": 1.02, "6M": 0.96, "1Y": 0.92}
        elif vrp > 0.03:
            return {"1M": 1.03, "3M": 1.00, "6M": 0.98, "1Y": 0.96}
        else:
            # normal upward-sloping
            return {"1M": 0.95, "3M": 1.00, "6M": 1.03, "1Y": 1.06}

    def _skew_for_tenor(self, days: int) -> Dict[str, float]:
        """
        Equity skew: OTM puts trade richer than OTM calls.
        Skew steepens for shorter tenors and in high-vol regimes.
        """
        base_skew = 0.04 * (self.vix_decimal / 0.20)  # scale to VIX
        decay = np.sqrt(30.0 / max(days, 1))  # steeper for short tenors
        return {
            "put_10d": base_skew * 2.0 * decay,
            "put_25d": base_skew * 1.0 * decay,
            "call_25d": -base_skew * 0.3 * decay,
            "call_10d": -base_skew * 0.5 * decay,
        }

    # -- queries ------------------------------------------------------------

    def get_atm_vol(self, tenor: str = "1M") -> float:
        """ATM implied vol for a given tenor."""
        return self.surface.get(tenor, {}).get("ATM", self.vix_decimal)

    def get_vol(self, tenor: str, delta_label: str) -> float:
        """Implied vol for a specific tenor and delta point."""
        return self.surface.get(tenor, {}).get(delta_label, self.vix_decimal)

    def skew_25d(self, tenor: str = "1M") -> float:
        """25-delta risk reversal: put IV minus call IV."""
        t = self.surface.get(tenor, {})
        return t.get("25d_put", 0) - t.get("25d_call", 0)

    def term_spread(self) -> float:
        """1Y ATM minus 1M ATM.  Negative = inverted."""
        return self.get_atm_vol("1Y") - self.get_atm_vol("1M")

    def detect_anomalies(self) -> List[str]:
        """Flag unusual surface conditions."""
        anomalies: List[str] = []
        if self.term_spread() < -0.02:
            anomalies.append("INVERTED_TERM_STRUCTURE")
        if self.skew_25d("1M") > 0.12:
            anomalies.append("EXTREME_SKEW_1M")
        vrp = self.vix_decimal - self.hv30
        if vrp > 0.10:
            anomalies.append("ELEVATED_VRP")
        if vrp < -0.03:
            anomalies.append("NEGATIVE_VRP")
        if self.vix_decimal > 0.35:
            anomalies.append("HIGH_VIX")
        return anomalies

    def interpolate_vol(self, tenor_days: int, moneyness: float) -> float:
        """
        Bilinear interpolation across tenor and moneyness.

        moneyness: K/S  (1.0 = ATM, <1 = OTM put, >1 = OTM call)
        """
        sorted_tenors = sorted(self.TENORS.items(), key=lambda x: x[1])
        tenor_labels = [t[0] for t in sorted_tenors]
        tenor_vals = np.array([t[1] for t in sorted_tenors], dtype=float)

        # clamp
        tenor_days = float(np.clip(tenor_days, tenor_vals[0], tenor_vals[-1]))

        # find bracketing tenors
        idx = int(np.searchsorted(tenor_vals, tenor_days, side="right"))
        idx = max(1, min(idx, len(tenor_vals) - 1))
        t0, t1 = tenor_labels[idx - 1], tenor_labels[idx]
        w = (tenor_days - tenor_vals[idx - 1]) / max(tenor_vals[idx] - tenor_vals[idx - 1], 1)

        # moneyness -> nearest delta label
        if moneyness <= 0.90:
            dlabel = "10d_put"
        elif moneyness <= 0.95:
            dlabel = "25d_put"
        elif moneyness <= 1.02:
            dlabel = "ATM"
        elif moneyness <= 1.05:
            dlabel = "25d_call"
        else:
            dlabel = "10d_call"

        v0 = self.surface[t0].get(dlabel, self.vix_decimal)
        v1 = self.surface[t1].get(dlabel, self.vix_decimal)
        return float(v0 * (1 - w) + v1 * w)


# ===================================================================
#  3. Options Strategy Builder  (~200 lines)
# ===================================================================

@dataclass
class StrategyLeg:
    """A single leg of a multi-leg options strategy."""
    option_type: str          # "call" or "put"
    strike: float
    expiry_days: int
    quantity: int             # positive = long, negative = short
    premium: float = 0.0     # per-contract premium (positive = cost)


@dataclass
class StrategyProfile:
    """Risk/reward profile of a strategy."""
    name: str
    legs: List[StrategyLeg]
    max_profit: float
    max_loss: float
    breakevens: List[float]
    probability_of_profit: float  # rough estimate
    net_premium: float            # negative = credit
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0


class OptionsStrategyBuilder:
    """
    Builds and evaluates multi-leg options strategies.

    Strategies are selected based on market regime and directional view.
    All pricing uses BlackScholesModel with a supplied vol surface.
    """

    BS = BlackScholesModel

    def __init__(self, spot: float, vol_surface: VolatilitySurface, risk_free: float = 0.05):
        self.S = spot
        self.vol = vol_surface
        self.r = risk_free

    # -- helpers -----------------------------------------------------------

    def _price(self, leg: StrategyLeg) -> float:
        T = leg.expiry_days / 365.0
        moneyness = leg.strike / self.S
        sigma = self.vol.interpolate_vol(leg.expiry_days, moneyness)
        if leg.option_type == "call":
            return self.BS.call_price(self.S, leg.strike, T, self.r, sigma)
        return self.BS.put_price(self.S, leg.strike, T, self.r, sigma)

    def _greeks(self, leg: StrategyLeg) -> Dict[str, float]:
        T = leg.expiry_days / 365.0
        moneyness = leg.strike / self.S
        sigma = self.vol.interpolate_vol(leg.expiry_days, moneyness)
        is_call = leg.option_type == "call"
        return {
            "delta": self.BS.delta(self.S, leg.strike, T, self.r, sigma, is_call) * leg.quantity,
            "gamma": self.BS.gamma(self.S, leg.strike, T, self.r, sigma) * leg.quantity,
            "theta": self.BS.theta(self.S, leg.strike, T, self.r, sigma, is_call) * leg.quantity,
            "vega": self.BS.vega(self.S, leg.strike, T, self.r, sigma) * leg.quantity,
        }

    def _build_profile(self, name: str, legs: List[StrategyLeg]) -> StrategyProfile:
        """Price all legs, compute payoff at expiry across a range of spots."""
        for leg in legs:
            leg.premium = self._price(leg) * leg.quantity

        net_premium = sum(leg.premium for leg in legs)

        # payoff grid at expiry
        spots = np.linspace(self.S * 0.5, self.S * 1.5, 1000)
        payoff = np.zeros_like(spots)
        for leg in legs:
            if leg.option_type == "call":
                intr = np.maximum(spots - leg.strike, 0)
            else:
                intr = np.maximum(leg.strike - spots, 0)
            payoff += intr * leg.quantity
        total_pnl = payoff - net_premium  # net_premium already has sign from quantity

        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))
        # cap infinite profit display
        if max_profit > self.S * 10:
            max_profit = float("inf")

        # breakevens: where PnL crosses zero
        sign_changes = np.where(np.diff(np.sign(total_pnl)))[0]
        breakevens = [float(spots[i]) for i in sign_changes]

        # rough probability of profit: fraction of spot range where PnL > 0
        # (assumes lognormal, very rough)
        prob_profit = float(np.mean(total_pnl > 0))

        # aggregate greeks
        agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in legs:
            g = self._greeks(leg)
            for k in agg:
                agg[k] += g[k]

        return StrategyProfile(
            name=name,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            probability_of_profit=prob_profit,
            net_premium=net_premium,
            net_delta=agg["delta"],
            net_gamma=agg["gamma"],
            net_theta=agg["theta"],
            net_vega=agg["vega"],
        )

    # -- strategy templates ------------------------------------------------

    def protective_put(self, expiry_days: int = 30, otm_pct: float = 0.05) -> StrategyProfile:
        """Buy a put to hedge downside.  OTM % below spot."""
        K = self.S * (1 - otm_pct)
        legs = [StrategyLeg("put", K, expiry_days, 1)]
        return self._build_profile("Protective Put", legs)

    def covered_call(self, expiry_days: int = 30, otm_pct: float = 0.05) -> StrategyProfile:
        """Sell a call against long stock (stock leg modelled as deep ITM call)."""
        K = self.S * (1 + otm_pct)
        legs = [StrategyLeg("call", K, expiry_days, -1)]
        return self._build_profile("Covered Call", legs)

    def bull_call_spread(self, expiry_days: int = 45, width_pct: float = 0.05) -> StrategyProfile:
        """Debit call spread: long lower, short higher."""
        K1 = self.S
        K2 = self.S * (1 + width_pct)
        legs = [
            StrategyLeg("call", K1, expiry_days, 1),
            StrategyLeg("call", K2, expiry_days, -1),
        ]
        return self._build_profile("Bull Call Spread", legs)

    def bear_put_spread(self, expiry_days: int = 45, width_pct: float = 0.05) -> StrategyProfile:
        """Debit put spread: long higher, short lower."""
        K1 = self.S
        K2 = self.S * (1 - width_pct)
        legs = [
            StrategyLeg("put", K1, expiry_days, 1),
            StrategyLeg("put", K2, expiry_days, -1),
        ]
        return self._build_profile("Bear Put Spread", legs)

    def bull_put_spread(self, expiry_days: int = 45, width_pct: float = 0.05) -> StrategyProfile:
        """Credit put spread: short higher put, long lower put."""
        K1 = self.S * (1 - width_pct * 0.5)
        K2 = self.S * (1 - width_pct * 1.5)
        legs = [
            StrategyLeg("put", K1, expiry_days, -1),
            StrategyLeg("put", K2, expiry_days, 1),
        ]
        return self._build_profile("Bull Put Spread", legs)

    def bear_call_spread(self, expiry_days: int = 45, width_pct: float = 0.05) -> StrategyProfile:
        """Credit call spread: short lower call, long higher call."""
        K1 = self.S * (1 + width_pct * 0.5)
        K2 = self.S * (1 + width_pct * 1.5)
        legs = [
            StrategyLeg("call", K1, expiry_days, -1),
            StrategyLeg("call", K2, expiry_days, 1),
        ]
        return self._build_profile("Bear Call Spread", legs)

    def iron_condor(self, expiry_days: int = 45, width_pct: float = 0.05, wing_pct: float = 0.10) -> StrategyProfile:
        """Short iron condor for range-bound markets."""
        legs = [
            StrategyLeg("put", self.S * (1 - wing_pct), expiry_days, 1),
            StrategyLeg("put", self.S * (1 - width_pct), expiry_days, -1),
            StrategyLeg("call", self.S * (1 + width_pct), expiry_days, -1),
            StrategyLeg("call", self.S * (1 + wing_pct), expiry_days, 1),
        ]
        return self._build_profile("Iron Condor", legs)

    def straddle(self, expiry_days: int = 30) -> StrategyProfile:
        """Long straddle: buy ATM call and put (volatility play)."""
        legs = [
            StrategyLeg("call", self.S, expiry_days, 1),
            StrategyLeg("put", self.S, expiry_days, 1),
        ]
        return self._build_profile("Straddle", legs)

    def strangle(self, expiry_days: int = 30, otm_pct: float = 0.05) -> StrategyProfile:
        """Long strangle: buy OTM call and put."""
        legs = [
            StrategyLeg("call", self.S * (1 + otm_pct), expiry_days, 1),
            StrategyLeg("put", self.S * (1 - otm_pct), expiry_days, 1),
        ]
        return self._build_profile("Strangle", legs)

    def short_strangle(self, expiry_days: int = 30, otm_pct: float = 0.05) -> StrategyProfile:
        """Short strangle: sell OTM call and put (theta harvesting).

        Profit if underlying stays within wings. Max(Θ+Γ) - c·V optimization.
        """
        legs = [
            StrategyLeg("call", self.S * (1 + otm_pct), expiry_days, -1),
            StrategyLeg("put", self.S * (1 - otm_pct), expiry_days, -1),
        ]
        return self._build_profile("Short Strangle", legs)

    def theta_gamma_optimize(self, expiry_days: int = 30, vega_cost: float = 0.5) -> dict:
        """Optimize max(Θ+Γ) - c·V for P4 sleeve allocation.

        Finds the best strike/expiry combination that maximizes theta+gamma
        income while penalizing vega exposure (c = vega cost parameter).

        Returns the optimal strategy profile and metrics.
        """
        best_score = -np.inf
        best_strategy = None
        best_metrics = {}

        # Scan OTM percentages for short strangles
        for otm in [0.03, 0.05, 0.07, 0.10, 0.15]:
            profile = self.short_strangle(expiry_days, otm)
            theta = profile.net_theta
            gamma = profile.net_gamma
            vega = abs(profile.net_vega)
            # max(Θ + Γ) - c·V
            score = (theta + gamma * 100) - vega_cost * vega
            if score > best_score:
                best_score = score
                best_strategy = profile
                best_metrics = {
                    "otm_pct": otm,
                    "theta": theta,
                    "gamma": gamma,
                    "vega": vega,
                    "score": score,
                }

        return {
            "strategy": best_strategy,
            "metrics": best_metrics,
            "formula": "max(Θ+Γ) - c·V",
            "vega_cost": vega_cost,
        }

    def collar(self, expiry_days: int = 30, put_otm: float = 0.05, call_otm: float = 0.05) -> StrategyProfile:
        """Collar: long put + short call (zero or low cost downside hedge)."""
        legs = [
            StrategyLeg("put", self.S * (1 - put_otm), expiry_days, 1),
            StrategyLeg("call", self.S * (1 + call_otm), expiry_days, -1),
        ]
        return self._build_profile("Collar", legs)

    def butterfly(self, expiry_days: int = 30, width_pct: float = 0.05) -> StrategyProfile:
        """Long butterfly: precise target play around ATM."""
        K_low = self.S * (1 - width_pct)
        K_mid = self.S
        K_high = self.S * (1 + width_pct)
        legs = [
            StrategyLeg("call", K_low, expiry_days, 1),
            StrategyLeg("call", K_mid, expiry_days, -2),
            StrategyLeg("call", K_high, expiry_days, 1),
        ]
        return self._build_profile("Butterfly", legs)

    def calendar_spread(self, near_days: int = 30, far_days: int = 60) -> StrategyProfile:
        """Calendar spread: short near-dated, long far-dated ATM call."""
        legs = [
            StrategyLeg("call", self.S, near_days, -1),
            StrategyLeg("call", self.S, far_days, 1),
        ]
        return self._build_profile("Calendar Spread", legs)

    # -- regime-based selection --------------------------------------------

    @staticmethod
    def regime_strategy_map() -> Dict[str, List[str]]:
        """Which strategies are preferred in each regime."""
        return {
            "RISK_ON": ["bull_call_spread", "bull_put_spread", "covered_call"],
            "NORMAL": ["iron_condor", "covered_call", "collar", "butterfly"],
            "CAUTIOUS": ["collar", "protective_put", "bear_call_spread"],
            "STRESS": ["protective_put", "bear_put_spread", "straddle"],
            "CRASH": ["protective_put", "straddle", "strangle"],
        }

    def select_for_regime(self, regime: str) -> List[StrategyProfile]:
        """Return evaluated strategies appropriate for the given regime."""
        mapping = self.regime_strategy_map()
        names = mapping.get(regime, mapping["NORMAL"])
        results: List[StrategyProfile] = []
        for name in names:
            method = getattr(self, name, None)
            if method:
                try:
                    results.append(method())
                except Exception as exc:
                    logger.warning("Strategy %s failed: %s", name, exc)
        return results


# ===================================================================
#  4. Convexity Hedge Manager  (~150 lines)
# ===================================================================

@dataclass
class HedgePosition:
    """A single hedge instrument in the tail-hedge book."""
    instrument: str       # e.g. "SPX_PUT_5pctOTM_30d"
    strike: float
    expiry_date: dt.date
    quantity: int
    entry_premium: float  # cost per contract
    current_value: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)


class ConvexityHedgeManager:
    """
    Manages tail-risk hedges:
      - SPX put ladder at various OTM levels
      - VIX call overlay
      - Roll scheduling and cost tracking
      - Regime-adaptive sizing
    """

    # OTM levels for the put ladder
    PUT_LADDER = [0.05, 0.10, 0.20]        # 5%, 10%, 20% OTM
    # Allocation weight to each rung (sums to 1)
    LADDER_WEIGHTS = [0.50, 0.30, 0.20]

    # Regime -> hedge budget as % of NAV (annualised)
    REGIME_HEDGE_BUDGET = {
        "RISK_ON": 0.003,    # 30 bps
        "NORMAL": 0.005,     # 50 bps
        "CAUTIOUS": 0.010,   # 100 bps
        "STRESS": 0.020,     # 200 bps
        "CRASH": 0.035,      # 350 bps
    }

    # Roll trigger: days to expiry
    ROLL_THRESHOLD_DAYS = 7

    def __init__(self, nav: float, regime: str = "NORMAL"):
        self.nav = nav
        self.regime = regime
        self.positions: List[HedgePosition] = []
        self.cumulative_cost: float = 0.0
        self.bs = BlackScholesModel

    # -- sizing -------------------------------------------------------------

    def hedge_budget(self) -> float:
        """Annualised hedge budget in dollars for current regime."""
        rate = self.REGIME_HEDGE_BUDGET.get(self.regime, 0.005)
        return self.nav * rate

    def monthly_budget(self) -> float:
        return self.hedge_budget() / 12.0

    # -- put ladder ---------------------------------------------------------

    def build_put_ladder(
        self,
        spot: float,
        vol: float,
        risk_free: float = 0.05,
        expiry_days: int = 30,
        target_date: Optional[dt.date] = None,
    ) -> List[HedgePosition]:
        """
        Construct a put ladder with the monthly hedge budget.

        Returns list of HedgePosition objects added.
        """
        budget = self.monthly_budget()
        T = expiry_days / 365.0
        exp_date = target_date or (dt.date.today() + dt.timedelta(days=expiry_days))

        new_positions: List[HedgePosition] = []
        for otm, weight in zip(self.PUT_LADDER, self.LADDER_WEIGHTS):
            K = spot * (1 - otm)
            premium = self.bs.put_price(spot, K, T, risk_free, vol)
            if premium < 0.01:
                premium = 0.01  # floor
            alloc = budget * weight
            qty = max(1, int(alloc / (premium * 100)))  # 100 multiplier
            pos = HedgePosition(
                instrument=f"SPX_PUT_{int(otm*100)}pctOTM_{expiry_days}d",
                strike=K,
                expiry_date=exp_date,
                quantity=qty,
                entry_premium=premium,
                current_value=premium,
                greeks={
                    "delta": self.bs.delta(spot, K, T, risk_free, vol, is_call=False) * qty,
                    "gamma": self.bs.gamma(spot, K, T, risk_free, vol) * qty,
                    "vega": self.bs.vega(spot, K, T, risk_free, vol) * qty,
                },
            )
            new_positions.append(pos)
            self.positions.append(pos)
            self.cumulative_cost += premium * qty * 100

        return new_positions

    # -- VIX overlay --------------------------------------------------------

    def build_vix_overlay(
        self,
        vix_spot: float,
        vol: float = 0.80,
        risk_free: float = 0.05,
        expiry_days: int = 30,
        budget_pct: float = 0.20,
    ) -> Optional[HedgePosition]:
        """
        Buy OTM VIX calls for crash convexity.
        budget_pct: fraction of monthly budget allocated to VIX calls.
        """
        budget = self.monthly_budget() * budget_pct
        K = vix_spot * 1.50  # 50% OTM VIX call
        T = expiry_days / 365.0
        premium = self.bs.call_price(vix_spot, K, T, risk_free, vol)
        if premium < 0.05:
            premium = 0.05
        qty = max(1, int(budget / (premium * 100)))
        exp_date = dt.date.today() + dt.timedelta(days=expiry_days)
        pos = HedgePosition(
            instrument=f"VIX_CALL_50pctOTM_{expiry_days}d",
            strike=K,
            expiry_date=exp_date,
            quantity=qty,
            entry_premium=premium,
            current_value=premium,
            greeks={
                "delta": self.bs.delta(vix_spot, K, T, risk_free, vol, is_call=True) * qty,
                "vega": self.bs.vega(vix_spot, K, T, risk_free, vol) * qty,
            },
        )
        self.positions.append(pos)
        self.cumulative_cost += premium * qty * 100
        return pos

    # -- roll management ----------------------------------------------------

    def positions_to_roll(self) -> List[HedgePosition]:
        """Identify positions approaching expiry that need rolling."""
        today = dt.date.today()
        to_roll: List[HedgePosition] = []
        for pos in self.positions:
            days_left = (pos.expiry_date - today).days
            if days_left <= self.ROLL_THRESHOLD_DAYS:
                to_roll.append(pos)
        return to_roll

    def roll_position(
        self,
        old: HedgePosition,
        spot: float,
        vol: float,
        new_expiry_days: int = 30,
        risk_free: float = 0.05,
    ) -> HedgePosition:
        """Roll an expiring position into a new tenor."""
        T = new_expiry_days / 365.0
        exp_date = dt.date.today() + dt.timedelta(days=new_expiry_days)
        premium = self.bs.put_price(spot, old.strike, T, risk_free, vol)
        new_pos = HedgePosition(
            instrument=old.instrument.rsplit("_", 1)[0] + f"_{new_expiry_days}d",
            strike=old.strike,
            expiry_date=exp_date,
            quantity=old.quantity,
            entry_premium=premium,
            current_value=premium,
        )
        # remove old, add new
        if old in self.positions:
            self.positions.remove(old)
        self.positions.append(new_pos)
        self.cumulative_cost += premium * old.quantity * 100
        return new_pos

    # -- reporting ----------------------------------------------------------

    def hedge_cost_drag(self) -> float:
        """Cumulative hedge cost as % of NAV (annualised drag)."""
        if self.nav <= 0:
            return 0.0
        return self.cumulative_cost / self.nav

    def total_hedge_delta(self) -> float:
        """Sum of delta across all hedge positions."""
        return sum(pos.greeks.get("delta", 0) for pos in self.positions)

    def update_regime(self, regime: str) -> None:
        self.regime = regime


# ===================================================================
#  5. Options Portfolio  (~100 lines)
# ===================================================================

@dataclass
class OptionPosition:
    """A single paper options position."""
    symbol: str
    option_type: str          # "call" or "put"
    strike: float
    expiry: dt.date
    quantity: int             # positive = long
    entry_price: float        # premium paid/received per contract
    current_price: float = 0.0
    underlying_price: float = 0.0
    iv: float = 0.20


class OptionsPortfolio:
    """
    Paper options portfolio tracker.

    Tracks positions, computes aggregate Greeks, decomposes P&L into
    delta / gamma / theta / vega components.
    """

    MARGIN_MULTIPLIER = 100  # standard US equity options

    def __init__(self):
        self.positions: List[OptionPosition] = []
        self.realised_pnl: float = 0.0
        self.bs = BlackScholesModel

    def add_position(self, pos: OptionPosition) -> None:
        self.positions.append(pos)

    def remove_expired(self) -> List[OptionPosition]:
        """Remove and settle expired positions."""
        today = dt.date.today()
        expired: List[OptionPosition] = []
        remaining: List[OptionPosition] = []
        for pos in self.positions:
            if pos.expiry <= today:
                # settle at intrinsic
                if pos.option_type == "call":
                    intrinsic = max(pos.underlying_price - pos.strike, 0)
                else:
                    intrinsic = max(pos.strike - pos.underlying_price, 0)
                pnl = (intrinsic - pos.entry_price) * pos.quantity * self.MARGIN_MULTIPLIER
                self.realised_pnl += pnl
                expired.append(pos)
            else:
                remaining.append(pos)
        self.positions = remaining
        return expired

    def _tte(self, pos: OptionPosition) -> float:
        """Time to expiry in years."""
        days = (pos.expiry - dt.date.today()).days
        return max(days, 1) / 365.0

    def aggregate_greeks(self, risk_free: float = 0.05) -> Dict[str, float]:
        """Portfolio-level Greeks."""
        agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for pos in self.positions:
            T = self._tte(pos)
            S = pos.underlying_price if pos.underlying_price > 0 else pos.strike
            is_call = pos.option_type == "call"
            mult = pos.quantity * self.MARGIN_MULTIPLIER
            agg["delta"] += self.bs.delta(S, pos.strike, T, risk_free, pos.iv, is_call) * mult
            agg["gamma"] += self.bs.gamma(S, pos.strike, T, risk_free, pos.iv) * mult
            agg["theta"] += self.bs.theta(S, pos.strike, T, risk_free, pos.iv, is_call) * mult
            agg["vega"] += self.bs.vega(S, pos.strike, T, risk_free, pos.iv) * mult
            agg["rho"] += self.bs.rho(S, pos.strike, T, risk_free, pos.iv, is_call) * mult
        return agg

    def pnl_attribution(self, spot_change: float, vol_change: float, risk_free: float = 0.05) -> Dict[str, float]:
        """
        Decompose P&L into Greek components for a given move.

        spot_change : absolute change in underlying
        vol_change  : absolute change in IV (e.g. +0.02 = +2 vol pts)
        """
        greeks = self.aggregate_greeks(risk_free)
        delta_pnl = greeks["delta"] * spot_change
        gamma_pnl = 0.5 * greeks["gamma"] * spot_change ** 2
        theta_pnl = greeks["theta"]  # one-day theta (already per calendar day)
        vega_pnl = greeks["vega"] * (vol_change * 100)  # vega is per 1%
        return {
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "vega_pnl": vega_pnl,
            "total_pnl": delta_pnl + gamma_pnl + theta_pnl + vega_pnl,
        }

    def margin_estimate(self, risk_free: float = 0.05) -> float:
        """
        Rough margin estimate: sum of |premium * qty * 100| for shorts,
        plus 20% of notional for naked shorts.
        """
        margin = 0.0
        for pos in self.positions:
            if pos.quantity < 0:
                notional = pos.underlying_price * abs(pos.quantity) * self.MARGIN_MULTIPLIER
                margin += abs(pos.current_price * pos.quantity * self.MARGIN_MULTIPLIER)
                margin += notional * 0.20
        return margin

    def unrealised_pnl(self) -> float:
        total = 0.0
        for pos in self.positions:
            total += (pos.current_price - pos.entry_price) * pos.quantity * self.MARGIN_MULTIPLIER
        return total


# ===================================================================
#  6. Predictive Options Signal  (~100 lines)
# ===================================================================

@dataclass
class OptionsSignal:
    """A signal derived from options-market data."""
    name: str
    direction: str     # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float    # 0..1
    description: str
    suggested_strategy: str = ""


class PredictiveOptionsSignal:
    """
    Generates trading signals from options-market observables:
      - Implied vs realised vol (vol risk premium)
      - Skew-based directional hints
      - Vol mean reversion
      - Earnings vol prediction
      - Put/call ratio
    """

    def __init__(self, vol_surface: VolatilitySurface):
        self.vol = vol_surface

    def vol_risk_premium_signal(self) -> OptionsSignal:
        """
        If implied >> realised, options are expensive -> sell vol.
        If implied << realised, options are cheap -> buy vol.
        """
        vrp = self.vol.vix_decimal - self.vol.hv30
        if vrp > 0.06:
            return OptionsSignal(
                "VRP", "NEUTRAL", min(vrp / 0.12, 1.0),
                f"High VRP ({vrp:.1%}): implied vol overpriced, sell vol strategies favoured",
                "iron_condor",
            )
        elif vrp < -0.02:
            return OptionsSignal(
                "VRP", "NEUTRAL", min(abs(vrp) / 0.06, 1.0),
                f"Negative VRP ({vrp:.1%}): implied vol cheap, buy vol strategies favoured",
                "straddle",
            )
        return OptionsSignal("VRP", "NEUTRAL", 0.2, "VRP in normal range", "")

    def skew_directional_signal(self) -> OptionsSignal:
        """
        Steep skew (puts much richer than calls) often precedes selloffs
        because smart money is hedging.
        """
        skew = self.vol.skew_25d("1M")
        if skew > 0.08:
            return OptionsSignal(
                "SKEW", "BEARISH", min(skew / 0.15, 1.0),
                f"Steep put skew ({skew:.1%}): heavy demand for downside protection",
                "protective_put",
            )
        elif skew < 0.02:
            return OptionsSignal(
                "SKEW", "BULLISH", 0.4,
                f"Flat skew ({skew:.1%}): complacency or bullish positioning",
                "bull_call_spread",
            )
        return OptionsSignal("SKEW", "NEUTRAL", 0.2, "Skew in normal range", "")

    def vol_mean_reversion_signal(self) -> OptionsSignal:
        """
        Vol tends to mean-revert.  Extremely high or low implied vol
        provides a signal.
        """
        iv = self.vol.get_atm_vol("1M")
        long_term_mean = 0.18  # approximate equity long-term vol
        z = (iv - long_term_mean) / 0.06  # rough z-score
        if z > 1.5:
            return OptionsSignal(
                "VOL_MR", "NEUTRAL", min(z / 3.0, 1.0),
                f"IV elevated ({iv:.1%}), mean-reversion likely -> sell vol",
                "covered_call",
            )
        elif z < -1.0:
            return OptionsSignal(
                "VOL_MR", "NEUTRAL", min(abs(z) / 3.0, 1.0),
                f"IV depressed ({iv:.1%}), mean-reversion likely -> buy vol",
                "strangle",
            )
        return OptionsSignal("VOL_MR", "NEUTRAL", 0.1, "IV near long-term average", "")

    def term_structure_signal(self) -> OptionsSignal:
        """
        Inverted vol term structure (short > long) is a warning sign.
        """
        spread = self.vol.term_spread()
        if spread < -0.03:
            return OptionsSignal(
                "TERM", "BEARISH", min(abs(spread) / 0.08, 1.0),
                f"Inverted term structure ({spread:.1%}): market pricing near-term risk",
                "protective_put",
            )
        elif spread > 0.04:
            return OptionsSignal(
                "TERM", "BULLISH", 0.3,
                f"Steep contango ({spread:.1%}): market expects calm near-term",
                "iron_condor",
            )
        return OptionsSignal("TERM", "NEUTRAL", 0.1, "Term structure normal", "")

    def put_call_ratio_signal(self, pcr: float) -> OptionsSignal:
        """
        Put/call volume ratio.
        >1.2 = heavy put buying (bearish or hedge-heavy)
        <0.6 = heavy call buying (bullish or complacent)
        """
        if pcr > 1.2:
            # contrarian: extreme put buying often marks bottoms
            return OptionsSignal(
                "PCR", "BULLISH", min((pcr - 1.0) / 0.5, 1.0),
                f"High P/C ratio ({pcr:.2f}): extreme hedging, contrarian bullish",
                "bull_put_spread",
            )
        elif pcr < 0.6:
            return OptionsSignal(
                "PCR", "BEARISH", min((0.8 - pcr) / 0.4, 1.0),
                f"Low P/C ratio ({pcr:.2f}): complacency, contrarian bearish",
                "bear_call_spread",
            )
        return OptionsSignal("PCR", "NEUTRAL", 0.1, f"P/C ratio normal ({pcr:.2f})", "")

    def earnings_vol_signal(self, symbol: str, implied_move_pct: float, hist_avg_move_pct: float) -> OptionsSignal:
        """
        Compare implied earnings move to historical average.
        If implied >> historical, options are overpricing the event.
        """
        ratio = implied_move_pct / max(hist_avg_move_pct, 0.01)
        if ratio > 1.3:
            return OptionsSignal(
                "EARNINGS", "NEUTRAL", min((ratio - 1.0) / 0.6, 1.0),
                f"{symbol}: implied move {implied_move_pct:.1%} vs hist {hist_avg_move_pct:.1%} "
                f"({ratio:.1f}x) -> sell earnings vol",
                "iron_condor",
            )
        elif ratio < 0.8:
            return OptionsSignal(
                "EARNINGS", "NEUTRAL", min((1.0 - ratio) / 0.4, 1.0),
                f"{symbol}: implied move {implied_move_pct:.1%} vs hist {hist_avg_move_pct:.1%} "
                f"({ratio:.1f}x) -> buy earnings vol",
                "straddle",
            )
        return OptionsSignal(
            "EARNINGS", "NEUTRAL", 0.1,
            f"{symbol}: implied move fairly priced ({ratio:.1f}x hist)", "",
        )

    def all_signals(self, pcr: float = 0.85) -> List[OptionsSignal]:
        """Return all non-earnings signals."""
        return [
            self.vol_risk_premium_signal(),
            self.skew_directional_signal(),
            self.vol_mean_reversion_signal(),
            self.term_structure_signal(),
            self.put_call_ratio_signal(pcr),
        ]


# ===================================================================
#  7. OptionsEngine — master orchestrator  (~150 lines)
# ===================================================================

@dataclass
class OptionOverlayAllocator:
    """Portfolio-level, contract-granular option budget allocator.

    The allocator starts with actual filled exposure plus accepted pending
    reservations.  It only grants a contract when overlay, initial margin,
    and delta-dollar room all remain.  It deliberately has no per-name
    minimum: one contract is valid when it is the only affordable amount.
    """
    nav: float
    existing_notional: float = 0.0
    existing_initial_margin: float = 0.0
    existing_delta_dollars: float = 0.0
    reserved_notional: float = 0.0
    reserved_initial_margin: float = 0.0
    reserved_delta_dollars: float = 0.0
    overlay_cap_pct: float = 0.22857142857
    initial_margin_cap_pct: float = 0.08
    delta_cap_pct: float = 0.20
    initial_margin_rate: float = 0.35

    def _remaining(self) -> Tuple[float, float, float]:
        return (
            max(0.0, self.nav * self.overlay_cap_pct - self.existing_notional - self.reserved_notional),
            max(0.0, self.nav * self.initial_margin_cap_pct - self.existing_initial_margin - self.reserved_initial_margin),
            max(0.0, self.nav * self.delta_cap_pct - self.existing_delta_dollars - self.reserved_delta_dollars),
        )

    def allocate(
        self, requested_contracts: int, contract_cost: float, delta_dollars_per_contract: float,
        initial_margin_per_contract: Optional[float] = None,
    ) -> int:
        """Reserve no more than the shared portfolio limits allow."""
        if self.nav <= 0 or requested_contracts <= 0 or contract_cost <= 0 or delta_dollars_per_contract < 0:
            return 0
        margin_per_contract = (
            initial_margin_per_contract
            if initial_margin_per_contract is not None
            else contract_cost * self.initial_margin_rate
        )
        if margin_per_contract < 0:
            return 0
        overlay_room, margin_room, delta_room = self._remaining()
        limits = [int(overlay_room // contract_cost)]
        if margin_per_contract > 0:
            limits.append(int(margin_room // margin_per_contract))
        if delta_dollars_per_contract > 0:
            limits.append(int(delta_room // delta_dollars_per_contract))
        contracts = min(int(requested_contracts), *limits)
        if contracts < 1:
            return 0
        self.reserved_notional += contracts * contract_cost
        self.reserved_initial_margin += contracts * margin_per_contract
        self.reserved_delta_dollars += contracts * delta_dollars_per_contract
        return contracts


def classify_single_leg_directional_candidate(
    option_type: str, instruction: str, option_symbol: str, legs: List[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    """Classify only the structure that is actually generated, else reject it."""
    if len(legs) != 1 or instruction != "BUY_TO_OPEN" or not option_symbol:
        return None
    leg = legs[0]
    instrument = leg.get("instrument", {}) if isinstance(leg, dict) else {}
    if not isinstance(instrument, dict):
        return None
    if leg.get("instruction") != "BUY_TO_OPEN" or instrument.get("symbol") != option_symbol:
        return None
    normalized_type = str(option_type).upper()
    if normalized_type == "CALL":
        kind = "SINGLE_LEG_LONG_CALL"
    elif normalized_type == "PUT":
        kind = "SINGLE_LEG_LONG_PUT"
    else:
        return None
    return {"kind": kind, "leg_count": 1, "generated_leg_count": 1}


class OptionsEngine:
    """
    Top-level orchestrator that ties together volatility surface, strategy
    selection, convexity hedging, portfolio tracking, and predictive signals.

    Paper mode only — no live execution is performed.

    Parameters
    ----------
    regime : current CubeRegime string (e.g. "NORMAL", "STRESS")
    nav    : total portfolio net asset value in USD
    """

    def __init__(
        self, regime: str = "NORMAL", nav: float = 1_000_000.0,
        market_data: Optional[object] = None,
    ):
        self.regime = regime
        self.nav = nav
        # Direct Schwab market-data client.  It is injected to keep tests
        # offline and to keep the options decision path broker-independent.
        self.market_data = market_data
        self.portfolio = OptionsPortfolio()
        self.hedge_mgr = ConvexityHedgeManager(nav, regime)
        self.vol_surface: Optional[VolatilitySurface] = None
        self.strategy_builder: Optional[OptionsStrategyBuilder] = None
        self.signal_gen: Optional[PredictiveOptionsSignal] = None
        self._last_spot: float = 0.0
        self._risk_free: float = 0.05
        self._portfolio_option_allocator: Optional[OptionOverlayAllocator] = None

    def begin_portfolio_option_allocation(
        self, allocation: Optional[Dict[str, object]] = None,
    ) -> None:
        """Start one scan-cycle budget shared by every underlying.

        Callers provide authoritative filled and pending exposures from their
        reconciliation layer.  The L7 risk engine independently repeats these
        checks immediately before acceptance, so an old scan can never bypass
        an intervening pending order.
        """
        allocation = allocation or {}
        self._portfolio_option_allocator = OptionOverlayAllocator(
            nav=self.nav,
            existing_notional=float(allocation.get("existing_option_notional", 0.0)),
            existing_initial_margin=float(allocation.get("existing_option_initial_margin", 0.0)),
            existing_delta_dollars=float(allocation.get("existing_option_delta_dollars", 0.0)),
            reserved_notional=float(allocation.get("reserved_option_notional", 0.0)),
            reserved_initial_margin=float(allocation.get("reserved_option_initial_margin", 0.0)),
            reserved_delta_dollars=float(allocation.get("reserved_option_delta_dollars", 0.0)),
            overlay_cap_pct=float(allocation.get("option_overlay_cap_pct", OptionOverlayAllocator.overlay_cap_pct)),
            initial_margin_cap_pct=float(allocation.get("option_initial_margin_cap_pct", OptionOverlayAllocator.initial_margin_cap_pct)),
            delta_cap_pct=float(allocation.get("option_delta_cap_pct", OptionOverlayAllocator.delta_cap_pct)),
            initial_margin_rate=float(allocation.get("option_initial_margin_rate", OptionOverlayAllocator.initial_margin_rate)),
        )

    # -- initialisation -----------------------------------------------------

    def update_market(
        self,
        spot: float,
        vix: float,
        hist_vol_30d: float,
        hist_vol_90d: float,
        risk_free: float = 0.05,
    ) -> None:
        """Refresh market data and rebuild derived objects."""
        self._last_spot = spot
        self._risk_free = risk_free
        self.vol_surface = VolatilitySurface(vix, hist_vol_30d, hist_vol_90d)
        self.strategy_builder = OptionsStrategyBuilder(spot, self.vol_surface, risk_free)
        self.signal_gen = PredictiveOptionsSignal(self.vol_surface)

    def update_regime(self, regime: str) -> None:
        self.regime = regime
        self.hedge_mgr.update_regime(regime)

    # -- vol surface export --------------------------------------------------

    def get_vol_surface(self) -> Dict[str, object]:
        """Return the implied-vol surface as a grid for the frontend.

        Builds from the normalized direct Schwab option chain when available.
        Falls back to the synthesised VolatilitySurface if a chain is unavailable.

        Returns
        -------
        dict  {"grid": [{"strike": float, "expiry": int, "iv": float}, ...],
               "source": "schwab" | "synthetic",
               "spot": float | None}
        """
        grid: List[Dict[str, object]] = []
        source = "synthetic"

        # --- Tier 1: normalized direct Schwab chain --------------------------
        if self.market_data is not None:
            try:
                chain = self.market_data.option_chain("SPY")
                spot = next((float(c.get("underlying_price", 0)) for c in chain if c.get("underlying_price")), 0.0)
                spot = spot or self._last_spot
                if spot > 0:
                    for expiry in [7, 30, 60, 90, 180]:
                        candidates = [c for c in chain if c.get("iv", 0) > 0 and abs(c.get("dte", 0) - expiry) <= max(5, expiry * .3)]
                        for moneyness in [0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15]:
                            if candidates:
                                c = min(candidates, key=lambda row: abs(float(row.get("strike", 0)) / spot - moneyness))
                                grid.append({"strike": round(moneyness, 2), "expiry": expiry, "iv": round(float(c["iv"]), 4)})
                    if len(grid) >= 15:
                        source = "schwab"
            except Exception:
                pass

        # --- Tier 2: synthesised from VolatilitySurface -----------------------
        if source == "synthetic":
            grid = []
            if self.vol_surface is not None:
                strikes = [0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15]
                expiries = [7, 30, 60, 90, 180]
                for exp in expiries:
                    for k in strikes:
                        iv = self.vol_surface.interpolate_vol(exp, k)
                        grid.append({"strike": round(k, 2), "expiry": exp, "iv": round(iv, 4)})
            else:
                # Last resort: basic smile model
                strikes = [0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15]
                expiries = [7, 30, 60, 90, 180]
                for exp in expiries:
                    for k in strikes:
                        base_iv = 0.20 + 0.05 * (exp / 365)
                        smile = 0.03 * (k - 1.0) ** 2 * 100
                        grid.append({"strike": round(k, 2), "expiry": exp, "iv": round(base_iv + smile, 4)})

        return {"grid": grid, "source": source, "spot": self._last_spot or None}

    def scan_opportunities(
        self, ticker: str, allocation: Optional[Dict[str, object]] = None,
        regime: Optional[str] = None, max_results: int = 10,
    ) -> List[Dict[str, object]]:
        """Create order-ready option candidates from the direct Schwab chain.

        This is the required Phase 4 hand-off.  Beta Corridor supplies
        directional bias, gamma efficiency plus Black-Scholes/Monte Carlo rank
        and validate contracts, and Kelly/budget controls determine sizing.
        Delta remains a measured Greek and risk limit, never a target exposure.
        """
        if self.market_data is None:
            return []
        selected_regime = regime or self.regime
        try:
            chain = self.market_data.option_chain(ticker)
        except Exception as exc:
            logger.warning("Schwab option chain unavailable for %s: %s", ticker, type(exc).__name__)
            return []
        if not chain:
            return []
        spot = next((float(row.get("underlying_price", 0)) for row in chain if row.get("underlying_price")), self._last_spot)
        if spot <= 0:
            return []
        allocation_data = allocation or {}
        current_beta = allocation_data.get("current_beta")
        target_beta = allocation_data.get("target_beta")
        if current_beta is None or target_beta is None:
            logger.warning(
                "Options scan for %s rejected: current_beta and target_beta are required",
                ticker,
            )
            return []
        current_beta = float(current_beta)
        target_beta = float(target_beta)
        beta_gap = target_beta - current_beta
        beta_threshold = max(float(allocation_data.get("beta_corridor_threshold", 0.02)), 0.0)
        if abs(beta_gap) <= beta_threshold:
            return []
        beta_full_scale = max(
            float(allocation_data.get("beta_corridor_full_scale", 0.25)),
            beta_threshold or 0.01,
        )
        directional_score = float(np.clip(beta_gap / beta_full_scale, -1.0, 1.0))

        ivs = [float(row["iv"]) for row in chain if 0.01 < float(row.get("iv", 0)) < 3.0]
        base_iv = float(np.median(ivs)) if ivs else 0.20
        self.update_market(spot=spot, vix=base_iv * 100, hist_vol_30d=base_iv * .9, hist_vol_90d=base_iv)
        assert self.strategy_builder is not None and self.signal_gen is not None
        signals = self.signal_gen.all_signals()
        predictive_score = sum(
            signal.strength if signal.direction == "BULLISH"
            else -signal.strength if signal.direction == "BEARISH"
            else 0.0
            for signal in signals
        )
        # The scanner emits only a single, long directional leg.  Multi-leg
        # research templates may inform analytics elsewhere but must never be
        # described as the generated order strategy.
        # Direction is exclusively controlled by the BetaCorridor gap.  The
        # options-derived predictive signals remain diagnostic and cannot also
        # manufacture the valuation input.
        existing_notional = sum(
            abs(pos.current_price * pos.quantity * OptionsPortfolio.MARGIN_MULTIPLIER)
            for pos in self.portfolio.positions
        )
        existing_margin = self.portfolio.margin_estimate()
        allocator = self._portfolio_option_allocator or OptionOverlayAllocator(
            nav=self.nav,
            existing_notional=float((allocation or {}).get("existing_option_notional", existing_notional)),
            existing_initial_margin=float((allocation or {}).get("existing_option_initial_margin", existing_margin)),
            existing_delta_dollars=float((allocation or {}).get("existing_option_delta_dollars", 0.0)),
            reserved_notional=float((allocation or {}).get("reserved_option_notional", 0.0)),
            reserved_initial_margin=float((allocation or {}).get("reserved_option_initial_margin", 0.0)),
            reserved_delta_dollars=float((allocation or {}).get("reserved_option_delta_dollars", 0.0)),
            overlay_cap_pct=float((allocation or {}).get("option_overlay_cap_pct", OptionOverlayAllocator.overlay_cap_pct)),
            initial_margin_cap_pct=float((allocation or {}).get("option_initial_margin_cap_pct", OptionOverlayAllocator.initial_margin_cap_pct)),
            delta_cap_pct=float((allocation or {}).get("option_delta_cap_pct", OptionOverlayAllocator.delta_cap_pct)),
            initial_margin_rate=float((allocation or {}).get("option_initial_margin_rate", OptionOverlayAllocator.initial_margin_rate)),
        )
        budget = allocator._remaining()[0]
        hedge = self.compute_hedge_requirements(nav=self.nav)
        hedge_plan: List[HedgePosition] = []
        hedge_underlying = str((allocation or {}).get("hedge_underlying", "SPY")).upper()
        if ticker.upper() == hedge_underlying:
            for old_position in list(self.hedge_mgr.positions_to_roll()):
                hedge_plan.append(self.hedge_mgr.roll_position(old_position, spot, base_iv, risk_free=self._risk_free))
            if hedge["need_more_hedges"] and not self.hedge_mgr.positions:
                hedge_plan.extend(self.hedge_mgr.build_put_ladder(spot, base_iv, risk_free=self._risk_free))
                vix_overlay = self.hedge_mgr.build_vix_overlay(
                    max(float(getattr(self, "_last_vix", 0.0) or base_iv * 100), 1.0),
                    risk_free=self._risk_free,
                )
                if vix_overlay is not None:
                    hedge_plan.append(vix_overlay)
            hedge = self.compute_hedge_requirements(nav=self.nav)
        hedge["planned_positions"] = [
            {
                "instrument": position.instrument,
                "strike": position.strike,
                "expiry_date": position.expiry_date.isoformat(),
                "quantity": position.quantity,
                "entry_premium": position.entry_premium,
            }
            for position in hedge_plan
        ]
        sizer = get_options_sizer()
        candidates: List[Dict[str, object]] = []
        liquid_chain = [
            row for row in chain
            if 14 <= int(row.get("dte", 0)) <= 90
            and .80 <= float(row.get("strike", 0)) / spot <= 1.20
            and (
                int(row.get("open_interest", 0)) > 0
                or int(row.get("volume", 0)) > 0
                or (float(row.get("bid", 0)) > 0 and float(row.get("ask", 0)) > 0)
            )
        ]
        liquid_chain.sort(
            key=lambda row: (
                int(row.get("open_interest", 0)) + int(row.get("volume", 0)),
                -abs(abs(float(row.get("delta", 0))) - .50),
            ),
            reverse=True,
        )
        for row in liquid_chain[:60]:
            mark_price = float(row.get("mark") or row.get("last") or 0)
            market_price = float(row.get("ask") or mark_price)
            strike, dte = float(row.get("strike", 0)), int(row.get("dte", 0))
            is_call = str(row.get("option_type", "")).upper() == "CALL"
            if not market_price > 0 or not strike > 0 or not 1 <= dte <= 180 or not row.get("symbol"):
                continue
            if directional_score > .15 and not is_call:
                continue
            if directional_score < -.15 and is_call:
                continue
            T = max(dte / 365.0, .001)
            quoted_iv = float(row.get("iv") or 0.0)
            if not 0.01 < quoted_iv < 3.0:
                quoted_iv = BlackScholesModel.implied_vol(
                    mark_price, spot, strike, T, self._risk_free, is_call,
                )
            if not np.isfinite(quoted_iv):
                continue
            # BS and Monte Carlo use the candidate's live IV.  The synthetic
            # surface may describe regime/skew but cannot be its own edge.
            model_iv = quoted_iv
            sizing = sizer.size_option(
                spot, strike, dte, model_iv, is_call, self.nav, budget,
                self._risk_free, market_price,
                signal_strength=abs(directional_score),
                fair_value_tolerance_bps=float(
                    allocation_data.get("fair_value_tolerance_bps", 100.0)
                ),
            )
            if sizing.get("rejected") or int(sizing.get("contracts", 0)) <= 0:
                continue
            theo = BlackScholesModel.call_price(spot, strike, T, self._risk_free, model_iv) if is_call else BlackScholesModel.put_price(spot, strike, T, self._risk_free, model_iv)
            greeks = {
                "delta": BlackScholesModel.delta(spot, strike, T, self._risk_free, model_iv, is_call),
                "gamma": BlackScholesModel.gamma(spot, strike, T, self._risk_free, model_iv),
                "theta": BlackScholesModel.theta(spot, strike, T, self._risk_free, model_iv, is_call),
                "vega": BlackScholesModel.vega(spot, strike, T, self._risk_free, model_iv),
            }
            instruction = "BUY_TO_OPEN"
            aligned_predictive_strength = sum(
                signal.strength
                for signal in signals
                if signal.direction == ("BULLISH" if is_call else "BEARISH")
            )
            gamma_per_premium = (
                abs(greeks["gamma"]) * spot * spot / max(market_price * 100, 0.01)
            )
            score = (
                abs(directional_score) * 1_000
                + gamma_per_premium * 100
                + float(sizing.get("edge_bps", 0))
            )
            explicit_sector = str(row.get("sector") or (allocation or {}).get("sector") or "").strip()
            validation = {
                "source": "OptionsEngine", "ticker": ticker.upper(), "regime": selected_regime,
                "sector": explicit_sector,
                "strategy_direction_score": directional_score,
                "signal_source": "BETA_CORRIDOR",
                "beta_corridor": {
                    "current_beta": current_beta,
                    "target_beta": target_beta,
                    "beta_gap": beta_gap,
                    "threshold": beta_threshold,
                },
                "predictive_direction_score": predictive_score,
                "predictive_signals": [{"name": s.name, "direction": s.direction, "strength": s.strength} for s in signals],
                "bsm": {
                    "spot": spot, "theoretical_price": theo,
                    "quoted_iv": quoted_iv, "model_iv": model_iv, "greeks": greeks,
                    "fair_value": sizing.get("fair_value"),
                    "black_scholes_price": sizing.get("bs_price"),
                    "monte_carlo_price": sizing.get("mc_price"),
                    "monte_carlo_std_error": sizing.get("mc_std_error"),
                    "gamma_per_premium": gamma_per_premium,
                },
                "sizing": sizing, "convexity_hedge": hedge,
            }
            legs = [{"instruction": instruction, "quantity": int(sizing["contracts"]),
                     "instrument": {"symbol": row["symbol"], "assetType": "OPTION"}}]
            strategy_classification = classify_single_leg_directional_candidate(
                "CALL" if is_call else "PUT", instruction, str(row["symbol"]), legs,
            )
            if strategy_classification is None:
                continue
            validation["strategy_classification"] = strategy_classification
            validation["estimated_initial_margin"] = round(
                float(sizing.get("cost_per_contract", market_price * 100))
                * int(sizing["contracts"]) * allocator.initial_margin_rate, 2
            )
            candidates.append({
                "option_symbol": row["symbol"], "ticker": ticker.upper(), "instruction": instruction,
                "quantity": int(sizing["contracts"]), "limit_price": market_price, "score": round(score, 4),
                "option_type": "CALL" if is_call else "PUT", "legs": legs,
                "sector": explicit_sector,
                "options_validation": validation,
            })
        sized_candidates: List[Dict[str, object]] = []
        for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
            validation = candidate["options_validation"]
            bsm = validation["bsm"]  # generated immediately above
            per_contract_delta_dollars = abs(float(bsm["greeks"]["delta"])) * float(bsm["spot"]) * 100
            contract_cost = float(validation["sizing"].get("cost_per_contract", candidate["limit_price"] * 100))
            requested = int(candidate["quantity"])
            contracts = allocator.allocate(requested, contract_cost, per_contract_delta_dollars)
            if contracts < 1:
                continue
            candidate["quantity"] = contracts
            candidate["legs"][0]["quantity"] = contracts
            validation["sizing"]["contracts"] = contracts
            validation["sizing"]["total_cost"] = round(contracts * contract_cost, 2)
            validation["estimated_initial_margin"] = round(
                contracts * contract_cost * allocator.initial_margin_rate, 2
            )
            sized_candidates.append(candidate)
        return sized_candidates[:max_results]

    # -- hedge requirements -------------------------------------------------

    def compute_hedge_requirements(
        self, cube_out: Optional[CubeOutput] = None, nav: Optional[float] = None
    ) -> Dict[str, object]:
        """
        Determine what hedges are needed based on regime, existing
        positions, and portfolio Greeks.

        Returns a dict describing recommended actions.
        """
        nav = nav or self.nav
        current_delta = self.hedge_mgr.total_hedge_delta()
        budget = self.hedge_mgr.monthly_budget()
        to_roll = self.hedge_mgr.positions_to_roll()
        cost_drag = self.hedge_mgr.hedge_cost_drag()

        # target hedge delta: more negative in stress regimes
        target_delta_map = {
            "RISK_ON": -0.02,
            "NORMAL": -0.05,
            "CAUTIOUS": -0.10,
            "STRESS": -0.18,
            "CRASH": -0.25,
        }
        target_delta = target_delta_map.get(self.regime, -0.05) * (nav / 100_000)
        delta_gap = target_delta - current_delta

        anomalies = self.vol_surface.detect_anomalies() if self.vol_surface else []

        return {
            "regime": self.regime,
            "current_hedge_delta": current_delta,
            "target_hedge_delta": target_delta,
            "delta_gap": delta_gap,
            "need_more_hedges": delta_gap < -1.0,
            "positions_to_roll": len(to_roll),
            "monthly_budget_usd": budget,
            "cumulative_cost_drag_pct": cost_drag,
            "vol_surface_anomalies": anomalies,
        }

    # -- tail hedge ---------------------------------------------------------

    def build_tail_hedge(
        self,
        nav: Optional[float] = None,
        crash_target: float = -0.20,
    ) -> Dict[str, object]:
        """
        Construct a tail-hedge package: SPX put ladder + VIX call overlay.

        crash_target : the drawdown level we are hedging against (e.g. -0.20 = -20%)
        """
        nav = nav or self.nav
        spot = self._last_spot
        if spot <= 0 or self.vol_surface is None:
            return {"error": "Market data not initialised — call update_market() first"}

        vol = self.vol_surface.get_atm_vol("1M")

        put_positions = self.hedge_mgr.build_put_ladder(
            spot, vol, self._risk_free, expiry_days=30
        )
        vix_spot = self.vol_surface.vix_decimal * 100  # rough VIX level
        vix_pos = self.hedge_mgr.build_vix_overlay(
            vix_spot, vol=0.80, risk_free=self._risk_free, expiry_days=30
        )

        return {
            "crash_target": crash_target,
            "put_ladder": [
                {
                    "instrument": p.instrument,
                    "strike": p.strike,
                    "qty": p.quantity,
                    "premium": p.entry_premium,
                }
                for p in put_positions
            ],
            "vix_overlay": {
                "instrument": vix_pos.instrument if vix_pos else None,
                "strike": vix_pos.strike if vix_pos else None,
                "qty": vix_pos.quantity if vix_pos else None,
            },
            "total_monthly_cost": self.hedge_mgr.monthly_budget(),
            "cost_drag_annualised": self.hedge_mgr.hedge_cost_drag(),
        }

    # -- strategy evaluation ------------------------------------------------

    def evaluate_strategy(
        self,
        strategy_name: str,
        **kwargs,
    ) -> Optional[StrategyProfile]:
        """Evaluate a named strategy with current market data."""
        if self.strategy_builder is None:
            logger.warning("Strategy builder not initialised")
            return None
        method = getattr(self.strategy_builder, strategy_name, None)
        if method is None:
            logger.warning("Unknown strategy: %s", strategy_name)
            return None
        return method(**kwargs)

    # -- portfolio Greeks ---------------------------------------------------

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Aggregate Greeks across the paper options portfolio."""
        return self.portfolio.aggregate_greeks(self._risk_free)

    # -- regime strategy matrix ---------------------------------------------

    def regime_strategy_matrix(self) -> Dict[str, List[str]]:
        """Return the full regime -> strategy mapping."""
        return OptionsStrategyBuilder.regime_strategy_map()

    # -- report -------------------------------------------------------------

    def get_options_report(self) -> str:
        """Generate an ASCII-formatted options engine report."""
        lines: List[str] = []
        w = 72
        lines.append("=" * w)
        lines.append("  METADRON CAPITAL — OPTIONS ENGINE REPORT (PAPER MODE)")
        lines.append("=" * w)
        lines.append(f"  Regime          : {self.regime}")
        lines.append(f"  NAV             : ${self.nav:,.0f}")
        lines.append(f"  Spot            : {self._last_spot:,.2f}")
        lines.append(f"  Risk-Free Rate  : {self._risk_free:.2%}")
        lines.append("")

        # vol surface
        if self.vol_surface:
            lines.append("-" * w)
            lines.append("  VOLATILITY SURFACE")
            lines.append("-" * w)
            lines.append(f"  VIX             : {self.vol_surface.vix:.1f}")
            lines.append(f"  HV30            : {self.vol_surface.hv30:.1%}")
            lines.append(f"  HV90            : {self.vol_surface.hv90:.1%}")
            lines.append(f"  Term Spread     : {self.vol_surface.term_spread():.1%}")
            lines.append(f"  1M 25d Skew     : {self.vol_surface.skew_25d('1M'):.1%}")
            lines.append("")
            lines.append(f"  {'Tenor':<8} {'10d Put':>8} {'25d Put':>8} {'ATM':>8} "
                         f"{'25d Call':>9} {'10d Call':>9}")
            for tenor in self.vol_surface.TENORS:
                t = self.vol_surface.surface.get(tenor, {})
                lines.append(
                    f"  {tenor:<8} {t.get('10d_put',0):>8.1%} {t.get('25d_put',0):>8.1%} "
                    f"{t.get('ATM',0):>8.1%} {t.get('25d_call',0):>9.1%} {t.get('10d_call',0):>9.1%}"
                )
            anomalies = self.vol_surface.detect_anomalies()
            if anomalies:
                lines.append(f"  Anomalies       : {', '.join(anomalies)}")
            lines.append("")

        # hedge book
        lines.append("-" * w)
        lines.append("  CONVEXITY HEDGE BOOK")
        lines.append("-" * w)
        lines.append(f"  Monthly Budget  : ${self.hedge_mgr.monthly_budget():,.0f}")
        lines.append(f"  Cum. Cost Drag  : {self.hedge_mgr.hedge_cost_drag():.2%}")
        lines.append(f"  Hedge Delta     : {self.hedge_mgr.total_hedge_delta():.2f}")
        lines.append(f"  Positions       : {len(self.hedge_mgr.positions)}")
        to_roll = self.hedge_mgr.positions_to_roll()
        if to_roll:
            lines.append(f"  ** {len(to_roll)} position(s) need rolling **")
        lines.append("")

        # portfolio greeks
        greeks = self.get_portfolio_greeks()
        lines.append("-" * w)
        lines.append("  PORTFOLIO GREEKS (paper positions)")
        lines.append("-" * w)
        lines.append(f"  Delta  : {greeks['delta']:>12.2f}")
        lines.append(f"  Gamma  : {greeks['gamma']:>12.4f}")
        lines.append(f"  Theta  : {greeks['theta']:>12.2f}  (per day)")
        lines.append(f"  Vega   : {greeks['vega']:>12.2f}  (per 1% vol)")
        lines.append(f"  Rho    : {greeks['rho']:>12.2f}  (per 1% rate)")
        lines.append(f"  Open positions   : {len(self.portfolio.positions)}")
        lines.append(f"  Unrealised P&L   : ${self.portfolio.unrealised_pnl():,.2f}")
        lines.append(f"  Realised P&L     : ${self.portfolio.realised_pnl:,.2f}")
        margin = self.portfolio.margin_estimate()
        if margin > 0:
            lines.append(f"  Margin estimate  : ${margin:,.0f}")
        lines.append("")

        # signals
        if self.signal_gen:
            lines.append("-" * w)
            lines.append("  PREDICTIVE SIGNALS")
            lines.append("-" * w)
            for sig in self.signal_gen.all_signals():
                arrow = {"BULLISH": "+", "BEARISH": "-", "NEUTRAL": "~"}.get(sig.direction, "?")
                lines.append(
                    f"  [{arrow}] {sig.name:<8} str={sig.strength:.0%}  {sig.description}"
                )
                if sig.suggested_strategy:
                    lines.append(f"{'':>14}-> suggested: {sig.suggested_strategy}")
            lines.append("")

        # regime strategy matrix
        lines.append("-" * w)
        lines.append("  REGIME -> STRATEGY MATRIX")
        lines.append("-" * w)
        for regime, strats in self.regime_strategy_matrix().items():
            marker = " <<" if regime == self.regime else ""
            lines.append(f"  {regime:<12} : {', '.join(strats)}{marker}")
        lines.append("")
        lines.append("=" * w)

        return "\n".join(lines)


# ===================================================================
#  Module-level convenience
# ===================================================================

def create_engine(regime: str = "NORMAL", nav: float = 1_000_000.0) -> OptionsEngine:
    """Factory for quick instantiation."""
    return OptionsEngine(regime=regime, nav=nav)

# ---------------------------------------------------------------------------
# Additional features merged from engine/options/black_scholes.py
# ---------------------------------------------------------------------------

@dataclass
class OptionPriceOutput:
    """Complete option pricing output with mispricing analysis."""
    underlying: str
    strike: float
    expiry: dt.date
    option_type: str  # "call" or "put"
    spot: float
    theoretical_price: float
    market_price: Optional[float]
    implied_vol: Optional[float]
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    moneyness: float
    intrinsic_value: float
    extrinsic_value: float

    @property
    def mispricing(self) -> Optional[float]:
        if self.market_price is not None:
            return self.theoretical_price - self.market_price
        return None

    @property
    def mispricing_pct(self) -> Optional[float]:
        if self.mispricing is not None and self.theoretical_price > 0:
            return self.mispricing / self.theoretical_price
        return None


def monte_carlo_option_price(
    S: float, K: float, T: float, sigma: float,
    is_call: bool = True, r: float = 0.05,
    n_sims: int = 10000, n_steps: int = 252,
    seed: Optional[int] = None,
) -> tuple:
    """
    Monte Carlo option pricing (handles path-dependent options).
    
    Returns (price, std_error).
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    # Antithetic variates reduce noise without mutating NumPy's process-global
    # RNG.  The previous fixed global seed made every contract reuse the same
    # path shocks and created correlated Monte Carlo pricing errors.
    half_sims = max(1, (n_sims + 1) // 2)
    half_Z = rng.standard_normal((half_sims, n_steps))
    Z = np.concatenate((half_Z, -half_Z), axis=0)[:n_sims]
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    S_T = S * np.exp(np.cumsum(log_returns, axis=1)[:, -1])
    
    if is_call:
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return float(price), float(std_error)


def scan_option_chain(
    spot: float,
    option_chain: list,
    min_mispricing_pct: float = 0.10,
    risk_free_rate: float = 0.05,
) -> list:
    """
    Scan an option chain for mispriced options using Black-Scholes.
    
    Args:
        spot: Current spot price
        option_chain: List of {strike, expiry (date), type (call/put), market_price}
        min_mispricing_pct: Minimum mispricing % to flag (default 10%)
    
    Returns:
        List of OptionPriceOutput with significant mispricing, sorted by magnitude.
    """
    opportunities = []
    bs = BlackScholesModel()
    
    for opt in option_chain:
        try:
            strike = opt["strike"]
            expiry = opt["expiry"] if isinstance(opt["expiry"], dt.date) else dt.date.fromisoformat(opt["expiry"])
            opt_type = opt.get("type", "call").lower()
            market_price = opt.get("market_price", opt.get("last_price", 0))
            
            if market_price <= 0:
                continue
            
            T = max((expiry - date.today()).days / 365.0, 0.001)
            is_call = opt_type == "call"
            
            # Solve for IV
            iv = bs.implied_vol(market_price, spot, strike, T, risk_free_rate, is_call)
            sigma = iv if iv else 0.20
            
            # Theoretical price
            if is_call:
                theo = bs.call_price(spot, strike, T, risk_free_rate, sigma)
            else:
                theo = bs.put_price(spot, strike, T, risk_free_rate, sigma)
            
            # Greeks
            d = bs.delta(spot, strike, T, risk_free_rate, sigma, is_call)
            g = bs.gamma(spot, strike, T, risk_free_rate, sigma)
            th = bs.theta(spot, strike, T, risk_free_rate, sigma, is_call)
            v = bs.vega(spot, strike, T, risk_free_rate, sigma)
            rh = bs.rho(spot, strike, T, risk_free_rate, sigma, is_call)
            
            # Intrinsic
            intrinsic = max(spot - strike, 0) if is_call else max(strike - spot, 0)
            
            result = OptionPriceOutput(
                underlying=opt.get("underlying", ""),
                strike=strike, expiry=expiry,
                option_type=opt_type, spot=spot,
                theoretical_price=theo, market_price=market_price,
                implied_vol=iv, delta=d, gamma=g, theta=th, vega=v, rho=rh,
                moneyness=spot / strike if strike > 0 else 0,
                intrinsic_value=intrinsic,
                extrinsic_value=max(theo - intrinsic, 0),
            )
            
            if result.mispricing_pct and abs(result.mispricing_pct) > min_mispricing_pct:
                opportunities.append(result)
                
        except Exception as e:
            logger.debug("Failed to price option: %s", e)
            continue
    
    opportunities.sort(key=lambda x: abs(x.mispricing_pct or 0), reverse=True)
    return opportunities


# ---------------------------------------------------------------------------
# Options Sizer — BS + Monte Carlo + Kelly for contract quantity
# ---------------------------------------------------------------------------

class OptionsSizer:
    """Compute option contract quantity using Black-Scholes, Monte Carlo, and Kelly.

    ONLY sizes positions where genuine mispricing is detected.
    If BS theoretical vs market price doesn't show sufficient edge,
    the position is REJECTED — no allocation.

    This is NOT a budget converter. It is an edge-gated sizer:
        1. BS + MC establish fair value
        2. Compare fair value vs market price → edge in bps
        3. If edge < MIN_EDGE_BPS → REJECT (no allocation)
        4. Kelly criterion sizes the position based on the detected edge
        5. A portfolio allocator may grant one contract when that is the only
           affordable amount under the shared overlay and margin caps.
    """

    CONTRACT_MULTIPLIER = 100
    KELLY_MULTIPLIER = 1.5
    MAX_CONTRACTS_PCT_NAV = 0.05
    MIN_CONTRACTS = 1
    MIN_EDGE_BPS = 200
    MC_SIMS = 10_000
    MC_STEPS = 63

    def __init__(self):
        self.bs = BlackScholesModel()

    def size_option(
        self,
        spot: float,
        strike: float,
        expiry_days: int,
        vol: float,
        is_call: bool,
        nav: float,
        budget_dollars: float,
        risk_free: float = 0.05,
        market_price: Optional[float] = None,
        signal_strength: Optional[float] = None,
        fair_value_tolerance_bps: float = 100.0,
    ) -> dict:
        """Compute optimal contract quantity for an option position.

        Parameters
        ----------
        spot : float
            Current underlying price.
        strike : float
            Option strike price.
        expiry_days : int
            Days to expiration.
        vol : float
            Annualized implied volatility.
        is_call : bool
            True for calls, False for puts.
        nav : float
            Portfolio net asset value.
        budget_dollars : float
            Dollar allocation from the allocation engine for this option bucket.
        risk_free : float
            Risk-free rate (annualized).
        market_price : float, optional
            Market price of the option (for mispricing edge calculation).

        Returns
        -------
        dict with keys:
            contracts, bs_price, mc_price, mc_std_error, mc_win_prob,
            kelly_fraction, kelly_contracts, dollar_cost, edge_bps,
            delta, gamma, theta, vega
        """
        T = max(expiry_days / 365.0, 0.001)

        # 1. Black-Scholes theoretical price
        if is_call:
            bs_price = self.bs.call_price(spot, strike, T, risk_free, vol)
        else:
            bs_price = self.bs.put_price(spot, strike, T, risk_free, vol)
        bs_price = max(bs_price, 0.01)

        # 2. Monte Carlo simulation — win probability and expected payoff
        seed = int(
            abs(spot * 1009 + strike * 917 + expiry_days * 101 + vol * 1_000_003)
        ) % (2**32)
        mc_price, mc_std_error = monte_carlo_option_price(
            S=spot, K=strike, T=T, sigma=vol,
            is_call=is_call, r=risk_free,
            n_sims=self.MC_SIMS, n_steps=self.MC_STEPS,
            seed=seed,
        )

        # Monte Carlo win probability: fraction of paths where option expires ITM
        dt_mc = T / self.MC_STEPS
        rng = np.random.default_rng(seed ^ 0x9E3779B9)
        Z = rng.standard_normal((self.MC_SIMS, self.MC_STEPS))
        log_returns = (risk_free - 0.5 * vol**2) * dt_mc + vol * np.sqrt(dt_mc) * Z
        S_T = spot * np.exp(np.cumsum(log_returns, axis=1)[:, -1])

        if is_call:
            itm_mask = S_T > strike
            payoffs = np.maximum(S_T - strike, 0)
        else:
            itm_mask = S_T < strike
            payoffs = np.maximum(strike - S_T, 0)

        win_prob = float(np.mean(itm_mask))
        avg_win_payoff = float(np.mean(payoffs[itm_mask])) if np.any(itm_mask) else 0.0

        # 3. Edge detection — ONLY proceed if mispricing exists
        # Fair value = average of BS and MC for robustness
        fair_value = (bs_price + mc_price) / 2.0
        entry_price = market_price if market_price and market_price > 0 else None

        if entry_price is None:
            return {
                "contracts": 0,
                "rejected": True,
                "reject_reason": "no market price — cannot assess mispricing",
                "bs_price": round(bs_price, 4),
                "mc_price": round(mc_price, 4),
                "edge_bps": 0.0,
            }

        # Edge: how much the option is underpriced (positive = we buy cheap)
        edge_bps = ((fair_value - entry_price) / entry_price * 10_000) if entry_price > 0 else 0.0

        corridor_mode = signal_strength is not None
        mc_band = 1.96 * mc_std_error
        tolerance = max(mc_band, fair_value * fair_value_tolerance_bps / 10_000)

        if corridor_mode and entry_price > fair_value + tolerance:
            return {
                "contracts": 0,
                "rejected": True,
                "reject_reason": (
                    f"executable ask {entry_price:.4f} exceeds fair value "
                    f"{fair_value:.4f} plus tolerance {tolerance:.4f}"
                ),
                "bs_price": round(bs_price, 4),
                "mc_price": round(mc_price, 4),
                "mc_std_error": round(mc_std_error, 4),
                "edge_bps": round(edge_bps, 1),
                "market_price": entry_price,
                "fair_value": round(fair_value, 4),
                "sizing_method": "BETA_CORRIDOR_GAMMA",
            }

        if not corridor_mode and abs(edge_bps) < self.MIN_EDGE_BPS:
            return {
                "contracts": 0,
                "rejected": True,
                "reject_reason": f"insufficient edge ({edge_bps:.0f}bps < {self.MIN_EDGE_BPS}bps minimum)",
                "bs_price": round(bs_price, 4),
                "mc_price": round(mc_price, 4),
                "edge_bps": round(edge_bps, 1),
                "market_price": entry_price,
                "fair_value": round(fair_value, 4),
            }

        # 4. Kelly criterion — size based on detected edge
        avg_net_win = max(avg_win_payoff - entry_price, 0.0)
        avg_loss = entry_price

        if not corridor_mode and avg_net_win > 0 and avg_loss > 0 and win_prob > 0:
            win_loss_ratio = avg_net_win / avg_loss
            q = 1.0 - win_prob
            kelly_f = max((win_prob * win_loss_ratio - q) / win_loss_ratio, 0.0)
        else:
            kelly_f = 0.0

        aggressive_kelly = kelly_f * self.KELLY_MULTIPLIER
        aggressive_kelly = min(aggressive_kelly, self.MAX_CONTRACTS_PCT_NAV)

        # 5. Contract count — Kelly-sized if edge, otherwise deploy budget on confirmed mispricing
        kelly_dollar_amount = aggressive_kelly * nav
        cost_per_contract = entry_price * self.CONTRACT_MULTIPLIER

        if corridor_mode:
            effective_budget = min(
                budget_dollars,
                self.MAX_CONTRACTS_PCT_NAV * nav,
            )
        elif kelly_f > 0 and kelly_dollar_amount > 0:
            effective_budget = min(kelly_dollar_amount, budget_dollars)
        else:
            effective_budget = budget_dollars

        if cost_per_contract > 0:
            contracts = int(effective_budget / cost_per_contract)
        else:
            contracts = 0

        # Enforce a one-contract minimum only when it fits the candidate cap.
        # Portfolio-level allocation below is authoritative across names.
        max_contracts_by_nav = int(
            (self.MAX_CONTRACTS_PCT_NAV * nav) / max(cost_per_contract, 1)
        )
        if max_contracts_by_nav < self.MIN_CONTRACTS:
            return {
                "contracts": 0,
                "rejected": True,
                "reject_reason": (
                    f"per-candidate cap permits only {max_contracts_by_nav} contracts "
                    f"(< {self.MIN_CONTRACTS} minimum)"
                ),
                "bs_price": round(bs_price, 4),
                "mc_price": round(mc_price, 4),
                "edge_bps": round(edge_bps, 1),
                "cost_per_contract": round(cost_per_contract, 2),
            }

        if contracts < self.MIN_CONTRACTS:
            if (
                budget_dollars >= self.MIN_CONTRACTS * cost_per_contract
                and max_contracts_by_nav >= self.MIN_CONTRACTS
            ):
                contracts = self.MIN_CONTRACTS
            else:
                return {
                    "contracts": 0,
                    "rejected": True,
                    "reject_reason": f"budget ${budget_dollars:.0f} cannot fund {self.MIN_CONTRACTS} contracts at ${cost_per_contract:.0f} each",
                    "bs_price": round(bs_price, 4),
                    "mc_price": round(mc_price, 4),
                    "edge_bps": round(edge_bps, 1),
                    "cost_per_contract": round(cost_per_contract, 2),
                }

        # Cap at max % of NAV; never restore the five-contract floor here.
        contracts = min(contracts, max_contracts_by_nav)

        # Greeks for the sized position
        greek_multiplier = contracts * self.CONTRACT_MULTIPLIER
        delta = self.bs.delta(spot, strike, T, risk_free, vol, is_call) * greek_multiplier
        gamma = self.bs.gamma(spot, strike, T, risk_free, vol) * greek_multiplier
        theta = self.bs.theta(spot, strike, T, risk_free, vol, is_call) * greek_multiplier
        vega = self.bs.vega(spot, strike, T, risk_free, vol) * greek_multiplier

        return {
            "contracts": contracts,
            "rejected": False,
            "bs_price": round(bs_price, 4),
            "mc_price": round(mc_price, 4),
            "mc_std_error": round(mc_std_error, 4),
            "mc_win_prob": round(win_prob, 4),
            "fair_value": round(fair_value, 4),
            "market_price": round(entry_price, 4),
            "kelly_fraction": round(kelly_f, 6),
            "kelly_aggressive": round(aggressive_kelly, 6),
            "kelly_dollar_amount": round(kelly_dollar_amount, 2),
            "effective_budget": round(effective_budget, 2),
            "cost_per_contract": round(cost_per_contract, 2),
            "total_cost": round(contracts * cost_per_contract, 2),
            "edge_bps": round(edge_bps, 1),
            "sizing_method": "BETA_CORRIDOR_GAMMA" if corridor_mode else "KELLY_MISPRICING",
            "signal_strength": round(float(signal_strength or 0.0), 6),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
        }

    def size_from_signal(
        self,
        ticker: str,
        spot: float,
        vol: float,
        nav: float,
        budget_dollars: float,
        market_price: float,
        is_call: bool = True,
        strike: Optional[float] = None,
        otm_pct: float = 0.05,
        expiry_days: int = 30,
        risk_free: float = 0.05,
    ) -> dict:
        """Size an option position — ONLY if mispricing is detected.

        Requires market_price for edge assessment. Without it, the position
        is rejected — we don't blindly allocate into fairly-priced options.

        Parameters
        ----------
        market_price : float
            Current market price of the option. Required for mispricing detection.
        strike : float, optional
            If not provided, computed as spot × (1 ± otm_pct).
        """
        if strike is None:
            strike = spot * (1 + otm_pct) if is_call else spot * (1 - otm_pct)

        result = self.size_option(
            spot=spot, strike=strike, expiry_days=expiry_days,
            vol=vol, is_call=is_call, nav=nav,
            budget_dollars=budget_dollars, risk_free=risk_free,
            market_price=market_price,
        )
        result["ticker"] = ticker
        result["strike"] = round(strike, 2)
        result["expiry_days"] = expiry_days
        result["is_call"] = is_call
        return result


# Module-level singleton
_options_sizer: Optional[OptionsSizer] = None


def get_options_sizer() -> OptionsSizer:
    global _options_sizer
    if _options_sizer is None:
        _options_sizer = OptionsSizer()
    return _options_sizer
