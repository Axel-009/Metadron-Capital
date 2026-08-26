# MSFT Options Methodology Lock

## Reference execution

The reference execution is the MSFT one-day call allocation submitted on
2026-08-25 at 18:17:06 UTC. The preserved submission record is
`Schwab-4806-and-0514-Intraday-Call-Submission.json`.

## Locked decision hierarchy

1. Beta Corridor supplies directional bias only: call, put, or neutral.
2. Gamma efficiency ranks convexity relative to premium.
3. Black-Scholes and Monte Carlo provide fair-value and selection checks.
4. Liquidity, spread, DTE, near-money, and risk filters reject unsuitable
   contracts.
5. Kelly, candidate budget, portfolio overlay, margin, and risk caps determine
   contract quantity.

## Delta rule

Delta is a measured Greek and may be used as a risk cap. The model must not:

- derive a target delta-dollar exposure from the Beta Corridor gap;
- treat option delta-dollars divided by NAV as portfolio beta;
- reverse direction or rotate positions to converge on a delta target; or
- cap a candidate's size against a target delta-dollar amount.

The MSFT-era `target_delta_dollars` field was sizing metadata, not an
authorized directional or portfolio-convergence target. It is removed from
the forward model interface to prevent that ambiguity.

## Valuation limitation

When Black-Scholes and Monte Carlo both use live quoted implied volatility,
their agreement with the option market is a pricing-consistency check rather
than independent evidence of mispricing. The executable ask and uncertainty
tolerance remain authoritative.

## Regression requirement

Focused tests must verify that:

- Beta Corridor sign selects call versus put direction;
- gamma and both valuation outputs remain in candidate metadata;
- `OptionsSizer.size_option` has no `target_delta_dollars` parameter; and
- candidate metadata contains no delta-exposure target.
