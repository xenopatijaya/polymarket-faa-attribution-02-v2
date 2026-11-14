#!/usr/bin/env python3
"""Two-space calibrated random-walk quantile forecaster (v2, "FAA
Attribution Forecaster v5") for polymarket-faa-attribution-02.

Forecasts, for every market in a case, the Yes price at each of the next
``horizon_minutes`` minutes as quantiles (5/25/50/75/95 percent).

Model
-----
One forecaster, run in two spaces and combined:

* In each space the centre is the last observed value, pulled back
  toward the value 90 minutes earlier (mean reversion, shrink 0.20,
  ramped in over 120 minutes) — but only when that 90-minute move is
  larger than half a sigma-scaled random walk over the same span
  (``|drift| > 0.5 * sigma * sqrt(90)``); smaller wiggles are treated
  as noise and not reverted.
* The scale is the market's *own* per-minute volatility in that space,
  estimated robustly: 1.4826 * MAD of the minute changes over the last
  1440 changes, blended 40/60 with the same estimate over the last 60
  changes; mean-absolute fallbacks when a MAD is exactly 0 (these
  markets sit still most minutes), a constant fallback when there is
  no usable history, then a floor and a cap.
* Quantiles: ``centre + z_tau * sigma * h**0.44 * tail(tau)`` with
  normal z-values and a piecewise tail widening (1.10 at the 25/75%
  points, 1.50 at the 5/95% points).

The two spaces are the raw price (linear) and the logit of the price.
The linear space is the v4 model; the logit space respects the [0, 1]
support — a market priced at 0.03 cannot fall as far as it can jump —
and produces the same construction there.  The submitted quantile is
the mix ``0.4 * q_linear + 0.6 * q_logit``.

Validation (exact local re-implementation of the official pinball
scorer, which reproduces the house baseline's published practice loss
0.363589 to all six decimals, over the six practice cases):
v4 (linear only, its own tuning) scored 0.273476.  The logit space
alone scored 0.270419 with v4's parameters.  The 40/60 mix scored
0.265358 before, and 0.263371 after, a light retune of the shared
parameters (exponent 0.42 -> 0.44, blend 0.7 -> 0.6, tails
1.00/1.65 -> 1.10/1.50, reversion lookback 60 -> 90 and shrink
0.15 -> 0.20, plus the reversion threshold above).  The mix improves
four of the six practice cases outright and is within 0.3% of v4 on
the other two — a broad gain, not one case carrying the rest.
Rejected during tuning, each on the same practice set: per-quantile
free multipliers (a large apparent gain that failed leave-one-case-out
validation and was discarded as overfitting), arcsine/probit spaces
alone, EWMA volatility, mean-price anchors, sibling-market drift, and
horizon-decaying vol blends.

Standard library only; fully deterministic (no randomness, no clock,
no network).
"""

import json
import math
import os
from statistics import NormalDist, median

# Shared parameters (both spaces).
SPREAD_EXPONENT = 0.44
LONG_WINDOW = 1440
SHORT_WINDOW = 60
SHORT_BLEND = 0.6
REVERSION_LOOKBACK = 90
REVERSION_SHRINK = 0.20
REVERSION_RAMP = 120
REVERSION_THRESHOLD = 0.5

# Per-space (floor, cap, no-history fallback) for the sigma estimate.
LIN_SPACE = (0.0002, 0.02, 0.0025)
LOG_SPACE = (0.002, 0.6, 0.01)
LOG_EPS = 0.0005

# Submitted quantile = (1 - LOG_WEIGHT) * linear + LOG_WEIGHT * logit.
LOG_WEIGHT = 0.6

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

_NORM = NormalDist()

# Tail widening as a piecewise-linear function of |z|: exactly 1.0 at the
# median, 1.10 at the 25/75% points and 1.50 at the 5/95% points.
_TAIL_POINTS = ((0.0, 1.0), (0.67448975, 1.10), (1.64485363, 1.50))


def _tail_factor(z):
    a = abs(z)
    pts = _TAIL_POINTS
    if a <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if a <= x1:
            return y0 + (y1 - y0) * (a - x0) / (x1 - x0)
    return pts[-1][1]


def _mad_sigma(window):
    """1.4826 * median absolute deviation of a window of changes."""
    if not window:
        return 0.0
    med = median(window)
    return 1.4826 * median([abs(v - med) for v in window])


def _mean_abs(window):
    if not window:
        return 0.0
    return sum(abs(v) for v in window) / len(window)


def _estimate_sigma(changes, floor, cap, fallback):
    """Blend a long-window and a short-window robust volatility estimate."""
    long_win = changes[-LONG_WINDOW:]
    sigma = _mad_sigma(long_win)
    if sigma == 0.0:
        sigma = _mean_abs(long_win)
    if sigma == 0.0:
        sigma = fallback

    short_win = changes[-SHORT_WINDOW:]
    sigma_short = _mad_sigma(short_win)
    if sigma_short == 0.0:
        sigma_short = _mean_abs(short_win)

    # A market born minutes before the checkpoint has too few recent
    # changes for the short window to mean anything; use the long
    # window alone until it has at least 10 recent changes.
    if sigma_short > 0.0 and len(short_win) >= 10:
        sigma = (1.0 - SHORT_BLEND) * sigma + SHORT_BLEND * sigma_short
    return min(cap, max(floor, sigma))


def _space_rows(series, empty_last, inverse, space, horizon, taus):
    """Quantile rows for one market in one transformed space.

    ``series`` is the full transformed price history (last value is the
    checkpoint value); ``inverse`` maps a forecast back to a price.
    """
    floor, cap, fallback = space
    last = series[-1] if series else empty_last
    if len(series) > REVERSION_LOOKBACK:
        drift = series[-1] - series[-REVERSION_LOOKBACK - 1]
    else:
        drift = 0.0
    changes = [series[i] - series[i - 1] for i in range(1, len(series))]
    changes = changes[-4320:]
    sigma = _estimate_sigma(changes, floor, cap, fallback)
    # Only revert moves too large to be ordinary random-walk noise.
    if abs(drift) <= REVERSION_THRESHOLD * sigma * math.sqrt(REVERSION_LOOKBACK):
        drift = 0.0

    zs = [(_NORM.inv_cdf(tau), _tail_factor(_NORM.inv_cdf(tau))) for tau in taus]
    rows = []
    for h in range(1, horizon + 1):
        centre = last - REVERSION_SHRINK * drift * min(h, REVERSION_RAMP) / REVERSION_RAMP
        spread = sigma * h ** SPREAD_EXPONENT
        row = []
        for z, tail in zs:
            v = inverse(centre + z * spread * tail)
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            row.append(v)
        rows.append(row)
    return rows


def _logit(p):
    p = LOG_EPS if p < LOG_EPS else (1.0 - LOG_EPS if p > 1.0 - LOG_EPS else p)
    return math.log(p / (1.0 - p))


def _sigmoid(v):
    # Branch on the sign so extreme centres from a jumpy market's logit
    # history can never overflow math.exp.
    if v >= 0.0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def forecast_market(prices, horizon, taus):
    """Return `horizon` rows of quantile forecasts for one market."""
    lin_rows = _space_rows(prices, 0.5, lambda v: v, LIN_SPACE, horizon, taus)
    log_series = [_logit(p) for p in prices]
    log_rows = _space_rows(log_series, 0.0, _sigmoid, LOG_SPACE, horizon, taus)
    rows = []
    for lin_row, log_row in zip(lin_rows, log_rows):
        row = [round((1.0 - LOG_WEIGHT) * a + LOG_WEIGHT * b, 6)
               for a, b in zip(lin_row, log_row)]
        row.sort()
        # Rounding can in principle break monotonicity for nearly equal
        # neighbours; a running maximum guarantees non-decreasing rows.
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case.get("horizon_minutes", 240))
        taus = [float(t) for t in case.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m.get("prices", []), horizon, taus)}
            for m in case["markets"]
        ]
        out_name = "{}.out".format(case["case"])
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump({"forecasts": forecasts}, fh, separators=(",", ":"))


if __name__ == "__main__":
    run(INPUT_DIR, OUTPUT_DIR)
