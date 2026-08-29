"""Distribution-drift signals without a heavyweight statistical dependency."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _finite_array(values: Iterable[float]) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array[np.isfinite(array)]


def _ks_statistic(current: np.ndarray, baseline: np.ndarray) -> float:
    points = np.unique(np.concatenate((current, baseline)))
    current_sorted = np.sort(current)
    baseline_sorted = np.sort(baseline)
    current_cdf = np.searchsorted(current_sorted, points, side="right") / current.size
    baseline_cdf = np.searchsorted(baseline_sorted, points, side="right") / baseline.size
    return float(np.max(np.abs(current_cdf - baseline_cdf)))


def _psi(current: np.ndarray, baseline: np.ndarray, bins: int = 10) -> float:
    if float(np.min(baseline)) == float(np.max(baseline)):
        value = float(baseline[0])
        edges = np.asarray([-np.inf, value, np.inf], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, num=min(bins + 1, baseline.size + 1))
        unique_values = np.unique(baseline)
        edges = np.unique(np.quantile(baseline, quantiles))
        if unique_values.size == 2:
            # Quantiles collapse for a binary/step-like baseline.  Adding the
            # midpoint lets PSI see a current population concentrated between
            # the two historical modes.
            midpoint = (float(unique_values[0]) + float(unique_values[1])) / 2.0
            edges = np.asarray(
                [-np.inf, unique_values[0], midpoint, unique_values[1], np.inf]
            )
        if edges.size < 2:
            value = float(baseline[0])
            edges = np.asarray([-np.inf, value, np.inf], dtype=float)
        else:
            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    baseline_probs = (baseline_counts + epsilon) / (baseline.size + epsilon * len(baseline_counts))
    current_probs = (current_counts + epsilon) / (current.size + epsilon * len(current_counts))
    return float(np.sum((current_probs - baseline_probs) * np.log(current_probs / baseline_probs)))


def _quantile_shape_signal(
    current: np.ndarray, baseline: np.ndarray
) -> tuple[float, bool, float, float]:
    """Return a robust shape score, decision, and spread values.

    With only a few observations, histogram/PSI bins are dominated by one
    sample.  A winsorized spread proxy is more stable and still catches a
    concentrated distribution with a similar mean.
    """
    base_q10, base_q25, base_median, base_q75, base_q90 = np.percentile(
        baseline, [10, 25, 50, 75, 90]
    )
    cur_q10, cur_q25, cur_median, cur_q75, cur_q90 = np.percentile(
        current, [10, 25, 50, 75, 90]
    )
    # When IQR is zero, use half of the central 80% range rather than a tiny
    # arbitrary floor. This avoids treating ordinary variation in a small
    # sample as a massive distribution shift.
    base_spread = max(float(base_q75 - base_q25), float((base_q90 - base_q10) * 0.5), 1e-12)
    cur_spread = max(float(cur_q75 - cur_q25), float((cur_q90 - cur_q10) * 0.5), 1e-12)
    spread_ratio = max(cur_spread / base_spread, base_spread / cur_spread)
    median_shift = abs(float(cur_median - base_median)) / base_spread
    score = max(spread_ratio / 2.0, median_shift / 3.0)
    return score, bool(spread_ratio >= 2.0 or median_shift >= 3.0), base_spread, cur_spread


def detect_distribution_shift(
    current_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    ratio_threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect location or shape drift using KS and population stability.

    ``ratio_threshold`` is retained for source compatibility.  The detector
    uses distribution-aware thresholds and only falls back to the old mean
    ratio as a diagnostic in the returned reason.
    """
    cur = _finite_array(current_values)
    base = _finite_array(baseline_values)
    if cur.size == 0 or base.size == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_psi",
            "reason": "empty_input",
        }

    ks = _ks_statistic(cur, base)
    critical = 1.36 * np.sqrt((cur.size + base.size) / (cur.size * base.size))
    psi = _psi(cur, base)
    shape_score, shape_anomaly, baseline_spread, current_spread = _quantile_shape_signal(cur, base)
    current_mean = float(np.mean(cur))
    baseline_mean = float(np.mean(base))
    if baseline_mean == 0:
        mean_ratio = float("inf") if current_mean != 0 else 1.0
    elif current_mean == 0:
        mean_ratio = float("inf")
    else:
        mean_ratio = max(abs(current_mean / baseline_mean), abs(baseline_mean / current_mean))

    ks_anomaly = bool(ks >= critical)
    # PSI is reliable only once each population has enough observations. For
    # small samples, use the robust quantile spread signal above instead.
    psi_usable = min(cur.size, base.size) >= 20
    psi_anomaly = bool(psi_usable and psi >= 0.20)
    mean_ratio_anomaly = bool(
        np.isfinite(mean_ratio)
        and abs(baseline_mean) > 1e-12
        and abs(current_mean) > 1e-12
        and mean_ratio >= ratio_threshold
    )
    # Normalizing by the decision boundaries gives callers a comparable score
    # while preserving the raw statistics in ``reason``.
    score_candidates = [
        float(ks / critical) if critical else 0.0,
        shape_score,
        float(mean_ratio / ratio_threshold) if np.isfinite(mean_ratio) else 0.0,
    ]
    if psi_usable:
        score_candidates.append(float(psi / 0.20))
    score = max(score_candidates)
    return {
        "is_anomaly": bool(
            ks_anomaly or psi_anomaly or shape_anomaly or mean_ratio_anomaly
        ),
        "score": float(score),
        "method": "ks_psi",
        "reason": (
            f"ks={ks:.4f}; ks_critical={critical:.4f}; psi={psi:.4f}; "
            f"psi_threshold=0.2; psi_usable={psi_usable}; "
            f"shape_score={shape_score:.4f}; shape_anomaly={shape_anomaly}; "
            f"baseline_spread={baseline_spread:.4f}; current_spread={current_spread:.4f}; "
            f"mean_ratio={mean_ratio:.3f}; mean_ratio_anomaly={mean_ratio_anomaly}; "
            f"legacy_ratio_threshold={ratio_threshold}"
        ),
    }
