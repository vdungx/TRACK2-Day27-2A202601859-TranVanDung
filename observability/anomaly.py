"""Robust, dependency-light anomaly detectors.

``zscore`` remains available as the transparent baseline used by the public
exercise.  ``auto`` adds context-aware segmentation and robust scale
estimation so one bad historical batch does not hide the next incident.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _finite_array(values: Iterable[float]) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array[np.isfinite(array)]


def _invalid_current(current: float) -> bool:
    try:
        return not bool(np.isfinite(float(current)))
    except (TypeError, ValueError):
        return True


def _invalid_result(method: str, reason: str) -> dict[str, Any]:
    return {
        "is_anomaly": True,
        "score": float("inf"),
        "method": method,
        "reason": reason,
    }


def zscore_detector(
    current: float, history: Iterable[float], threshold: float = 3.0
) -> dict[str, Any]:
    values = _finite_array(history)
    if _invalid_current(current):
        return _invalid_result("zscore", "current_value_is_not_finite")
    if values.size < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "zscore",
            "reason": "insufficient_history",
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        score = float("inf") if float(current) != mean else 0.0
    else:
        score = abs(float(current) - mean) / std
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def _robust_score(current: float, values: np.ndarray) -> tuple[float, str]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return 0.6745 * abs(float(current) - median) / mad, (
            f"median={median:.3f}, mad={mad:.3f}"
        )

    # A zero MAD is common for counters with repeated values. Try a non-zero
    # IQR/std scale before declaring every non-identical value anomalous.
    q1, q3 = np.percentile(values, [25, 75])
    iqr_scale = float(q3 - q1) / 1.349
    if iqr_scale > 0:
        return abs(float(current) - median) / iqr_scale, (
            f"median={median:.3f}, mad=0, iqr_scale={iqr_scale:.3f}"
        )
    std = float(np.std(values))
    if std > 0:
        return abs(float(current) - median) / std, (
            f"median={median:.3f}, mad=0, std_scale={std:.3f}"
        )
    return (0.0 if float(current) == median else float("inf")), (
        f"median={median:.3f}, mad=0, constant_baseline=true"
    )


def mad_detector(
    current: float, history: Iterable[float], threshold: float = 3.5
) -> dict[str, Any]:
    values = _finite_array(history)
    if _invalid_current(current):
        return _invalid_result("mad", "current_value_is_not_finite")
    if values.size < 5:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "mad",
            "reason": "insufficient_history",
        }
    score, baseline = _robust_score(float(current), values)
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "mad",
        "reason": f"{baseline}, threshold={threshold}",
    }


def _context_segment(context: Mapping[str, Any] | None) -> np.ndarray:
    if not context:
        return np.asarray([], dtype=float)

    segment = context.get("same_segment_history")
    if segment is not None:
        values = _finite_array(segment)
        if values.size:
            return values

    history_by_day = context.get("history_by_day")
    day = context.get("day_of_week")
    if isinstance(history_by_day, Mapping) and day is not None:
        segment = history_by_day.get(day, history_by_day.get(str(day), []))
        return _finite_array(segment)
    return np.asarray([], dtype=float)


def _trend_score(
    current: float, values: np.ndarray, trend: Any
) -> tuple[float, str, float]:
    """Score the residual from a supplied or learned linear trend."""
    x = np.arange(values.size, dtype=float)
    slope: float | None = None
    if isinstance(trend, Mapping):
        candidate = trend.get("slope")
        if candidate is not None:
            try:
                candidate_float = float(candidate)
            except (TypeError, ValueError):
                candidate_float = float("nan")
            if np.isfinite(candidate_float):
                slope = candidate_float
    elif isinstance(trend, (int, float, np.number)) and not isinstance(trend, bool):
        candidate_float = float(trend)
        if np.isfinite(candidate_float):
            slope = candidate_float

    if slope is None:
        slope = float(np.polyfit(x, values, 1)[0]) if values.size >= 2 else 0.0
    intercept = float(np.mean(values - slope * x))
    predicted = intercept + slope * values.size
    fitted = intercept + slope * x
    residuals = values - fitted
    score, baseline = _robust_score(float(current) - predicted, residuals)
    return score, f"slope={slope:.6f}; predicted={predicted:.3f}; {baseline}", predicted


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect a point anomaly while preserving the stable lab API.

    ``auto`` uses a same-segment baseline when supplied, then a robust
    median/MAD score. An explicitly known event is recorded as expected and
    does not page. The original z-score path is intentionally unchanged in
    its threshold semantics.
    """
    if method == "mad":
        # Preserve the explicit MAD detector's original 3.5 modified-z
        # threshold; ``threshold`` is the auto/z-score knob.
        return mad_detector(current, history)
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)
    if method != "auto":
        raise ValueError(f"Unsupported method: {method}")

    if context and context.get("known_event"):
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto:known_event",
            "reason": f"known_event={context['known_event']}; signal_suppressed",
        }

    global_values = _finite_array(history)
    segment_values = _context_segment(context)
    values = segment_values if segment_values.size >= 3 else global_values
    if _invalid_current(current):
        return _invalid_result("auto:robust", "current_value_is_not_finite")
    if values.size < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "auto:robust",
            "reason": "insufficient_history",
        }

    trend = context.get("trend") if context else None
    if trend:
        score, baseline, _ = _trend_score(float(current), values, trend)
        method_name = "auto:trend"
        trend_note = f"; trend_context={trend}"
    else:
        score, baseline = _robust_score(float(current), values)
        method_name = "auto:mad"
        trend_note = ""
    source = "same_segment_history" if segment_values.size >= 3 else "history"
    mad = float(np.median(np.abs(values - np.median(values))))
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": method_name if trend else ("auto:mad" if mad > 0 else "auto:robust"),
        "reason": f"baseline_source={source}; {baseline}; threshold={threshold}{trend_note}",
    }
