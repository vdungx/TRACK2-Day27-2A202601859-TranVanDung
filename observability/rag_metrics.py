from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import detect_anomaly


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    # Deliberately simple proxy; no tokenizer/model download needed.
    lengths: list[int] = []
    for value in texts:
        if value is None or value.__class__.__name__ in {"NAType", "NaTType"}:
            value = ""
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            value = ""
        lengths.append(len(str(value).split()))
    return lengths


def detect_text_length_shift(
    current_texts: Iterable[str],
    baseline_batch_means: Iterable[float],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    lengths = approximate_token_lengths(current_texts)
    current_mean = float(np.mean(lengths)) if lengths else 0.0
    result = detect_anomaly(
        current_mean,
        baseline_batch_means,
        method="auto",
        threshold=threshold,
        context={"metric_name": "mean_text_length"},
    )
    result["metric"] = "mean_text_length"
    result["current_mean"] = current_mean
    return result


def _coerce_finite(values: Iterable[float]) -> tuple[np.ndarray, int]:
    """Coerce norm observations independently and retain invalid-count evidence."""

    try:
        raw_values = list(values)
    except TypeError:
        return np.asarray([], dtype=float), 0
    finite: list[float] = []
    invalid = 0
    for value in raw_values:
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        if not np.isfinite(converted):
            invalid += 1
            continue
        finite.append(converted)
    return np.asarray(finite, dtype=float), invalid


def detect_embedding_norm_shift(
    current_norms: Iterable[float], baseline_norms: Iterable[float]
) -> dict[str, Any]:
    """Detect a shift in the mean embedding norm.

    Norms are precomputed inputs, so this check needs no embedding model.  A
    robust scale handles ordinary model noise while a relative-change guard
    still catches drift when the historical norms are constant.
    """
    current, current_invalid = _coerce_finite(current_norms)
    baseline, baseline_invalid = _coerce_finite(baseline_norms)
    if current_invalid or baseline_invalid:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "embedding:embedding_norm_robust",
            "metric": "embedding_norm_distribution",
            "reason": (
                "invalid_numeric_input; "
                f"current_invalid={current_invalid}; baseline_invalid={baseline_invalid}"
            ),
        }
    if current.size == 0 or baseline.size == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding:embedding_norm_robust",
            "metric": "embedding_norm_distribution",
            "reason": "empty_input",
        }
    if bool((current < 0).any()) or bool((baseline < 0).any()):
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "embedding:embedding_norm_robust",
            "metric": "embedding_norm_distribution",
            "reason": "negative_embedding_norm",
        }

    current_mean = float(np.mean(current))
    baseline_median = float(np.median(baseline))
    baseline_mean = float(np.mean(baseline))
    mad = float(np.median(np.abs(baseline - baseline_median)))
    if mad > 0:
        scale = mad / 0.6745
        scale_name = "mad_scale"
    else:
        scale = float(np.std(baseline))
        scale_name = "std_scale"

    difference = abs(current_mean - baseline_median)
    # A perfectly constant norm baseline has no empirical scale. Use the
    # relative guard in that case instead of treating harmless floating-point
    # noise as an infinite robust score.
    robust_score = difference / scale if scale > 0 else 0.0
    denominator = max(abs(baseline_median), 1e-12)
    relative_shift = difference / denominator
    score = max(float(robust_score), float(relative_shift / 0.20))
    return {
        "is_anomaly": bool(robust_score > 3.0 or relative_shift >= 0.20),
        "score": float(score),
        "method": "embedding:embedding_norm_robust",
        "metric": "embedding_norm_distribution",
        "reason": (
            f"baseline_mean={baseline_mean:.6f}; baseline_median={baseline_median:.6f}; "
            f"current_mean={current_mean:.6f}; {scale_name}={scale:.6f}; "
            f"relative_shift={relative_shift:.6f}; thresholds=3.0_or_0.2"
        ),
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
    }
