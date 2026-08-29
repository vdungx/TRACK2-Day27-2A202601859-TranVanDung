from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import detect_anomaly


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    # Deliberately simple proxy; no tokenizer/model download needed.
    return [len(str(t).split()) for t in texts]


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


def _finite_array(values: Iterable[float]) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array[np.isfinite(array)]


def detect_embedding_norm_shift(
    current_norms: Iterable[float], baseline_norms: Iterable[float]
) -> dict[str, Any]:
    """Detect a shift in the mean embedding norm.

    Norms are precomputed inputs, so this check needs no embedding model.  A
    robust scale handles ordinary model noise while a relative-change guard
    still catches drift when the historical norms are constant.
    """
    current = _finite_array(current_norms)
    baseline = _finite_array(baseline_norms)
    if current.size == 0 or baseline.size == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding_norm_robust",
            "reason": "empty_input",
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
    robust_score = difference / scale if scale > 0 else (0.0 if difference == 0 else float("inf"))
    denominator = max(abs(baseline_median), 1e-12)
    relative_shift = difference / denominator
    score = max(float(robust_score), float(relative_shift / 0.20))
    return {
        "is_anomaly": bool(robust_score > 3.0 or relative_shift >= 0.20),
        "score": float(score),
        "method": "embedding_norm_robust",
        "reason": (
            f"baseline_mean={baseline_mean:.6f}; baseline_median={baseline_median:.6f}; "
            f"current_mean={current_mean:.6f}; {scale_name}={scale:.6f}; "
            f"relative_shift={relative_shift:.6f}; thresholds=3.0_or_0.2"
        ),
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
    }
