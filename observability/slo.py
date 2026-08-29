from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


def calculate_slo(target: float, bad_events: int, total_events: int) -> dict[str, Any]:
    """Calculate error budget consumption for a proportion-based SLO."""
    try:
        target = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError("target must be between 0 and 1 (exclusive)") from exc
    if not 0 < target < 1 or not math.isfinite(target):
        raise ValueError("target must be between 0 and 1 (exclusive)")

    try:
        bad_value = float(bad_events)
        total_value = float(total_events)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid event counts") from exc
    if (
        not math.isfinite(bad_value)
        or not math.isfinite(total_value)
        or bad_value < 0
        or total_value < 0
        or bad_value > total_value
        or not bad_value.is_integer()
        or not total_value.is_integer()
    ):
        raise ValueError("invalid event counts")
    bad_events = int(bad_value)
    total_events = int(total_value)

    # Decimal keeps common decimal SLOs exact at the public API boundary. This
    # avoids returning 3.9999999999999964 for the mathematically exact 4x burn.
    try:
        target_decimal = Decimal(str(target))
        allowed_decimal = Decimal(1) - target_decimal
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("target must be between 0 and 1 (exclusive)") from exc
    allowed_bad_rate = float(allowed_decimal)
    if total_events == 0:
        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }

    actual_decimal = Decimal(bad_events) / Decimal(total_events)
    actual_bad_rate = float(actual_decimal)
    at_budget_boundary = actual_decimal == allowed_decimal
    burn_rate = (
        1.0
        if at_budget_boundary
        else float(actual_decimal / allowed_decimal)
    )
    consumed_fraction = (
        1.0
        if at_budget_boundary or actual_decimal > allowed_decimal
        else float(actual_decimal / allowed_decimal)
    )
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_decimal > allowed_decimal and not at_budget_boundary),
    }


def evaluate_multiwindow_burn(
    *,
    short_window_burn: float,
    long_window_burn: float,
    policy: str = "default",
) -> dict[str, Any]:
    """Evaluate a two-window burn policy.

    The policy follows the common SRE two-window shape: a fast short-window
    burn is only a critical page when a slower window confirms it. A lower
    sustained burn is still actionable as a warning page, while a short-only
    spike is informational and must not wake an operator.
    """
    try:
        short = float(short_window_burn)
        long = float(long_window_burn)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("burn rates must be finite non-negative numbers") from exc
    if not math.isfinite(short) or not math.isfinite(long) or short < 0 or long < 0:
        raise ValueError("burn rates must be finite non-negative numbers")

    critical_thresholds = {"short": 14.4, "long": 6.0}
    warning_thresholds = {"short": 6.0, "long": 3.0}

    if short >= critical_thresholds["short"] and long >= critical_thresholds["long"]:
        page, severity, reason = True, "critical", "sustained_fast_burn"
    elif short >= warning_thresholds["short"] and long >= warning_thresholds["long"]:
        page, severity, reason = True, "warning", "sustained_elevated_burn"
    elif short >= warning_thresholds["short"]:
        page, severity, reason = False, "info", "transient_short_window_spike"
    else:
        page, severity, reason = False, "info", "within_burn_policy"

    return {
        "page": page,
        "severity": severity,
        "reason": reason,
        "policy": policy,
        "short_window_burn": short,
        "long_window_burn": long,
        "thresholds": {
            "critical": critical_thresholds,
            "warning": warning_thresholds,
        },
        # Keep the original aliases for callers that consumed the starter
        # response while exposing the complete two-window policy above.
        "page_threshold": critical_thresholds["short"],
        "ticket_threshold": critical_thresholds["long"],
    }


def evaluate_slo_history(
    good_events: Iterable[bool],
    *,
    target: float,
    short_window: int = 5,
    long_window: int = 30,
    min_short_samples: int = 3,
    min_long_samples: int = 5,
) -> dict[str, Any]:
    """Evaluate SLO burn over recent windows without cold-start paging.

    A single failed event in a newly-created history is useful evidence, but
    it is not enough to page. The returned object keeps both window SLO
    calculations so the caller can explain the decision and audit the sample
    sizes used by the policy.
    """
    try:
        short_window = int(short_window)
        long_window = int(long_window)
        min_short_samples = int(min_short_samples)
        min_long_samples = int(min_long_samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("SLO window sizes must be positive integers") from exc
    if (
        short_window <= 0
        or long_window <= 0
        or min_short_samples < 0
        or min_long_samples < 0
    ):
        raise ValueError("SLO window sizes must be positive integers")

    events = [bool(value) for value in good_events]
    short = events[-short_window:]
    long = events[-long_window:]
    short_status = calculate_slo(target, short.count(False), len(short))
    long_status = calculate_slo(target, long.count(False), len(long))
    alert = evaluate_multiwindow_burn(
        short_window_burn=short_status["burn_rate"],
        long_window_burn=long_status["burn_rate"],
    )
    enough_data = (
        len(short) >= min_short_samples and len(long) >= min_long_samples
    )
    if not enough_data:
        alert = {
            **alert,
            "page": False,
            "severity": "info",
            "reason": "insufficient_window_history",
        }
    return {
        "target": target,
        "sample_count": len(events),
        "short_window": {**short_status, "samples": len(short)},
        "long_window": {**long_status, "samples": len(long)},
        "alert": alert,
    }
