from __future__ import annotations

import math
from typing import Any


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

    allowed_bad_rate = 1.0 - target
    if total_events == 0:
        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }

    actual_bad_rate = bad_events / total_events
    burn_rate = actual_bad_rate / allowed_bad_rate
    consumed_fraction = min(1.0, actual_bad_rate / allowed_bad_rate)
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_bad_rate > allowed_bad_rate),
    }


def evaluate_multiwindow_burn(
    *,
    short_window_burn: float,
    long_window_burn: float,
    policy: str = "starter",
) -> dict[str, Any]:
    """Evaluate a two-window burn policy.

    The default thresholds follow the commonly used SRE policy: both windows
    must show a fast burn (14.4x) before paging.  A 6x sustained burn is a
    warning/ticket, while a single-window spike is deliberately not a page.
    """
    del policy  # Kept for compatibility with the starter implementation.
    try:
        short = float(short_window_burn)
        long = float(long_window_burn)
    except (TypeError, ValueError):
        return {
            "page": False,
            "severity": "warning",
            "reason": "invalid_burn_input",
            "short_window_burn": short_window_burn,
            "long_window_burn": long_window_burn,
        }
    if not math.isfinite(short) or not math.isfinite(long) or short < 0 or long < 0:
        return {
            "page": False,
            "severity": "warning",
            "reason": "invalid_burn_input",
            "short_window_burn": short_window_burn,
            "long_window_burn": long_window_burn,
        }

    page_threshold = 14.4
    ticket_threshold = 6.0
    sustained_page = short >= page_threshold and long >= page_threshold
    sustained_warning = short >= ticket_threshold and long >= ticket_threshold
    one_window_high = max(short, long) >= ticket_threshold

    if sustained_page:
        severity = "critical"
        reason = (
            f"sustained_fast_burn; short={short:.3f}, long={long:.3f}; "
            f"both_windows>={page_threshold}"
        )
    elif sustained_warning:
        severity = "warning"
        reason = (
            f"sustained_elevated_burn_without_page; short={short:.3f}, "
            f"long={long:.3f}; both_windows>={ticket_threshold}"
        )
    elif one_window_high:
        severity = "warning"
        reason = (
            f"transient_or_unconfirmed_spike; short={short:.3f}, long={long:.3f}; "
            "both_windows_required_for_page"
        )
    else:
        severity = "info"
        reason = f"within_policy; short={short:.3f}, long={long:.3f}"

    return {
        "page": bool(sustained_page),
        "severity": severity,
        "reason": reason,
        "short_window_burn": short,
        "long_window_burn": long,
        "page_threshold": page_threshold,
        "ticket_threshold": ticket_threshold,
    }
