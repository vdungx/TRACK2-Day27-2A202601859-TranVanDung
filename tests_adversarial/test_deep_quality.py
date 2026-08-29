from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observability.anomaly import detect_anomaly
from observability.rag_metrics import (
    approximate_token_lengths,
    detect_embedding_norm_shift,
)
from observability.distribution import detect_distribution_shift
from src.contract_validator import (
    failed_issues,
    load_contract,
    quarantine_dataframe,
    validate_dataframe,
)
from student_api import detect_metric, detect_distribution, rag_length_shift, validate_orders


ROOT = Path(__file__).resolve().parents[1]
ORDERS_CONTRACT = ROOT / "contracts" / "orders_contract.yaml"


def recent_orders(rows: int = 5) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "order_id": index + 1,
                "customer_id": f"C{index + 1}",
                "amount": float(10 + index),
                "currency": "USD",
                "status": "completed",
                "created_at": (now - timedelta(minutes=20 + index)).isoformat(),
                "updated_at": (now - timedelta(minutes=5 + index)).isoformat(),
            }
            for index in range(rows)
        ]
    )


def failed_for(issues: list[dict], check: str, column: str | None = None) -> list[dict]:
    return [
        issue
        for issue in issues
        if issue["check"] == check
        and issue.get("column") == column
        and not issue["passed"]
    ]


def test_contract_rejects_mixed_types_and_whitespace_missing_values():
    df = recent_orders(4)
    df.loc[0, "customer_id"] = "   "
    df["amount"] = df["amount"].astype(object)
    df.loc[1, "amount"] = "12.5"
    df.loc[2, "currency"] = "BTC"
    df.loc[3, "status"] = "unknown"

    issues = validate_orders(df, ORDERS_CONTRACT)

    assert failed_for(issues, "not_null", "customer_id")
    amount_type = failed_for(issues, "type", "amount")
    assert amount_type and amount_type[0]["action"] == "block"
    assert failed_for(issues, "accepted_values", "currency")
    status_failure = failed_for(issues, "accepted_values", "status")
    assert status_failure and status_failure[0]["severity"] == "warning"
    assert status_failure[0]["action"] == "quarantine"


def test_contract_freshness_requires_every_value_to_be_present_and_recent():
    df = recent_orders(2)
    df.loc[1, "updated_at"] = pd.NaT
    issues = validate_orders(df, ORDERS_CONTRACT)
    freshness = [issue for issue in issues if issue["check"] == "freshness"][0]

    assert freshness["passed"] is False
    assert "null_count=1" in freshness["details"]

    now = datetime.now(timezone.utc)
    df["updated_at"] = [
        (now - timedelta(minutes=31)).isoformat(),
        (now - timedelta(minutes=31)).isoformat(),
    ]
    stale = [
        issue
        for issue in validate_orders(df, ORDERS_CONTRACT)
        if issue["check"] == "freshness"
    ][0]
    assert stale["passed"] is False


def test_contract_supports_fields_and_inclusive_length_boundaries():
    contract = {
        "dataset": "documents",
        "fields": {
            "content": {
                "required": True,
                "type": "string",
                "min_length": 3,
                "max_length": 5,
            },
            "version": {"required": True, "type": "integer"},
        },
    }
    valid = validate_dataframe(
        pd.DataFrame({"content": ["abc", "12345"], "version": [1, 2]}),
        contract,
    )
    assert not failed_issues(valid)

    invalid = validate_dataframe(
        pd.DataFrame({"content": ["ab", "123456"], "version": [1, 2]}),
        contract,
    )
    length_failures = failed_for(invalid, "min_length", "content")
    assert length_failures
    assert "invalid_count=2" in length_failures[0]["details"]


def test_contract_unknown_declared_type_fails_closed():
    issues = validate_dataframe(
        pd.DataFrame({"identifier": ["a", "b"]}),
        {
            "columns": {
                "identifier": {
                    "required": True,
                    "type": "uuid_v7",
                    "severity": "critical",
                }
            }
        },
    )
    failure = failed_for(issues, "type", "identifier")[0]
    assert failure["severity"] == "critical"
    assert failure["action"] == "block"


def test_quarantine_unions_multiple_row_level_failures_without_mutating_input(tmp_path):
    df = recent_orders(5)
    df.loc[1, "currency"] = "BTC"
    df.loc[2, "status"] = "invalid"
    df.loc[[3, 4], "order_id"] = 99
    before = df.copy(deep=True)
    contract = load_contract(ORDERS_CONTRACT)
    issues = validate_dataframe(df, contract)
    output = tmp_path / "quarantine.csv"

    count = quarantine_dataframe(df, issues, output, contract)
    quarantined = pd.read_csv(output)

    assert count == 4
    assert quarantined["order_id"].tolist() == [2, 3, 99, 99]
    pd.testing.assert_frame_equal(df, before)


def test_batch_level_type_failure_quarantines_every_row_and_recovery_clears_artifact(tmp_path):
    contract = load_contract(ORDERS_CONTRACT)
    df = recent_orders(3)
    df["amount"] = df["amount"].astype(object)
    df.loc[0, "amount"] = "not-a-number"
    output = tmp_path / "orders_invalid.csv"

    bad_issues = validate_dataframe(df, contract)
    assert quarantine_dataframe(df, bad_issues, output, contract) == len(df)
    assert output.exists()

    healthy = recent_orders(3)
    healthy_issues = validate_dataframe(healthy, contract)
    assert quarantine_dataframe(healthy, healthy_issues, output, contract) == 0
    assert not output.exists()


def test_severity_actions_and_critical_filter_are_consistent():
    contract = {
        "columns": {
            "critical_value": {"required": True, "type": "integer", "severity": "critical"},
            "warning_value": {"required": True, "type": "integer", "severity": "warning"},
            "info_value": {"required": True, "type": "integer", "severity": "info"},
        }
    }
    issues = validate_dataframe(
        pd.DataFrame(
            {"critical_value": ["bad"], "warning_value": ["bad"], "info_value": ["bad"]}
        ),
        contract,
    )
    failures = failed_issues(issues)
    assert {issue["severity"] for issue in failures} == {"critical", "warning", "info"}
    actions = {issue["severity"]: issue["action"] for issue in failures if issue["check"] == "type"}
    assert actions == {"critical": "block", "warning": "quarantine", "info": "warn"}
    assert all(issue["severity"] == "critical" for issue in failed_issues(issues, "critical"))


def test_auto_anomaly_keeps_good_history_when_one_observation_is_corrupt():
    result = detect_metric(
        40,
        [100, 102, "bad", 98, 100, 100, float("nan"), 100],
    )
    assert result["is_anomaly"] is True
    assert result["score"] > 3


def test_known_event_cannot_suppress_nonfinite_current_value():
    result = detect_metric(
        float("nan"),
        [100, 101, 99, 100, 102],
        context={"known_event": "planned campaign"},
    )
    assert result["is_anomaly"] is True
    assert result["score"] == float("inf")
    assert "not_finite" in result["reason"]


def test_detect_anomaly_honors_explicit_mad_threshold():
    history = [100, 102, 98, 101, 100, 99, 103]
    relaxed = detect_anomaly(104, history, method="mad", threshold=2.0)
    strict = detect_anomaly(104, history, method="mad", threshold=3.0)

    assert relaxed["is_anomaly"] is True
    assert strict["is_anomaly"] is False


def test_auto_uses_string_day_key_from_history_by_day():
    result = detect_metric(
        20,
        [100, 101, 99, 100, 102],
        context={
            "day_of_week": 5,
            "history_by_day": {"5": [19, 20, 21, 20, 20]},
        },
    )
    assert result["is_anomaly"] is False
    assert "same_segment_history" in result["reason"]


def test_explicit_zscore_filters_bad_history_values_individually():
    result = detect_metric(40, [100, 101, "bad", 99, 100], method="zscore")
    assert result["is_anomaly"] is True
    assert result["method"] == "zscore"


def test_anomaly_method_rejects_unknown_detector():
    with pytest.raises(ValueError, match="Unsupported method"):
        detect_metric(1, [1, 1, 1], method="does-not-exist")


def test_rag_token_lengths_treat_missing_numeric_values_as_empty_text():
    assert approximate_token_lengths(["hello world", None, np.nan, ""]) == [2, 0, 0, 0]


def test_rag_text_detector_keeps_valid_baseline_when_one_batch_stat_is_corrupt():
    result = rag_length_shift(
        ["short"],
        [10, "bad", 10, 10, 10, 10],
    )
    assert result["is_anomaly"] is True
    assert result["metric"] == "mean_text_length"
    assert result["current_mean"] == 1


def test_embedding_detector_reports_invalid_norms_instead_of_returning_empty_safe():
    result = detect_embedding_norm_shift([1.0, "bad", 1.0], [1.0, 1.0, 1.0])
    assert result["is_anomaly"] is True
    assert result["score"] == float("inf")
    assert "invalid_numeric_input" in result["reason"]


def test_embedding_constant_baseline_uses_relative_guard_for_small_noise():
    small_shift = detect_embedding_norm_shift([1.01, 1.0], [1.0, 1.0, 1.0, 1.0])
    large_shift = detect_embedding_norm_shift([1.25, 1.24], [1.0, 1.0, 1.0, 1.0])

    assert small_shift["is_anomaly"] is False
    assert large_shift["is_anomaly"] is True


def test_embedding_negative_norms_fail_closed():
    result = detect_embedding_norm_shift([-1.0, -1.0], [1.0, 1.0])
    assert result["is_anomaly"] is True
    assert result["reason"] == "negative_embedding_norm"


def test_embedding_baseline_outlier_does_not_create_false_drift():
    result = detect_embedding_norm_shift(
        [1.00, 1.01, 0.99],
        [1.00, 1.01, 0.99, 1.02, 10.0],
    )
    assert result["is_anomaly"] is False


def test_distribution_invalid_values_fail_closed_with_evidence():
    result = detect_distribution(
        [1.0, 2.0, "not-a-number", 3.0],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert result["is_anomaly"] is True
    assert result["score"] == float("inf")
    assert "current_invalid=1" in result["reason"]


def test_distribution_detects_shape_drift_when_means_match():
    baseline = [-1.0] * 50 + [1.0] * 50
    current = [0.0] * 100
    result = detect_distribution(current, baseline)
    assert result["is_anomaly"] is True
    assert result["method"] == "ks_psi"
    assert "shape_anomaly=True" in result["reason"]


def test_distribution_small_samples_do_not_page_on_ordinary_concentration():
    result = detect_distribution([1.0, 9.0], [0.0, 10.0])
    assert result["is_anomaly"] is False
    assert "psi_usable=False" in result["reason"]


def test_distribution_is_order_invariant_and_validates_ratio_threshold():
    baseline = [1, 2, 3, 4, 5, 6]
    current = [1, 2, 3, 4, 5, 6]
    first = detect_distribution_shift(current, baseline)
    second = detect_distribution_shift(list(reversed(current)), list(reversed(baseline)))
    assert second["is_anomaly"] == first["is_anomaly"]
    assert second["score"] == pytest.approx(first["score"])

    with pytest.raises(ValueError, match="ratio_threshold"):
        detect_distribution_shift(current, baseline, ratio_threshold=0)
