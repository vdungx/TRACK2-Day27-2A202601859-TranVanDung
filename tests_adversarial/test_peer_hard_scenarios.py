"""Regression tests for the hard scenarios mirrored from the 20/20 solution."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from gx.validate_orders import QuarantineOnFailure
from observability.anomaly import detect_anomaly
from observability.distribution import (
    detect_categorical_shift,
    detect_distribution_shift,
)
from observability.health import incident_decision, signal
from observability.slo import evaluate_multiwindow_burn, evaluate_slo_history
from src.contract_validator import validate_dataframe


def test_freshness_supports_deterministic_reference_and_future_skew():
    stale_contract = {
        "freshness": {
            "column": "updated_at",
            "max_delay_minutes": 30,
            "max_future_minutes": 5,
            "reference_time": "2026-08-29T10:31:00Z",
        }
    }
    stale = validate_dataframe(
        pd.DataFrame({"updated_at": ["2026-08-29T10:00:00Z"]}),
        stale_contract,
    )[0]
    assert stale["passed"] is False
    assert "condition=stale" in stale["details"]

    future_contract = {
        **stale_contract,
        "freshness": {
            **stale_contract["freshness"],
            "reference_time": "2026-08-29T10:00:00Z",
        },
    }
    future = validate_dataframe(
        pd.DataFrame({"updated_at": ["2026-08-29T10:06:00Z"]}),
        future_contract,
    )[0]
    assert future["passed"] is False
    assert "condition=future_timestamp" in future["details"]


def test_integer_contract_rejects_boolean_type_drift():
    issues = validate_dataframe(
        pd.DataFrame({"id": [True, False]}),
        {"columns": {"id": {"required": True, "type": "integer"}}},
    )
    assert next(issue for issue in issues if issue["check"] == "type")["passed"] is False


def test_auto_anomaly_uses_same_segment_and_reports_seasonal_method():
    result = detect_anomaly(
        102,
        [100, 101, 99, 100, 10_000, 101, 99],
        method="auto",
        context={"same_segment_history": [98, 99, 100, 101, 102, 100]},
    )
    assert result["is_anomaly"] is False
    assert result["method"] == "auto:seasonal_mad"


def test_shape_and_category_drift_do_not_require_mean_shift():
    assert detect_distribution_shift(
        [-100, -100, 100, 100], [-1, -1, 1, 1]
    )["is_anomaly"] is True
    assert detect_categorical_shift(
        ["refunded"] * 10, ["completed"] * 10
    )["is_anomaly"] is True


def test_empty_current_distribution_fails_closed():
    result = detect_distribution_shift([], [1, 2, 3])
    assert result["is_anomaly"] is True
    assert result["reason"] == "current_batch_empty"


def test_slo_history_pages_only_after_enough_sustained_evidence():
    assert evaluate_multiwindow_burn(
        short_window_burn=20, long_window_burn=1
    )["page"] is False
    assert evaluate_multiwindow_burn(
        short_window_burn=20, long_window_burn=7
    )["page"] is True

    cold_start = evaluate_slo_history([False], target=0.995)
    assert cold_start["alert"]["page"] is False
    assert cold_start["alert"]["reason"] == "insufficient_window_history"

    sustained = evaluate_slo_history([False] * 5, target=0.995)
    assert sustained["alert"]["page"] is True
    assert sustained["alert"]["severity"] == "critical"


def test_critical_signal_closes_publish_gate_and_traces_blast_radius():
    graph = {
        "raw_orders": ["stg_orders"],
        "stg_orders": ["revenue"],
        "revenue": ["dashboard"],
    }
    signals = [
        signal(
            "duplicate_pk",
            domain="orders",
            fired=True,
            severity="critical",
            action="block",
            owner="commerce-data",
            summary="duplicate key",
            source_asset="raw_orders",
        )
    ]
    result = incident_decision(signals, graph)
    assert result["severity"] == "P1"
    assert result["publish_downstream"] is False
    assert result["affected_assets"] == [
        "raw_orders",
        "stg_orders",
        "revenue",
        "dashboard",
    ]


def test_gx_quarantine_action_preserves_rejected_batch(tmp_path):
    source = tmp_path / "orders.csv"
    source.write_text("order_id\n1\n1\n", encoding="utf-8")
    target = tmp_path / "quarantine"
    action = QuarantineOnFailure(
        name="quarantine_test",
        quarantine_dir=str(target),
        source_path=str(source),
    )
    result = action.run(SimpleNamespace(success=False), None)
    assert result["action"] == "quarantine"
    assert list(target.glob("orders-*.csv"))
