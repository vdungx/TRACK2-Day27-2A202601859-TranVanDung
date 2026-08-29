from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from student_api import (
    column_downstream,
    detect_distribution,
    detect_metric,
    downstream_assets,
    multiwindow_burn,
    rag_embedding_shift,
    rag_length_shift,
    slo_status,
    validate_orders,
)
from src.contract_validator import (
    failed_issues,
    load_contract,
    quarantine_dataframe,
    validate_dataframe,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
ORDERS_CONTRACT = ROOT / "contracts" / "orders_contract.yaml"


def recent_orders(rows: int = 2) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "order_id": index + 1,
                "customer_id": f"C{index + 1}",
                "amount": 10.0 + index,
                "currency": "USD",
                "status": "completed",
                "created_at": (now - timedelta(minutes=10 + index)).isoformat(),
                "updated_at": (now - timedelta(minutes=5 + index)).isoformat(),
            }
            for index in range(rows)
        ]
    )


def test_contract_detects_strict_type_drift_and_preserves_action():
    df = recent_orders()
    df["order_id"] = ["1", "2"]
    issues = validate_orders(df, ORDERS_CONTRACT)
    type_failures = [issue for issue in issues if issue["check"] == "type" and not issue["passed"]]
    assert type_failures
    assert type_failures[0]["severity"] == "critical"
    assert type_failures[0]["action"] == "block"


def test_contract_detects_missing_required_column_and_stale_freshness():
    df = recent_orders().drop(columns=["currency"])
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    df["updated_at"] = old.isoformat()
    issues = validate_orders(df, ORDERS_CONTRACT)
    assert any(issue["check"] == "required_column" and not issue["passed"] for issue in issues)
    assert any(issue["check"] == "freshness" and not issue["passed"] for issue in issues)


def test_kb_fields_and_min_length_are_supported():
    contract = load_contract(ROOT / "contracts" / "kb_contract.yaml")
    now = datetime.now(timezone.utc).isoformat()
    df = pd.DataFrame(
        [{
            "doc_id": "doc-1",
            "version": 1,
            "effective_at": now,
            "published_at": now,
            "source_uri": "support/doc.md",
            "content": "too short",
        }]
    )
    issues = validate_dataframe(df, contract)
    assert any(issue["check"] == "min_length" and not issue["passed"] for issue in issues)


def test_failed_issues_can_focus_on_critical_severity():
    df = recent_orders()
    df.loc[0, "status"] = "unknown"
    issues = validate_orders(df, ORDERS_CONTRACT)
    assert failed_issues(issues)
    assert all(issue["severity"] == "critical" for issue in failed_issues(issues, "critical"))


def test_quarantine_contains_only_offending_rows(tmp_path):
    df = recent_orders()
    df.loc[1, "currency"] = "BTC"
    contract = load_contract(ORDERS_CONTRACT)
    issues = validate_dataframe(df, contract)
    output = tmp_path / "orders_invalid.csv"
    assert quarantine_dataframe(df, issues, output, contract) == 1
    quarantined = pd.read_csv(output)
    assert quarantined["order_id"].tolist() == [2]


def test_auto_uses_same_segment_and_avoids_weekend_false_positive():
    result = detect_metric(
        250,
        [600, 610, 590, 605, 1000],
        context={
            "metric_name": "row_count",
            "day_of_week": 5,
            "same_segment_history": [240, 252, 248, 255, 245],
        },
    )
    assert result["is_anomaly"] is False
    assert "same_segment_history" in result["reason"]


def test_auto_mad_catches_current_drop_despite_historical_outlier():
    result = detect_metric(40, [100, 102, 98, 101, 1000], method="auto")
    assert result["is_anomaly"] is True
    assert result["method"].startswith("auto:")


def test_auto_handles_zero_mad_without_hiding_change():
    assert detect_metric(100, [100, 100, 100, 100, 100], method="auto")["is_anomaly"] is False
    assert detect_metric(40, [100, 100, 100, 100, 100], method="auto")["is_anomaly"] is True


def test_known_event_is_annotated_without_page():
    result = detect_metric(10, [100, 101, 99, 100, 102], context={"known_event": "campaign"})
    assert result["is_anomaly"] is False
    assert result["method"] == "auto:known_event"


def test_trend_context_scores_against_expected_next_value():
    continuation = detect_metric(125, [100, 105, 110, 115, 120], context={"trend": "up"})
    break_in_trend = detect_metric(180, [100, 105, 110, 115, 120], context={"trend": "up"})
    assert continuation["is_anomaly"] is False
    assert break_in_trend["is_anomaly"] is True


def test_nonfinite_metric_fails_closed():
    result = detect_metric(float("nan"), [1, 2, 3])
    assert result["is_anomaly"] is True
    assert result["score"] == float("inf")


def test_distribution_shape_shift_is_detected_even_with_similar_mean():
    baseline = [0, 0, 0, 10, 10, 10]
    current = [3, 3, 3, 7, 7, 7]
    result = detect_distribution(current, baseline)
    assert result["is_anomaly"] is True
    assert result["method"] == "ks_psi"


def test_identical_distribution_is_not_anomaly():
    values = [1, 2, 3, 4, 5, 6]
    assert detect_distribution(values, values)["is_anomaly"] is False


def test_lineage_traversal_is_transitive_and_cycle_safe():
    graph = {"a": ["b", "b"], "b": ["c"], "c": ["a", "d"], "d": []}
    assert downstream_assets(graph, "a") == ["b", "c", "d"]
    assert column_downstream(graph, "a") == ["b", "c", "d"]


def test_lineage_accepts_complete_graph_envelope():
    graph = {"dataset_lineage": {"raw": ["model"]}, "column_lineage": {"raw.x": ["model.y"]}}
    assert downstream_assets(graph, "raw") == ["model"]
    assert column_downstream(graph, "raw.x") == ["model.y"]


def test_slo_exact_budget_is_not_a_breach():
    result = slo_status(0.99, bad_events=1, total_events=100)
    assert result["actual_bad_rate"] == pytest.approx(result["allowed_bad_rate"])
    assert result["burn_rate"] == pytest.approx(1)
    assert result["breached"] is False


def test_multiwindow_requires_sustained_fast_burn_to_page():
    assert multiwindow_burn(20, 20)["page"] is True
    assert multiwindow_burn(20, 2)["page"] is False
    assert multiwindow_burn(2, 20)["page"] is False


def test_rag_length_and_embedding_drift():
    assert rag_length_shift(["a b", "c d"], [40, 42, 39, 41, 43])["is_anomaly"] is True
    assert rag_length_shift(["one two three four"], [4, 4, 4, 4, 4])["is_anomaly"] is False
    assert rag_embedding_shift([1.5, 1.6, 1.55], [1.0, 1.01, 0.99, 1.02])["is_anomaly"] is True
    assert rag_embedding_shift([1.0, 1.01], [1.0, 1.01, 0.99])["is_anomaly"] is False
