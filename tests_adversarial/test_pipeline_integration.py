"""End-to-end adversarial checks for the reliability baseline pipeline.

These tests deliberately run ``scripts.run_baseline.main`` against a complete
temporary mini-repository.  The production data tree is never changed: the
pipeline root is monkeypatched and every input/output is materialised below
pytest's ``tmp_path``.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from scripts import run_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _now_iso(minutes_ago: float = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _orders(
    count: int,
    *,
    updated_minutes_ago: float = 1,
    amount_values: list[object] | None = None,
    duplicate_order_id: bool = False,
    invalid_currency_row: int | None = None,
    drop_amount: bool = False,
) -> pd.DataFrame:
    if amount_values is None:
        amount_values = [float(100 + index) for index in range(count)]
    assert len(amount_values) == count

    rows = []
    for index in range(count):
        rows.append(
            {
                "order_id": 100000 + index,
                "customer_id": f"C{index + 1:04d}",
                "amount": amount_values[index],
                "currency": "BTC" if index == invalid_currency_row else "USD",
                "status": "completed",
                "created_at": _now_iso(20),
                "updated_at": _now_iso(updated_minutes_ago),
            }
        )

    frame = pd.DataFrame(rows)
    if duplicate_order_id and count >= 2:
        frame.loc[1, "order_id"] = frame.loc[0, "order_id"]
    if drop_amount:
        frame = frame.drop(columns=["amount"])
    return frame


def _documents(
    count: int = 4,
    *,
    published_minutes_ago: float = 1,
    content: str = "alpha beta gamma delta epsilon zeta eta theta",
    embedding_norms: list[float] | None = None,
) -> list[dict[str, object]]:
    if embedding_norms is None:
        embedding_norms = [1.0] * count
    assert len(embedding_norms) == count

    effective_at = _now_iso(5)
    published_at = _now_iso(published_minutes_ago)
    return [
        {
            "doc_id": f"doc-{index + 1}",
            "version": 1,
            "effective_at": effective_at,
            "published_at": published_at,
            "source_uri": f"support/doc-{index + 1}.md",
            "content": content,
            "embedding_norm": embedding_norms[index],
        }
        for index in range(count)
    ]


def _history(
    row_count: int,
    *,
    text_mean: float = 8.0,
    embedding_mean: float = 1.0,
) -> pd.DataFrame:
    # The final Saturday makes the inferred reference date Sunday.  The prior
    # Sundays give run_baseline a real same-weekday segment with >= 3 points.
    dates = [date(2026, 7, 5) + timedelta(days=7 * index) for index in range(8)]
    dates.append(date(2026, 8, 29))
    return pd.DataFrame(
        {
            "date": [value.isoformat() for value in dates],
            "day_of_week": [value.weekday() for value in dates],
            "row_count": [row_count] * len(dates),
            "null_rate": [0.01] * len(dates),
            "avg_amount": [100.0] * len(dates),
            "mean_text_length": [text_mean] * len(dates),
            "embedding_norm_mean": [embedding_mean] * len(dates),
        }
    )


def _lineage_payload() -> dict[str, object]:
    return {
        "dataset_lineage": {
            "raw_orders": ["stg_orders"],
            "stg_orders": ["fct_daily_revenue"],
            "fct_daily_revenue": ["ceo_revenue_dashboard"],
            "kb_documents": ["kb_active_docs"],
            "kb_active_docs": ["rag_index"],
            "rag_index": ["support_agent"],
        },
        "column_lineage": {
            "raw_orders.amount": ["stg_orders.amount_usd"],
            "stg_orders.amount_usd": ["fct_daily_revenue.daily_revenue"],
            "fct_daily_revenue.daily_revenue": ["ceo_revenue_dashboard.revenue"],
            "kb_documents.content": ["kb_active_docs.content"],
            "kb_active_docs.content": ["rag_index.embedding"],
            "rag_index.embedding": ["support_agent.answer"],
        },
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "child_map": {
            "model.data_reliability_lab.stg_orders": [
                "model.data_reliability_lab.fct_daily_revenue"
            ],
            "model.data_reliability_lab.fct_daily_revenue": [],
        }
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _materialise_fixture(
    root: Path,
    *,
    orders: pd.DataFrame,
    baseline_orders: pd.DataFrame,
    documents: list[dict[str, object]],
    history: pd.DataFrame,
) -> None:
    for directory in (
        root / "contracts",
        root / "data" / "incoming",
        root / "data" / "baseline",
        root / "data" / "history",
        root / "dbt_project" / "target",
        root / "reports",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for name in ("orders_contract.yaml", "kb_contract.yaml"):
        (root / "contracts" / name).write_text(
            (PROJECT_ROOT / "contracts" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "lab_config.yaml").write_text(
        (PROJECT_ROOT / "lab_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    orders.to_csv(root / "data" / "incoming" / "orders.csv", index=False)
    baseline_orders.to_csv(root / "data" / "baseline" / "orders.csv", index=False)
    history.to_csv(root / "data" / "history" / "metrics_history.csv", index=False)
    _write_jsonl(root / "data" / "incoming" / "kb_documents.jsonl", documents)
    (root / "data" / "baseline" / "lineage_graph.json").write_text(
        json.dumps(_lineage_payload(), indent=2), encoding="utf-8"
    )
    (root / "dbt_project" / "target" / "manifest.json").write_text(
        json.dumps(_manifest_payload(), indent=2), encoding="utf-8"
    )


def _run_pipeline(root: Path, monkeypatch) -> dict[str, object]:
    monkeypatch.setattr(run_baseline, "ROOT", root)
    run_baseline.main()
    report_path = root / "reports" / "latest_metrics.json"
    assert report_path.exists()
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_pipeline_healthy_path_keeps_all_signals_green(tmp_path, monkeypatch):
    orders = _orders(10)
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=orders.copy(),
        documents=_documents(),
        history=_history(10),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    assert report["orders_rows"] == 10
    assert report["failed_contract_checks"] == 0
    assert report["critical_contract_failures"] == 0
    assert report["contract_action"] == "allow"
    assert report["kb_failed_contract_checks"] == 0
    assert report["kb_action"] == "allow"
    assert report["row_count_anomaly"]["is_anomaly"] is False
    assert report["amount_distribution_shift"]["is_anomaly"] is False
    assert report["kb_text_length_signal"]["is_anomaly"] is False
    assert report["rag_embedding_signal"]["is_anomaly"] is False
    assert report["quarantine"] == {
        "orders_path": None,
        "orders_rows": 0,
        "kb_path": None,
        "kb_rows": 0,
    }
    assert not (tmp_path / "data" / "quarantine").exists()


def test_pipeline_duplicate_contract_blocks_and_quarantines_only_duplicates(
    tmp_path, monkeypatch
):
    orders = _orders(6, duplicate_order_id=True)
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(6),
        documents=_documents(),
        history=_history(6),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    assert report["contract_action"] == "block"
    assert report["critical_contract_failures"] >= 1
    assert any(
        issue["check"] == "unique"
        and issue["column"] == "order_id"
        and issue["passed"] is False
        for issue in report["contract_issues"]
    )
    assert report["row_count_anomaly"]["is_anomaly"] is False
    assert report["quarantine"]["orders_rows"] == 2

    quarantined = pd.read_csv(tmp_path / "data" / "quarantine" / "orders_invalid.csv")
    assert quarantined["order_id"].tolist() == [100000, 100000]
    assert report["amount_distribution_shift"]["is_anomaly"] is False


def test_pipeline_volume_drop_uses_same_weekday_history_without_contract_failure(
    tmp_path, monkeypatch
):
    orders = _orders(2, amount_values=[100.0, 101.0])
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(2, amount_values=[100.0, 101.0]),
        documents=_documents(),
        history=_history(10),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    assert report["failed_contract_checks"] == 0
    assert report["contract_action"] == "allow"
    assert report["row_count_anomaly"]["is_anomaly"] is True
    assert report["row_count_anomaly"]["method"].startswith("auto:")
    assert "same_segment_history" in report["row_count_anomaly"]["reason"] or (
        "baseline_source=same_segment_history" in report["row_count_anomaly"]["reason"]
    )
    assert report["amount_distribution_shift"]["is_anomaly"] is False
    assert report["quarantine"]["orders_rows"] == 0


def test_pipeline_distribution_shape_drift_is_detected_with_similar_mean(
    tmp_path, monkeypatch
):
    current_amounts = [50.5] * 100
    baseline_amounts = [1.0] * 50 + [100.0] * 50
    orders = _orders(100, amount_values=current_amounts)
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(100, amount_values=baseline_amounts),
        documents=_documents(),
        history=_history(100),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    distribution = report["amount_distribution_shift"]
    assert report["contract_action"] == "allow"
    assert distribution["is_anomaly"] is True
    assert distribution["method"] == "ks_psi"
    assert "shape_anomaly=True" in distribution["reason"]
    assert "mean_ratio_anomaly=False" in distribution["reason"]
    assert report["quarantine"]["orders_rows"] == 0


def test_pipeline_stale_kb_surfaces_freshness_and_both_rag_signals(
    tmp_path, monkeypatch
):
    orders = _orders(8)
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(8),
        documents=_documents(
            published_minutes_ago=180,
            content="tiny knowledge text remains valid",
            embedding_norms=[2.0] * 4,
        ),
        history=_history(8, text_mean=40.0, embedding_mean=1.0),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    freshness_issues = [
        issue
        for issue in report["kb_contract_issues"]
        if issue["check"] == "freshness" and issue["passed"] is False
    ]
    assert freshness_issues
    assert report["kb_freshness_minutes"] > 60
    assert report["kb_action"] == "quarantine"
    assert report["kb_text_length_signal"]["is_anomaly"] is True
    assert report["rag_embedding_signal"]["is_anomaly"] is True
    assert report["kb_slo"]["breached"] is True
    assert report["quarantine"]["kb_rows"] == 4

    quarantined = pd.read_csv(tmp_path / "data" / "quarantine" / "kb_documents_invalid.csv")
    assert len(quarantined) == 4
    assert report["contract_action"] == "allow"
    assert report["orders_rows"] == 8


def test_pipeline_missing_amount_column_fails_closed_without_distribution_crash(
    tmp_path, monkeypatch
):
    orders = _orders(5, drop_amount=True)
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(5),
        documents=_documents(),
        history=_history(5),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    assert report["contract_action"] == "block"
    assert any(
        issue["check"] == "required_column"
        and issue["column"] == "amount"
        and issue["passed"] is False
        for issue in report["contract_issues"]
    )
    assert report["amount_distribution_shift"]["is_anomaly"] is False
    assert report["amount_distribution_shift"]["reason"] == "empty_input"
    assert report["quarantine"]["orders_rows"] == 5
    assert (tmp_path / "data" / "quarantine" / "orders_invalid.csv").exists()


def test_pipeline_combines_contract_rag_and_distribution_failures_safely(
    tmp_path, monkeypatch
):
    orders = _orders(6, duplicate_order_id=True, invalid_currency_row=3)
    orders.loc[2, "amount"] = None
    _materialise_fixture(
        tmp_path,
        orders=orders,
        baseline_orders=_orders(6),
        documents=_documents(
            count=3,
            published_minutes_ago=180,
            content="tiny knowledge text remains valid",
            embedding_norms=[2.0] * 3,
        ),
        history=_history(6, text_mean=40.0, embedding_mean=1.0),
    )

    report = _run_pipeline(tmp_path, monkeypatch)

    assert report["contract_action"] == "block"
    assert report["critical_contract_failures"] >= 3
    failed_checks = {
        (issue["check"], issue["column"])
        for issue in report["contract_issues"]
        if issue["passed"] is False
    }
    assert ("unique", "order_id") in failed_checks
    assert ("not_null", "amount") in failed_checks
    assert ("accepted_values", "currency") in failed_checks
    assert report["amount_distribution_shift"]["is_anomaly"] is True
    assert "invalid_numeric_input" in report["amount_distribution_shift"]["reason"]
    assert report["kb_action"] == "quarantine"
    assert report["kb_text_length_signal"]["is_anomaly"] is True
    assert report["rag_embedding_signal"]["is_anomaly"] is True
    assert report["multiwindow_burn"]["page"] is True
    assert report["quarantine"]["orders_rows"] == 4
    assert report["quarantine"]["kb_rows"] == 3

    quarantined_orders = pd.read_csv(
        tmp_path / "data" / "quarantine" / "orders_invalid.csv"
    )
    assert quarantined_orders["order_id"].tolist() == [100000, 100000, 100002, 100003]


def test_pipeline_recovery_removes_old_quarantine_artifacts(tmp_path, monkeypatch):
    bad_orders = _orders(4, invalid_currency_row=0)
    _materialise_fixture(
        tmp_path,
        orders=bad_orders,
        baseline_orders=_orders(4),
        documents=_documents(published_minutes_ago=180),
        history=_history(4),
    )
    bad_report = _run_pipeline(tmp_path, monkeypatch)
    order_quarantine = tmp_path / "data" / "quarantine" / "orders_invalid.csv"
    kb_quarantine = tmp_path / "data" / "quarantine" / "kb_documents_invalid.csv"
    assert bad_report["contract_action"] == "block"
    assert bad_report["kb_action"] == "quarantine"
    assert order_quarantine.exists()
    assert kb_quarantine.exists()

    healthy_orders = _orders(4)
    _materialise_fixture(
        tmp_path,
        orders=healthy_orders,
        baseline_orders=_orders(4),
        documents=_documents(),
        history=_history(4),
    )
    recovered_report = _run_pipeline(tmp_path, monkeypatch)

    assert recovered_report["failed_contract_checks"] == 0
    assert recovered_report["kb_failed_contract_checks"] == 0
    assert recovered_report["contract_action"] == "allow"
    assert recovered_report["kb_action"] == "allow"
    assert recovered_report["quarantine"]["orders_rows"] == 0
    assert recovered_report["quarantine"]["kb_rows"] == 0
    assert recovered_report["quarantine"]["orders_path"] is None
    assert recovered_report["quarantine"]["kb_path"] is None
    assert not order_quarantine.exists()
    assert not kb_quarantine.exists()
