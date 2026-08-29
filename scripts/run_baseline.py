#!/usr/bin/env python3
"""Run the lab's evidence-first reliability checks and write a JSON report."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observability.anomaly import detect_anomaly
from observability.distribution import detect_distribution_shift
from observability.lineage import (
    extract_dbt_dataset_graph,
    get_column_downstream,
    get_downstream_assets,
)
from observability.rag_metrics import (
    detect_embedding_norm_shift,
    detect_text_length_shift,
)
from observability.slo import calculate_slo, evaluate_multiwindow_burn
from src.contract_validator import (
    failed_issues,
    load_contract,
    quarantine_dataframe,
    validate_dataframe,
)
from src.io_utils import load_jsonl, load_yaml


def _freshness_minutes(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    parsed = df[column].map(lambda value: pd.to_datetime(value, utc=True, errors="coerce"))
    parsed = parsed.dropna()
    if parsed.empty:
        return None
    latest = parsed.max()
    return max(0.0, (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60.0)


def _reference_date(history: pd.DataFrame) -> tuple[str, int]:
    """Infer the intended next batch date from the supplied history.

    The synthetic incoming batch is generated independently of the machine's
    wall clock. Using the day after the last historical record keeps the
    seasonality comparison reproducible when a lab is run later.
    """
    dates = pd.to_datetime(history.get("date", pd.Series(dtype=str)), errors="coerce").dropna()
    if dates.empty:
        value = pd.Timestamp.now(tz="UTC").date()
    else:
        value = (dates.max() + pd.Timedelta(days=1)).date()
    return value.isoformat(), int(value.weekday())


def _history_segment(history: pd.DataFrame, column: str, day_of_week: int) -> list[float]:
    if "day_of_week" in history.columns:
        segment = history.loc[history["day_of_week"] == day_of_week, column].dropna().tail(8)
        if len(segment) >= 3:
            return segment.astype(float).tolist()
    return history[column].dropna().tail(14).astype(float).tolist()


def _action_for_failures(issues: list[dict[str, Any]]) -> str:
    failed = failed_issues(issues)
    if any(issue.get("severity") == "critical" for issue in failed):
        return "block"
    if failed:
        return "quarantine"
    return "allow"


def _optional_embedding_signal(docs: list[dict[str, Any]], history: pd.DataFrame) -> dict[str, Any]:
    current_norms = [doc["embedding_norm"] for doc in docs if "embedding_norm" in doc]
    if not current_norms or "embedding_norm_mean" not in history.columns:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding_norm_robust",
            "reason": "no_precomputed_embedding_norms",
        }
    return detect_embedding_norm_shift(
        current_norms,
        history["embedding_norm_mean"].dropna().tail(14).tolist(),
    )


def main() -> None:
    config = load_yaml(ROOT / "lab_config.yaml")
    orders = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    baseline_orders = pd.read_csv(ROOT / "data" / "baseline" / "orders.csv")
    history = pd.read_csv(ROOT / "data" / "history" / "metrics_history.csv")

    orders_contract = load_contract(ROOT / "contracts" / "orders_contract.yaml")
    order_issues = validate_dataframe(orders, orders_contract)
    order_failed = failed_issues(order_issues)
    order_critical_failed = failed_issues(order_issues, min_severity="critical")

    docs = load_jsonl(ROOT / "data" / "incoming" / "kb_documents.jsonl")
    kb_df = pd.DataFrame(docs)
    kb_contract = load_contract(ROOT / "contracts" / "kb_contract.yaml")
    kb_issues = validate_dataframe(kb_df, kb_contract)
    kb_failed = failed_issues(kb_issues)
    kb_critical_failed = failed_issues(kb_issues, min_severity="critical")

    reference_date, reference_dow = _reference_date(history)
    row_history = _history_segment(history, "row_count", reference_dow)
    row_result = detect_anomaly(
        len(orders),
        row_history,
        method="auto",
        context={
            "metric_name": "row_count",
            "day_of_week": reference_dow,
            "same_segment_history": row_history,
        },
    )

    amount_result = detect_distribution_shift(
        orders["amount"].dropna().tolist(),
        baseline_orders["amount"].dropna().tolist(),
    )
    text_result = detect_text_length_shift(
        [str(doc.get("content", "")) for doc in docs],
        history["mean_text_length"].dropna().tail(14).tolist(),
    )
    embedding_result = _optional_embedding_signal(docs, history)

    critical_bad = 1 if order_critical_failed or kb_critical_failed else 0
    contract_target = float(
        config.get("slo", {}).get("critical_contract_pass", {}).get("target", 0.999)
    )
    contract_slo = calculate_slo(contract_target, bad_events=critical_bad, total_events=1)
    kb_target = float(config.get("slo", {}).get("rag_index_freshness", {}).get("target", 0.99))
    kb_slo = calculate_slo(kb_target, bad_events=int(bool(kb_failed)), total_events=1)
    burn = evaluate_multiwindow_burn(
        short_window_burn=contract_slo["burn_rate"],
        long_window_burn=contract_slo["burn_rate"],
    )

    with open(ROOT / "data" / "baseline" / "lineage_graph.json", "r", encoding="utf-8") as f:
        lineage_payload = json.load(f)
    dataset_lineage = lineage_payload.get("dataset_lineage", lineage_payload)
    column_lineage = lineage_payload.get("column_lineage", {})
    blast_radius = get_downstream_assets(dataset_lineage, "stg_orders")
    column_blast_radius = get_column_downstream(column_lineage, "raw_orders.amount")
    manifest_graph = extract_dbt_dataset_graph(
        ROOT / "dbt_project" / "target" / "manifest.json"
    )
    dbt_blast_radius = get_downstream_assets(manifest_graph, "model.data_reliability_lab.stg_orders")

    quarantine_dir = ROOT / "data" / "quarantine"
    order_quarantine_path = quarantine_dir / "orders_invalid.csv"
    kb_quarantine_path = quarantine_dir / "kb_documents_invalid.csv"
    order_quarantine_rows = quarantine_dataframe(
        orders, order_issues, order_quarantine_path, orders_contract
    )
    kb_quarantine_rows = quarantine_dataframe(
        kb_df, kb_issues, kb_quarantine_path, kb_contract
    )

    order_freshness = _freshness_minutes(orders, "updated_at")
    kb_freshness = _freshness_minutes(kb_df, "published_at")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "orders_rows": int(len(orders)),
        "failed_contract_checks": len(order_failed),
        "critical_contract_failures": len(order_critical_failed),
        "contract_action": _action_for_failures(order_issues),
        "contract_issues": order_issues,
        "kb_failed_contract_checks": len(kb_failed),
        "kb_critical_contract_failures": len(kb_critical_failed),
        "kb_action": _action_for_failures(kb_issues),
        "kb_contract_issues": kb_issues,
        "row_count_anomaly": row_result,
        "amount_distribution_shift": amount_result,
        "freshness_minutes": order_freshness,
        "kb_freshness_minutes": kb_freshness,
        "kb_text_length_signal": text_result,
        "rag_embedding_signal": embedding_result,
        "contract_slo": contract_slo,
        "kb_slo": kb_slo,
        "multiwindow_burn": burn,
        "sample_blast_radius_from_stg_orders": blast_radius,
        "column_blast_radius_from_raw_orders_amount": column_blast_radius,
        "dbt_blast_radius_from_stg_orders": dbt_blast_radius,
        "seasonality_reference_date": reference_date,
        "seasonality_reference_day_of_week": reference_dow,
        "quarantine": {
            "orders_path": str(order_quarantine_path.relative_to(ROOT)) if order_quarantine_rows else None,
            "orders_rows": order_quarantine_rows,
            "kb_path": str(kb_quarantine_path.relative_to(ROOT)) if kb_quarantine_rows else None,
            "kb_rows": kb_quarantine_rows,
        },
    }
    out = ROOT / "reports" / "latest_metrics.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== DATA RELIABILITY BASELINE ===")
    print(f"orders rows              : {len(orders)}")
    print(f"contract failed checks   : {len(order_failed)}")
    print(f"critical contract fails  : {len(order_critical_failed)}")
    print(f"contract action          : {_action_for_failures(order_issues)}")
    print(
        f"row-count anomaly        : {row_result['is_anomaly']} "
        f"({row_result['method']}, score={row_result['score']:.2f})"
    )
    print(f"freshness minutes        : {order_freshness if order_freshness is not None else 'unknown'}")
    print(f"KB failed checks         : {len(kb_failed)}")
    print(f"KB freshness minutes    : {kb_freshness if kb_freshness is not None else 'unknown'}")
    print(f"KB length anomaly        : {text_result['is_anomaly']}")
    print(f"distribution anomaly     : {amount_result['is_anomaly']}")
    print(f"multi-window page        : {burn['page']} ({burn['severity']})")
    print(f"sample blast radius      : {', '.join(blast_radius)}")
    print(f"column blast radius      : {', '.join(column_blast_radius)}")
    print(f"report                    : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
