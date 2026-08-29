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
from observability.distribution import (
    detect_categorical_shift,
    detect_distribution_shift,
)
from observability.health import incident_decision, signal
from observability.lineage import (
    extract_dbt_dataset_graph,
    get_column_downstream,
    get_downstream_assets,
)
from observability.rag_metrics import (
    detect_embedding_norm_shift,
    detect_text_length_shift,
)
from observability.slo import (
    calculate_slo,
    evaluate_multiwindow_burn,
    evaluate_slo_history,
)
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
    if column not in history.columns:
        return []
    numeric = pd.to_numeric(history[column], errors="coerce")
    if "day_of_week" in history.columns:
        segment = numeric.loc[history["day_of_week"] == day_of_week].dropna().tail(8)
        if len(segment) >= 3:
            return segment.tolist()
    return numeric.dropna().tail(14).tolist()


def _column_values(df: pd.DataFrame, column: str) -> list[Any]:
    """Return a metric column without crashing on a contract-level omission."""
    if column not in df.columns:
        return []
    return df[column].tolist()


def _failed_check(
    issues: list[dict[str, Any]], check: str
) -> dict[str, Any] | None:
    return next(
        (
            issue
            for issue in issues
            if issue.get("check") == check and not issue.get("passed")
        ),
        None,
    )


def _read_optional_csv(path: Path) -> pd.DataFrame:
    """Load an optional control-plane input without weakening core checks."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _kb_version_rollbacks(
    current_docs: list[dict[str, Any]], baseline_docs: list[dict[str, Any]]
) -> list[str]:
    baseline_versions = {
        str(doc.get("doc_id")): doc.get("version")
        for doc in baseline_docs
        if isinstance(doc, dict)
    }
    rollback_ids: list[str] = []
    for doc in current_docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id"))
        baseline_version = baseline_versions.get(doc_id)
        try:
            if baseline_version is not None and int(doc.get("version")) < int(
                baseline_version
            ):
                rollback_ids.append(doc_id)
        except (TypeError, ValueError):
            continue
    return sorted(set(rollback_ids))


def _read_run_history() -> list[dict[str, Any]]:
    path = ROOT / "reports" / "monitoring_history.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("slis"), dict):
            rows.append(payload)
    return rows[-199:]


def _write_run_history(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "reports" / "monitoring_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(
        json.dumps(row, sort_keys=True, default=str) for row in rows[-200:]
    )
    path.write_text(rendered + ("\n" if rendered else ""), encoding="utf-8")


def _action_for_failures(issues: list[dict[str, Any]]) -> str:
    failed = failed_issues(issues)
    if any(issue.get("severity") == "critical" for issue in failed):
        return "block"
    if failed:
        return "quarantine"
    return "allow"


def _optional_embedding_signal(docs: list[dict[str, Any]], history: pd.DataFrame) -> dict[str, Any]:
    current_norms = [
        doc["embedding_norm"]
        for doc in docs
        if isinstance(doc, dict) and "embedding_norm" in doc
    ]
    if not current_norms or "embedding_norm_mean" not in history.columns:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding:embedding_norm_robust",
            "metric": "embedding_norm_distribution",
            "reason": "no_precomputed_embedding_norms",
        }
    return detect_embedding_norm_shift(
        current_norms,
        history["embedding_norm_mean"].dropna().tail(14).tolist(),
    )


def main() -> None:
    now = datetime.now(timezone.utc)
    config = load_yaml(ROOT / "lab_config.yaml")
    orders = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    customers = _read_optional_csv(ROOT / "data" / "incoming" / "customers.csv")
    baseline_orders = pd.read_csv(ROOT / "data" / "baseline" / "orders.csv")
    history = pd.read_csv(ROOT / "data" / "history" / "metrics_history.csv")

    orders_contract = load_contract(ROOT / "contracts" / "orders_contract.yaml")
    order_issues = validate_dataframe(orders, orders_contract)
    order_failed = failed_issues(order_issues)
    order_critical_failed = failed_issues(order_issues, min_severity="critical")

    docs = load_jsonl(ROOT / "data" / "incoming" / "kb_documents.jsonl")
    baseline_kb_path = ROOT / "data" / "baseline" / "kb_documents.jsonl"
    baseline_docs = load_jsonl(baseline_kb_path) if baseline_kb_path.exists() else []
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

    # A missing contract column is a structural failure, not the same thing
    # as a present-but-empty metric batch. Keep the distribution signal
    # neutral here because the contract/quarantine result already owns the
    # blocking decision; direct detector callers still fail closed on an
    # explicitly empty current batch.
    if "amount" not in orders.columns or "amount" not in baseline_orders.columns:
        amount_result = {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_psi",
            "reason": "empty_input",
        }
    else:
        amount_result = detect_distribution_shift(
            _column_values(orders, "amount"),
            _column_values(baseline_orders, "amount"),
        )
    status_result = detect_categorical_shift(
        _column_values(orders, "status"),
        _column_values(baseline_orders, "status"),
    )
    observed_columns = [
        str(name)
        for name in orders_contract.get("columns", {})
        if name in orders.columns
    ]
    null_rate = (
        0.0
        if orders.empty or not observed_columns
        else float(orders[observed_columns].isna().sum().sum())
        / (len(orders) * len(observed_columns))
    )
    null_history = (
        pd.to_numeric(history["null_rate"], errors="coerce").dropna().tail(30).tolist()
        if "null_rate" in history.columns
        else []
    )
    null_rate_result = detect_anomaly(
        null_rate,
        null_history,
        method="auto",
        context={"metric_name": "null_rate"},
    )
    rollback_ids = _kb_version_rollbacks(docs, baseline_docs)
    active_mask = (
        customers.get("is_active", pd.Series(False, index=customers.index))
        .astype(str)
        .str.lower()
        .eq("true")
    )
    active_counts = (
        customers.loc[active_mask].groupby("customer_id").size()
        if "customer_id" in customers.columns
        else pd.Series(dtype=int)
    )
    duplicate_active_customers = sorted(
        map(str, active_counts[active_counts > 1].index.tolist())
    )
    customer_ids = set(
        customers.get("customer_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    order_customer_ids = set(
        orders.get("customer_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    orphan_customer_ids = sorted(order_customer_ids - customer_ids) if customer_ids else []
    text_result = detect_text_length_shift(
        [doc.get("content", "") if isinstance(doc, dict) else "" for doc in docs],
        history.get("mean_text_length", pd.Series(dtype=float)).dropna().tail(14).tolist(),
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

    order_freshness_failure = _failed_check(order_issues, "freshness")
    kb_freshness_failure = _failed_check(kb_issues, "freshness")
    current_slis = {
        "critical_contract_pass": not bool(
            order_critical_failed or kb_critical_failed
        ),
        "revenue_freshness": order_freshness_failure is None,
        "rag_index_freshness": kb_freshness_failure is None,
    }
    run_history = _read_run_history()
    run_history.append({"timestamp": now.isoformat(), "slis": current_slis})
    _write_run_history(run_history)

    slo_config = config.get("slo", {})
    targets = {
        "critical_contract_pass": float(
            slo_config.get("critical_contract_pass", {}).get("target", 0.999)
        ),
        "revenue_freshness": float(
            slo_config.get("revenue_freshness", {}).get("target", 0.995)
        ),
        "rag_index_freshness": float(
            slo_config.get("rag_index_freshness", {}).get("target", 0.99)
        ),
    }
    alerting = config.get("alerting", {})
    short_window = int(alerting.get("short_window_checks", 5))
    long_window = int(alerting.get("long_window_checks", 30))
    slo_windows = {
        name: evaluate_slo_history(
            [bool(item["slis"].get(name, True)) for item in run_history],
            target=target,
            short_window=short_window,
            long_window=long_window,
        )
        for name, target in targets.items()
    }

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

    runbook = "docs/OBSERVABILITY_RUNBOOK.md"
    signals = [
        signal(
            "orders_contract_critical",
            domain="orders",
            fired=bool(order_critical_failed),
            severity="critical",
            action="block",
            owner="commerce-data",
            summary="Critical orders contract violation",
            evidence={"failed_checks": order_critical_failed},
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "orders_freshness",
            domain="orders",
            fired=order_freshness_failure is not None,
            severity="warning",
            action="quarantine",
            owner="commerce-data",
            summary="Orders batch exceeded its freshness SLO",
            evidence={
                "freshness_minutes": _freshness_minutes(orders, "updated_at"),
                "failure": order_freshness_failure,
            },
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "orders_volume",
            domain="orders",
            fired=row_result["is_anomaly"],
            severity="warning",
            action="quarantine",
            owner="commerce-data",
            summary="Seasonality-aware order volume anomaly",
            evidence=row_result,
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "orders_amount_distribution",
            domain="orders",
            fired=amount_result["is_anomaly"],
            severity="warning",
            action="quarantine",
            owner="commerce-data",
            summary="Order amount distribution drift",
            evidence=amount_result,
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "orders_status_mix",
            domain="orders",
            fired=status_result["is_anomaly"],
            severity="warning",
            action="investigate",
            owner="commerce-data",
            summary="Order status mix drift",
            evidence=status_result,
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "orders_null_rate",
            domain="orders",
            fired=null_rate_result["is_anomaly"],
            severity="warning",
            action="quarantine",
            owner="commerce-data",
            summary="Null-rate anomaly",
            evidence={**null_rate_result, "current_null_rate": null_rate},
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "customer_scd_overlap",
            domain="customers",
            fired=bool(duplicate_active_customers),
            severity="critical",
            action="block",
            owner="commerce-data",
            summary="Multiple active SCD rows can inflate revenue",
            evidence={
                "customer_ids": duplicate_active_customers,
                "count": len(duplicate_active_customers),
            },
            source_asset="raw_customers",
            runbook=runbook,
        ),
        signal(
            "orders_orphan_customer",
            domain="customers",
            fired=bool(orphan_customer_ids),
            severity="critical",
            action="block",
            owner="commerce-data",
            summary="Orders reference missing customers",
            evidence={
                "customer_ids": orphan_customer_ids[:20],
                "count": len(orphan_customer_ids),
            },
            source_asset="raw_orders",
            runbook=runbook,
        ),
        signal(
            "kb_contract_critical",
            domain="rag",
            fired=bool(kb_critical_failed),
            severity="critical",
            action="block",
            owner="support-ai",
            summary="Critical knowledge-base contract violation",
            evidence={"failed_checks": kb_critical_failed},
            source_asset="kb_documents",
            runbook=runbook,
        ),
        signal(
            "kb_freshness",
            domain="rag",
            fired=kb_freshness_failure is not None,
            severity="warning",
            action="quarantine",
            owner="support-ai",
            summary="Knowledge base exceeded its freshness SLO",
            evidence={"failure": kb_freshness_failure},
            source_asset="kb_documents",
            runbook=runbook,
        ),
        signal(
            "kb_text_length",
            domain="rag",
            fired=text_result["is_anomaly"],
            severity="warning",
            action="quarantine",
            owner="support-ai",
            summary="Knowledge-base content length drift",
            evidence=text_result,
            source_asset="kb_documents",
            runbook=runbook,
        ),
        signal(
            "kb_version_rollback",
            domain="rag",
            fired=bool(rollback_ids),
            severity="critical",
            action="block",
            owner="support-ai",
            summary="Knowledge-base document version regressed",
            evidence={"doc_ids": rollback_ids, "count": len(rollback_ids)},
            source_asset="kb_documents",
            runbook=runbook,
        ),
    ]
    for name, status in slo_windows.items():
        signals.append(
            signal(
                f"{name}_fast_burn",
                domain="slo",
                fired=bool(status["alert"]["page"]),
                severity="critical",
                action="page",
                owner="data-reliability",
                summary=f"Sustained multi-window error-budget burn: {name}",
                evidence=status,
                runbook=runbook,
            )
        )
    system_status = incident_decision(signals, dataset_lineage)

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
        "schema_version": 2,
        "timestamp": now.isoformat(),
        "system_status": system_status,
        "signals": signals,
        "slis": current_slis,
        "slo_windows": slo_windows,
        "telemetry_coverage": {
            "embedding_norm_current_batch": {
                "status": (
                    "available"
                    if any(
                        isinstance(doc, dict) and "embedding_norm" in doc
                        for doc in docs
                    )
                    else "not_instrumented"
                ),
                "impact": (
                    "Embedding-model drift is evaluated from supplied norms"
                    if embedding_result.get("is_anomaly") is not False
                    or embedding_result.get("reason") != "no_precomputed_embedding_norms"
                    else "Embedding-model drift cannot be verified from the supplied current dataset"
                ),
                "action": "Export current embedding norms and call rag_embedding_shift",
                "owner": "support-ai",
            }
        },
        "orders_rows": int(len(orders)),
        "failed_contract_checks": len(order_failed),
        "critical_contract_failures": len(order_critical_failed),
        "contract_action": _action_for_failures(order_issues),
        "contract_issues": order_issues,
        "order_contract_results": order_issues,
        "kb_failed_contract_checks": len(kb_failed),
        "kb_critical_contract_failures": len(kb_critical_failed),
        "kb_action": _action_for_failures(kb_issues),
        "kb_contract_issues": kb_issues,
        "kb_contract_results": kb_issues,
        "row_count_anomaly": row_result,
        "amount_distribution_shift": amount_result,
        "amount_distribution_signal": amount_result,
        "status_distribution_signal": status_result,
        "null_rate_signal": {**null_rate_result, "current_null_rate": null_rate},
        "freshness_minutes": order_freshness,
        "kb_freshness_minutes": kb_freshness,
        "kb_text_length_signal": text_result,
        "rag_embedding_signal": embedding_result,
        "customer_integrity": {
            "duplicate_active_customer_ids": duplicate_active_customers,
            "orphan_customer_ids": orphan_customer_ids,
        },
        "kb_version_rollback_ids": rollback_ids,
        "contract_slo": contract_slo,
        "kb_slo": kb_slo,
        "multiwindow_burn": burn,
        "sample_blast_radius_from_stg_orders": blast_radius,
        "column_blast_radius_from_raw_orders_amount": column_blast_radius,
        "column_blast_radius_from_order_amount": column_blast_radius,
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
