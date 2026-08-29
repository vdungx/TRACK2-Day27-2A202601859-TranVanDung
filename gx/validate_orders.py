#!/usr/bin/env python3
"""Reusable Great Expectations validation for the orders contract.

The lab's custom validator remains the source of truth for type, freshness
and severity-aware actions.  This module adds a native GX Suite,
ValidationDefinition and Checkpoint so the same batch is protected by a
standard validation flow as well.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import great_expectations as gx
    from great_expectations.checkpoint import ValidationAction
    from great_expectations.checkpoint.actions import UpdateDataDocsAction
except ImportError as exc:  # friendlier classroom failure
    raise SystemExit(
        "great_expectations is not installed. Run: pip install -r requirements.txt"
    ) from exc

from src.contract_validator import failed_issues, load_contract, validate_dataframe


class QuarantineOnFailure(ValidationAction):
    """Persist the rejected source batch when a GX checkpoint fails.

    The custom contract still decides whether a failure is a warning or a
    hard block. This action only provides the durable GX-side evidence and is
    intentionally reusable by instructor-side checks.
    """

    type: Literal["quarantine_on_critical_failure"] = "quarantine_on_critical_failure"
    quarantine_dir: str
    source_path: str

    def run(self, checkpoint_result: Any, action_context: Any) -> dict[str, Any]:
        del action_context
        target = Path(self.quarantine_dir)
        target.mkdir(parents=True, exist_ok=True)
        failed = checkpoint_result.success is False
        quarantine_path: Path | None = None
        if failed:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine_path = target / f"orders-{stamp}.csv"
            shutil.copy2(self.source_path, quarantine_path)
        payload = {
            "checkpoint_success": bool(checkpoint_result.success),
            "action": "quarantine" if failed else "allow",
            "reason": (
                "orders_suite_failure" if failed else "all_suite_checks_passed"
            ),
            "quarantine_path": str(quarantine_path) if quarantine_path else None,
        }
        (target / "latest_gx_action.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload


def _severity_meta(severity: str) -> dict[str, str]:
    action = {
        "critical": "block",
        "warning": "quarantine",
        "info": "warn",
    }.get(str(severity).lower(), "warn")
    return {"severity": str(severity).lower(), "action": action}


def build_expectation_suite() -> gx.ExpectationSuite:
    """Build the reusable GX suite for the deterministic orders checks."""
    suite = gx.ExpectationSuite(
        name="orders_contract_suite",
        meta={
            "owner": "commerce-data",
            "purpose": "orders_contract_validation",
        },
    )

    critical = _severity_meta("critical")
    warning = _severity_meta("warning")
    expectations = [
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=[
                "order_id",
                "customer_id",
                "amount",
                "currency",
                "status",
                "created_at",
                "updated_at",
            ],
            exact_match=True,
            severity="critical",
            meta=critical,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id", severity="critical", meta=critical
        ),
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="order_id", severity="critical", meta=critical
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0, severity="critical", meta=critical
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="currency",
            value_set=["USD", "VND"],
            severity="critical",
            meta=critical,
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["pending", "completed", "refunded", "cancelled"],
            severity="warning",
            meta=warning,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id", severity="critical", meta=critical
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="updated_at", severity="critical", meta=critical
        ),
    ]
    for expectation in expectations:
        suite.add_expectation(expectation)
    return suite


def run_checkpoint(df: pd.DataFrame) -> dict[str, Any]:
    """Run a fresh ephemeral GX checkpoint against ``df``."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("orders_pandas")
    asset = data_source.add_dataframe_asset(name="orders_dataframe")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_orders")
    suite = build_expectation_suite()
    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        data=batch_definition,
        suite=suite,
        name="orders_contract_validation",
    )
    context.validation_definitions.add(validation_definition)

    checkpoint = gx.Checkpoint(
        name="orders_contract_checkpoint",
        validation_definitions=[validation_definition],
        actions=[UpdateDataDocsAction(name="update_data_docs", site_names=[])],
        result_format="SUMMARY",
    )
    context.checkpoints.add(checkpoint)
    result = checkpoint.run(batch_parameters={"dataframe": df})

    return {
        "success": bool(result.success),
        "checkpoint": checkpoint.name,
        "suite": suite.name,
        "action_names": ["update_data_docs"],
        "result": result,
    }


def validate_orders(df: pd.DataFrame) -> dict[str, Any]:
    """Run GX and custom contract validation and return JSON-friendly output."""
    contract = load_contract(ROOT / "contracts" / "orders_contract.yaml")
    contract_issues = validate_dataframe(df, contract)
    gx_result = run_checkpoint(df)
    failed = failed_issues(contract_issues)
    return {
        "success": bool(gx_result["success"] and not failed),
        "gx_success": gx_result["success"],
        "contract_success": not failed,
        "checkpoint": gx_result["checkpoint"],
        "suite": gx_result["suite"],
        "actions": gx_result["action_names"],
        "contract_issues": contract_issues,
        "failed_issues": failed,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    result = validate_orders(df)
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"contract_issues", "failed_issues"}
    }
    print(json.dumps(printable, indent=2, default=_json_default))
    for issue in result["contract_issues"]:
        print(
            f"contract {issue['check']:<18} column={issue['column']!s:<12} "
            f"passed={issue['passed']} severity={issue['severity']} "
            f"action={issue['action']}"
        )
    print("GX/custom validation:", "PASS" if result["success"] else "FAIL")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
