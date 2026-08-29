from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from gx.validate_orders import validate_orders


def valid_orders(rows: int = 2) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "order_id": index + 1,
                "customer_id": f"C{index + 1}",
                "amount": 10.0 + index,
                "currency": "USD",
                "status": "completed",
                "created_at": (now - timedelta(minutes=20 + index)).isoformat(),
                "updated_at": (now - timedelta(minutes=5 + index)).isoformat(),
            }
            for index in range(rows)
        ]
    )


def test_gx_checkpoint_and_custom_contract_agree_on_healthy_batch():
    result = validate_orders(valid_orders())

    assert result["success"] is True
    assert result["gx_success"] is True
    assert result["contract_success"] is True
    assert result["checkpoint"] == "orders_contract_checkpoint"
    assert result["suite"] == "orders_contract_suite"
    assert result["actions"] == ["update_data_docs"]


def test_gx_and_custom_contract_both_fail_duplicate_primary_key():
    df = valid_orders()
    df.loc[1, "order_id"] = df.loc[0, "order_id"]

    result = validate_orders(df)

    assert result["success"] is False
    assert result["gx_success"] is False
    assert result["contract_success"] is False
    unique_failures = [
        issue
        for issue in result["failed_issues"]
        if issue["check"] == "unique" and issue["column"] == "order_id"
    ]
    assert unique_failures
    assert unique_failures[0]["severity"] == "critical"
    assert unique_failures[0]["action"] == "block"


def test_gx_checkpoint_handles_missing_column_as_failed_validation():
    result = validate_orders(valid_orders().drop(columns=["amount"]))

    assert result["success"] is False
    assert result["gx_success"] is False
    assert result["contract_success"] is False
    assert any(
        issue["check"] == "required_column" and issue["column"] == "amount"
        for issue in result["failed_issues"]
    )
