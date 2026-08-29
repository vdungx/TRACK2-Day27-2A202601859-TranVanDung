from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "latest_metrics.json"
HISTORY = ROOT / "data" / "history" / "metrics_history.csv"

st.set_page_config(page_title="Data Reliability Lab", layout="wide")
st.title("Data Reliability Game Day")
st.caption("Evidence-first view of contracts, anomalies, SLOs and blast radius.")

if not REPORT.exists():
    st.warning("Run `make baseline` first to generate reports/latest_metrics.json")
    st.stop()

report = json.loads(REPORT.read_text(encoding="utf-8"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders rows", report["orders_rows"])
c2.metric("Orders freshness (min)", f"{report['freshness_minutes']:.1f}")
c3.metric("Contract failures", report["failed_contract_checks"])
c4.metric("Critical failures", report["critical_contract_failures"])
c5.metric("KB failures", report.get("kb_failed_contract_checks", 0))

if report.get("contract_action") == "block":
    st.error("Orders pipeline action: BLOCK")
elif report.get("contract_action") == "quarantine":
    st.warning("Orders pipeline action: QUARANTINE")
if report.get("kb_action") in {"block", "quarantine"}:
    st.warning(f"KB pipeline action: {report['kb_action'].upper()}")

st.subheader("Current signals")
st.json({
    "row_count_anomaly": report["row_count_anomaly"],
    "amount_distribution_shift": report.get("amount_distribution_shift"),
    "kb_text_length_signal": report["kb_text_length_signal"],
    "rag_embedding_signal": report.get("rag_embedding_signal"),
    "contract_slo": report["contract_slo"],
    "kb_slo": report.get("kb_slo"),
    "multiwindow_burn": report.get("multiwindow_burn"),
})

history = pd.read_csv(HISTORY)
st.subheader("Historical row count")
st.line_chart(history.set_index("date")[["row_count"]])

st.subheader("Example blast radius")
st.write("stg_orders -> " + " -> ".join(report["sample_blast_radius_from_stg_orders"]))
st.write(
    "raw_orders.amount -> "
    + " -> ".join(report.get("column_blast_radius_from_raw_orders_amount", []))
)

quarantine = report.get("quarantine", {})
if quarantine.get("orders_rows") or quarantine.get("kb_rows"):
    st.warning(
        f"Quarantine rows: orders={quarantine.get('orders_rows', 0)}, "
        f"kb={quarantine.get('kb_rows', 0)}"
    )
