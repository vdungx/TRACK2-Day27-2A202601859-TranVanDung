# Observability Runbook

## Triage order

1. Check `reports/latest_metrics.json` and identify active signals in
   `system_status.active_signals`.
2. Treat critical contract, customer-integrity, rollback, and sustained SLO
   signals as blocking evidence.
3. Use `affected_assets` and the column blast radius to decide which outputs
   must remain unpublished.
4. Inspect the corresponding contract issues and quarantined batch before
   rerunning dbt or refreshing the RAG index.

## Actions

- `block`: stop downstream publication and quarantine the rejected source.
- `quarantine`: isolate the batch, notify the owning team, and investigate.
- `warn`/`investigate`: continue only when no blocking signal is active.
- A short-window burn without long-window confirmation is informational; a
  sustained warning or critical burn is page-worthy.

## Recovery checklist

- Re-run contract and GX validation on the repaired batch.
- Confirm the quarantine artifact is removed only after a clean validation.
- Run dbt tests and verify the relevant anomaly signals return to baseline.
- Confirm `system_status.publish_downstream` is true before activation.
