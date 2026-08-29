# Incident Report — KB freshness incident

## Severity

P2 / warning: the support knowledge base violated its freshness SLO and was quarantined before activation.

## Summary

The orders/revenue pipeline was healthy, but the incoming KB batch was stale. The newest `published_at` timestamp was approximately 190 minutes old against a 60-minute contract limit. This explains why a Support Agent could serve outdated policy content while the main data pipeline still reported success.

## Detection

- Signal: KB contract `freshness` check failed; KB action was `quarantine`.
- First observed time: `2026-08-29T04:45:26.975306+00:00` in the captured evidence run.
- Supporting signals: orders freshness was approximately 5 minutes, row-count anomaly was false, and distribution/text-length checks were healthy.

## Root Cause

The incoming KB publish timestamps were delayed beyond the support-data freshness SLA. All five documents were affected, so this was a batch-level freshness issue rather than a single malformed document.

## Evidence

1. `kb_documents` freshness: `latest age = 190.035 minutes`, `max_delay = 60 minutes`, `passed = false`.
2. KB SLO: target `99%`, actual bad rate `100%`, burn rate approximately `100x`, breached.
3. Contract action: `quarantine`; `data/quarantine/kb_documents_invalid.csv` contained 5 rows.
4. Orders controls remained healthy: 600 rows, zero contract failures, orders freshness approximately 5 minutes, no row-count anomaly.

## Blast Radius

```text
kb_documents
-> kb_active_docs
-> rag_index
-> support_agent
```

Column path:

```text
kb_documents.content
-> kb_active_docs.content
-> rag_index.embedding
-> support_agent.answer
```

## Mitigation

- Quarantine the stale KB batch and prevent it from becoming the active KB/index input.
- Continue serving the last known-good active KB while the publisher is repaired.
- Notify the support-AI owner with the freshness evidence and affected document count.

## Recovery

After re-anchoring the incoming data and rerunning validation, the KB freshness check returned to approximately 10 minutes, the KB contract had zero failures, and the quarantine action cleared.

## Verification

- [x] Contract healthy after recovery
- [x] dbt tests healthy
- [x] anomaly returned to expected range
- [x] SLO healthy / budget understood
- [x] downstream output and lineage verified

## Prevention / Action Items

| Action | Owner | Deadline | Why |
|---|---|---|---|
| Enforce KB freshness contract before activation | Support AI | Next sprint | Prevent stale policy from entering the index |
| Page/ticket on sustained KB freshness burn | Data Reliability | Next sprint | Make the SLO actionable before users report stale answers |
| Keep quarantine output and evidence in each run report | Data Platform | Next sprint | Preserve triage and recovery auditability |
| Add publisher lag metric and dashboard panel | Support AI | Next sprint | Distinguish source delay from indexing failure |
