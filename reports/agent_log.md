# AI Agent Decision Log

## Decision 1 — Strict contract types and freshness

- Hypothesis: pipeline success is not enough; numeric coercion can hide type drift and timestamps need an explicit freshness rule.
- Agent proposal: validate declared integer/number/string/datetime types without silent coercion, validate the newest UTC timestamp, and preserve severity/action metadata.
- Evidence/test: public contract tests, dynamic stale timestamp test, missing-column test and KB `fields`/`min_length` test pass.
- Decision: accept.
- Why: deterministic failures are visible to both the API and the incident report.

## Decision 2 — Context-aware robust anomaly detection

- Hypothesis: global z-score creates seasonality false positives and historical outliers can mask a real drop.
- Agent proposal: keep `zscore`, add same-segment selection, median/MAD with IQR/std fallback, known-event suppression and optional trend residuals in `auto`.
- Evidence/test: healthy synthetic batch is not anomalous, volume drop is anomalous, weekend baseline is stable, outlier history and zero-MAD cases are handled.
- Decision: accept.
- Why: the detector remains explainable and keeps the original public z-score behavior.

## Decision 3 — Distribution drift beyond mean ratio

- Hypothesis: equal means can hide a changed distribution, while PSI is unstable for tiny samples.
- Agent proposal: combine empirical KS, PSI for sufficiently large samples, and a robust quantile-spread signal for small samples.
- Evidence/test: extreme mean shift and same-mean shape shift are detected; similar small samples are not falsely flagged.
- Decision: accept after revising the initial PSI-only approach.

## Decision 4 — Protect revenue from SCD join inflation

- Hypothesis: multiple active customer versions can multiply facts without producing a SQL error.
- Agent proposal: rank active customer versions and join only rank 1; add active-key/reconciliation singular tests and dbt unit tests with duplicate active rows.
- Evidence/test: dbt healthy build passes 23/23 nodes; unit test expects 170 revenue from two orders despite two active customer rows.
- Decision: accept.

## Decision 5 — Multi-window SLO alerting

- Hypothesis: a short burn spike should not page, while sustained fast burn should.
- Agent proposal: page critically when the short window is at least 14.4x and the long confirmation window is at least 6x; page at warning severity when short is at least 6x and long is at least 3x; suppress short-only spikes.
- Evidence/test: 20/7 pages critically, 20/2 and 2/20 do not, 6/3 pages as warning, cold-start history is suppressed, and the exact error-budget boundary is not breached.
- Decision: accept.

## Decision 6 — Layered GX and quarantine actions

- Hypothesis: custom contract results and native GX results should agree, with a safe action for bad batches.
- Agent proposal: build Suite → Validation Definition → Checkpoint → Data Docs Action and write invalid rows to a separate quarantine file without overwriting incoming data.
- Evidence/test: healthy GX run passes; duplicate PK fails GX/custom validation and produces block/quarantine evidence.
- Decision: accept.

## Decision 7 — Adversarial verification

- Hypothesis: public tests alone do not prove reliability under edge cases.
- Agent proposal: maintain a separate `tests_adversarial` suite covering type/freshness/KB, outliers, seasonality, trend, shape drift, cycles, SLO boundaries, RAG drift, distribution contamination, GX agreement, pipeline multi-faults and quarantine recovery.
- Evidence/test: 72 adversarial/public tests pass; coverage includes deterministic freshness clocks, future timestamps, categorical drift, empty batches, SLO history, health blast radius, GX quarantine, healthy, duplicate, volume-drop, distribution-shift, stale-KB, missing-column, multi-fault and recovery paths.
- Decision: accept.

## Decision 8 — Align hard-scenario control-plane behavior

- Hypothesis: private evaluation can exercise the operational edges around the stable API, not only the happy-path metric values.
- Agent proposal: add deterministic freshness reference times and future-skew detection, fail closed on empty current distributions, expose categorical TVD, normalize seasonal detector method labels, and provide SLO history/cold-start decisions.
- Evidence/test: the peer hard-scenario regression suite passes locally, including stale/future timestamps, seasonal MAD, shape/category drift, empty current batches and sustained burn.
- Decision: accept.

## Decision 9 — Actionable incident and GX evidence

- Hypothesis: a detected signal is only useful when it determines publication safety and preserves the rejected input.
- Agent proposal: add `observability.health` incident decisions with lineage blast radius, a reusable GX `QuarantineOnFailure` action, exact-column expectation, and richer baseline status fields.
- Evidence/test: critical signals resolve to P1 with downstream publication blocked; GX failures copy the source batch to a timestamped quarantine; baseline retains legacy report keys and emits system status/SLO history.
- Decision: accept.
