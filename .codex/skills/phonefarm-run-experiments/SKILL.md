---
name: phonefarm-run-experiments
description: Run reproducible PhoneFarm benchmark, profiling, partitioning, thermal, and failover experiments. Use when collecting JSONL metrics, generating CSV summaries, comparing 2-5 worker routes, or evaluating static and adaptive schedulers.
---

# PhoneFarm experiments

Measure first. Keep raw events append-only and make every reported result traceable to a run manifest.

## Workflow

1. Record model checksum, runtime versions, worker capabilities, route, and prompt set in a run manifest.
2. Warm up before measured requests.
3. Capture stage timings, bytes, RAM, temperature, and failures as JSONL.
4. Aggregate only from raw JSONL to CSV.
5. Compare equal partition against profile-aware and adaptive mappings on identical inputs.

Read `references/metrics.md` before adding a metric or interpreting a result.
