---
pretty_name: Shared emergence ICL modalities - 128 trials per task
task_categories:
- text-generation
tags:
- in-context-learning
- reproducibility
- multimodal
---

# Shared-emergence ICL replication at T=128

This dataset contains the complete raw result archive for the paper
“Many Next-Token Predictors are In-Context Learners.”

The campaign evaluates a fixed suite of 100 program-synthesis tasks using 128
sampled prompts per task, for every clean and deranged shot cell described by
the paper:

- 21 run keys;
- 281 experiment cells;
- 12,800 predictions per cell;
- 3,596,800 predictions in total.

The archive expands to a top-level `results_128/` directory. It contains raw
per-task and per-trial JSON records, including predictions, exact-match
correctness, edit distances, and run configuration metadata. Summary JSON
files are present for the Evo2 runner but are not counted as experiment cells.

Use the code, complete run manifest, exact model revisions, strict validator,
analysis scripts, and per-file hashes in
[`N8python/shared-emergence-icl-modalities`](https://github.com/N8python/shared-emergence-icl-modalities).

## Integrity

Archive: `icl_128_results_2026-07-28.tar.gz`

SHA-256:
`aa4c7b5331092582b14ce7cc1c025ef55f4a05a26696299cd69973e8da942931`

Compressed size: 718,954,210 bytes.

The repository fetcher verifies both the byte size and SHA-256 before
extracting, and the validator checks the complete 281-cell grid and all
3,596,800 trial records.
