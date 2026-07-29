# T=128 result artifact

The paper-facing result is the complete `T=128` campaign:

- 21 run keys: six representative models, 13 additional model-scale rows, and
  two controls;
- 281 clean/deranged model-by-shot cells;
- 100 tasks per cell;
- 128 sampled prompts per task;
- 3,596,800 recorded predictions.

The 719 MB compressed raw archive is stored separately because it is too large
for ordinary Git history. `artifact_manifest.json` records its URL, byte size,
SHA-256, expected layout, and evaluation counts. Fetch and verify it with:

```bash
python scripts/fetch_t128_results.py
```

Extraction creates `data/t128/results_128/`. The default validation and
analysis commands read from that directory.

The small files committed here are sufficient to audit the campaign structure
and reported summaries without downloading the raw trial records:

- `model_revisions.json`: exact upstream checkpoint revisions;
- `raw_results.sha256`: one SHA-256 entry for each raw result file;
- `canonical_semantic_clusters_k7.json`: the fixed 100-task, seven-cluster
  assignment used for the cluster analysis;
- `analysis/`: validated aggregate CSV/JSON reports and preview plots.

The prior eight-trial artifact remains available under `data/legacy_t8/`.
