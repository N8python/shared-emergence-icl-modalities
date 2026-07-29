# Shared emergence of in-context learning across modalities

Reproducibility artifact for “Many Next-Token Predictors are In-Context
Learners.”

The paper-facing release uses **128 independently sampled prompts per task**.
It covers the complete paper grid, rather than only each model’s best-shot
cell:

- 100 fixed program-synthesis tasks;
- 21 model/run keys;
- every clean and deranged shot cell in the paper;
- 281 experiment cells;
- 12,800 predictions per complete cell;
- 3,596,800 predictions in total.

The earlier eight-trial pilot is retained explicitly under `data/legacy_t8/`
and `results/legacy_t8/`. It is not the default.

## Disclosure notes

- **Code generation.** For responsible disclosure, the code in this repository
  was generated using OpenAI GPT-5.6.
- **Final-paper metric.** The z-gap was dropped from the final paper; it remains
  in this repository only as a historical and reproducibility output.

## Reproduce the published analyses

The committed `data/t128/analysis/` directory contains compact validated
summaries and preview plots. The full raw archive is external because it is
719 MB compressed and approximately 2.5 GB extracted.

Install the analysis dependencies, fetch the checksum-pinned raw artifact, and
rebuild all paper-facing results:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/fetch_t128_results.py
python scripts/run_t128_analysis.py
```

The fetcher downloads
[`N8Programs/shared-emergence-icl-modalities-128`](https://huggingface.co/datasets/N8Programs/shared-emergence-icl-modalities-128),
verifies the 718,954,210-byte archive against SHA-256
`aa4c7b5331092582b14ce7cc1c025ef55f4a05a26696299cd69973e8da942931`,
checks every archive member before extraction, and creates
`data/t128/results_128/`.

For a faster code-path smoke test with reduced Monte Carlo counts:

```bash
python scripts/run_t128_analysis.py --quick
```

The full command verifies all 287 archived JSON files against
`data/t128/raw_results.sha256`, strictly validates all 281 experiment cells and
3,596,800 predictions, then regenerates:

- full-grid cell, paired-gap, and best-shot summaries;
- Table 3’s exact maximum-shot paired tests;
- Figure 3 and its table;
- Figure 4 clean-minus-deranged per-task correlations, one-sided permutation
  tests, and hatching;
- Figure 5 semantic-cluster clean-minus-deranged effects;
- Figure 6 model-scale clean-minus-deranged effects;
- Figure 7 chess and MusicRoll controls;
- the finite-Bernoulli-sampling correlation calibration.

Generated files are written under `results/t128/analysis/`.

To recompute only the representative clean/deranged numerical summary using
the Python standard library:

```bash
python scripts/replicate_principal_results.py
```

## Audit the experiment grid

The complete, executable experiment declaration is
`configs/paper_runs_t128.json`. It records every model, condition, shot count,
seed, backend option, and output path. Check its cardinality without loading a
model:

```bash
python scripts/inspect_run_manifest.py
```

Expected output:

```text
run keys:             21
experiment cells:     281
tasks per cell:        100
trials per task:       128
expected predictions:  3596800
```

`data/t128/model_revisions.json` records the exact upstream Hugging Face commit
used for all 21 checkpoints.

## Rerun model cells

The command builder defaults to the complete T=128 manifest and prints commands
without executing them:

```bash
python scripts/run_program_synth_config.py \
  --run qwen3 \
  --condition both \
  --shots 1 2 4 8 16 32 64
```

Execute selected cells by adding `--execute`:

```bash
python scripts/run_program_synth_config.py \
  --run qwen3 \
  --condition both \
  --shots 1 2 4 8 16 32 64 \
  --execute
```

The runner is safe to resume: it validates and skips accepted complete cells,
and refuses to overwrite malformed or partial files. `--overwrite` is an
explicit escape hatch for an inspected, intentionally selected cell.

The shorter representative/control-only T=128 declaration remains at
`configs/principal_runs.json`. The exact former pilot config is
`configs/legacy_t8_principal_runs.json`.

### Converted ImageGPT and ProGen2 checkpoints

ImageGPT-small/medium and ProGen2-small/medium/large/xlarge use the custom MLX
model implementations in `src/model_implementations/`. Prepare their pinned
upstream revisions with:

```bash
python -m pip install -r requirements-conversion.txt
python scripts/prepare_converted_models.py
```

This writes the paths already named by the complete run config under `models/`.
Individual prepared locations can instead be supplied with:

```text
IMAGEGPT_SMALL_BF16_MODEL
IMAGEGPT_MEDIUM_BF16_MODEL
PROGEN2_SMALL_BF16_MODEL
PROGEN2_MEDIUM_BF16_MODEL
PROGEN2_LARGE_BF16_MODEL
PROGEN2_XLARGE_BF16_MODEL
```

The remaining default model IDs can likewise be overridden by the
`model_env` field associated with each run.

### MLX environment

Qwen3, NextTerm, ImageGPT, TimesFM, ProGen2, ChessGPT, and MusicRoll use the MLX
path. The exact campaign versions are pinned because the per-prompt constrained
batch generator depends on `mlx-lm` batching semantics:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install --no-deps mlx-cuda==0.29.4
python -m pip install --force-reinstall \
  nvidia-cublas-cu12==12.8.4.1 \
  nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cudnn-cu12==9.8.0.87 \
  nvidia-nccl-cu12==2.27.7
python scripts/patch_mlx_cuda_sm90.py
```

The CUDA package pins are also collected in `requirements-mlx-h100.txt`; they
are installed separately because `mlx-cuda==0.29.4` declares CUDA 12.9
dependencies, while the successful H100 run used the listed CUDA 12.8 wheels.
The patch is idempotent and is needed only if an MLX CUDA wheel encounters the
H100/SM90 NVRTC `__nv_fp8_e8m0` header mismatch. Complete `pip freeze` snapshots
from the successful MLX and Evo2 environments are preserved under
`data/t128/environment/`.

### Evo2 H100 environment

Evo2 uses a separate CUDA/Torch stack:

```bash
python3 -m venv .venv_evo2
. .venv_evo2/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install numpy tqdm requests huggingface-hub safetensors psutil ninja packaging
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=16
python -m pip install flash-attn==2.8.0.post2 --no-build-isolation
python -m pip install evo2 --no-deps
python -m pip install biopython vtx==1.1.0 einops==0.8.1 rich
export CUDNN_ROOT=$PWD/.venv_evo2/lib/python3.11/site-packages/nvidia/cudnn
export CPATH=$CUDNN_ROOT/include:${CPATH:-}
export LIBRARY_PATH=$CUDNN_ROOT/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:${LD_LIBRARY_PATH:-}
python -m pip install "transformer_engine[pytorch]==2.3.0" --no-build-isolation
```

Keep `LD_LIBRARY_PATH` set when executing the Evo2 rows.

## Statistical analyses

Table 3’s modality-specific `p_task` values test clean greater than deranged at
the maximum shot count. The statistic is the mean of the 100 within-task
clean-minus-deranged differences. Under the sharp paired null, clean/deranged
labels are exchangeable within each task, so the exact one-sided distribution
is obtained by enumerating the `2^m` sign assignments of the `m` nonzero task
differences with dynamic programming. These pre-specified modality-specific
tests are reported unadjusted.

Figure 4 correlates the 100-task **clean-minus-deranged** vectors, not raw clean
accuracies. For each modality pair, the one-sided correlation p-value holds one
rank vector fixed and randomly permutes the other task labels one million
times. The add-one estimate is `(b + 1) / (B + 1)`. Figure hatching uses the
unadjusted 0.05 threshold; the significant/nonsignificant classification is
unchanged under Holm correction across all 15 pairs.

The paper’s z-gap is the independently selected best-clean accuracy minus
best-deranged accuracy divided by the root-sum-square of their task-cluster
standard errors. Model-scale plots instead select each model’s peak-clean shot
and subtract deranged accuracy at that same shot.

## Artifact layout

```text
configs/
  paper_runs_t128.json             complete 281-cell default
  principal_runs.json              representative/control T=128 subset
  legacy_t8_principal_runs.json    former eight-trial config
data/
  t128/
    artifact_manifest.json
    model_revisions.json
    raw_results.sha256
    canonical_semantic_clusters_k7.json
    environment/                   successful MLX and Evo2 pip freezes
    analysis/                      committed validated summaries
    results_128/                   downloaded raw JSONs, ignored by Git
  legacy_t8/principal_results/     bundled former pilot
scripts/
  analysis/                        individual validation/analysis programs
  fetch_t128_results.py
  run_t128_analysis.py
  run_program_synth_config.py
src/
  program_synth.py
  run_evo2_h100_clean_deranged.py
  model_implementations/
```
