# ICLManyReplication
(README.md + code by GPT-5.5)

Minimal replication artifact for the main result table in "Many Next-Token
Predictors are In-Context Learners".

This first pass intentionally avoids plotting and model-checkpoint plumbing. It
recomputes the principal clean-vs-deranged numerical summaries from the raw
per-run JSON artifacts bundled in this repository:

- Qwen3
- Evo2
- NextTerm
- ImageGPT
- TimesFM
- ProGen2
- ChessGPT negative control
- MusicRoll negative control

## Usage

From this directory:

```bash
python scripts/replicate_principal_results.py
```

By default, the script reads:

```text
data/principal_results/
```

and writes:

```text
results/principal_results/summary.csv
results/principal_results/by_shot.csv
results/principal_results/summary.json
```

To point at a different data folder or output folder:

```bash
python scripts/replicate_principal_results.py \
  --data-root /path/to/principal_results_data \
  --output-dir results/principal_results
```

The script uses only the Python standard library.

`--source-root` is accepted as a backwards-compatible alias for `--data-root`,
but the intended default is fully local and does not require the original
`bitstringProgsynth` checkout.

## Bundled Data Layout

The required result JSONs are stored as:

```text
data/principal_results/<run-key>/clean/*.json
data/principal_results/<run-key>/deranged/*.json
```

The bundled run keys are:

```text
qwen3, evo2, nextterm, imagegpt, timesfm, progen2, chessgpt, musicroll
```

## What This Recomputes

For each model family, it loads matched clean and deranged JSON files, aligns
the shared shot counts, and recomputes:

- best clean exact accuracy over shared shots
- best deranged exact accuracy over shared shots
- best exact-accuracy z-gap
- best clean edit distance over shared shots
- best deranged edit distance over shared shots
- best edit-distance z-gap, when edit-distance standard errors are present
- per-shot clean/deranged exact accuracy and edit distance

The z-gap denominator is:

```text
sqrt(clean_stderr^2 + deranged_stderr^2)
```

matching the paper-facing utility in the source catalog.

## Scope Boundary

The bundled data script above is the stable, inspectable result-reproduction
layer.

## Required Libraries

The summary-only reproduction path:

```bash
python scripts/replicate_principal_results.py
```

uses only the Python standard library.

Full model reruns use two Python environments:

```text
MLX rows: numpy, requests, tqdm, huggingface-hub, python-chess,
          transformers, mlx, mlx-lm

Evo2 row: numpy, requests, tqdm, huggingface-hub, safetensors, psutil,
          ninja, packaging, torch, flash-attn, evo2, biopython, vtx,
          einops, rich, transformer-engine
```

The MLX rows cover Qwen3, NextTerm, ImageGPT, TimesFM, ProGen2, ChessGPT, and
MusicRoll. Evo2 is kept separate because it uses a different CUDA/Torch stack.
The exact install commands used for the verified H100 rerun are listed in
the remote setup notes below.

## Model Rerun Harness

The core evaluation harness has also been copied into:

```text
src/program_synth.py
src/dsl.py
src/curated_transformations.jsonl
src/program_synth_evo_smartbatch.py
src/run_evo2_h100_clean_deranged.py
src/timemoe_mlx.py
src/timesfm_mlx.py
src/music_backend.py
```

Run recipes for the principal rows live in:

```text
configs/principal_runs.json
```

To print the exact rerun commands without running models:

```bash
python scripts/run_program_synth_config.py --run qwen3 --condition clean --shots 1
```

Most rows use `src/program_synth.py`. Evo2 uses the canonical local/H100 runner:

```text
src/run_evo2_h100_clean_deranged.py
```

To execute selected commands:

```bash
python scripts/run_program_synth_config.py \
  --run qwen3 \
  --condition clean \
  --shots 1 \
  --execute
```

Model paths can be overridden with environment variables:

```text
QWEN3_14B_MODEL
EVO2_MODEL
NEXTTERM_440M_MODEL
IMAGEGPT_LARGE_BF16_MODEL
TIMESFM_2_5_MODEL
PROGEN2_BASE_BF16_MODEL
CHESSGPT_50M_MODEL
MUSICROLL_50M_MODEL
```

By default, `configs/principal_runs.json` points these rows at Hugging Face
model IDs rather than machine-local model directories.

Evo2 reruns do not use an API key. The config uses the Hugging Face model ID
`arcinstitute/evo2_40b`; the H100 runner maps that to Evo2's package-level
alias `evo2_40b` only at load time.

## Remote H100 Setup Notes

The principal reruns have two dependency stacks. On a fresh Linux/H100 machine,
use one environment for the MLX-backed rows and a separate environment for
Evo2.

For Qwen3, NextTerm, ImageGPT, TimesFM, ProGen2, ChessGPT, and MusicRoll:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install numpy requests tqdm huggingface-hub python-chess
python -m pip install "transformers<5,>=4.40" "mlx-lm[cuda13]"
python scripts/patch_mlx_cuda_sm90.py
```

The MLX patch script is idempotent. It is only needed when the installed MLX
CUDA wheel hits the H100/SM90 NVRTC `__nv_fp8_e8m0` header mismatch.

For Evo2 40B on two H100s:

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

When running Evo2, keep the cuDNN library path exported:

```bash
export CUDNN_ROOT=$PWD/.venv_evo2/lib/python3.11/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0,1 python src/run_evo2_h100_clean_deranged.py ...
```
