#!/usr/bin/env bash
# Rebuttal appendix: encoding-resampling runs for language (Qwen3-14B, 64 shots)
# and image (ImageGPT-large, 31 shots). T=128, 100 tasks, clean + deranged.
#
# usage: run_encoding_variants.sh <python> <image|language> <conditions> [variant ...]
#   conditions: "clean", "deranged", or "clean deranged"
# Resume-safe: a cell whose output file exists is skipped.
set -u
PY="$1"; MODALITY="$2"; CONDITIONS="$3"; shift 3
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/results_rebuttal/$MODALITY"; LOGS="$ROOT/logs/rebuttal"
mkdir -p "$OUT" "$LOGS"
export MLX_CUDA_GRAPH_CACHE_SIZE="${MLX_CUDA_GRAPH_CACHE_SIZE:-4000}"
COMMON=(--programs "$ROOT/src/curated_transformations.jsonl" --trials-per-program 128 --bit-length 8 --max-new-tokens 8)

variant_args() {  # prints: seed then extra args
  case "$MODALITY:$1" in
    image:reference_sameseed) echo "42 --imagegpt-row-layout left-pad" ;;
    image:reference_reseed)   echo "20261111 --imagegpt-row-layout left-pad" ;;
    image:right_sep)          echo "20261112 --imagegpt-row-layout right-pad" ;;
    image:middle_sep)         echo "20261113 --imagegpt-row-layout middle-pad" ;;
    image:left_two_sep)       echo "20261114 --imagegpt-row-layout left-two-sep" ;;
    image:right_two_sep)      echo "20261115 --imagegpt-row-layout right-two-sep" ;;
    language:reference_sameseed) echo "1958817553 --lmshuffle" ;;
    language:reference_reseed)   echo "20261121 --lmshuffle" ;;
    language:no_sep)             echo "20261122 --raw-bits --symbol-bits 0123456789" ;;
    language:two_sep)            echo "20261123 --lmshuffle --lm-two-separators" ;;
    language:nondigit)           echo "20261124 --lmshuffle --force-binary-tokens --lm-token-pool $ROOT/configs/rebuttal/qwen3_top1000_tokens.json" ;;
    language:nondigit_two_sep)   echo "20261125 --lmshuffle --lm-two-separators --force-binary-tokens --lm-token-pool $ROOT/configs/rebuttal/qwen3_top1000_tokens.json" ;;
    *) return 1 ;;
  esac
}

if [ "$MODALITY" = image ]; then
  MODEL_ARGS=(--backend imagegpt --model "${IMAGEGPT_LARGE_BF16_MODEL:-N8Programs/imagegpt-large-bf16}" --in-context-examples 31 --batch-size-completion "${IMAGE_BATCH:-32}")
  DEFAULT_VARIANTS=(reference_sameseed reference_reseed right_sep middle_sep left_two_sep right_two_sep)
else
  MODEL_ARGS=(--backend mlx --model "${QWEN3_14B_MODEL:-Qwen/Qwen3-14B-Base}" --in-context-examples 64 --per-trial-random --batch-programs --batch-size-prefill "${LM_PREFILL:-32}" --batch-size-completion "${LM_COMPLETION:-64}")
  DEFAULT_VARIANTS=(reference_sameseed reference_reseed two_sep no_sep nondigit nondigit_two_sep)
fi
[ "$#" -gt 0 ] && VARIANTS=("$@") || VARIANTS=("${DEFAULT_VARIANTS[@]}")

cd "$ROOT/src"
for variant in "${VARIANTS[@]}"; do
  spec="$(variant_args "$variant")" || { echo "unknown variant $variant"; exit 2; }
  seed="${spec%% *}"; extra="${spec#* }"
  for cond in $CONDITIONS; do
    out="$OUT/$variant/$cond.json"
    if [ -e "$out" ]; then echo "[skip] $out"; continue; fi
    mkdir -p "$OUT/$variant"
    ablate=(); [ "$cond" = deranged ] && ablate=(--ablate-labels)
    echo "[start $(date -Is)] $MODALITY $variant $cond seed=$seed"
    # shellcheck disable=SC2086
    "$PY" program_synth.py "${COMMON[@]}" "${MODEL_ARGS[@]}" --seed "$seed" $extra "${ablate[@]}" --output "$out" \
      > "$LOGS/${MODALITY}_${variant}_${cond}.log" 2>&1
    status=$?
    echo "[done  $(date -Is)] $MODALITY $variant $cond exit=$status $(grep -E '^Mean accuracy' "$LOGS/${MODALITY}_${variant}_${cond}.log" | tail -1)"
  done
done
echo "QUEUE_COMPLETE $MODALITY $CONDITIONS"
