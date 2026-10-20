#!/usr/bin/env bash
# Bench throughput for the SM120 attention matrix:
#   {FA2, FA4} x {no speculation, DFlash2}
#
# Each case is a separate `vllm bench throughput` run (fresh engine), so the
# numbers are comparable: identical model, prompt/output lengths and seed.
#
# Usage: ./bench_attention_matrix.sh [OUTDIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-/home/danielk/models/Qwen3.8-27B-FP8}"
DRAFT="${DRAFT:-/home/danielk/models/incoai-Qwen3.8-27B-DFlash2}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
IN_LEN="${IN_LEN:-512}"
OUT_LEN="${OUT_LEN:-256}"
MAX_LEN="${MAX_LEN:-8192}"
GMEM="${GMEM:-0.85}"
MAX_SEQS="${MAX_SEQS:-32}"
OUTDIR="${1:-$REPO_ROOT/bench-results/$(date +%Y%m%d-%H%M%S)}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM="${VLLM:-$REPO_ROOT/.venv/bin/vllm}"

mkdir -p "$OUTDIR"

# Ambient env vars would silently change which flash-attention tree is used.
unset VLLM_FA4_SM120_DIR VLLM_FLASH_ATTN_SRC_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SPEC_DFLASH="{\"method\": \"dflash\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": 7}"

run_case() {
  local name="$1" fa_version="$2" spec="$3"
  local json="$OUTDIR/$name.json" log="$OUTDIR/$name.log"
  local -a args=(
    "$VLLM" bench throughput
    --model "$MODEL"
    --dataset-name random
    --num-prompts "$NUM_PROMPTS"
    --random-input-len "$IN_LEN"
    --random-output-len "$OUT_LEN"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GMEM"
    --max-num-seqs "$MAX_SEQS"
    --seed 0
    --attention-config "{\"backend\": \"FLASH_ATTN\", \"flash_attn_version\": $fa_version}"
    --output-json "$json"
  )
  if [ -n "$spec" ]; then
    args+=(--speculative-config "$spec")
  fi

  echo "=== $name (FA$fa_version${spec:+, dflash2})"
  if "$PYTHON" "${args[@]}" >"$log" 2>&1; then
    "$PYTHON" - "$json" "$name" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as exc:
    print(f"  {name}: could not read JSON: {exc}")
    raise SystemExit(0)
elapsed = float(d.get("elapsed_time", 0) or 0)
reqs = int(d.get("num_requests", 0) or 0)
tok_s = float(d.get("tokens_per_second", 0) or 0)
req_s = float(d.get("requests_per_second", 0) or 0)
total = int(d.get("total_num_tokens", 0) or 0)
line = (
    f"  {name}: {tok_s:8.2f} tok/s | {req_s:6.2f} req/s"
    f" | {reqs} reqs, {total} tok, {elapsed:6.2f}s"
)
spec = d.get("spec_decode") or {}
if spec:
    line += f" | accept {spec.get('mean_acceptance_length', float('nan')):.2f}"
print(line)
PY
  else
    echo "  $name: FAILED (see $log)"
    tail -5 "$log" | sed 's/^/    /'
  fi
}

echo "model:   $MODEL"
echo "draft:   $DRAFT"
echo "workload: $NUM_PROMPTS prompts, $IN_LEN in / $OUT_LEN out, max-len $MAX_LEN"
echo "outdir:  $OUTDIR"
echo

run_case fa2-plain        2 ""
run_case fa2-dflash2      2 "$SPEC_DFLASH"
run_case fa4-plain        4 ""
run_case fa4-dflash2      4 "$SPEC_DFLASH"

echo
echo "Results in $OUTDIR"
