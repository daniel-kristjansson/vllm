#!/usr/bin/env bash
# Throughput/latency sweep for the SM120 attention matrix:
#   {FA2, FA4} x {no speculation, DFlash2} x concurrency {1,2,4,8}
#
# Starts one vLLM server per (attention backend, speculation) combination and
# runs `vllm bench serve` against it at each concurrency level, so the only
# thing that changes between rows is concurrency.
#
# Prompts are natural language (see make_bench_prompts.py). `--dataset-name
# random` synthesizes an arithmetic ramp over the vocabulary, which no drafter
# can predict: measured DFlash2 draft acceptance was 1.4-3.2% on it versus 28%
# on real text, so speculative numbers from that dataset are meaningless.
#
# Usage: ./bench_attention_matrix.sh [OUTDIR]
#   CONCURRENCY="1 2 4 8"  override the sweep
#   NUM_PROMPTS=64         prompts per concurrency level
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-/home/danielk/models/Qwen3.8-27B-FP8}"
DRAFT="${DRAFT:-/home/danielk/models/incoai-Qwen3.8-27B-DFlash2}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
OUT_LEN="${OUT_LEN:-256}"
MAX_LEN="${MAX_LEN:-8192}"
GMEM="${GMEM:-0.85}"
MAX_SEQS="${MAX_SEQS:-32}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8}"
PORT="${PORT:-18095}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
OUTDIR="${1:-$REPO_ROOT/bench-results/$(date +%Y%m%d-%H%M%S)}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM="${VLLM:-$REPO_ROOT/.venv/bin/vllm}"

# Ambient env vars would silently change which flash-attention tree is used.
unset VLLM_FA4_SM120_DIR VLLM_FLASH_ATTN_SRC_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTDIR"

DATASET_NAME="${DATASET_NAME:-code}"
DATASET="$OUTDIR/prompts.json"
if [ ! -f "$DATASET" ]; then
  case "$DATASET_NAME" in
    code)  GENERATOR="$REPO_ROOT/make_code_bench_prompts.py" ;;
    prose) GENERATOR="$REPO_ROOT/make_bench_prompts.py" ;;
    *) echo "error: DATASET_NAME must be 'code' or 'prose'" >&2; exit 1 ;;
  esac
  "$PYTHON" "$GENERATOR" \
    --out "$DATASET" --num-prompts "$NUM_PROMPTS" --output-len "$OUT_LEN"
fi

SPEC_DFLASH="{\"method\": \"dflash\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": 7}"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    # `vllm serve` forks an EngineCore child; killing the wrapper alone leaves
    # the device allocated and the next case fails its memory check.
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    # Wait for all its descendants to disappear before the next case.
    for _ in $(seq 1 30); do
      pgrep -g "$SERVER_PID" >/dev/null 2>&1 || break
      sleep 2
    done
    sleep 5
  fi
  SERVER_PID=""
}
trap 'stop_server' EXIT

start_server() {
  local name="$1" attn_cfg="$2" spec="$3"
  local log="$OUTDIR/$name.server.log"
  local -a args=(
    serve "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --served-model-name bench-target
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GMEM"
    --max-num-seqs "$MAX_SEQS"
    --seed 0
    --attention-config "$attn_cfg"
  )
  if [ -n "$spec" ]; then
    args+=(--speculative-config "$spec")
  fi
  echo "  starting server ($name)..."
  "$VLLM" "${args[@]}" >"$log" 2>&1 &
  SERVER_PID=$!

  local waited=0
  until curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  FAILED to start ($name); see $log"
      tail -5 "$log" | sed 's/^/    /'
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$STARTUP_TIMEOUT" ]; then
      echo "  FAILED to start within ${STARTUP_TIMEOUT}s ($name)"
      return 1
    fi
  done
  return 0
}

run_case() {
  local name="$1" attn_cfg="$2" spec="$3"
  local spec_label=""
  if [ -n "$spec" ]; then
    case "$spec" in
      *dflash*) spec_label=", dflash2" ;;
      *)        spec_label=", spec" ;;
    esac
  fi
  echo "=== $name ($attn_cfg$spec_label)"
  if ! start_server "$name" "$attn_cfg" "$spec"; then
    stop_server
    return 0
  fi

  for conc in $CONCURRENCY; do
    local out="$OUTDIR/$name-c$conc"
    "$VLLM" bench serve \
      --backend openai-chat \
      --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL" \
      --served-model-name bench-target \
      --endpoint /v1/chat/completions \
      --dataset-name sharegpt --dataset-path "$DATASET" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$conc" \
      --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result --result-filename "$out.json" \
      >"$out.log" 2>&1 || { echo "  c=$conc FAILED"; tail -3 "$out.log" | sed 's/^/    /'; continue; }

    local json
    json="$(ls -1 "$out"*.json 2>/dev/null | head -1)"
    if [ -n "$json" ]; then
      "$PYTHON" - "$json" "$name" "$conc" <<'PY'
import json, sys
path, name, conc = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
def g(*ks, default=float("nan")):
    for k in ks:
        if k in d and d[k] is not None:
            return float(d[k])
    return default
tput = g("output_throughput", "tokens_per_second")
rps = g("request_throughput", "requests_per_second")
ttft = g("median_ttft_ms", "mean_ttft_ms")
tpot = g("median_tpot_ms", "mean_tpot_ms")
print(f"  c={conc}: {tput:8.2f} tok/s | {rps:6.2f} req/s | ttft {ttft:7.1f} ms | tpot {tpot:6.2f} ms")
PY
    else
      grep -E "Output token throughput|Request throughput" "$out.log" | sed 's/^/  c='"$conc"': /'
    fi
  done

  # Draft acceptance, when speculation was enabled for this case.
  if [ -n "$spec" ]; then
    grep -h "Mean acceptance length" "$OUTDIR/$name.server.log" 2>/dev/null | tail -1 |
      sed 's/.*SpecDecoding metrics:/  spec:/' || true
  fi

  stop_server
}

echo "model:       $MODEL"
echo "draft:       $DRAFT"
echo "workload:    $NUM_PROMPTS $DATASET_NAME prompts, $OUT_LEN out tokens"
if [ "${SMOKE:-0}" = "1" ]; then
  CONCURRENCY="1"
fi
echo "concurrency: $CONCURRENCY"
echo "outdir:      $OUTDIR"
echo

FA2_CFG='{"backend": "FLASH_ATTN", "flash_attn_version": 2}'
FA4_CFG='{"backend": "FLASH_ATTN", "flash_attn_version": 4}'
TRITON_CFG='{"backend": "TRITON_ATTN"}'
FI_CFG='{"backend": "FLASHINFER"}'

# Limit the sweep via BACKENDS="fa2 fa4" (default: all four).
BACKENDS="${BACKENDS:-fa2 fa4 triton flashinfer}"
for backend in $BACKENDS; do
  case "$backend" in
    fa2)        cfg="$FA2_CFG" ;;
    fa4)        cfg="$FA4_CFG" ;;
    triton)     cfg="$TRITON_CFG" ;;
    flashinfer) cfg="$FI_CFG" ;;
    *) echo "error: unknown BACKENDS entry '$backend'" >&2; exit 1 ;;
  esac
  run_case "$backend-plain"   "$cfg" ""
  run_case "$backend-dflash2" "$cfg" "$SPEC_DFLASH"
done

echo
echo "Results in $OUTDIR"
