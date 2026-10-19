#!/usr/bin/env bash
# Clean rebuild of vLLM against the SM120 flash-attention fork.
#
# The flash-attention source tree provides BOTH halves of the SM120 work:
#   - FA2: csrc/flash_attn (C++/CUDA) -> _vllm_fa2_C.abi3.so
#   - FA4: flash_attn/cute (CuTe DSL)  -> symlinked into vllm/vllm_flash_attn/cute
#
# `VLLM_FLASH_ATTN_SRC_DIR` makes vLLM's CMake fetch flash-attn from that path
# instead of the pinned GIT_TAG, and symlinks flash_attn/cute into the package.
#
# FA2 must be compiled to native sm_120 SASS: a PTX-only build fails at runtime
# on this host with cudaErrorUnsupportedPtxVersion (nvcc 13.4 vs older driver).
#
# Usage:
#   ./build.sh              # clean build (removes build/ and stale .so first)
#   ./build.sh --incremental  # keep build/ and existing .so
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

FA_SRC="${VLLM_FLASH_ATTN_SRC_DIR:-/home/danielk/code/flash-attention-prod-sm120}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
BUILD_DIR="$REPO_ROOT/build"
INCREMENTAL=0
[ "${1:-}" = "--incremental" ] && INCREMENTAL=1

if [ ! -x "$PYTHON" ]; then
  echo "error: python not found at $PYTHON" >&2
  exit 1
fi
if [ ! -f "$FA_SRC/flash_attn/cute/interface.py" ]; then
  echo "error: FA source tree not found at $FA_SRC" >&2
  echo "       (set VLLM_FLASH_ATTN_SRC_DIR to a flash-attention checkout)" >&2
  exit 1
fi

# The PyTorch stable-ABI build needs the matching torch headers from $PYTHON.
export VLLM_FLASH_ATTN_SRC_DIR="$FA_SRC"
# Native SM120 codegen for FA2 (see header note). vLLM's own kernels use the
# default arch detection, so this only offsets the flash-attn sub-build flags.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

echo "== vLLM clean build =="
echo "  repo:        $REPO_ROOT"
echo "  FA source:   $VLLM_FLASH_ATTN_SRC_DIR"
echo "  python:      $PYTHON"
echo "  MAX_JOBS:    $MAX_JOBS"
echo "  incremental: $INCREMENTAL"

if [ "$INCREMENTAL" -eq 0 ]; then
  echo "  removing $BUILD_DIR and stale flash-attn extensions"
  rm -rf "$BUILD_DIR"
  rm -f vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
  # The CuTe package is symlinked from the source tree; drop any stale copy.
  if [ -L vllm/vllm_flash_attn/cute ]; then
    rm -f vllm/vllm_flash_attn/cute
  fi
fi

# flash-attn's own submodules (CUTLASS) may be uninitialised in a fresh clone.
if [ -f "$VLLM_FLASH_ATTN_SRC_DIR/.gitmodules" ]; then
  git -C "$VLLM_FLASH_ATTN_SRC_DIR" submodule update --init --recursive
fi

# Editable install; vLLM's CMake builds _vllm_fa2_C from the source tree.
"$PYTHON" -m pip install -e . --no-build-isolation --no-deps

echo
echo "== post-build checks =="
"$PYTHON" - <<'PY'
import os
import torch

from vllm.vllm_flash_attn import flash_attn_interface as fai

src = os.environ["VLLM_FLASH_ATTN_SRC_DIR"]
print(f"  FA2 available: {fai.FA2_AVAILABLE} ({fai.FA2_UNAVAILABLE_REASON})")
print(f"  FA3 available: {fai.FA3_AVAILABLE}")
print(f"  FA4 available: {fai.FA4_AVAILABLE} ({fai.FA4_UNAVAILABLE_REASON})")

cute = fai.__file__.replace("flash_attn_interface.py", "cute/interface.py")
print(f"  FA4 interface: {cute}")
print(f"  FA4 from src:  {os.path.abspath(cute).startswith(os.path.abspath(src))}")

so = os.path.join(os.path.dirname(fai.__file__), "_vllm_fa2_C.abi3.so")
print(f"  FA2 extension: {so} ({os.path.getsize(so) / 1e6:.0f} MB)")

cap = torch.cuda.get_device_capability()
print(f"  device:        {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
print(f"  fa2 supported: {fai._is_fa2_supported()}")
print(f"  fa4 supported: {fai._is_fa4_supported()}")
PY

echo
echo "Build complete."
