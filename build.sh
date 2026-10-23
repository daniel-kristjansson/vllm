#!/usr/bin/env bash
# Clean rebuild of vLLM against the SM120 flash-attention fork.
#
# The flash-attention source tree provides BOTH halves of the SM120 work:
#   - FA2: csrc/flash_attn (C++/CUDA) -> _vllm_fa2_C.abi3.so
#   - FA4: flash_attn/cute (CuTe DSL)  -> copied into vllm/vllm_flash_attn/cute
#                                        (imports rewritten to the vllm package)
#
# `VLLM_FLASH_ATTN_SRC_DIR` makes vLLM's CMake build flash-attn from that path
# instead of the pinned GIT_TAG.
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

# Default to the prod-sm120 checkout. Note: do NOT inherit an ambient
# VLLM_FLASH_ATTN_SRC_DIR from an interactive shell (it may point at an older
# flash-attention tree); override explicitly via FA_SRC=... instead.
FA_SRC="${FA_SRC:-/home/danielk/code/flash-attention-prod-sm120}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
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
# setup.py launches CMake in a subprocess; also pass the source path as an
# explicit CMake cache argument so FetchContent and the FA4 install/symlink
# rule cannot silently fall back to the pinned upstream GIT_TAG.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DVLLM_FLASH_ATTN_SRC_DIR=$FA_SRC"
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

# Keep a FIXED build directory across builds. ccache keys include the absolute
# paths of the build dir (via -I/-MMD flags), so a new /tmp/tmpXXXX.build-temp
# per run misses 100% of the cache even with hash_dir=false. A stable path is
# what actually makes rebuilds incremental and cache-hit.
export BUILD_TEMP_DIR="${BUILD_TEMP_DIR:-$REPO_ROOT/.build-temp}"

if [ "$INCREMENTAL" -eq 0 ]; then
  echo "  removing stale flash-attn extensions and installed CuTe tree"
  rm -f vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
  rm -rf vllm/vllm_flash_attn/cute
  # A "clean" build keeps BUILD_TEMP_DIR so ccache can reuse objects; use
  # BUILD_TEMP_DIR=... with a fresh path (or WIPE=1) for a truly cold build.
  if [ "${WIPE:-0}" = "1" ]; then
    echo "  WIPE=1: removing $BUILD_TEMP_DIR (cold build, no ccache reuse)"
    rm -rf "$BUILD_TEMP_DIR"
  fi
fi

# flash-attn's own submodules (CUTLASS) may be uninitialised in a fresh clone.
if [ -f "$VLLM_FLASH_ATTN_SRC_DIR/.gitmodules" ]; then
  git -C "$VLLM_FLASH_ATTN_SRC_DIR" submodule update --init --recursive
fi

# Editable install; vLLM's CMake builds _vllm_fa2_C from the source tree.
# The venv is managed by uv and has no pip module, so drive it through uv.
# `--config-settings=--build-option=...` pins setuptools' build_temp so the
# absolute paths embedded in compile commands stay stable between runs (this is
# what lets ccache hit), and the CMake build tree is reused incrementally.
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" -e . --no-build-isolation --no-deps \
    --config-settings="--build-option=build_ext --build-temp $BUILD_TEMP_DIR"
else
  "$PYTHON" -m pip install -e . --no-build-isolation --no-deps \
    --config-settings="--build-option=build_ext --build-temp $BUILD_TEMP_DIR"
fi

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

cute_interface = os.path.join(
    os.path.dirname(fai.__file__), "cute", "interface.py"
)
print(f"  FA4 interface: {cute_interface}")
# The CuTe sources import themselves as `flash_attn.cute.*`; CMake rewrites
# that to `vllm.vllm_flash_attn.cute` when installing. Verify the installed
# copy is exactly the selected checkout after that rewrite (no stale tree, no
# un-rewritten imports).
with open(cute_interface) as f:
    installed_text = f.read()
with open(os.path.join(src, "flash_attn", "cute", "interface.py")) as f:
    source_text = f.read()
expected_text = source_text.replace("flash_attn.cute", "vllm.vllm_flash_attn.cute")
assert installed_text == expected_text, (
    "installed CuTe interface does not match VLLM_FLASH_ATTN_SRC_DIR "
    "(after the flash_attn.cute -> vllm.vllm_flash_attn.cute rewrite)"
)
assert "from flash_attn.cute" not in installed_text, (
    "installed CuTe package still has un-rewritten flash_attn.cute imports"
)
print("  FA4 source:    exact match to checkout (imports rewritten)")
try:
    import vllm.vllm_flash_attn.cute.interface as _cute_iface
    print(f"  FA4 import:    OK ({_cute_iface.__file__})")
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"  FA4 import:    FAILED: {exc}")

so = os.path.join(os.path.dirname(fai.__file__), "_vllm_fa2_C.abi3.so")
print(f"  FA2 extension: {so} ({os.path.getsize(so) / 1e6:.0f} MB)")

cap = torch.cuda.get_device_capability()
print(f"  device:        {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
print(f"  fa2 supported: {fai._is_fa2_supported()}")
print(f"  fa4 supported: {fai._is_fa4_supported()}")
PY

echo
echo "Build complete."
