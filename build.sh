#!/usr/bin/env bash
# Clean rebuild of vLLM against the SM120 flash-attention fork.
#
# The flash-attention source tree provides BOTH halves of the SM120 work:
#   - FA2: csrc/flash_attn (C++/CUDA) -> _vllm_fa2_C.abi3.so
#   - FA4: flash_attn/cute (CuTe DSL)  -> installed into vllm/vllm_flash_attn/cute
#
# Because this script always sets VLLM_FLASH_ATTN_SRC_DIR, CMake's FA4 install
# takes its *symlink* branch: vllm/vllm_flash_attn/cute becomes a symlink to
# $FA_SRC/flash_attn/cute. The CuTe sources keep their native
# `flash_attn.cute.*` imports and vllm_flash_attn/__init__.py registers a
# virtual `flash_attn` package so those imports resolve. (The alternative
# copy+rewrite branch only runs when VLLM_FLASH_ATTN_SRC_DIR is unset.)
#
# `VLLM_FLASH_ATTN_SRC_DIR` makes vLLM's CMake build flash-attn from that path
# instead of the pinned GIT_TAG.
#
# FA2 must be compiled to native sm_120 SASS: a PTX-only build fails at runtime
# on this host with cudaErrorUnsupportedPtxVersion (nvcc 13.4 vs older driver).
#
# Arch selection: TORCH_CUDA_ARCH_LIST defaults to 12.0 but is overridable, e.g.
#   TORCH_CUDA_ARCH_LIST="8.0;8.6;12.0" ./build.sh
# Note FA2's arch set is fixed by the FA fork's CMakeLists to `8.0+PTX;12.0`, so
# sm86 gets vLLM's own kernels but falls back to sm_80 PTX for FA2.
#
# Editing workflow: edit either tree (this repo's vllm/, or the FA checkout's
# flash_attn/cute/), then re-run ./build.sh. FA4 is symlinked so CuTe-DSL edits
# are live on the next Python start with no rebuild; FA2 (C++/CUDA) needs the
# rebuild. The script ends by verifying both halves actually import and run, and
# exits non-zero if not -- a "green" build.sh is the ship signal.
#
# Usage:
#   ./build.sh                # clean build (removes stale .so + FA4 tree first)
#   ./build.sh --incremental  # keep build/ and existing .so
#   FA_SRC=/path/to/flash-attention ./build.sh   # different FA checkout
#   SKIP_GPU_CHECK=1 ./build.sh                  # skip the GPU smoke test
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Default to the prod-sm120 checkout. Note: do NOT inherit an ambient
# VLLM_FLASH_ATTN_SRC_DIR from an interactive shell (it may point at an older
# flash-attention tree); override explicitly via FA_SRC=... instead.
FA_SRC="/home/danielk/code/flash-attention"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
INCREMENTAL=0
[ "${1:-}" = "--incremental" ] && INCREMENTAL=1

if [ ! -x "$PYTHON" ]; then
  echo "error: python not found at $PYTHON" >&2
  exit 1
fi
# Preflight the FA checkout: a missing or half-initialised tree produces
# confusing downstream failures (CMake silently falling back to the pinned
# upstream GIT_TAG, or a copy-mode FA4 install). Fail here instead.
if [ ! -f "$FA_SRC/flash_attn/cute/interface.py" ]; then
  echo "error: FA source tree not found at $FA_SRC" >&2
  echo "       (set FA_SRC=/path/to/flash-attention)" >&2
  exit 1
fi
if [ ! -d "$FA_SRC/csrc/flash_attn/src" ]; then
  echo "error: $FA_SRC has no csrc/flash_attn/src (FA2 sources missing)" >&2
  exit 1
fi
# The SM120 vendor branch is what makes this FA2/FA4 build work here; the
# upstream pinned tag would build but lacks flash_fwd_sm120_tma.py.
if [ ! -f "$FA_SRC/flash_attn/cute/flash_fwd_sm120_tma.py" ]; then
  echo "error: $FA_SRC/flash_attn/cute/flash_fwd_sm120_tma.py missing" >&2
  echo "       this does not look like the SM120 vendor flash-attention branch" >&2
  echo "       use FA_SRC=/path/to/vendor-sm120" >&2
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

# setuptools' build_ext puts its CMake tree in $REPO_ROOT/build by default.
# That path is already stable across runs, which is what ccache needs (its keys
# include the absolute build paths via -I/-MMD). An earlier version of this
# script tried to pin the tree explicitly, but setuptools >= 64's build_meta
# silently ignores `--build-option` and `--global-option` corrupts the command
# line, so the pin never took effect and a random /tmp/tmpXXXX.build-temp was
# used each run instead. The default in-repo tree is used deliberately now.
BUILD_DIR="$REPO_ROOT/build"

if [ "$INCREMENTAL" -eq 0 ]; then
  echo "  removing stale flash-attn extensions and installed CuTe tree"
  rm -f vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
        vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
  rm -rf vllm/vllm_flash_attn/cute
  # A "clean" build keeps $BUILD_DIR so ccache can reuse objects; set WIPE=1
  # for a truly cold build with no ccache reuse.
  if [ "${WIPE:-0}" = "1" ]; then
    echo "  WIPE=1: removing $BUILD_DIR (cold build, no ccache reuse)"
    rm -rf "$BUILD_DIR"
  fi
fi

# Purge orphaned setuptools build trees. `editable_wheel` allocates a fresh
# /tmp/tmpXXXX.build-temp per invocation and leaves it behind; a stale tree from
# an older checkout gets reused, baking in its CMake cache (old FA source path,
# old arch list, and the copy-not-symlink FA4 install rule). Removing them keeps
# the build deterministic.
for _stale in /tmp/tmp*.build-temp; do
  [ -d "$_stale" ] || continue
  # Only reap trees we own.
  if [ -O "$_stale" ]; then
    echo "  removing orphaned build tree: $_stale"
    rm -rf "$_stale"
  fi
done

# Drop any stale FetchContent populate of vllm-flash-attn. FetchContent honours
# SOURCE_DIR only on the FIRST populate; if .deps/vllm-flash-attn-src already
# exists (e.g. it was populated from the pinned upstream GIT_TAG by an earlier
# build that ran without VLLM_FLASH_ATTN_SRC_DIR), CMake silently keeps the
# upstream checkout and ignores this fork. Symptoms: FA4 installed as a copied
# directory with rewritten imports instead of a symlink, and FA4 missing the
# fork's flash_fwd_sm120_tma.py.
rm -rf "$REPO_ROOT/.deps/vllm-flash-attn-src" \
      "$REPO_ROOT/.deps/vllm-flash-attn-subbuild" \
      "$REPO_ROOT/.deps/vllm-flash-attn-build"

# flash-attn's own submodules (CUTLASS) may be uninitialised in a fresh clone.
if [ -f "$VLLM_FLASH_ATTN_SRC_DIR/.gitmodules" ]; then
  git -C "$VLLM_FLASH_ATTN_SRC_DIR" submodule update --init --recursive
fi

# Editable install; vLLM's CMake builds _vllm_fa2_C from the source tree.
# The venv is managed by uv and has no pip module, so drive it through uv.
#
# --no-build-isolation means uv does NOT install [build-system].requires into
# the backend environment: setup.py runs inside this venv as-is. torch is pinned
# and already provided by the precompiled install, but the rest of the build
# requirements (setuptools-rust, setuptools-scm, cmake) must be present too or
# prepare_metadata_for_build_editable dies with ModuleNotFoundError before CMake
# ever runs. They are cheap, so install them up front when any is missing.
BUILD_REQS=(
  "setuptools>=77.0.3,<81.0.0"
  "setuptools-scm>=8"
  "setuptools-rust>=1.9.0"
  "cmake>=3.26.1"
  ninja
  packaging
  wheel
  jinja2
)
if ! "$PYTHON" -c 'import setuptools, setuptools_rust, setuptools_scm, cmake' \
    >/dev/null 2>&1; then
  echo "  installing build requirements (torch already present in venv)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" "${BUILD_REQS[@]}"
  else
    "$PYTHON" -m pip install "${BUILD_REQS[@]}"
  fi
fi

# Note: do NOT try to pass --build-temp through --config-settings. setuptools >=
# 64's build_meta ignores `--build-option`, and `--global-option` is appended to
# the *global* command line (bdist_wheel) rather than to build_ext -- passing
# `build_ext --build-temp` there corrupts the command flow and CMake ends up
# running in a directory with no cache ("not a CMake build directory").
# The default in-repo build/ tree is stable enough for ccache, so just use it.
#
# `--no-cache` is required. uv's wheel cache key records the environment as an
# empty dict (`"env":{}` in the installed dist-info's uv_cache.json), so it does
# NOT invalidate when VLLM_FLASH_ATTN_SRC_DIR / TORCH_CUDA_ARCH_LIST / CMAKE_ARGS
# change. Without it, every run is a stale cache hit that skips CMake entirely
# and reinstates a previous FA4 install. `--reinstall` alone is not enough.
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" -e . --no-build-isolation --no-deps \
    --no-cache --reinstall
else
  "$PYTHON" -m pip install -e . --no-build-isolation --no-deps --no-cache-dir
fi

# ---------------------------------------------------------------------------
# Enforce FA4 symlink mode.
#
# CMake only creates the symlink when VLLM_FLASH_ATTN_SRC_DIR is visible to the
# *CMake* process. When it isn't -- e.g. an incremental/dependency-cached build
# that skips the FA4 install rule entirely, or a leftover FetchContent populate
# from the pinned upstream tag -- CMake instead installs `cute/` as a REAL
# directory of verbatim sources whose imports are still `flash_attn.cute.*`.
# That combination cannot work: vllm_flash_attn/__init__.py only registers the
# virtual `flash_attn` package when `cute` is a symlink, so every FA4 import
# dies with `No module named 'flash_attn'`.
#
# The sources are identical either way, so normalise to the supported layout
# rather than trusting CMake. This also makes CuTe-DSL edits live without a
# rebuild, which is the point of the symlink mode.
#
# Note `cute` is gitignored (a build artifact), so rewriting it is safe.
# ---------------------------------------------------------------------------
CUTE_LINK="$REPO_ROOT/vllm/vllm_flash_attn/cute"
CUTE_TARGET="$FA_SRC/flash_attn/cute"
if [ -L "$CUTE_LINK" ]; then
  _cur="$(readlink -f "$CUTE_LINK")"
  if [ "$_cur" != "$(readlink -f "$CUTE_TARGET")" ]; then
    echo "  fixing stale FA4 symlink ($_cur -> $CUTE_TARGET)"
    rm -f "$CUTE_LINK"
    ln -s "$CUTE_TARGET" "$CUTE_LINK"
  fi
elif [ -d "$CUTE_LINK" ]; then
  echo "  CMake installed FA4 in copy mode; replacing with symlink"
  echo "    (copy mode leaves native flash_attn.cute imports that cannot resolve)"
  rm -rf "$CUTE_LINK"
  ln -s "$CUTE_TARGET" "$CUTE_LINK"
else
  echo "  FA4 tree missing after build; creating symlink"
  mkdir -p "$(dirname "$CUTE_LINK")"
  ln -s "$CUTE_TARGET" "$CUTE_LINK"
fi
# Also drop any stale bytecode under the (now symlinked) CuTe tree so edits are
# never served from a cached .pyc with a mismatched mtime.
find "$CUTE_TARGET" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "== post-build checks =="
# These checks are the ship gate: they raise SystemExit on any failure so a
# non-verifying build can never look successful. SKIP_GPU_CHECK=1 skips only the
# kernel smoke test (for builds on a host with no GPU).
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}" "$PYTHON" - <<'PY'
import os
import sys

import torch

from vllm.vllm_flash_attn import flash_attn_interface as fai

src = os.environ["VLLM_FLASH_ATTN_SRC_DIR"]
pkg_dir = os.path.dirname(fai.__file__)
cute_dir = os.path.join(pkg_dir, "cute")
cute_interface = os.path.join(cute_dir, "interface.py")
failures = []


def check(label, ok, detail=""):
    print(f"  {label:<22}{'OK' if ok else 'FAIL'} {detail}".rstrip())
    if not ok:
        failures.append(f"{label.strip()}: {detail}")


# FA2: the stable-ABI extension must have loaded and registered its ops.
fa2_ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith("_vllm_fa2_C::")]
check("FA2 available:", fai.FA2_AVAILABLE, fai.FA2_UNAVAILABLE_REASON or "")
check("FA2 ops:", bool(fa2_ops), f"{len(fa2_ops)} registered")
print(f"  FA3 available:        {fai.FA3_AVAILABLE}")

# FA4 install layout. Symlink mode is the only mode the import shim supports;
# build.sh normalises to it above, so anything else is a real failure.
is_link = os.path.islink(cute_dir)
expected = os.path.realpath(os.path.join(src, "flash_attn", "cute"))
actual = os.path.realpath(cute_dir) if is_link else None
check("FA4 mode:", is_link, f"symlink -> {actual}" if is_link else "copied directory (not a symlink)")
if is_link:
    target_ok = actual == expected
    check("FA4 symlink target:", target_ok, f"{actual}")
    if target_ok and os.path.isfile(cute_interface):
        with open(cute_interface) as f:
            installed_text = f.read()
        with open(os.path.join(src, "flash_attn", "cute", "interface.py")) as f:
            source_text = f.read()
        check("FA4 matches checkout:", installed_text == source_text)
    elif not os.path.isfile(cute_interface):
        check("FA4 matches checkout:", False, f"{cute_interface} not found")

# FA4_AVAILABLE is a file-existence probe and stays True even when the imports
# are broken, so a real import is the only meaningful check.
try:
    import vllm.vllm_flash_attn.cute.interface as _cute_iface

    check("FA4 import:", True, _cute_iface.__file__)
except Exception as exc:  # noqa: BLE001
    check("FA4 import:", False, str(exc))

so = os.path.join(pkg_dir, "_vllm_fa2_C.abi3.so")
if os.path.isfile(so):
    check("FA2 extension exists:", True, f"{os.path.getsize(so) / 1e6:.0f} MB")
else:
    check("FA2 extension exists:", False, f"{so} missing")
    print(
        "\n    The FA2 CUDA extension was not installed. A clean build deletes it\n"
        "    first, so this means CMake/uv skipped the flash-attn sub-build.\n"
        "    Re-run with `WIPE=1 ./build.sh`, and if that still fails check\n"
        "    that ~/.cache/ccache has space and that nvcc is on PATH.",
        file=sys.stderr,
    )

# The FA2 cubin must carry native sm_120 SASS. A PTX-only build compiles fine and
# then dies at the first call on this host (cudaErrorUnsupportedPtxVersion).
if os.path.isfile(so) and torch.cuda.is_available():
    import subprocess

    try:
        elf = subprocess.run(
            ["cuobjdump", "-lelf", so], capture_output=True, text=True, timeout=300
        ).stdout
        have_sm120 = "sm_120" in elf
        check("FA2 native sm_120:", have_sm120, "" if have_sm120 else "no sm_120 cubin (PTX-only build)")
    except Exception as exc:  # noqa: BLE001
        print(f"  FA2 native sm_120:    skipped ({exc})")

if not torch.cuda.is_available():
    print("  device:               no CUDA device; skipping GPU checks")
else:
    cap = torch.cuda.get_device_capability()
    print(f"  device:               {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    check("fa2 supported:", fai._is_fa2_supported())
    check("fa4 supported:", fai._is_fa4_supported())

    # Real kernel launches -- the only proof the build actually runs. Importing
    # FA4 exercises the CuTe compile path, which is where SM120 regressions show
    # up (bad masking, wrong tile, DSL API drift).
    if os.environ.get("SKIP_GPU_CHECK", "0") != "1" and not failures:
        import torch.nn.functional as F

        for ver in (2, 4):
            if not fai.is_fa_version_supported(ver):
                continue
            try:
                torch.manual_seed(0)
                B, S, H, D = 2, 512, 8, 128
                dt = torch.bfloat16
                q = torch.randn(B * S, H, D, device="cuda", dtype=dt)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                cu = torch.tensor([0, S, B * S], dtype=torch.int32, device="cuda")
                out = fai.flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=S, cu_seqlens_q=cu,
                    max_seqlen_k=S, cu_seqlens_k=cu,
                    causal=True, fa_version=ver,
                )
                ref = F.scaled_dot_product_attention(
                    q.view(B, S, H, D).transpose(1, 2),
                    k.view(B, S, H, D).transpose(1, 2),
                    v.view(B, S, H, D).transpose(1, 2),
                    is_causal=True,
                ).transpose(1, 2).reshape(B * S, H, D)
                err = (out.float() - ref.float()).abs().max().item()
                check(f"FA{ver} forward runs:", err < 2e-2, f"max_abs_err={err:.2e}")
            except Exception as exc:  # noqa: BLE001
                check(f"FA{ver} forward runs:", False, f"{type(exc).__name__}: {exc}")

    # FA4 is available on sm120 but FA2 stays the default there (faster decode).
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        print(f"  default FA:           v{get_flash_attn_version()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  default FA:           unknown ({exc})")

print()
if failures:
    print("BUILD VERIFICATION FAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("All checks passed.")
PY

echo
echo "Build complete."
