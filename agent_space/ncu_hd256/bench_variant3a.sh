#!/usr/bin/env bash
set -euo pipefail

# A/B bench for Variant 3a (TMA bulk store epilogue) vs the loop-reorder
# baseline. Adapted from agent_space/bench_pr_delta.sh.
#
#   A = 1748c9a (loop-reorder commit, pre-variant-3a)
#   B = current branch tip (variant 3a applied)
#
# Run on B200 GPU 0 with clocks locked at 1755 MHz.

GPU=0
A_REF=1748c9a
B_REF=$(git rev-parse HEAD)
SEQLENS="4096,8192,16384,32768,65536,131072"
COMMON="--headdim 256 --bwd --backend fa4 --seqlen $SEQLENS --causal both --warmup 5 --rep 50 --total-seqlen 131072"
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)
START_HEAD=$(git rev-parse HEAD)

cleanup() {
    cur=$(git rev-parse HEAD 2>/dev/null || true)
    if [[ "$cur" != "$START_HEAD" ]]; then
        git checkout -q "$START_BRANCH"
    fi
    nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

nvidia-smi -i $GPU -lgc 1755

OUTDIR=/tmp/fa4_v3a
mkdir -p $OUTDIR

run_bench() {
    local tag=$1; shift
    echo "--- $tag ---"
    CUDA_VISIBLE_DEVICES=$GPU python benchmarks/benchmark_attn.py \
        $COMMON "$@" 2>/dev/null | tee "$OUTDIR/${tag}.txt"
}

echo "=== AFTER (variant 3a, $B_REF) ==="
run_bench "after_mha" --nheads 32 --nheads-kv 32
run_bench "after_gqa" --nheads 32 --nheads-kv 2

echo "=== BEFORE (baseline, $A_REF) ==="
git checkout -q "$A_REF"
run_bench "before_mha" --nheads 32 --nheads-kv 32
run_bench "before_gqa" --nheads 32 --nheads-kv 2
git checkout -q "$START_BRANCH"

echo
echo "=== outputs ==="
ls -lh "$OUTDIR"
