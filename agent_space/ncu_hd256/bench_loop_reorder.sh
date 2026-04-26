#!/usr/bin/env bash
set -euo pipefail
# Compare working-tree (loop reorder applied) vs HEAD (PR baseline w/o reorder).

GPU=0
SEQLENS="4096,8192,16384,32768"
COMMON="--headdim 256 --bwd --backend fa4 --seqlen $SEQLENS --causal both --warmup 5 --rep 50 --total-seqlen 32768"

cleanup() {
    if git stash list 2>/dev/null | grep -q "loop-reorder bench"; then
        git stash pop -q || true
    fi
    nvidia-smi -i $GPU -rgc >/dev/null 2>&1 || true
}
trap cleanup EXIT

nvidia-smi -i $GPU -lgc 1755

run_bench() {
    local tag=$1; shift
    CUDA_VISIBLE_DEVICES=$GPU python benchmarks/benchmark_attn.py \
        $COMMON "$@" 2>/dev/null | tee /tmp/fa4_lr_${tag}.txt
}

echo "=== AFTER (working tree, loop-reorder applied) ==="
run_bench "after_mha"  --nheads 32 --nheads-kv 32
run_bench "after_gqa"  --nheads 32 --nheads-kv 2

echo "=== BEFORE (HEAD, no reorder) ==="
git stash push -m "loop-reorder bench" -q
run_bench "before_mha" --nheads 32 --nheads-kv 32
run_bench "before_gqa" --nheads 32 --nheads-kv 2
git stash pop -q
