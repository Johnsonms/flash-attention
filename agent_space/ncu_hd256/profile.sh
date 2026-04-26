#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/ncu
SECTIONS="--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats --section SchedulerStats --section SourceCounters --section LaunchStats"

run_ncu() {
    local tag=$1; local kfilter=$2; local causal=$3
    echo "=== profiling $tag (kernel=$kfilter causal=$causal) ==="
    CFG_CAUSAL=$causal CUDA_VISIBLE_DEVICES=0 ncu \
        --target-processes all \
        --kernel-name regex:"$kfilter" \
        --launch-count 1 \
        $SECTIONS \
        -o /tmp/ncu/${tag} -f \
        python agent_space/ncu_hd256/repro.py 2>&1 | tail -5
}

# Config A: MHA non-causal
run_ncu "A_dq"   "dqkernel"   "0"
run_ncu "A_dkdv" "dkdvkernel" "0"

# Config B: MHA causal
run_ncu "B_dq"   "dqkernel"   "1"
run_ncu "B_dkdv" "dkdvkernel" "1"

echo ""
echo "=== reports ==="
ls -lh /tmp/ncu/*.ncu-rep
