#!/bin/bash
# kernel cycles of FlashAttentionBackwardSm100 for a given shim (worktree) on DSL 4.8 + quack 0.6.5 (the main stack).
# usage: measure3187.sh <gpu> <shim-name> <label> [shapes...]   shapes: 8k_c 8k_nc 32k_c 2k_c (default: 8k_c 8k_nc)
#   shim-name = directory under /home/johnson/hd256-bench-local/shims/ whose flash_attn/cute points at the tree under test.
# Appends "<label> <shape> <Mcycles> <MHz> n=<launches> cluster_x=[..]" to logs-2026-11-19/fix/results.tsv and prints it.
gpu=$1; shim=$2; label=$3; shift 3; shapes=${@:-8k_c 8k_nc}
H=/home/johnson/sandbox/flash-attention/issues/issue-3187-fa4-bwd-below-paper; D=/home/johnson/hd256-bench-local
py=/home/johnson/venvs/cutlass-dsl-4.8/bin/python; pp=$D/stub:$D/shims/$shim:/home/johnson/fa-pr2891-shims/deps
M=launch__cluster_dim_x,gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,launch__grid_size
mkdir -p $H/logs-2026-11-19/fix; rc=0
for name in $shapes; do
  case $name in 8k_c) args="--seqlen 8192 --causal";; 8k_nc) args="--seqlen 8192";; 32k_c) args="--seqlen 32768 --causal";; 2k_c) args="--seqlen 2048 --causal";; *) echo "unknown shape $name"; exit 2;; esac
  csv=$H/logs-2026-11-19/fix/${label}_${name}.csv
  (cd /tmp && PYTHONPATH=$pp CUDA_VISIBLE_DEVICES=$gpu timeout 900 ncu --clock-control none --metrics $M -k regex:BackwardSm100 --launch-skip 2 --launch-count 3 --csv $py $H/one_bwd3187.py $args 2>$csv.err | grep -v '^==' > $csv)
  line=$(python3 - $csv <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]))); h = [r for r in rows if r and r[0] == 'ID']
if not h: print("FAIL"); sys.exit()
h = h[0]; im, iv = h.index("Metric Name"), h.index("Metric Value"); d = {}
for r in rows:
    if len(r) == len(h) and r[0].isdigit(): d.setdefault(r[0], {})[r[im]] = float(r[iv].replace(',', ''))
cyc = [m['sm__cycles_elapsed.avg'] for m in d.values()]; mhz = [m['sm__cycles_elapsed.avg.per_second'] for m in d.values()]
cl = sorted({m['launch__cluster_dim_x'] for m in d.values()})
print(f"{min(cyc)/1e6:.4f} {sum(mhz)/len(mhz)/1e6:.0f} n={len(cyc)} cluster_x={cl}")
PY
)
  [ "$line" = FAIL ] && { rc=1; tail -3 $csv.err; }
  echo "$label $name $line" | tee -a $H/logs-2026-11-19/fix/results.tsv
done
exit $rc
