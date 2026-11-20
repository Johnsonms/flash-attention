#!/bin/bash
# usage: measure.sh <gpu> <tree> <label> shapes...   (local port of measure3187.sh)
gpu=$1; tree=$2; label=$3; shift 3; shapes=${@:-8k_c 8k_nc}
W=$HOME/work3187; S=$HOME/handoff-3187/scripts; py=$HOME/venvs/fa3187/bin/python; NCU=/usr/local/cuda-13.2/bin/ncu
pp=$W/stub:$W/$tree; L=$W/logs/v1; mkdir -p $L
M=launch__cluster_dim_x,gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,launch__grid_size
for name in $shapes; do
  case $name in 8k_c) args="--seqlen 8192 --causal";; 8k_nc) args="--seqlen 8192";; 32k_c) args="--seqlen 32768 --causal";; 2k_c) args="--seqlen 2048 --causal";; *) echo bad; exit 2;; esac
  csv=$L/${label}_${name}.csv
  (cd /tmp && PYTHONPATH=$pp CUDA_VISIBLE_DEVICES=$gpu timeout 900 $NCU --clock-control none --metrics $M -k regex:BackwardSm100 --launch-skip 2 --launch-count 3 --csv $py $S/one_bwd3187.py $args 2>$csv.err | grep -v '^==' > $csv)
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
  echo "$label $name $line" | tee -a $W/logs/v1/results.tsv
done
