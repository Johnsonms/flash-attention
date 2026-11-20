#!/bin/bash
# V1: ncu kernel cycles, 4 trees x 2 rounds x 4 shapes, one GPU, interleaved. Run as: sudo bash ~/work3187/v1_sudo.sh [gpu]
set -u
gpu=${1:-1}; U=johnson; UH=/home/$U; W=$UH/work3187; S=$UH/handoff-3187/scripts
py=$UH/venvs/fa3187/bin/python; NCU=/usr/local/cuda-13.2/bin/ncu; L=$W/logs/v1; mkdir -p $L /tmp/root3187
export HOME=/tmp/root3187 XDG_CACHE_HOME=/tmp/root3187/cache PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$gpu
M=launch__cluster_dim_x,gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,launch__grid_size
cd /tmp
for r in 1 2; do for tree in main pr3200 mov pr3200mov; do for name in 8k_c 32k_c 8k_nc 2k_c; do
  case $name in 8k_c) a="--seqlen 8192 --causal";; 8k_nc) a="--seqlen 8192";; 32k_c) a="--seqlen 32768 --causal";; 2k_c) a="--seqlen 2048 --causal";; esac
  csv=$L/v1r${r}_${tree}_${name}.csv
  PYTHONPATH=$W/stub:$W/$tree timeout 900 $NCU --clock-control none --metrics $M -k regex:BackwardSm100 --launch-skip 2 --launch-count 3 --csv \
    $py -W ignore $S/one_bwd3187.py $a 2>$csv.err | grep -v '^==' > $csv
  line=$($py -I - $csv <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]))); h = [r for r in rows if r and r[0] == 'ID']
if not h: print("FAIL"); sys.exit()
h = h[0]; im, iv = h.index("Metric Name"), h.index("Metric Value"); d = {}
for r in rows:
    if len(r) == len(h) and r[0].isdigit(): d.setdefault(r[0], {})[r[im]] = float(r[iv].replace(',', ''))
cyc = [m['sm__cycles_elapsed.avg'] for m in d.values()]; mhz = [m['sm__cycles_elapsed.avg.per_second'] for m in d.values()]
print(f"{min(cyc)/1e6:.4f} {sum(mhz)/len(mhz)/1e6:.0f} n={len(cyc)} cluster_x={sorted({m['launch__cluster_dim_x'] for m in d.values()})}")
PY
)
  echo "v1r${r} $tree $name $line" | tee -a $L/v1.tsv
done; done; done
chown -R $U:$U $L
echo "done: $L/v1.tsv"
