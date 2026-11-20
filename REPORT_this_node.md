# #3187 "彻底查清楚" report (2026-11-20, GPUs 2-7, B200 HGX, driver 580.159.04, VBIOS 97.00.E4.00.1E; all scripts opt2/, logs opt2/logs, details opt2/NOTES.md)

## 1. What the paper protocol measures: a boost plateau, not sustained throughput
`benchmarks/benchmark_attn.py` @ paper tag a365a190: repeats = 10, `Timer.timeit(10)` after `time.sleep(1)  # Sleep to avoid residual power throttling
from the previous benchmark`. One 8k x 4 backward is 2.3-4.2 ms, so a data point is a 25-45 ms burst from idle (16k / 32k: 45-170 ms).
Per-iteration CUDA-event timing after an idle gap (burst_profile.py, GPU 3, main; effective clock = known kernel cycles / time):

| 8k non-causal | iterations 1-9 (to ~42 ms) | it 19 (88 ms) | it 49-99 | it 299 (1.4 s) |
|---|---|---|---|---|
| ms / TFLOPS | 4.14-4.17 / 1320-1330 | 4.73 / 1160 | 4.65-5.05 / 1090-1180 | 4.67-4.87 / 1130-1180 |
| effective clock | ~1490 MHz (kernel ~1550) | ~1310 | 1230-1330 | 1270-1325 |

8k causal 1154-1163 for ~45 ms then 1005-1044; 32k causal (8.3 ms/call) iterations 1-4 at 1325-1337, 9+ at 1174-1182; forward 8k nc 1520-1540 then 1320-1390.
- The first ~45 ms after idle run on a plateau; then the 1000 W SW power cap (reason 0x4) pulls the clock down ~12 % for good. Sustained throughput is 12-15 % below the burst numbers at every size.
- 1k-8k windows (< 45 ms) sit inside the plateau: stable to 1 %. 16k-32k windows straddle the edge: +-5 % run to run (table x3: 32k causal 1225 / 1231 / 1291).
- With 3 reps instead of 10 (window back inside the plateau) 32k nc reads 1384-1421 = the paper's 1412; 16k nc 1380-1392 (paper 1457); 32k causal 1339-1349 (paper 1446).
- Idle 1 s = 5 s = 30 s = 180 s "cold"; neighbour load (5 GPUs concurrently) changes nothing; GPU-to-GPU spread on this node 1 % (1k-4k), 3 % (8k), 4-5 % (16k-32k).

## 2. What sets the plateau height (and therefore "paper vs us")
- Clock lock (GPU 3, 8k nc, Timer / plateau): unlocked 1299 / 1308-1319; -lgc 1965 1314 / 1319-1330; 1800 1318; 1700 1321; 1600 1320 / 1326-1336; 1500 1271 / 1273-1284; 1400 1191.
  The plateau is lock-independent down to 1600 MHz: an instantaneous limiter of the card holds the kernel at ~1550 MHz while nvidia-smi and DCGM report 1845 MHz
  (neither resolves it; the DCGM violation counters return sentinels on this stack). Sustained numbers are lock-independent too (power-bound).
- Power limit (GPU 2, Timer, main 8k nc / causal): 1000 W 1329 / 1172; 900 W 1243 / 1111; 800 W 1148 / 1045; 700 W 1029 / 966; 600 W 864 / 853: ~7-9 % per 100 W.
- Best-case protocol (idle 30 s, 3 reps) vs the paper, PR #3200 tree: causal -15 / -12 / -9 / -8 / -7 / -5 %, non-causal -11 / -9 / -8 / -8 / -5 / +1 % (1k..32k).
  The gap is ~8 % at 1k-8k (both boxes fully inside their plateau: pure plateau-clock difference) and shrinks to 0-5 % at 32k (the paper's own 10-rep window
  must also have run partly outside its plateau). Reading: the paper's card held a ~8 % higher plateau AND a plateau longer than our 45 ms. Both are
  card / VBIOS / driver properties (paper era 2026-03 drivers < our 580 < the reporter's 610); not reachable from code. Nothing else is left on the table.

## 3. Corrections to what we posted
1. "sustained clock under the 1000 W cap" (my 11-19 comment and the consolidated comment): the measurement never reaches the sustained regime; it is the
   plateau height of the first ~45 ms. The measured kernel clock (1517-1563 MHz under ncu) and the back-calculated paper clock (1580-1680) stand.
2. "+2 BAR.SYNC +1 FENCE +1 MEMBAR per dQ-drain iteration on DSL >= 4.6" (#3907 title / Status / first comment): wrong. Codex counted the dynamic
   named barriers in the 4.5.3 and 4.8 profiles: identical (8,667,136); ncu merely files them under different exec classes. The residual is only the
   per-use UR->R re-materialisation in the compute warps.

## 4. The residual is recoverable from the source side (Codex, GPU 5; verified on GPUs 4 + 5)
28-line change in flash_bwd_sm100.py (codex/residual.diff, worktree codexB = PR #3200 + diff): for causal 2CTA hdim 128, materialise the two broadcast
smem addresses (LSE / dPsum) through an identity `mov.b32` inline PTX before the compute loop and rebuild the tensors from them; DSL 4.8 then emits
`LDS.128 [UR+imm]` instead of re-materialising the uniform offset into a GPR before every load: compute class 421 -> 385 instructions/iteration, 128 regs.
Kernel cycles vs #3200 (GPU 5 / GPU 4, 2 rounds each): 8k causal 3.412 -> 3.358-3.369 (-1.3..-1.5 %), 32k causal 11.95 -> 11.78 (-1.5 %), 2k -1.0 %,
GQA 16/4 -1.4 %, non-causal / hd96 / 1CTA unchanged (guarded). Numerics vs #3200: out / dK / dV bitwise on 7 shapes, dQ <= 4.9e-4 bf16 (main's own noise).
Tests: dense causal hdim 128 all variants 620 passed (same 4 pre-existing sink+gqa failures as #3200 / main); targeted bwd set 196 passed, 4 skipped. ruff clean.
TFLOPS (GPU 5, paper method): main 552/807/1025/1183/1296/1289 -> #3200 557/821/1045/1210/1328/1308 -> +mov 558/824/1038/1223/1325/1336 causal.
Rejected by Codex: static stage indices (0), batching the dQ drain (+6.7 %), paired chunks (+0.9 %), plain pointer hoist (-0.2 %), identity shuffle (-1.1 %).
It is an inline-asm workaround for a compiler behaviour, so: separate follow-up PR (not folded into #3200), with the compiler statement for NVIDIA.

## 5. Where #3187 stands after today
- Causal-only regression since the paper: source-side redundant work exposed by DSL >= 4.6: fixed by #3200 (+#3909 in the DSL); the last 1.5 % by the mov.b32 follow-up.
- Common -7..-10 % to the paper (same in causal / non-causal): the card's boost-plateau height and length; measurable only by cycles / time; cannot be changed in code.
- Sustained throughput is 12-15 % below any burst number; the paper protocol was designed (sleep 1 s) to measure the burst.
