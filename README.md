# #3187 handoff: verify the FA4 hdim 128 backward findings and fixes on a second B200 machine

Written 2026-11-20 from the slinky-0 B200 HGX node (driver 580.159.04, VBIOS 97.00.E4.00.1E). Everything below was
measured there; the point of the second machine is to confirm (a) the code-side gains of the three fixes and (b) whether
the remaining gap to the paper numbers is a per-card property, as concluded here. Full detail: REPORT_this_node.md,
results_this_node.tsv, and the sandbox `opt2/NOTES.md` / `logs-2026-11-19/NOTES.md` on slinky-0.

## 1. The problem and where it stands

Issue Dao-AILab/flash-attention#3187: FA4 hdim 128 backward on B200 is below the paper's numbers
(paper, hdim 128, 16 heads, 32k tokens/shape, 1k..32k: causal 614/887/1102/1284/1402/1446 TFLOPS,
non-causal 936/1166/1331/1430/1457/1412). The regression is causal-only in the code; a second, size-independent gap
is common to causal and non-causal.

Three findings, three artefacts (all opened as GitHub user Johnsonms):

| # | what | where | effect on this node |
|---|---|---|---|
| A | CuTe DSL >= 4.6 lowers `cute.arch.*` cluster-rank / pair-mask arguments of the SM100 pipelines (and the causal mask-gate) to runtime values, exposing redundant per-iteration work in the FA source that DSL 4.5.3 folded away | NVIDIA/cutlass#3907 (issue, "compiler statement"); fix in FA: Dao-AILab/flash-attention#3200 (branch `Johnsonms:sm100-bwd-causal-mask-gate-const-cluster`); DSL-side counterpart: NVIDIA/cutlass#3909 (branch `Johnsonms:dsl-2cta-constant-cluster-targets` @ 68e43622, one file `python/CuTeDSL/cutlass/pipeline/sm100.py`) | #3200: causal −3.9 % (8k), −4.6 % (32k), GQA 16/4 −4.2 %, hd96 −5.9 %, hd256 causal −4.6 %, non-causal ≤ −1 %, 1CTA 0 |
| B | residual DSL 4.8 lowering: compute warps re-materialise the uniform smem offset of the broadcast LSE / dPsum `LDS.128` into a GPR before every use | follow-up branch `Johnsonms:sm100-bwd-materialize-lse-dpsum-smem-addr` @ 9800aa66 = main dbabcc78 + `scripts/fa-mov.diff` (28 lines, identity `mov.b32` inline PTX; PR NOT opened yet) | on main: 8k causal −2.0 %, 32k causal −2.2 %, 2k −1.4 %, GQA −2.3 %, non-causal 0 (guarded). Adds on top of #3200 (8k causal 3.547 → 3.410 → 3.36 Mcycles) |
| C | the paper protocol measures a ~45 ms boost plateau after idle (kernel ~1550 MHz, lock-independent down to -lgc 1600), after which the 1000 W SW power cap (reason 0x4) drops the clock ~12 % for good. Our card's plateau is ~8 % lower than the paper's and shorter; the common −7..−10 % gap is this, not code | documented in #3187 consolidated comment + #3907; nothing to fix in code | best case (idle 30 s, 3 reps) PR #3200 vs paper: causal −15/−12/−9/−8/−7/−5 %, nc −11/−9/−8/−8/−5/+1 % (1k..32k) |

Two statements that were posted earlier and then corrected (do not re-derive them): "sustained clock under the
1000 W cap" (it is the plateau height, the measurement never reaches the sustained regime) and "+2 BAR.SYNC +1 FENCE
+1 MEMBAR per dQ-drain iteration" (dynamic named-barrier counts are identical on 4.5.3 and 4.8: 8,667,136; it was an
ncu exec-class artefact).

## 2. Environment to reproduce

- B200 (sm_100), any recent driver; record `nvidia-smi --query-gpu=driver_version,vbios_version,power.limit --format=csv`.
- Python 3.10 venv: `torch==2.14.0+cu130` (cu128 also fine), `nvidia-cutlass-dsl==4.8.0`, `quack-kernels==0.6.5`,
  `einops`, `torch_c_dlpack_ext` (pip; FA's cute interface imports quack + einops). ncu from CUDA 13 (or 12.8+).
- FA trees (clone `https://github.com/Johnsonms/flash-attention`, or apply the diffs in `scripts/` to the given base):
  - `main` = Dao-AILab main @ dbabcc78 (base of everything below)
  - `pr3200` = branch `sm100-bwd-causal-mask-gate-const-cluster` (`scripts/fa-3200.diff` on dbabcc78 applies cleanly)
  - `mov` = branch `sm100-bwd-materialize-lse-dpsum-smem-addr` @ 9800aa66 (= main + `scripts/fa-mov.diff`)
  - `pr3200+mov` = pr3200 + `scripts/fa-mov.diff` (what Codex validated here; applies cleanly)
- No install needed: run from a neutral cwd (e.g. `/tmp`) with `PYTHONPATH=<tree>` so `flash_attn/cute` is the tree under test
  (each tree is its own checkout or `git worktree`). The `flash_attn/__init__.py` of main imports the FA2 extension lazily;
  if an import of `flash_attn_2_cuda` fails, put a directory containing an empty `flash_attn_2_cuda.py` first on PYTHONPATH.
- DSL-side fix (#3909) without rebuilding the wheel: copy `site-packages/cutlass` to a scratch dir, apply
  `scripts/cutlass-3909.diff` (strip `python/CuTeDSL/`), and put that scratch dir first on PYTHONPATH (or a `sitecustomize.py`
  that inserts it). `python -c "import cutlass; print(cutlass.__file__)"` must show the copy.

## 3. Measurement rules (learned the hard way here)

1. Kernel cycles are the only clock-independent metric: `ncu --clock-control none --metrics sm__cycles_elapsed.avg,...`
   (`scripts/measure3187.sh`). Do not compare wall-clock A/B at different times of day.
2. For TFLOPS use the paper's own method (`scripts/paper_timer2.py` = `benchmark_attn.py` Timer: `sleep 1`, then
   `Timer.timeit(10)`), with the clocks locked `nvidia-smi -lgc 1965,1965 -i N` (reset `-rgc`). Expect ±1 % run-to-run at 1k–8k,
   ±5 % at 16k–32k (plateau edge). 3 reps instead of 10 moves 16k–32k back inside the plateau.
3. nvidia-smi / DCGM clock readouts do not resolve the plateau (they report 1845 MHz while the kernel runs ~1550). Only
   `cycles / time` (`scripts/burst_profile.py`, known kernel cycles) measures the real clock.
4. Do not read anything from the first iteration after an idle gap (launch / cold). Idle 1 s = 5 s = 30 s = 180 s.
5. GPU-to-GPU spread on one node: 1 % (1k–4k), 3 % (8k), 4–5 % (16k–32k). Always A/B on the same GPU, interleaved.
6. Power limit matters ~7–9 % per 100 W; check `nvidia-smi -q -d POWER` and the active throttle reason
   (`clocks_throttle_reasons.active`, 0x4 = SW power cap) during the run.

## 4. What to verify, with commands and expected results

All commands: `cd /tmp; export CUDA_VISIBLE_DEVICES=N; PY=<venv>/bin/python; S=<this dir>/scripts`.

### V1  Kernel cycles: main vs pr3200 vs mov vs pr3200+mov (the code-side claims)

```
for tree in main pr3200 mov pr3200mov; do for r in 1 2; do
  PYTHONPATH=/path/$tree bash $S/measure3187.sh $GPU <ignored> ${tree}_r$r 8k_c 32k_c 8k_nc 2k_c
done; done
```
(`measure3187.sh` builds PYTHONPATH from a shim dir; on the new machine edit the two path lines at its top, or run its
inner `ncu ... one_bwd3187.py` line directly: `ncu --clock-control none --metrics launch__cluster_dim_x,gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,launch__grid_size -k regex:BackwardSm100 --launch-skip 2 --launch-count 3 --csv $PY $S/one_bwd3187.py --seqlen 8192 --causal`, take the min `sm__cycles_elapsed.avg`.)

Expected (Mcycles, this node, 2 rounds each; `cluster_x` must print `[2.0]`):

| shape | main | pr3200 | main+mov | pr3200+mov |
|---|---|---|---|---|
| 8k causal | 3.549–3.553 | 3.408–3.412 | 3.479–3.482 | 3.358–3.369 |
| 32k causal | 12.53 | 11.95 | 12.25 | 11.77–11.78 |
| 2k causal | 1.224 | – | 1.206 | – |
| 8k non-causal | 6.151 | 6.08–6.09 | 6.146–6.153 | 6.07–6.10 |

Pass = same ordering and gains within ±0.5 % of the above. GQA (`--nheads 16 --nheads_kv 4 --seqlen 8192 --causal`,
needs `one_bwd3187.py` extended with `--nheads_kv`, or use `numcheck3187b.py` shapes): main 3.639 → mov 3.555 (−2.3 %);
pr3200 −4.2 % vs main.

### V2  Paper-method TFLOPS: main vs pr3200 (and vs the paper)

```
sudo nvidia-smi -lgc 1965,1965 -i $GPU
for r in 1 2 3; do for tree in main pr3200; do
  PYTHONPATH=/path/$tree $PY $S/paper_timer2.py --reps 10 --sleep 1 > tf_${tree}_r$r.log; done; done
sudo nvidia-smi -rgc -i $GPU
```
Expected (median of 3, GPU 7 here), causal 1k..32k: main 550/800/1010/1166/1267/1254 → pr3200 555/811/1028/1191/1298/1293;
non-causal within noise (862/1091/1242/1325/1302/1288 → 866/1094/1245/1318/1308/1285).
The interesting number on the new machine is the absolute level: if main alone already reads the paper's
614/887/1102/… at 1k–8k, that card's plateau is the paper's and finding C is confirmed; if it reads our ~550/800/1010, the
gap is the same on a second card.

### V3  The plateau (finding C)

```
PYTHONPATH=/path/main $PY $S/burst_profile.py --seqlen 8192 --idle 5 --iters 300            # non-causal
PYTHONPATH=/path/main $PY $S/burst_profile.py --seqlen 8192 --idle 5 --iters 300 --causal
PYTHONPATH=/path/main $PY $S/burst_profile.py --seqlen 8192 --idle 5 --iters 300 --fwd
```
Expected here (8k nc): iterations 1–9 (first ~42 ms) 4.14–4.17 ms / 1320–1330 TFLOPS, then 4.65–5.05 ms / 1090–1180 from
~45 ms on, never recovering within the burst. Record (a) the plateau length in ms, (b) plateau vs sustained TFLOPS ratio,
(c) the throttle reason that appears (sample `nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -lms 20` alongside).
Also repeat V2 with `--reps 3 --sleep 30` (best case): here 32k nc then reaches 1384–1421 = the paper's 1412.
Optional: `nvidia-smi -pl 900/800 -i N` (restore afterwards) should move V2 by ~7–9 % per 100 W; `-lgc 1800/1700/1600`
should NOT change the plateau numbers, `-lgc 1500` should (−3 %).

### V4  Numerics (both fixes)

```
for tree in main pr3200 mov pr3200mov; do
  PYTHONPATH=/path/$tree $PY $S/numcheck3187.py num_$tree.pt
  PYTHONPATH=/path/$tree $PY $S/numcheck3187b.py numb_$tree.pt            # GQA / hd64/96/256 shapes
  FA_DISABLE_2CTA=1 PYTHONPATH=/path/$tree $PY $S/numcheck3187b.py numb1_$tree.pt   # 1CTA path
done
$PY $S/numcompare.py num_main.pt num_mov.pt      # etc.
```
Expected vs main: out / dK / dV bitwise (0.000e+00) on every shape; dQ ≤ ~5e-4 (fp32 atomic reduce order; two runs of
main itself differ by the same amount, so also compare `num_main.pt` against a second main run).

### V5  Tests (FA tree, GPU, ~6 + ~5 min)

```
cd /path/<tree>; PYTHONPATH=. $PY -m pytest tests/cute/test_flash_attn.py -x -q -k "test_flash_attn_output and -128-True-"
PYTHONPATH=. $PY -m pytest tests/cute/test_flash_attn.py -q -k "clc_2cta or hd256_sm100_deterministic or bwd_preallocated or lse_grad or sm100_bwd_unused_kv_tail or small_head_dim or pack_gqa_padded or empty_backward or seqlen_k_zero or local_huge_window or varlen_backward_non_aligned or learnable_sink_backward"
```
Expected on pr3200+mov and on main: first set 620 passed with the same 4 pre-existing `learnable_sink` + gqa failures
(drop `-x`), second set 196 passed, 4 skipped. pr3200 alone, non-causal dense: 4988 passed / 4 pre-existing.
With the #3909 DSL copy on PYTHONPATH: 6232 passed / 8 pre-existing (dense campaign), and the CuTeDSL
`examples/python/CuTeDSL/blackwell/dense_gemm` tutorials still run.

### V6  (optional) #3909 alone and the compiler statement

With the patched `cutlass` copy first on PYTHONPATH and the `main` FA tree: V1 8k causal should land close to pr3200
(this node: 3.3563 Mcycles for paper-era a00ddeb2 on DSL 4.5.3, i.e. the fix recovers the 4.5.3 level); #3909 does not
address finding B, so `mov` still adds its ~2 % on top. If a DSL 4.5.3 venv is available: the SAME FA source
(e.g. 68bd095c, before FA's 4.6 bump) gives the same TFLOPS on 4.5.3 and on 4.8+#3200, which is the whole argument of #3907.

## 5. What a result on the second machine would change

- V1/V4/V5 reproduce → the three fixes are machine-independent; the mov.b32 follow-up PR can be opened with two-node data.
- V2 absolute level at 1k–8k ≥ paper on `main` → finding C confirmed as a card/VBIOS/driver property; post the second
  node's driver/VBIOS/plateau numbers to #3187 as the explanation of the remaining gap.
- V2 still ~8 % below paper AND V3 shows a plateau ≥ 100 ms or no plateau → finding C is wrong; the common gap needs a
  new explanation (then look at the forward too: here fwd 8k nc shows the same 1520–1540 → 1320–1390 plateau).
- Anything in V1 not reproducing (ordering or > 1 % off) → report the ncu CSVs and `cluster_x`; a `cluster_x=[1.0]`
  means the 2CTA path was not taken (wrong shape / env), not a perf difference.
