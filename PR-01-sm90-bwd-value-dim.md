# [CuTe, SM90] Backward: fix head_dim_v < head_dim (dO smem layout, preprocess dPsum overrun)

Branch: `Johnsonms:johnson/sm90-bwd-sdo-hdimv` (one commit on `main` 965c9212)
Category: correctness (silent wrong gradients + compile failure). SM100 counterpart: #2975 (dV epilogue for asymmetric head dims), #2552.

---

## Summary

On SM90 the CuTe backward with `head_dim_v < head_dim` has two independent bugs:

| (head_dim, head_dim_v) | main | this PR |
|---|---|---|
| (128, 96) | compile error in the bwd kernel | correct |
| (128, 64) | runs, **dQ/dK silently wrong for some heads** (write-order race) | correct |
| (96, 64), (192, 128), equal dims | correct | unchanged |

1. **dO smem layout** (`flash_bwd_sm90.py`). The dO layout took its major-mode size from `tile_hdim` instead of `tile_hdimv`:

   ```
   error: expected top-level shape equivalence between the SMEM layout and the CTA V-map,
   but got 'S<3,4,3> o 0 o ((8,10),(64,2)):((64,512),(1,5120))' and '(80,96):(1@0,1@1)'
   ```

   dO only ever has `head_dim_v` columns (A operand of dP = dO·Vᵀ, B operand of dV = Pᵀ·dO), so the size is `tile_hdimv // wg_d_dKV`. Equal-dim configs are unchanged by construction. For (192, 128) the swizzle atom stays 128B, and perf is unchanged (below).

2. **Preprocess dPsum overrun** (`flash_bwd_preprocess.py`). The O/dO tiled copy covers `num_threads / threads_per_row` rows per pass: 64 rows at `head_dim_v` 96, 32 at 64, 16 at 128. The dense hdim-128 SM90 config uses `tile_m = 80`, which is not a multiple of 64 or 32. The partition therefore spans rows past `tile_m`, and with padded offsets the write loop stores `dPsum = 0` for those rows. Inside a slice they are re-written with the right values by the next m block. From the last m block of a (batch, head) slice they land in **rows 0..47 (dv 96) / 0..15 (dv 64) of the next slice**. Whichever CTA writes last wins. The fix skips rows `>= tile_m` when the copy does not tile `tile_m` (constexpr, so configs where it does are unchanged).

## Evidence (H100 80GB, CUDA 13.3 driver, torch 2.14.1+cu130, cutlass-dsl from `flash_attn/cute[cu13]`)

dPsum captured right after the preprocess, compared with `rowsum(dO * O)`. Shape: b=2, sq=80, sk=256, h=4, d=128, non-causal, 10 seeds:

| tree | dv=96 wrong entries per run | dv=64 | dv=128 |
|---|---|---|---|
| main + layout fix only | 175, 175, 112, 168, 200, 168, 144, 152, 152, 152 | 0, 0, 0, 4, 0, 12, 12, 0, 4, 12 | all 0 |
| perturbation control (same-shape guard `row < (1 << 20)`) | 183, 159, 128, 160, 208, 168, 168, 152, 136, 152 | 0, 4, 4, 4, 0, 16, 0, 16, 0, 0 | all 0 |
| **this PR** | **all 0** | **all 0** | all 0 |

Every wrong entry is exactly 0.0, in rows [0, 47] for dv 96 and [12, 15] for dv 64, which is what the mechanism predicts. Slice (0, 0) is never hit.

## Tests

- New `test_flash_attn_sm90_value_dim_smaller`: (128, 96), (128, 64), (96, 64) × dense/causal × dense/varlen × seqlen (80, 211) / (257, 513), against an fp64 reference, using the existing `check_sm90_hdim_padding` helper. seqlen_q = 80 is one m block per head, so every head borders the next. With 4 m blocks per head the race is usually won by the right writer, which is why the existing (257, 513) padding tests never caught it.
  - this PR: 24 passed
  - unfixed preprocess: 3 failed (`dense-80-211-128-96`, `dense-80-211-128-64`, varlen `dense-80-211-128-96`)
- Existing `-k sm90_hdim_padding`: 162 passed (with this PR's test included).
- Sampled `test_flash_attn_output` / `test_flash_attn_varlen_output` at d ∈ {64, 96, 128}: see "Regression run" below.

## Performance (bwd, b=4, s=4096, h=16, bf16; rough: shared GPU)

| config | main | this PR |
|---|---|---|
| d192 dv128 | 464 TFLOP/s | 474 |
| d192 dv128 causal | 413 | 413 |
| d128 dv64 | 511 | 519 |
| d128 dv128 | 587 | 587 |

## Regression run

PENDING. Will be filled in when the sampled run completes.

## Not covered

- SM90 GQA backward still asserts `head_dim == head_dim_v` (`flash_bwd_sm90.py:116`). That's a feature, separate PR.
- (64, 128) still overflows smem on SM90 bwd (the config is keyed on head_dim only). Separate PR.
- SASS diff of fixed vs unfixed preprocess not done. The perturbation control above stands in for it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
