#!/usr/bin/env python
"""SM100 Blackwell head_dim=256 paged KV TMA benchmark.

Compares paged KV performance across kernel paths (all use TMA):
  A. Generic SM100 paged TMA  (current default fallback)
  B. hd256 2CTA paged TMA     (our new specialized implementation)

Also shows contiguous (non-paged) baselines for reference:
  C. hd256 2CTA non-paged     (varlen scheduler)

Usage:
    python benchmarks/bench_sm100_hd256_paged_tma.py
    python benchmarks/bench_sm100_hd256_paged_tma.py --seqlen 2048,4096,8192
    python benchmarks/bench_sm100_hd256_paged_tma.py --batch 4 --nheads 8
    python benchmarks/bench_sm100_hd256_paged_tma.py --correctness-only
"""
import argparse
import sys

import torch

from flash_attn.cute.interface import (
    _flash_attn_fwd,
    flash_attn_fwd_sm100_hd256_2cta,
)


# ── Constants ─────────────────────────────────────────────────────────────

HEAD_DIM = 256
PAGE_SIZE = 128  # must equal tile_n for TMA paged KV


# ── Helpers ───────────────────────────────────────────────────────────────


def csv_ints(s):
    return [int(x.strip()) for x in s.split(",")]


def auto_batch(seqlen, batch_arg, total_tokens=32768):
    return batch_arg if batch_arg > 0 else max(1, total_tokens // seqlen)


def fwd_flops(batch, nheads, seqlen, hdim):
    return batch * nheads * 2 * seqlen * seqlen * (hdim + hdim)


def bench(fn, warmup=20, rep=50):
    """Benchmark a CUDA function with trimmed-mean timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        torch.cuda.synchronize()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    trim = max(1, rep // 10)
    return sum(times[trim:-trim]) / len(times[trim:-trim])


def check_sm100():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.", file=sys.stderr)
        sys.exit(1)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if cap[0] not in (10, 11):
        print(
            f"WARNING: This benchmark targets SM100/SM110 (Blackwell). "
            f"Current GPU: {name} (SM{cap[0]}{cap[1]}).",
            file=sys.stderr,
        )
    else:
        print(f"GPU: {name} (SM{cap[0]}{cap[1]})")


def make_paged_kv(k_cont, v_cont, B, S, H, d, dtype, device):
    """Build paged KV tensors from contiguous KV for correctness checking."""
    num_pages_per_seq = (S + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = B * num_pages_per_seq
    k_paged = torch.zeros(total_pages, PAGE_SIZE, H, d, dtype=dtype, device=device)
    v_paged = torch.zeros(total_pages, PAGE_SIZE, H, d, dtype=dtype, device=device)
    for b in range(B):
        for s in range(S):
            pi = b * num_pages_per_seq + s // PAGE_SIZE
            po = s % PAGE_SIZE
            k_paged[pi, po] = k_cont[b * S + s]
            v_paged[pi, po] = v_cont[b * S + s]
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        B, num_pages_per_seq
    )
    return k_paged, v_paged, page_table


def make_random_paged_kv(B, S, H, d, dtype, device):
    """Create random paged KV tensors (for benchmarking, no correctness link)."""
    num_pages_per_seq = (S + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = B * num_pages_per_seq
    k_paged = torch.randn(total_pages, PAGE_SIZE, H, d, dtype=dtype, device=device)
    v_paged = torch.randn(total_pages, PAGE_SIZE, H, d, dtype=dtype, device=device)
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        B, num_pages_per_seq
    )
    return k_paged, v_paged, page_table


# ── Runners ───────────────────────────────────────────────────────────────


def call_hd256_2cta_nonpaged(q, k, v, cu_q, cu_k, S):
    """hd256 2CTA kernel, non-paged varlen."""
    return flash_attn_fwd_sm100_hd256_2cta(
        q, k, v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        seqused_q=None, seqused_k=None,
        max_seqlen_q=S, max_seqlen_k=S,
    )


def call_hd256_2cta_paged(q, k_paged, v_paged, cu_q, S, page_table):
    """hd256 2CTA kernel, paged TMA (our new implementation)."""
    return flash_attn_fwd_sm100_hd256_2cta(
        q, k_paged, v_paged,
        cu_seqlens_q=cu_q, cu_seqlens_k=None,
        seqused_q=None, seqused_k=None,
        max_seqlen_q=S, max_seqlen_k=S,
        page_table=page_table,
    )


def call_generic_sm100_paged(q_batch, k_paged, v_paged, S, page_table, seqused_k):
    """Generic SM100 kernel, paged TMA (the fallback path)."""
    return _flash_attn_fwd(
        q_batch, k_paged, v_paged,
        seqused_k=seqused_k,
        page_table=page_table,
        causal=False,
        return_lse=True,
    )


# ── Correctness ───────────────────────────────────────────────────────────


def run_correctness(args):
    """Verify all three paths produce matching results."""
    dtype = torch.bfloat16
    device = "cuda"
    H = args.nheads
    d = HEAD_DIM

    print("\nCorrectness: all paths vs hd256 2CTA non-paged reference")
    print("-" * 70)

    all_pass = True
    for S in args.seqlen:
        B = auto_batch(S, args.batch)
        torch.manual_seed(42)

        q = torch.randn(B * S, H, d, dtype=dtype, device=device)
        k_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)
        v_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
        cu_k = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S

        k_paged, v_paged, page_table = make_paged_kv(
            k_cont, v_cont, B, S, H, d, dtype, device
        )
        seqused_k = torch.full((B,), S, dtype=torch.int32, device=device)

        out_ref, _ = call_hd256_2cta_nonpaged(q, k_cont, v_cont, cu_q, cu_k, S)
        out_2cta, _ = call_hd256_2cta_paged(q, k_paged, v_paged, cu_q, S, page_table)

        q_batch = q.view(B, S, H, d)
        out_generic, _ = call_generic_sm100_paged(
            q_batch, k_paged, v_paged, S, page_table, seqused_k
        )
        out_generic = out_generic.view(B * S, H, d)

        diff_2cta = (out_2cta - out_ref).abs().max().item()
        diff_generic = (out_generic - out_ref).abs().max().item()

        ok_2cta = diff_2cta < 0.01
        ok_generic = diff_generic < 0.01
        ok = ok_2cta and ok_generic
        all_pass = all_pass and ok

        label = f"B={B} S={S} H={H}"
        tag_2cta = "EXACT" if torch.equal(out_2cta, out_ref) else f"diff={diff_2cta:.4f}"
        tag_gen = "EXACT" if torch.equal(out_generic, out_ref) else f"diff={diff_generic:.4f}"
        print(f"  {label:<23}  2CTA paged: {tag_2cta:<12}  Generic paged: {tag_gen:<12}  [{'OK' if ok else 'FAIL'}]")

    print()
    if all_pass:
        print("All correctness checks PASSED.")
    else:
        print("Some correctness checks FAILED!", file=sys.stderr)
    return all_pass


# ── Benchmark ─────────────────────────────────────────────────────────────


def run_benchmark(args):
    """Benchmark all three paths with clear comparison."""
    dtype = torch.bfloat16
    device = "cuda"
    H = args.nheads
    d = HEAD_DIM

    # ── Section 1: Paged comparison (Generic vs 2CTA) ─────────────────────
    print("\n")
    print("=" * 80)
    print("  PAGED KV-CACHE COMPARISON: Generic SM100 (baseline) vs hd256 2CTA (new)")
    print("  Both use TMA for KV loads. Difference is the attention kernel itself.")
    print("=" * 80)
    print()
    print(f"  {'Config':<23} {'Generic SM100':>14} {'hd256 2CTA':>14} {'Delta':>10} {'Speedup':>9}")
    print(f"  {'':23} {'ms':>8} {'TF/s':>5} {'ms':>8} {'TF/s':>5} {'ms':>10} {'':>9}")
    print(f"  {'-'*75}")

    results = []
    for S in args.seqlen:
        B = auto_batch(S, args.batch)
        flops = fwd_flops(B, H, S, d)
        torch.manual_seed(42)

        q = torch.randn(B * S, H, d, dtype=dtype, device=device)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
        k_paged, v_paged, page_table = make_random_paged_kv(B, S, H, d, dtype, device)
        seqused_k = torch.full((B,), S, dtype=torch.int32, device=device)
        q_batch = q.view(B, S, H, d)

        # Generic SM100 paged TMA (baseline)
        fn_gen = lambda: call_generic_sm100_paged(
            q_batch, k_paged, v_paged, S, page_table, seqused_k
        )
        fn_gen()
        ms_gen = bench(fn_gen)
        tf_gen = flops / (ms_gen * 1e-3) / 1e12

        # hd256 2CTA paged TMA (new)
        fn_2cta = lambda: call_hd256_2cta_paged(q, k_paged, v_paged, cu_q, S, page_table)
        fn_2cta()
        ms_2cta = bench(fn_2cta)
        tf_2cta = flops / (ms_2cta * 1e-3) / 1e12

        delta_ms = ms_2cta - ms_gen
        if ms_gen > 0:
            speedup = ms_gen / ms_2cta
            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
            speedup = 0

        sign = "+" if delta_ms > 0 else ""
        label = f"B={B} S={S} H={H}"
        print(
            f"  {label:<23} {ms_gen:>8.3f} {tf_gen:>5.0f} "
            f"{ms_2cta:>8.3f} {tf_2cta:>5.0f} "
            f"{sign}{delta_ms:>9.3f} {speedup_str:>9}"
        )
        results.append((B, S, ms_gen, ms_2cta, speedup))

    print()
    # Summary line
    wins = sum(1 for _, _, mg, m2, _ in results if m2 < mg)
    losses = sum(1 for _, _, mg, m2, _ in results if m2 > mg)
    print(f"  2CTA paged faster in {wins}/{len(results)} configs, "
          f"slower in {losses}/{len(results)} configs")

    # ── Section 2: Non-paged baseline ─────────────────────────────────────
    print()
    print("=" * 80)
    print("  REFERENCE: hd256 2CTA non-paged (varlen) baseline")
    print("=" * 80)
    print()
    print(f"  {'Config':<23} {'non-paged':>14} {'paged TMA':>14} {'Paged overhead':>16}")
    print(f"  {'':23} {'ms':>8} {'TF/s':>5} {'ms':>8} {'TF/s':>5} {'':>16}")
    print(f"  {'-'*68}")

    for i, S in enumerate(args.seqlen):
        B = auto_batch(S, args.batch)
        flops = fwd_flops(B, H, S, d)
        torch.manual_seed(42)

        q = torch.randn(B * S, H, d, dtype=dtype, device=device)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
        cu_k = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
        k_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)
        v_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)

        fn_np = lambda: call_hd256_2cta_nonpaged(q, k_cont, v_cont, cu_q, cu_k, S)
        fn_np()
        ms_np = bench(fn_np)
        tf_np = flops / (ms_np * 1e-3) / 1e12

        _, _, _, ms_2cta_paged, _ = results[i]
        tf_2cta_paged = flops / (ms_2cta_paged * 1e-3) / 1e12
        overhead = (ms_2cta_paged / ms_np - 1) * 100

        label = f"B={B} S={S} H={H}"
        print(
            f"  {label:<23} {ms_np:>8.3f} {tf_np:>5.0f} "
            f"{ms_2cta_paged:>8.3f} {tf_2cta_paged:>5.0f} "
            f"{overhead:>+15.1f}%"
        )

    print()
    print("  NOTE: non-paged uses varlen scheduler (cu_seqlens_k), paged uses")
    print("  batch scheduler (no cu_seqlens_k). Negative overhead = paged is faster")
    print("  due to simpler scheduler path, not paged KV itself.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hd256 paged TMA: specialized 2CTA vs generic SM100 fallback"
    )
    parser.add_argument(
        "--seqlen",
        type=csv_ints,
        default=[1024, 2048, 4096, 8192],
        help="Comma-separated sequence lengths (default: 1024,2048,4096,8192)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch size (0 = auto-scale to ~32k tokens, default: 0)",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=16,
        help="Number of attention heads (default: 16)",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run correctness check only, no timing",
    )
    args = parser.parse_args()

    check_sm100()

    if args.correctness_only:
        ok = run_correctness(args)
        sys.exit(0 if ok else 1)

    run_correctness(args)
    run_benchmark(args)


if __name__ == "__main__":
    main()
