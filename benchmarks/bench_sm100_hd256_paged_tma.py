#!/usr/bin/env python
"""SM100 Blackwell head_dim=256 paged KV TMA benchmark.

Targets DeepSeek-V2/V3/R1 style MLA (Multi-head Latent Attention) after
KV absorption: nheads=128, head_dim_k=512, head_dim_v=256. Since d_k and
d_v differ, we benchmark with d=256 (the value-head dimension that drives
the 2CTA kernel specialization).

Compares kernel paths:
  A. Generic SM100 paged TMA  (current default fallback)
  B. hd256 2CTA paged TMA     (specialized implementation)

Usage:
    # DeepSeek-V3 decode (default): MHA 128 heads, d=256
    python benchmarks/bench_sm100_hd256_paged_tma.py

    # Llama-style GQA with d=256
    python benchmarks/bench_sm100_hd256_paged_tma.py --nheads-q 32 --nheads-k 8

    # Custom sweep
    python benchmarks/bench_sm100_hd256_paged_tma.py --batch 1,8,64,128 --seqlen 1024,4096,16384

    # Prefill benchmark
    python benchmarks/bench_sm100_hd256_paged_tma.py --mode prefill

    # Correctness only
    python benchmarks/bench_sm100_hd256_paged_tma.py --correctness-only
"""
import argparse
import math
import sys

import torch

from flash_attn.cute.interface import (
    _flash_attn_fwd,
    flash_attn_fwd_sm100_hd256_2cta,
)


# ── Constants ─────────────────────────────────────────────────────────────

HEAD_DIM = 256
PAGE_SIZE = 128  # must equal tile_n for TMA paged KV

# DeepSeek-V2/V3/R1 after MLA absorption: 128 heads, MHA (nheads_q == nheads_k)
DEFAULT_NHEADS_Q = 128
DEFAULT_NHEADS_K = 128

# Typical production decode batch sizes
DEFAULT_DECODE_BATCHES = [1, 8, 32, 64, 128]
# Typical KV cache lengths for long-context LLM serving
DEFAULT_DECODE_SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768]
# Prefill sequence lengths
DEFAULT_PREFILL_SEQLENS = [1024, 2048, 4096, 8192]


# ── Helpers ───────────────────────────────────────────────────────────────


def csv_ints(s):
    return [int(x.strip()) for x in s.split(",")]


def auto_batch(seqlen, batch_arg, total_tokens=32768):
    return batch_arg if batch_arg > 0 else max(1, total_tokens // seqlen)


def fwd_tflops(batch, nheads_q, seqlen_q, seqlen_k, hdim, ms):
    flops = 4 * batch * seqlen_q * seqlen_k * nheads_q * hdim
    return flops / (ms * 1e-3) / 1e12


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


def call_hd256_2cta(q, k, v, cu_q, cu_k, max_sq, max_sk,
                    page_table=None, seqused_k=None):
    """hd256 2CTA kernel (paged or non-paged)."""
    return flash_attn_fwd_sm100_hd256_2cta(
        q, k, v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        seqused_q=None, seqused_k=seqused_k,
        max_seqlen_q=max_sq, max_seqlen_k=max_sk,
        page_table=page_table,
    )


def call_generic_sm100(q, k, v, max_sk, page_table=None, seqused_k=None):
    """Generic SM100 kernel (paged or non-paged)."""
    return _flash_attn_fwd(
        q, k, v,
        seqused_k=seqused_k,
        max_seqlen_k=max_sk,
        page_table=page_table,
        causal=False,
        return_lse=True,
    )


# ── Correctness ───────────────────────────────────────────────────────────


def run_correctness(args):
    """Verify paged paths match non-paged reference and seqused_k vs manual attention."""
    dtype = torch.bfloat16
    device = "cuda"
    H = 8  # MHA for correctness (avoids pack_gqa layout differences)
    d = HEAD_DIM

    print("\nCorrectness checks (MHA, H=8)")
    print("-" * 70)

    all_pass = True

    # Test 1: Prefill — paged vs non-paged varlen (exact match expected)
    for S in [256, 512]:
        B = 2
        torch.manual_seed(42)

        q = torch.randn(B * S, H, d, dtype=dtype, device=device)
        k_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)
        v_cont = torch.randn(B * S, H, d, dtype=dtype, device=device)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
        cu_k = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S

        k_paged, v_paged, page_table = make_paged_kv(
            k_cont, v_cont, B, S, H, d, dtype, device
        )

        out_ref, _ = call_hd256_2cta(q, k_cont, v_cont, cu_q, cu_k, S, S)
        out_2cta, _ = call_hd256_2cta(
            q, k_paged, v_paged, cu_q, None, S, S, page_table=page_table
        )

        diff = (out_2cta - out_ref).abs().max().item()
        ok = diff < 0.01
        all_pass = all_pass and ok
        tag = "EXACT" if torch.equal(out_2cta, out_ref) else f"diff={diff:.4f}"
        print(f"  prefill  B={B} S={S:<5} H={H}  paged vs non-paged: {tag:<12} [{'OK' if ok else 'FAIL'}]")

    # Test 2: Decode — paged + seqused_k vs manual reference
    for B in [4]:
        for S in [256, 512, 1024]:
            torch.manual_seed(42)
            seqlen_q = 1
            q = torch.randn(B, seqlen_q, H, d, dtype=dtype, device=device)
            k = torch.randn(B, S, H, d, dtype=dtype, device=device)
            v = torch.randn(B, S, H, d, dtype=dtype, device=device)

            k_paged, v_paged, page_table = make_paged_kv(
                k.reshape(B * S, H, d), v.reshape(B * S, H, d),
                B, S, H, d, dtype, device,
            )
            seqused_k = torch.tensor(
                [S * 3 // 4, S, S // 2, S], dtype=torch.int32, device=device
            )[:B]

            out, _ = call_hd256_2cta(
                q, k_paged, v_paged, None, None, seqlen_q, S,
                page_table=page_table, seqused_k=seqused_k,
            )

            scale = 1.0 / math.sqrt(d)
            max_diff = 0.0
            for b in range(B):
                sk = seqused_k[b].item()
                scores = torch.einsum('qhd,khd->hqk', q[b].float(), k[b, :sk].float()) * scale
                attn = torch.softmax(scores, dim=-1)
                ref = torch.einsum('hqk,khd->qhd', attn, v[b, :sk].float()).to(dtype)
                max_diff = max(max_diff, (out[b, 0] - ref[0]).abs().max().item())

            ok = max_diff < 0.01
            all_pass = all_pass and ok
            tag = f"max_diff={max_diff:.4f}"
            print(f"  decode   B={B} Sk={S:<5} H={H}  paged+seqused_k:  {tag:<12} [{'OK' if ok else 'FAIL'}]")

    print()
    if all_pass:
        print("All correctness checks PASSED.")
    else:
        print("Some correctness checks FAILED!", file=sys.stderr)
    return all_pass


# ── Benchmark: Inference (seqlen_q=1) ────────────────────────────────────


def run_inference_benchmark(args):
    """Benchmark decode inference: seqlen_q=1, paged KV, seqused_k.

    Models this benchmark targets:
      - DeepSeek-V2/V3/R1 (MLA absorbed): nheads=128, d_v=256, MHA
      - Any model with d=256 and paged KV cache
    """
    dtype = torch.bfloat16
    device = "cuda"
    nheads_q = args.nheads_q
    nheads_k = args.nheads_k
    d = HEAD_DIM
    seqlen_q = 1

    batches = args.batch if args.batch else DEFAULT_DECODE_BATCHES
    seqlens = args.seqlen

    gqa_str = "MHA" if nheads_q == nheads_k else f"GQA {nheads_q}/{nheads_k}"

    print()
    print("=" * 95)
    print(f"  DECODE: Sq=1, {gqa_str}, H={nheads_q}, d={d}, paged KV (page_size={PAGE_SIZE})")
    print("=" * 95)

    for section_title, use_seqused_k in [
        (f"Paged KV + seqused_k  (production decode)", True),
        (f"Paged KV  (uniform seqlen, no seqused_k)", False),
        (f"Non-paged batch mode  (contiguous KV reference)", None),
    ]:
        print(f"\n  {section_title}")
        print(f"  {'-'*88}")
        print(f"  {'Config':<28} {'2CTA ms':>9} {'TF/s':>7} {'Generic ms':>11} {'TF/s':>7} {'Speedup':>9}")
        print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*11} {'-'*7} {'-'*9}")

        for B in batches:
            for S in seqlens:
                # Estimate memory: KV tensors dominate (~2 * B * S * nheads_k * d * 2 bytes)
                kv_bytes = 2 * B * S * nheads_k * d * 2  # 2 tensors, bf16
                free_mem = torch.cuda.mem_get_info()[0]
                if kv_bytes > free_mem * 0.7:
                    label = f"B={B:<4} Sk={S:<6}"
                    print(f"  {label:<28}  (skipped — {kv_bytes / 2**30:.1f} GiB KV > available)")
                    continue

                torch.manual_seed(42)
                q = torch.randn(B, seqlen_q, nheads_q, d, dtype=dtype, device=device)

                if use_seqused_k is None:
                    # Non-paged contiguous KV
                    k = torch.randn(B, S, nheads_k, d, dtype=dtype, device=device)
                    v = torch.randn(B, S, nheads_k, d, dtype=dtype, device=device)
                    seqused_k = None

                    fn_2cta = lambda: call_hd256_2cta(
                        q, k, v, None, None, seqlen_q, S
                    )
                    fn_gen = lambda: call_generic_sm100(q, k, v, S)
                else:
                    k_paged, v_paged, page_table = make_random_paged_kv(
                        B, S, nheads_k, d, dtype, device
                    )
                    seqused_k = (
                        torch.full((B,), S * 3 // 4, dtype=torch.int32, device=device)
                        if use_seqused_k else None
                    )
                    fn_2cta = lambda: call_hd256_2cta(
                        q, k_paged, v_paged, None, None, seqlen_q, S,
                        page_table=page_table, seqused_k=seqused_k,
                    )
                    fn_gen = lambda: call_generic_sm100(
                        q, k_paged, v_paged, S,
                        page_table=page_table, seqused_k=seqused_k,
                    )

                fn_2cta()
                ms_2cta = bench(fn_2cta)
                tf_2cta = fwd_tflops(B, nheads_q, seqlen_q, S, d, ms_2cta)

                fn_gen()
                ms_gen = bench(fn_gen)
                tf_gen = fwd_tflops(B, nheads_q, seqlen_q, S, d, ms_gen)

                speedup = ms_gen / ms_2cta
                label = f"B={B:<4} Sk={S:<6}"
                print(
                    f"  {label:<28} {ms_2cta:>9.3f} {tf_2cta:>7.1f}"
                    f" {ms_gen:>11.3f} {tf_gen:>7.1f}"
                    f" {speedup:>8.2f}x"
                )

                # Free memory for next config
                del q
                if use_seqused_k is None:
                    del k, v
                else:
                    del k_paged, v_paged, page_table
                    if seqused_k is not None:
                        del seqused_k
                torch.cuda.empty_cache()


# ── Benchmark: Prefill (seqlen_q=seqlen_k) ──────────────────────────────


def run_prefill_benchmark(args):
    """Benchmark prefill: seqlen_q=seqlen_k, paged and non-paged."""
    dtype = torch.bfloat16
    device = "cuda"
    nheads_q = args.nheads_q
    nheads_k = args.nheads_k
    d = HEAD_DIM

    gqa_str = "MHA" if nheads_q == nheads_k else f"GQA {nheads_q}/{nheads_k}"

    print()
    print("=" * 95)
    print(f"  PREFILL: Sq=Sk, {gqa_str}, H={nheads_q}, d={d}")
    print("=" * 95)

    for section_title, paged in [
        ("Paged KV  (paged prefill)", True),
        ("Non-paged  (contiguous prefill)", False),
    ]:
        print(f"\n  {section_title}")
        print(f"  {'-'*88}")
        print(f"  {'Config':<28} {'2CTA ms':>9} {'TF/s':>7} {'Generic ms':>11} {'TF/s':>7} {'Speedup':>9}")
        print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*11} {'-'*7} {'-'*9}")

        for S in args.seqlen:
            B = auto_batch(S, args.batch[0] if args.batch else 0)
            torch.manual_seed(42)

            if paged:
                q = torch.randn(B * S, nheads_q, d, dtype=dtype, device=device)
                cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
                k_paged, v_paged, page_table = make_random_paged_kv(
                    B, S, nheads_k, d, dtype, device
                )
                seqused_k = torch.full((B,), S, dtype=torch.int32, device=device)
                q_batch = q.view(B, S, nheads_q, d)

                fn_2cta = lambda: call_hd256_2cta(
                    q, k_paged, v_paged, cu_q, None, S, S, page_table=page_table,
                )
                fn_gen = lambda: call_generic_sm100(
                    q_batch, k_paged, v_paged, S,
                    page_table=page_table, seqused_k=seqused_k,
                )
            else:
                q = torch.randn(B * S, nheads_q, d, dtype=dtype, device=device)
                k = torch.randn(B * S, nheads_k, d, dtype=dtype, device=device)
                v = torch.randn(B * S, nheads_k, d, dtype=dtype, device=device)
                cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
                cu_k = torch.arange(0, B + 1, dtype=torch.int32, device=device) * S
                q_batch = q.view(B, S, nheads_q, d)
                k_batch = k.view(B, S, nheads_k, d)
                v_batch = v.view(B, S, nheads_k, d)

                fn_2cta = lambda: call_hd256_2cta(
                    q, k, v, cu_q, cu_k, S, S,
                )
                fn_gen = lambda: call_generic_sm100(q_batch, k_batch, v_batch, S)

            fn_2cta()
            ms_2cta = bench(fn_2cta)
            tf_2cta = fwd_tflops(B, nheads_q, S, S, d, ms_2cta)

            fn_gen()
            ms_gen = bench(fn_gen)
            tf_gen = fwd_tflops(B, nheads_q, S, S, d, ms_gen)

            speedup = ms_gen / ms_2cta
            label = f"B={B:<3} S={S:<5}"
            print(
                f"  {label:<28} {ms_2cta:>9.3f} {tf_2cta:>7.1f}"
                f" {ms_gen:>11.3f} {tf_gen:>7.1f}"
                f" {speedup:>8.2f}x"
            )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hd256 paged TMA: 2CTA vs generic SM100. "
        "Default config matches DeepSeek-V2/V3/R1 MLA (128 heads, d=256)."
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "prefill", "all"],
        default="inference",
        help="Benchmark mode (default: inference)",
    )
    parser.add_argument(
        "--seqlen",
        type=csv_ints,
        default=None,
        help="Comma-separated KV sequence lengths (default: mode-dependent)",
    )
    parser.add_argument(
        "--batch",
        type=csv_ints,
        default=None,
        help="Comma-separated batch sizes (default: mode-dependent; 0=auto for prefill)",
    )
    parser.add_argument(
        "--nheads-q",
        type=int,
        default=DEFAULT_NHEADS_Q,
        help=f"Number of Q attention heads (default: {DEFAULT_NHEADS_Q})",
    )
    parser.add_argument(
        "--nheads-k",
        type=int,
        default=DEFAULT_NHEADS_K,
        help=f"Number of KV attention heads (default: {DEFAULT_NHEADS_K})",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run correctness check only, no timing",
    )
    args = parser.parse_args()

    # Defaults per mode
    if args.seqlen is None:
        if args.mode == "prefill":
            args.seqlen = DEFAULT_PREFILL_SEQLENS
        else:
            args.seqlen = DEFAULT_DECODE_SEQLENS

    check_sm100()

    if args.correctness_only:
        ok = run_correctness(args)
        sys.exit(0 if ok else 1)

    run_correctness(args)

    if args.mode in ("inference", "all"):
        run_inference_benchmark(args)
    if args.mode in ("prefill", "all"):
        run_prefill_benchmark(args)

    print()


if __name__ == "__main__":
    main()
