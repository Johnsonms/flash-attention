#!/usr/bin/env python
"""SM100 hd256 2CTA forward benchmark covering dense / paged / paged+seqused_k.

Representative MLA-style decode shape (DeepSeek-V3 post-absorption):
  nheads_q = 128, head_dim = 256, bfloat16.

Three paths compared:
  A. Dense (non-varlen) non-causal — exercises persistent scheduling + cluster
     tile scheduler (PR 3), since is_persistent gates on cu_seqlens_q is None
     and not causal.
  B. Paged — page_size=128 (tile_n) varlen path, non-persistent due to
     cu_seqlens_q.
  C. Paged + seqused_k — MLA decode pattern with per-batch variable KV lengths.

Usage:
    python benchmarks/bench_sm100_hd256_paged.py
    python benchmarks/bench_sm100_hd256_paged.py --batch 1,8,32
    python benchmarks/bench_sm100_hd256_paged.py --seqlen-kv 1024,4096,16384
"""
import argparse
import itertools
import time

import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func


def flops_fwd(batch, seqlen_q, seqlen_kv, nheads, head_dim):
    # QK: 2 * b * h * s_q * s_kv * d; PV: 2 * b * h * s_q * s_kv * d.
    return 4 * batch * nheads * seqlen_q * seqlen_kv * head_dim


def time_fn(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_dense(batch, seqlen_q, seqlen_kv, nheads, head_dim, dtype, device):
    q = torch.randn(batch, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_kv, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_kv, nheads, head_dim, device=device, dtype=dtype)
    lat = time_fn(lambda: flash_attn_func(q, k, v, causal=False))
    return lat


def bench_paged(batch, seqlen_q, seqlen_kv, nheads, head_dim, dtype, device, with_seqused_k):
    page_size = 128
    assert seqlen_kv % page_size == 0
    num_pages_per_seq = seqlen_kv // page_size
    total_pages = batch * num_pages_per_seq

    q = torch.randn(batch * seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_paged = torch.randn(total_pages, page_size, nheads, head_dim, device=device, dtype=dtype)
    v_paged = torch.randn(total_pages, page_size, nheads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen_q
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch, num_pages_per_seq
    )

    if with_seqused_k:
        # MLA decode pattern: varied per-batch lengths, clipped to a fraction of full.
        seqused_k = torch.tensor(
            [max(page_size, int(seqlen_kv * (0.5 + 0.5 * (i + 1) / batch))) for i in range(batch)],
            dtype=torch.int32, device=device,
        )
    else:
        seqused_k = None

    def run():
        flash_attn_varlen_func(
            q, k_paged, v_paged,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_kv,
            page_table=page_table, seqused_k=seqused_k,
        )

    lat = time_fn(run)
    return lat, seqused_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="1,8,32")
    parser.add_argument("--seqlen-q", type=str, default="128")
    parser.add_argument("--seqlen-kv", type=str, default="1024,4096,16384")
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    batches = [int(x) for x in args.batch.split(",")]
    seqlens_q = [int(x) for x in args.seqlen_q.split(",")]
    seqlens_kv = [int(x) for x in args.seqlen_kv.split(",")]
    dtype = getattr(torch, args.dtype)
    device = "cuda"
    torch.random.manual_seed(0)

    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability()
    assert cap[0] == 10, f"This benchmark targets SM100 (B200), got capability {cap}"

    print(
        f"\nhd256 2CTA benchmark — GPU={torch.cuda.get_device_name(0)}, "
        f"dtype={args.dtype}, nheads={args.nheads}, d={args.head_dim}\n"
    )
    header = (
        f"{'batch':>5} {'s_q':>5} {'s_kv':>6} {'dense(ms)':>10} {'paged(ms)':>10} "
        f"{'paged+sk(ms)':>13} {'dense TFLOP/s':>14} {'paged TFLOP/s':>14}"
    )
    print(header)
    print("-" * len(header))
    for batch, s_q, s_kv in itertools.product(batches, seqlens_q, seqlens_kv):
        lat_dense = bench_dense(batch, s_q, s_kv, args.nheads, args.head_dim, dtype, device)
        lat_paged, _ = bench_paged(batch, s_q, s_kv, args.nheads, args.head_dim, dtype, device, False)
        lat_paged_sk, _ = bench_paged(batch, s_q, s_kv, args.nheads, args.head_dim, dtype, device, True)
        fl = flops_fwd(batch, s_q, s_kv, args.nheads, args.head_dim)
        tflops_dense = fl / lat_dense / 1e12
        tflops_paged = fl / lat_paged / 1e12
        print(
            f"{batch:>5} {s_q:>5} {s_kv:>6} {lat_dense*1e3:>10.3f} {lat_paged*1e3:>10.3f} "
            f"{lat_paged_sk*1e3:>13.3f} {tflops_dense:>14.2f} {tflops_paged:>14.2f}"
        )


if __name__ == "__main__":
    main()
