# Copyright (c) 2025, Johnsonms.

# We recommend locking GPU clocks before running the benchmark to ensure consistent results.
# This can be done using the following commands (2619 MHz is the max clock for B200):
# sudo nvidia-smi -i 0 -pm 1
# sudo nvidia-smi -i 0 --lock-gpu-clocks 2619,2619
# See more here: https://github.com/triton-lang/triton/blob/d9f10ebdc5da53f73eb852fde73d8d7d80b679d1/python/triton/testing.py#L487

# Usage:
#   python benchmark_mla_paged_kv.py                    # run all benchmarks
#   python benchmark_mla_paged_kv.py --mode decode       # decode only (seqlen_q=1)
#   python benchmark_mla_paged_kv.py --mode prefill      # prefill only (seqlen_q=seqlen_k)
#   python benchmark_mla_paged_kv.py --mode splitkv      # splitkv sweep only
#   python benchmark_mla_paged_kv.py --mode paged        # paged KV sweep only

import argparse
import time
import torch

from triton.testing import do_bench

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func


device = "cuda"
dtype = torch.bfloat16
nheads_q = 128
nheads_kv = 1  # MQA-128
headdim = 64
headdim_v = 512
causal = True

torch.manual_seed(0)


def make_paged_kv(k, v, batch_size, seqlen, page_size):
    num_pages_per_seq = (seqlen + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    k_paged = torch.zeros(total_pages, page_size, nheads_kv, headdim, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, headdim_v, device=device, dtype=dtype)
    page_table = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            page_idx = b * num_pages_per_seq + p
            start = p * page_size
            end = min(start + page_size, seqlen)
            k_offset = b * seqlen
            if start < seqlen:
                k_paged[page_idx, :end - start] = k[k_offset + start:k_offset + end]
                v_paged[page_idx, :end - start] = v[k_offset + start:k_offset + end]
            page_table[b, p] = page_idx
    seqused_k = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)
    return k_paged, v_paged, page_table, seqused_k


def compute_metrics(seqlen_q, seqlen_k, batch_size):
    total_seqlen = seqlen_k * batch_size
    total_q = seqlen_q * batch_size
    mem_io = (
        total_seqlen * nheads_kv * (headdim + headdim_v) * 2  # K + V read
        + total_q * nheads_q * headdim * 2  # Q read
        + total_q * nheads_q * headdim_v * 2  # QV read
        + total_q * nheads_q * headdim_v * 2  # O write
    )
    flops = seqlen_q * total_seqlen * nheads_q * (headdim + headdim_v * 2) * 2
    return mem_io, flops


def run_bench(fn, label, mem_io, flops, seqlen_k):
    fn()  # warmup / compile
    time.sleep(1)
    t = do_bench(fn, warmup=1, rep=10)
    print(
        f"  seqlen_k={seqlen_k:>6d}, {label:>30s}: {t * 1e3:8.1f} us, "
        f"{mem_io * 1e-9 / (t * 1e-3):6.0f} GB/s, "
        f"{flops * 1e-12 / (t * 1e-3):6.0f} TFLOPS/s"
    )
    return t


def bench_paged_kv():
    """Paged KV benchmark: decode (seqlen_q=1), sweep page_size."""
    seqlen_q = 1
    batch_size = 128
    page_sizes = [None, 16, 64, 128]

    print(f"\n{'='*90}")
    print(f"Paged KV Decode: batch={batch_size}, seqlen_q={seqlen_q}, causal={causal}")
    print(f"{'='*90}")

    for seqlen_k in [s * 1024 for s in [1, 2, 4, 8, 16, 32, 64]]:
        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k

        try:
            q = torch.randn(total_q, nheads_q, headdim, dtype=dtype, device=device)
            k = torch.randn(total_k, nheads_kv, headdim, dtype=dtype, device=device)
            v = torch.randn(total_k, nheads_kv, headdim_v, dtype=dtype, device=device)
            qv = torch.randn(total_q, nheads_q, headdim_v, dtype=dtype, device=device)
        except torch.OutOfMemoryError:
            continue

        cu_seqlens_q = torch.arange(0, total_q + seqlen_q, seqlen_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.arange(0, total_k + seqlen_k, seqlen_k, dtype=torch.int32, device=device)
        mem_io, flops = compute_metrics(seqlen_q, seqlen_k, batch_size)

        for page_size in page_sizes:
            if page_size is None:
                fn = lambda: flash_attn_varlen_func(
                    q, k, v, qv=qv,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
                    causal=causal,
                )
                label = "non-paged"
            else:
                k_paged, v_paged, page_table, seqused_k = make_paged_kv(k, v, batch_size, seqlen_k, page_size)
                path = "TMA" if page_size == 128 else "cp.async"
                fn = lambda kp=k_paged, vp=v_paged, pt=page_table, su=seqused_k: flash_attn_varlen_func(
                    q, kp, vp, qv=qv,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
                    max_seqlen_q=seqlen_q, max_seqlen_k=None,
                    seqused_k=su, page_table=pt,
                    causal=causal,
                )
                label = f"paged-{page_size} ({path})"

            run_bench(fn, label, mem_io, flops, seqlen_k)

        print(f"  arithmetic intensity: {flops / mem_io:.1f}")
        print()


def bench_splitkv():
    """SplitKV benchmark: prefill and decode with num_splits sweep."""
    print(f"\n{'='*90}")
    print(f"SplitKV: causal={causal}")
    print(f"{'='*90}")

    configs = [
        # (seqlen_q, seqlen_k, batch_size, label)
        (1, 4096, 128, "decode bs=128"),
        (1, 16384, 128, "decode bs=128"),
        (1, 65536, 32, "decode bs=32"),
        (128, 4096, 16, "prefill bs=16"),
        (128, 16384, 8, "prefill bs=8"),
        (256, 8192, 8, "prefill bs=8"),
        (512, 16384, 4, "prefill bs=4"),
        (1024, 16384, 2, "prefill bs=2"),
        (4096, 4096, 4, "prefill bs=4"),
    ]
    num_splits_list = [1, 2, 3, 4, 0]  # 0 = auto heuristic

    for seqlen_q, seqlen_k, batch_size, desc in configs:
        print(f"\n  --- {desc}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k} ---")

        try:
            q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
            k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
            v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim_v, dtype=dtype, device=device)
            qv = torch.randn(batch_size, seqlen_q, nheads_q, headdim_v, dtype=dtype, device=device)
        except torch.OutOfMemoryError:
            print("    OOM, skipping")
            continue

        mem_io, flops = compute_metrics(seqlen_q, seqlen_k, batch_size)

        for num_splits in num_splits_list:
            split_label = f"num_splits={num_splits}" if num_splits > 0 else "num_splits=auto"
            fn = lambda ns=num_splits: flash_attn_func(
                q, k, v, qv=qv,
                causal=causal,
                num_splits=ns,
            )
            run_bench(fn, split_label, mem_io, flops, seqlen_k)


def bench_splitkv_paged():
    """SplitKV + paged KV: decode with page_size and num_splits sweep."""
    seqlen_q = 1
    batch_size = 128
    page_sizes = [16, 64, 128]
    num_splits_list = [1, 3, 0]  # 0 = auto

    print(f"\n{'='*90}")
    print(f"SplitKV + Paged KV Decode: batch={batch_size}, seqlen_q={seqlen_q}, causal={causal}")
    print(f"{'='*90}")

    for seqlen_k in [4096, 16384, 65536]:
        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k

        try:
            q = torch.randn(total_q, nheads_q, headdim, dtype=dtype, device=device)
            k = torch.randn(total_k, nheads_kv, headdim, dtype=dtype, device=device)
            v = torch.randn(total_k, nheads_kv, headdim_v, dtype=dtype, device=device)
            qv = torch.randn(total_q, nheads_q, headdim_v, dtype=dtype, device=device)
        except torch.OutOfMemoryError:
            continue

        cu_seqlens_q = torch.arange(0, total_q + seqlen_q, seqlen_q, dtype=torch.int32, device=device)
        mem_io, flops = compute_metrics(seqlen_q, seqlen_k, batch_size)

        print(f"\n  --- seqlen_k={seqlen_k} ---")

        for page_size in page_sizes:
            k_paged, v_paged, page_table, seqused_k = make_paged_kv(k, v, batch_size, seqlen_k, page_size)
            path = "TMA" if page_size == 128 else "cp.async"

            for num_splits in num_splits_list:
                split_label = f"num_splits={num_splits}" if num_splits > 0 else "num_splits=auto"
                label = f"paged-{page_size} ({path}), {split_label}"
                fn = lambda kp=k_paged, vp=v_paged, pt=page_table, su=seqused_k, ns=num_splits: flash_attn_varlen_func(
                    q, kp, vp, qv=qv,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
                    max_seqlen_q=seqlen_q, max_seqlen_k=None,
                    seqused_k=su, page_table=pt,
                    causal=causal,
                    num_splits=ns,
                )
                run_bench(fn, label, mem_io, flops, seqlen_k)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLA benchmark: paged KV + SplitKV")
    parser.add_argument("--mode", choices=["all", "paged", "splitkv", "decode", "prefill"], default="all")
    args = parser.parse_args()

    print(f"MLA benchmark: nheads_q={nheads_q}, nheads_kv={nheads_kv}, "
          f"headdim={headdim}, headdim_v={headdim_v}, causal={causal}")

    if args.mode in ("all", "paged", "decode"):
        bench_paged_kv()
    if args.mode in ("all", "splitkv", "prefill"):
        bench_splitkv()
    if args.mode in ("all", "splitkv", "decode"):
        bench_splitkv_paged()
