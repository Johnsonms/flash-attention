"""Verify Codex's dq plumbing is bit-identical to HEAD.

Strategy: capture (dq, dk, dv) tensors with current dq (Codex's changes) and
again after `git checkout HEAD -- dqkernel.py`, then compare bit-by-bit.
Driver script handles git swap externally; this script just runs once and
saves a tagged tensor snapshot.

Usage:
  python3 verify_dq_bitidentical.py <out_tag>
"""
import os, sys, torch
torch.cuda.set_device(0)
torch.manual_seed(0)

from flash_attn.cute.interface import flash_attn_func

CASES = [
    (1, 256, 4, 4, 256, False),
    (1, 256, 4, 4, 256, True),
    (2, 1024, 8, 8, 256, False),
    (2, 1024, 8, 8, 256, True),
    (1, 1024, 8, 2, 256, False),
    (1, 1024, 8, 2, 256, True),
]

tag = sys.argv[1]
out_path = f"/tmp/dq_verify_{tag}.pt"
results = {}

for B, S, H, HKV, D, causal in CASES:
    key = f"B{B}_S{S}_H{H}_HKV{HKV}_D{D}_c{int(causal)}"
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    o = flash_attn_func(q, k, v, causal=causal)
    if isinstance(o, tuple):
        o = o[0]
    o.backward(g)
    results[key] = {
        "out": o.detach().clone().cpu(),
        "dq": q.grad.detach().clone().cpu(),
        "dk": k.grad.detach().clone().cpu(),
        "dv": v.grad.detach().clone().cpu(),
    }
    print(f"  {key}: ok")

torch.save(results, out_path)
print(f"saved → {out_path}")
