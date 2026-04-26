"""Correctness check: FA4 hd256 bwd vs PyTorch reference attention.

Computes forward+backward through both, compares dQ/dK/dV gradients.
Tolerances are bf16-typical (rtol=0.05, atol=0.05 for absolute small values).
"""
import os, torch
import torch.nn.functional as F
torch.cuda.set_device(0)
torch.manual_seed(0)

from flash_attn.cute.interface import flash_attn_func

CASES = [
    # (B, S, H, HKV, D, causal)
    (1, 256, 4, 4, 256, False),
    (1, 256, 4, 4, 256, True),
    (2, 1024, 8, 8, 256, False),
    (2, 1024, 8, 8, 256, True),
    (1, 4096, 4, 4, 256, False),
    (1, 4096, 4, 4, 256, True),
    (1, 1024, 8, 2, 256, False),  # GQA
    (1, 1024, 8, 2, 256, True),   # GQA causal
]

def ref_attention(q, k, v, causal):
    """Reference using torch SDPA in fp32."""
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    # Expand kv heads to match q heads (GQA broadcast)
    if k.shape[-2] != q.shape[-2]:
        rep = q.shape[-2] // k.shape[-2]
        kf = kf.repeat_interleave(rep, dim=-2)
        vf = vf.repeat_interleave(rep, dim=-2)
    # (B, S, H, D) -> (B, H, S, D)
    qf = qf.transpose(1, 2)
    kf = kf.transpose(1, 2)
    vf = vf.transpose(1, 2)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)
    return out.transpose(1, 2).to(q.dtype)

def run_case(B, S, H, HKV, D, causal):
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

    # FA4 path
    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    o1 = flash_attn_func(q1, k1, v1, causal=causal)
    if isinstance(o1, tuple):
        o1 = o1[0]
    o1.backward(g)
    fa_dq, fa_dk, fa_dv = q1.grad, k1.grad, v1.grad

    # Reference path (fp32 internally)
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    o2 = ref_attention(q2, k2, v2, causal)
    o2.backward(g)
    ref_dq, ref_dk, ref_dv = q2.grad, k2.grad, v2.grad

    def cmp(a, b, name):
        diff = (a.float() - b.float()).abs()
        rel = diff / (b.float().abs() + 1e-6)
        return f"{name}: max_abs={diff.max().item():.4e} max_rel={rel.max().item():.4e}"

    print(f"  {cmp(o1, o2, 'out')}")
    print(f"  {cmp(fa_dq, ref_dq, 'dq')}")
    print(f"  {cmp(fa_dk, ref_dk, 'dk')}")
    print(f"  {cmp(fa_dv, ref_dv, 'dv')}")

    # bf16 has ~3 decimal digits; tolerate up to ~5% relative
    ok = (
        torch.allclose(o1.float(), o2.float(), rtol=0.02, atol=0.02)
        and torch.allclose(fa_dq.float(), ref_dq.float(), rtol=0.05, atol=0.05)
        and torch.allclose(fa_dk.float(), ref_dk.float(), rtol=0.05, atol=0.05)
        and torch.allclose(fa_dv.float(), ref_dv.float(), rtol=0.05, atol=0.05)
    )
    return ok

failed = []
for case in CASES:
    print(f"\n=== B={case[0]} S={case[1]} H={case[2]} HKV={case[3]} D={case[4]} causal={case[5]} ===")
    ok = run_case(*case)
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append(case)

print()
if failed:
    print(f"{len(failed)} case(s) failed:")
    for c in failed:
        print(f"  {c}")
    raise SystemExit(1)
print("All cases passed.")
