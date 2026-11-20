"""dq/dk/dv for a few causal + non-causal shapes, saved to a file for A/B between two trees. usage: numcheck3187.py out.pt"""
import sys, torch
from flash_attn.cute.interface import flash_attn_func
out = {}
for (bs, sq, sk, h, hd, causal) in [(4, 8192, 8192, 16, 128, True), (4, 8192, 8192, 16, 128, False), (2, 4096, 4096, 16, 128, True),
                                    (2, 3000, 4096, 16, 128, True), (2, 4096, 3000, 16, 128, True), (2, 1000, 1000, 8, 128, True),
                                    (2, 2048, 2048, 16, 128, True)]:
    torch.manual_seed(0)
    q = torch.randn(bs, sq, h, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(bs, sk, h, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(bs, sk, h, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(bs, sq, h, hd, device="cuda", dtype=torch.bfloat16)
    y = flash_attn_func(q, k, v, causal=causal); y = y[0] if isinstance(y, tuple) else y
    y.backward(g)
    out[(bs, sq, sk, h, hd, causal)] = (y.detach().float().cpu(), q.grad.float().cpu(), k.grad.float().cpu(), v.grad.float().cpu())
torch.save(out, sys.argv[1]); print("saved", sys.argv[1])
