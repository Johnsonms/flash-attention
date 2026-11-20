"""GQA / packed-head and 1CTA numerics for the mask gate. usage: numcheck3187b.py out.pt  (set FA_DISABLE_2CTA=1 outside for the 1CTA set)"""
import sys, torch
from flash_attn.cute.interface import flash_attn_func
out = {}
for (bs, sq, sk, h, hk, hd, causal) in [(2, 8192, 8192, 16, 4, 128, True), (2, 4096, 4096, 16, 4, 128, False), (2, 3000, 4096, 16, 4, 128, True),
                                        (2, 4096, 3000, 16, 2, 128, True), (2, 2000, 2000, 8, 1, 128, True), (2, 4096, 4096, 16, 8, 64, True),
                                        (2, 4096, 4096, 16, 4, 96, True)] + ([] if __import__("os").environ.get("FA_DISABLE_2CTA") else [(1, 4096, 4096, 8, 2, 256, True)]):
    torch.manual_seed(0)
    q = torch.randn(bs, sq, h, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(bs, sk, hk, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(bs, sk, hk, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(bs, sq, h, hd, device="cuda", dtype=torch.bfloat16)
    y = flash_attn_func(q, k, v, causal=causal); y = y[0] if isinstance(y, tuple) else y
    y.backward(g)
    out[(bs, sq, sk, h, hk, hd, causal)] = (y.detach().float().cpu(), q.grad.float().cpu(), k.grad.float().cpu(), v.grad.float().cpu())
torch.save(out, sys.argv[1]); print("saved", sys.argv[1])
