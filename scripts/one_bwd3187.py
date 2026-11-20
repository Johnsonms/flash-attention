"""fwd once + N bwd (or N fwd) for ncu. usage: one_bwd3187.py --seqlen 32768 [--causal] [--iters 3] [--mode bwd|fwd] [--nheads 16] [--bs N]"""
import argparse, torch
from flash_attn.cute.interface import flash_attn_func
p = argparse.ArgumentParser(); p.add_argument("--seqlen", type=int, default=8192); p.add_argument("--causal", action="store_true"); p.add_argument("--iters", type=int, default=3); p.add_argument("--headdim", type=int, default=128)
p.add_argument("--mode", default="bwd"); p.add_argument("--nheads", type=int, default=None); p.add_argument("--bs", type=int, default=None)
a = p.parse_args(); hd = a.headdim; nheads = a.nheads or (32 if hd <= 64 else 16); bs = a.bs or 32768 // a.seqlen
torch.manual_seed(0)
q, k, v = [torch.randn(bs, a.seqlen, nheads, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
g = torch.randn_like(q)
if a.mode == "fwd":
    with torch.no_grad():
        for _ in range(a.iters + 2): y = flash_attn_func(q, k, v, causal=a.causal)
else:
    y = flash_attn_func(q, k, v, causal=a.causal); y = y[0] if isinstance(y, tuple) else y
    for _ in range(a.iters):
        for x in (q, k, v): x.grad = None
        y.backward(g, retain_graph=True)
torch.cuda.synchronize(); print("done")
