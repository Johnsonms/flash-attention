"""Per-iteration backward time inside one long burst after an idle gap: shows the clock ramp-down (effective MHz = known cycles / time).
usage: burst_profile.py --seqlen 8192 [--causal] [--idle 5] [--iters 300] [--fwd]   prints: iter  t_since_start_ms  iter_ms  TFLOPS"""
import argparse, time, torch
from flash_attn.cute.interface import flash_attn_func
p = argparse.ArgumentParser(); p.add_argument("--seqlen", type=int, default=8192); p.add_argument("--causal", action="store_true")
p.add_argument("--idle", type=float, default=5); p.add_argument("--iters", type=int, default=300); p.add_argument("--fwd", action="store_true"); p.add_argument("--headdim", type=int, default=128)
a = p.parse_args(); hd = a.headdim; nheads = 16; bs = 32768 // a.seqlen; torch.manual_seed(0)
q, k, v = [torch.randn(bs, a.seqlen, nheads, hd, device="cuda", dtype=torch.bfloat16, requires_grad=not a.fwd) for _ in range(3)]
flops = bs * nheads * 2 * a.seqlen * (a.seqlen / 2 if a.causal else a.seqlen) * 2 * hd * (1 if a.fwd else 2.5)
if a.fwd:
    def f(): flash_attn_func(q, k, v, causal=a.causal)
else:
    g = torch.randn_like(q); y = flash_attn_func(q, k, v, causal=a.causal); y = y[0] if isinstance(y, tuple) else y
    def f():
        for x in (q, k, v): x.grad = None
        y.backward(g, retain_graph=True)
for _ in range(3): f()
torch.cuda.synchronize(); time.sleep(a.idle)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters + 1)]
ev[0].record()
for i in range(a.iters):
    f(); ev[i + 1].record()
torch.cuda.synchronize()
print(f"# seqlen={a.seqlen} causal={a.causal} fwd={a.fwd} idle={a.idle}s iters={a.iters}")
t = 0.0
for i in range(a.iters):
    d = ev[i].elapsed_time(ev[i + 1]); t += d
    print(f"{i:4d} {t:9.2f} {d:8.3f} {flops/(d*1e-3)*1e-12:6.0f}")
