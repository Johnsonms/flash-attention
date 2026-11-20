"""paper-method timing with knobs + epoch timestamps so 20 ms nvidia-smi samples can be aligned to each burst.
usage: paper_timer2.py [--causal both|true|false] [--seqlens ...] [--reps 10] [--sleep 1.0] [--loops 1] [--fwd] [--headdim 128]
Prints one line per (loop, causal, seqlen): t0 t1 (epoch s) of the Timer.timeit window, ms, TFLOPS."""
import argparse, time, torch
from torch.utils import benchmark
from flash_attn.cute.interface import flash_attn_func
p = argparse.ArgumentParser()
p.add_argument("--causal", default="both"); p.add_argument("--seqlens", default="1024,2048,4096,8192,16384,32768")
p.add_argument("--reps", type=int, default=10); p.add_argument("--sleep", type=float, default=1.0); p.add_argument("--loops", type=int, default=1)
p.add_argument("--fwd", action="store_true"); p.add_argument("--headdim", type=int, default=128); p.add_argument("--presleep", type=float, default=0)
a = p.parse_args(); hd = a.headdim; nheads = 32 if hd <= 64 else 16
causals = [False, True] if a.causal == "both" else [a.causal == "true"]
print(f"# reps={a.reps} sleep={a.sleep} loops={a.loops} fwd={a.fwd} presleep={a.presleep}")
print(f"{'loop':>4} {'seqlen':>6} {'causal':>6} {'batch':>5} {'t0':>14} {'t1':>14} {'ms':>9} {'TFLOPS':>6}", flush=True)
time.sleep(a.presleep)
for loop in range(a.loops):
    for causal in causals:
        for seqlen in [int(s) for s in a.seqlens.split(",")]:
            bs = 32768 // seqlen; torch.manual_seed(0)
            q, k, v = [torch.randn(bs, seqlen, nheads, hd, device="cuda", dtype=torch.bfloat16, requires_grad=not a.fwd) for _ in range(3)]
            flops = bs * nheads * 2 * seqlen * (seqlen / 2 if causal else seqlen) * 2 * hd * (1 if a.fwd else 2.5)
            if a.fwd:
                def f(): flash_attn_func(q, k, v, causal=causal)
            else:
                g = torch.randn(bs, seqlen, nheads, hd, device="cuda", dtype=torch.bfloat16)
                y = flash_attn_func(q, k, v, causal=causal); y = y[0] if isinstance(y, tuple) else y
                def f():
                    for x in (q, k, v): x.grad = None
                    y.backward(g, retain_graph=True)
            f(); torch.cuda.synchronize(); time.sleep(a.sleep)
            t = benchmark.Timer(stmt="f()", globals={"f": f}, num_threads=torch.get_num_threads())
            t0 = time.time(); m = t.timeit(a.reps); torch.cuda.synchronize(); t1 = time.time()
            print(f"{loop:>4} {seqlen:>6} {str(causal):>6} {bs:>5} {t0:14.3f} {t1:14.3f} {m.mean*1e3:9.3f} {flops/m.mean*1e-12:6.0f}", flush=True)
            del q, k, v
            if not a.fwd: del g, y
