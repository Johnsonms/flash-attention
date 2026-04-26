"""One-shot repro of FA4 hd256 backward for ncu profiling.
Configurable via env vars (defaults = config A: MHA non-causal, B=8, S=16384).
"""
import os, torch
from flash_attn.cute.interface import flash_attn_func

torch.cuda.set_device(0)

B = int(os.environ.get("CFG_B", "8"))
S = int(os.environ.get("CFG_S", "16384"))
H = int(os.environ.get("CFG_H", "32"))
HKV = int(os.environ.get("CFG_HKV", "32"))
D = int(os.environ.get("CFG_D", "256"))
CAUSAL = os.environ.get("CFG_CAUSAL", "0") == "1"
WARMUP = int(os.environ.get("CFG_WARMUP", "3"))
PROFILED_ITERS = int(os.environ.get("CFG_ITERS", "3"))

torch.manual_seed(0)
q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, S, HKV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

print(f"[repro] B={B} S={S} H={H} HKV={HKV} D={D} causal={CAUSAL} warmup={WARMUP} iters={PROFILED_ITERS}")

def step():
    out = flash_attn_func(q, k, v, causal=CAUSAL)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(g, retain_graph=True)

# Warmup: trigger JIT compile + warm caches
for i in range(WARMUP):
    step()
torch.cuda.synchronize()
print("[repro] warmup done; profiled iters begin")

# Profiled iters
for i in range(PROFILED_ITERS):
    step()
torch.cuda.synchronize()
print("[repro] done")
