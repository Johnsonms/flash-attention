"""compare two numcheck .pt files (dict shape -> (out, dq, dk, dv)). usage: numcompare.py a.pt b.pt"""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for key in a:
    diffs = [(x - y).abs().max().item() for x, y in zip(a[key], b[key])]
    print(key, " ".join(f"{n}={d:.3e}" for n, d in zip(("out", "dq", "dk", "dv"), diffs)))
