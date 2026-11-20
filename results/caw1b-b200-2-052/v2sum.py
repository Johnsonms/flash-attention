import sys, glob, statistics as st, collections
# usage: v2sum.py <glob-prefix>  -> median TFLOPS per tree/causal/seqlen
d = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] + "*_r*.log")):
    tree = f.rsplit("/",1)[1].rsplit("_r",1)[0]
    for l in open(f):
        p = l.split()
        if len(p) == 8 and p[0].isdigit(): d[(tree, p[2], int(p[1]))].append(int(p[7]))
paper = {"True": [614,887,1102,1284,1402,1446], "False": [936,1166,1331,1430,1457,1412]}
for c in ["True", "False"]:
    print("causal" if c == "True" else "non-causal", "(median [min-max], n)   seqlen 1k..32k")
    for tree in sorted({k[0] for k in d}):
        row = [d[(tree, c, s)] for s in [1024,2048,4096,8192,16384,32768]]
        print(f"  {tree:22s}", "  ".join(f"{st.median(r):.0f}[{min(r)}-{max(r)}]" for r in row), f"n={len(row[0])}")
    print(f"  {'paper':22s}", "  ".join(map(str, paper[c])))
