"""Aggregate ncu per-instruction metric to per-source-line.

Usage: python per_line.py <report.ncu-rep> <metric_name>
"""
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.4.1/extras/python")
import ncu_report

if len(sys.argv) < 3:
    print("usage: per_line.py <report.ncu-rep> <metric_name> [top_n]")
    sys.exit(1)

report_path, metric_name = sys.argv[1], sys.argv[2]
top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

ctx = ncu_report.load_report(report_path)
print(f"# report: {report_path}")
print(f"# ranges: {ctx.num_ranges()}")

for r_idx in range(ctx.num_ranges()):
    rng = ctx.range_by_idx(r_idx)
    print(f"# range {r_idx}: {rng.num_actions()} actions")
    for a_idx in range(rng.num_actions()):
        act = rng.action_by_idx(a_idx)
        kname = act.name()
        print(f"\n## action {a_idx}: {kname[:120]}")
        m = act.metric_by_name(metric_name)
        if m is None:
            print(f"  (metric '{metric_name}' not found)")
            continue
        n = m.num_instances()
        print(f"  metric '{metric_name}' has {n} instances; has_correlation_ids={m.has_correlation_ids()}")
        if not m.has_correlation_ids():
            print(f"  total value: {m.value()}")
            continue
        corr = m.correlation_ids()
        # Aggregate per (file, line)
        per_line = {}
        for i in range(n):
            v = m.value(i)
            if v is None or v == 0:
                continue
            addr = corr.value(i)
            si = act.source_info(int(addr))
            if si is None:
                key = ("<no src>", 0)
            else:
                key = (si.file_name(), si.line())
            per_line[key] = per_line.get(key, 0) + v
        # Sort and print top N
        ranked = sorted(per_line.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(per_line.values())
        print(f"  total over lines: {total:,.0f}")
        print(f"  top {top_n}:")
        print(f"  {'rank':>4}  {'value':>16}  {'%':>6}  file:line")
        for i, ((fn, ln), v) in enumerate(ranked[:top_n], 1):
            pct = 100 * v / total if total else 0
            short_fn = fn.split("/")[-1] if fn else "<none>"
            print(f"  {i:>4}  {v:>16,.0f}  {pct:>5.1f}%  {short_fn}:{ln}")
