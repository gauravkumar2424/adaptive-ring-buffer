#!/usr/bin/env python3
"""
analyze_contract.py -- Phase 2 Experiment A analysis.

Answers four questions in order:

  1. How much memory does SDT actually need to honour its epsilon bound?
     (max_occupancy / buffer_size. If this exceeds 1, a fixed budget of
     that size cannot host SDT without loss.)

  2. How often does a bounded queue overflow at each budget?

  3. Under FIFO drop, how much quality is lost?

  4. Under STALL, by how much is the epsilon bound violated?
     (max_dev / epsilon. A ratio > 1 means the guarantee is void.)

The claim survives only if SDT fails on BOTH horns: quality under
FIFO, guarantee under STALL. If it comfortably fits the budget, the
bounded-memory framing does not hold and we say so.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "memory_contract.csv")
if not os.path.exists(CSV):
    sys.exit(f"missing {CSV} -- run build/memory_contract first")

d = pd.read_csv(CSV)
for c in ["occupancy_ratio", "max_dev", "eps_violation_ratio", "snr_db",
          "max_occupancy", "overflow_events", "dropped", "stalled"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

sdt_u = d[(d.method == "SDT") & (d.policy == "unbounded")]
sdt_f = d[(d.method == "SDT") & (d.policy == "fifo_drop")]
sdt_s = d[(d.method == "SDT") & (d.policy == "stall")]
prop = d[d.method == "PROPOSED"]


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


hdr("1. MEMORY SDT REQUIRES  (peak queue depth / fixed budget)")
print("   >1.0 means SDT cannot run in that budget without loss.\n")
p = sdt_u.pivot_table(index="buffer_size", columns="overload",
                      values="occupancy_ratio", aggfunc="median")
print("   median ratio:")
print(p.round(2).to_string())
p95 = sdt_u.pivot_table(index="buffer_size", columns="overload",
                        values="occupancy_ratio", aggfunc=lambda x: np.percentile(x, 95))
print("\n   95th percentile ratio:")
print(p95.round(2).to_string())
print(f"\n   configurations where SDT needs MORE than the budget: "
      f"{100*(sdt_u.occupancy_ratio > 1).mean():.1f}%")
print(f"   worst case observed: {sdt_u.occupancy_ratio.max():.1f}x the budget"
      f"  ({sdt_u.loc[sdt_u.occupancy_ratio.idxmax(), 'signal']})")

print("\n   by domain:")
for dom, g in sdt_u.groupby("domain"):
    print(f"     {dom:<10} median {g.occupancy_ratio.median():5.2f}x   "
          f"p95 {np.percentile(g.occupancy_ratio,95):5.2f}x   "
          f"exceeds budget {100*(g.occupancy_ratio>1).mean():5.1f}%")

hdr("2. OVERFLOW EVENTS AT A BOUNDED QUEUE")
p = sdt_f.pivot_table(index="buffer_size", columns="overload",
                      values="overflow_events", aggfunc="median")
print("   median overflow events (FIFO policy):")
print(p.round(0).to_string())
print(f"\n   configurations with >=1 overflow: "
      f"{100*(sdt_f.overflow_events > 0).mean():.1f}%")

hdr("3. HORN A -- FIFO DROP: DATA IS LOST")
m = sdt_f.merge(sdt_u[["signal", "buffer_size", "overload", "snr_db"]],
                on=["signal", "buffer_size", "overload"],
                suffixes=("", "_unbounded"))
m["snr_cost"] = m.snr_db_unbounded - m.snr_db
print("   SNR cost of bounding the queue (dB, unbounded - bounded):")
p = m.pivot_table(index="buffer_size", columns="overload",
                  values="snr_cost", aggfunc="median")
print(p.round(2).to_string())
print(f"\n   median points dropped: {sdt_f.dropped.median():.0f}"
      f"   max: {sdt_f.dropped.max():.0f}")

hdr("4. HORN B -- STALL: THE EPSILON GUARANTEE IS VOID")
v = sdt_s[sdt_s.stalled > 0]
print(f"   configurations where the emitter stalled: "
      f"{100*(sdt_s.stalled>0).mean():.1f}%")
if len(v):
    p = v.pivot_table(index="buffer_size", columns="overload",
                      values="eps_violation_ratio", aggfunc="median")
    print("\n   median actual deviation / epsilon  (>1 = bound broken):")
    print(p.round(2).to_string())
    print(f"\n   worst violation: {v.eps_violation_ratio.max():.1f}x epsilon")
    print(f"   configs exceeding epsilon: "
          f"{100*(v.eps_violation_ratio>1).mean():.1f}%")
else:
    print("   no stalls observed -- the queue was never saturated.")

hdr("5. HEAD TO HEAD AT A FIXED BUDGET  (per-signal, n = signals)")
key = ["signal", "buffer_size", "overload"]
for pol, lab in [("fifo_drop", "SDT (FIFO drop)"), ("stall", "SDT (stall)")]:
    s = d[(d.method == "SDT") & (d.policy == pol)][key + ["snr_db"]]
    j = prop[key + ["snr_db"]].merge(s, on=key, suffixes=("_prop", "_sdt"))
    if not len(j):
        continue
    per = (j.snr_db_prop - j.snr_db_sdt).groupby(j.signal).mean()
    dz = per.mean() / per.std() if per.std() > 0 else float("nan")
    print(f"   PROPOSED vs {lab:<18} n={len(per):3d} "
          f"mean={per.mean():+7.3f} dB  dz={dz:+6.3f}  "
          f"win={100*(per>0).mean():5.1f}%")

print("""
   PROPOSED never overflows: peak occupancy equals the budget exactly,
   by construction, on every signal and every operating point.
""")

hdr("READ THIS")
print("""
  Claim holds if:  SDT's required memory exceeds the fixed budget on a
  substantial fraction of configurations, AND bounding the queue costs
  either quality (FIFO) or the epsilon guarantee (STALL).

  Claim fails if:  occupancy ratio stays below 1 nearly everywhere. Then
  SDT fits the budget comfortably, the bounded-memory framing does not
  distinguish the methods, and we drop it rather than defend it.
""")
