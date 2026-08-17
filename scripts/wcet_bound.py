#!/usr/bin/env python3
"""
wcet_bound.py -- item #4: a WCET bound that is actually a bound.

THE PROBLEM WITH THE OLD ONE
  Progress Reports v2/v3 claim

      WCET <= 9*floor(log2 N) + 16 operations per eviction

  At N=256 that is 664. The MEASURED average on hardware is 729 cycles
  and the measured maximum is 1285. A bound violated by every single
  measurement is not a bound; it is an operation count. At a real-time
  venue that is a category error, and the derived "maximum guaranteed
  sample rate" of 54,789 Hz inherits the error -- it was computed from
  an observed maximum over 744 samples, which is an estimate.

WHAT THIS SCRIPT DOES
  1. Builds a CYCLE-level bound from the Cortex-M4 instruction costs,
     structured as
         WCET(N) = C_fixed + C_level * (3 * floor(log2 N))
     The factor 3 is the three sift chains per eviction implied by the
     2-neighbour invariant: one extract-min sift-down plus two
     neighbour reheapifications.
  2. Calibrates C_fixed and C_level by LEAST SQUARES ON THE MEASURED
     MAXIMA, then inflates by the safety factor needed so the bound
     dominates every observation.
  3. VALIDATES: reports the tightness ratio bound/measured_max for
     every configuration. A bound that is never violated and is within
     ~2x is defensible. One that is violated anywhere is not a bound
     and the script says so.

  Calibrating against measurements and then declaring the result a
  bound is a measurement-based WCET estimate, NOT a static analysis.
  The script labels it that way and the paper must too. A sound bound
  needs the disassembly; the structure here is what that analysis would
  produce, with the constants fitted rather than derived.

INPUT
  Paste the v19 GDB dump into results/v19_results.txt, or let the
  script fall back to the values recorded from the 2026-08-11 run.

OUTPUT
  results/wcet_bound.csv
"""

import os
import re
import sys
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# Cortex-M4 (ARMv7E-M) costs, from the technical reference manual.
# Listed so a referee can audit the model rather than trust a number.
COST = {
    "ldr":      2,   # load, no wait states (ART enabled, 5 WS flash)
    "str":      2,
    "cmp":      1,
    "branch":   3,   # taken branch: pipeline refill
    "vldr":     2,   # FPU load
    "vcmp":     1,
    "vsub":     1,
    "vmul":     1,
    "vdiv":    14,   # VDIV.F32 -- dominant single instruction
    "vabs":     1,
    "alu":      1,
}

# Per sift level: 2 child loads, 2 float compares, tie-break integer
# compare, swap (4 loads + 4 stores), loop branch.
SIFT_LEVEL = (2 * COST["vldr"] + 2 * COST["vcmp"] + COST["cmp"]
              + 4 * COST["ldr"] + 4 * COST["str"] + COST["branch"])

# Fixed work per eviction: extract-min bookkeeping, unlink (4 pointer
# writes), two neighbour ie() recomputations (each: 6 loads, 1 VDIV,
# 2 VSUB, 1 VMUL, 1 VABS), heap slot bookkeeping, call overhead.
IE_COST = (6 * COST["ldr"] + COST["vdiv"] + 2 * COST["vsub"]
           + COST["vmul"] + COST["vabs"])
FIXED = (8 * COST["ldr"] + 8 * COST["str"] + 2 * IE_COST
         + 6 * COST["alu"] + 2 * COST["branch"])


# Measured maxima from the v19 run, LOGN mode, cycles.
# buffer -> {signal: max_cyc}
FALLBACK = {
    32:   dict(ecg100=944,  ecg105=951,  vib_normal=991,  vib_inner=946,
               sine=992,  har_mixed=913),
    64:   dict(ecg100=1123, ecg105=1122, vib_normal=1093, vib_inner=1044,
               sine=1145, har_mixed=1146),
    128:  dict(ecg100=1168, ecg105=1177, vib_normal=1201, vib_inner=1164,
               sine=1301, har_mixed=1211),
    256:  dict(ecg100=1285, ecg105=1381, vib_normal=1289, vib_inner=1282,
               sine=1442, har_mixed=1355),
    512:  dict(ecg100=1460, ecg105=1560, vib_normal=1446, vib_inner=1396,
               sine=1701, har_mixed=1433),
    1024: dict(ecg100=1647, ecg105=1700, vib_normal=1645, vib_inner=1532,
               sine=1815, har_mixed=1540),
    2048: dict(ecg100=1867, ecg105=1828, vib_normal=1698, vib_inner=1609,
               sine=2194, har_mixed=1657),
    4096: dict(ecg100=2190, ecg105=1960, vib_normal=1713, vib_inner=1753,
               sine=2360, har_mixed=1640),
}

# Streaming maxima, LOGN, incl. ISR entry/exit and consumer contention.
STREAM_MAX = {256: 1758, 1024: 2040, 4096: 2332}


def parse_gdb(path):
    """Extract (buffer, max_cyc) for LOGN rows from a GDB array dump."""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {}
    pat = re.compile(
        r'signal\s*=\s*"([^"\\]+)[^,]*,\s*mode\s*=\s*"LOGN[^,]*,\s*'
        r'buf\s*=\s*(\d+).*?max_cyc\s*=\s*(\d+)', re.S)
    for sig, buf, mx in pat.findall(txt):
        out.setdefault(int(buf), {})[sig] = int(mx)
    return out or None


def main():
    print("=" * 78)
    print("WCET BOUND -- cycle level, validated")
    print("=" * 78)

    data = parse_gdb(os.path.join(R, "v19_results.txt"))
    if data:
        print(f"\nParsed {sum(len(v) for v in data.values())} LOGN rows "
              f"from results/v19_results.txt")
    else:
        data = FALLBACK
        print("\nUsing recorded v19 maxima (results/v19_results.txt not found)")

    print(f"\nInstruction-count model:")
    print(f"  per sift level : {SIFT_LEVEL} cycles "
          f"(2 VLDR, 2 VCMP, CMP, 4 LDR, 4 STR, branch)")
    print(f"  ie() recompute : {IE_COST} cycles "
          f"(6 LDR, VDIV={COST['vdiv']}, 2 VSUB, VMUL, VABS)")
    print(f"  fixed per evict: {FIXED} cycles "
          f"(unlink, 2 x ie(), bookkeeping)")

    bufs = sorted(data.keys())
    x = np.array([3 * math.floor(math.log2(b)) for b in bufs], float)
    y = np.array([max(data[b].values()) for b in bufs], float)

    A = np.vstack([np.ones_like(x), x]).T
    c_fixed, c_level = np.linalg.lstsq(A, y, rcond=None)[0]

    fitted = c_fixed + c_level * x
    infl = float(np.max(y / fitted))
    c_fixed *= infl
    c_level *= infl

    # ISR overhead term. The static path never pays exception entry/exit:
    # stacking r0-r3/r12/LR/PC/xPSR plus lazy FPU context save and the
    # returning pipeline refill. Calibrated so the bound dominates the
    # measured streaming maxima too, then rounded up.
    isr_over = 0.0
    for _b, _mx in STREAM_MAX.items():
        _bd = c_fixed + c_level * (3 * math.floor(math.log2(_b)))
        isr_over = max(isr_over, _mx - _bd)
    C_ISR = math.ceil(isr_over / 10.0) * 10 + 20   # margin, rounded
    c_fixed += C_ISR
    print(f"  ISR overhead term: +{C_ISR} cycles "
          f"(exception entry/exit, FPU context, pipeline refill)")

    print(f"\nCalibrated (least squares on measured maxima, then inflated "
          f"by {infl:.3f} so the bound dominates):")
    print(f"  WCET(N) = {c_fixed:.0f} + {c_level:.1f} * 3*floor(log2 N)  cycles")
    print(f"  model predicts {FIXED} fixed / {SIFT_LEVEL} per level; "
          f"the gap is stack traffic, ISR prologue and flash access "
          f"the instruction model omits.")

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print(f"\n{'buf':>6} {'3*log2N':>8} {'bound':>8} {'meas max':>9} "
          f"{'meas avg':>9} {'ratio':>7}  status")
    print("-" * 62)

    rows, ok = [], True
    for b in bufs:
        lv = 3 * math.floor(math.log2(b))
        bound = c_fixed + c_level * lv
        mx = max(data[b].values())
        av = float(np.mean(list(data[b].values())))
        ratio = bound / mx
        good = bound >= mx
        ok &= good
        print(f"{b:>6} {lv:>8} {bound:>8.0f} {mx:>9} {av:>9.0f} "
              f"{ratio:>7.2f}  {'ok' if good else 'VIOLATED'}")
        rows.append((b, lv, bound, mx, av, ratio, good))

    print("\n" + "=" * 78)
    print("STREAMING (ISR entry/exit + consumer contention)")
    print("=" * 78)
    print(f"\n{'buf':>6} {'bound':>8} {'ISR max':>9} {'ratio':>7}  status")
    print("-" * 40)
    for b, mx in sorted(STREAM_MAX.items()):
        bound = c_fixed + c_level * (3 * math.floor(math.log2(b)))
        good = bound >= mx
        print(f"{b:>6} {bound:>8.0f} {mx:>9} {bound/mx:>7.2f}  "
              f"{'ok' if good else 'VIOLATED -- add ISR overhead'}")
        if not good:
            ok = False

    if ok:
        print("\n  Bound holds on every configuration, static and streaming.")
    else:
        print("\n  BOUND VIOLATED. Increase the inflation factor or add an")
        print("  explicit ISR-overhead term before using this in the paper.")

    print("\n" + "=" * 78)
    print("MAXIMUM GUARANTEED SAMPLE RATE  (168 MHz / WCET)")
    print("=" * 78)
    print(f"\n{'buf':>6} {'WCET':>8} {'max rate':>12}  headroom at 48 kHz")
    print("-" * 52)
    for b in bufs:
        bound = c_fixed + c_level * (3 * math.floor(math.log2(b)))
        rate = 168e6 / bound
        dl = 168e6 / 48000
        print(f"{b:>6} {bound:>8.0f} {rate/1000:>10.1f} kHz  "
              f"{100*(1 - bound/dl):>6.1f}%")

    out = os.path.join(R, "wcet_bound.csv")
    with open(out, "w") as f:
        f.write("buffer,levels,bound_cycles,measured_max,measured_avg,"
                "tightness_ratio,holds,max_rate_hz\n")
        for b, lv, bd, mx, av, rt, gd in rows:
            f.write(f"{b},{lv},{bd:.1f},{mx},{av:.1f},{rt:.3f},"
                    f"{int(gd)},{168e6/bd:.0f}\n")
    print(f"\nWrote {os.path.relpath(out, ROOT)}")

    print(f"""
FOR THE PAPER
  * Call it a MEASUREMENT-BASED WCET ESTIMATE, not a static bound. The
    structure C_fixed + C_level*3*floor(log2 N) follows from the
    2-neighbour invariant; the constants are calibrated against
    {sum(len(v) for v in data.values())} measured maxima and inflated to dominate all of them.
  * Delete "9*floor(log2 N) + 16 operations" and the 54,789 Hz figure.
    That expression is violated by every measurement.
  * State what IS guaranteed: bounded per-eviction cost logarithmic in
    N, fixed memory with no dynamic allocation, deterministic decisions
    identical between the O(N) and O(log N) implementations.
  * A sound static bound needs the disassembly and a flash/bus model.
    Say so as a limitation rather than overclaiming.
""")


if __name__ == "__main__":
    main()
