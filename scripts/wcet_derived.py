#!/usr/bin/env python3
"""
wcet_derived.py -- WCET bound derived from the compiled binary.

REPLACES
  wcet_bound.py, whose constants were least-squares fitted to measured
  maxima and then inflated until nothing was violated. That produced a
  negative intercept before inflation, which is a tell: a regression
  that happens to dominate 48 observations is not a cycle model. It
  also replaces the v2/v3 claim

      WCET <= 9*floor(log2 N) + 16 operations

  which counted operations rather than cycles and was violated by every
  measurement (664 "operations" at N=256 against a measured max of
  1285 cycles).

SOURCE
  arm-none-eabi-objdump -d ring_buffer_eval.elf, built -O2 with
  -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard.
  Instruction costs from the Cortex-M4 Technical Reference Manual.
  Every cycle below is traceable to a specific address range in the
  binary, listed in COUNTS.

STRUCTURE
  Per eviction the 2-neighbour update invariant gives exactly three
  heap traversals: one sift-down from extract-min, plus two neighbour
  reheapifications, each hfix() = hup() then hdn().

  hfix costs at most floor(log2 N) levels, not 2*floor(log2 N): if
  hup() moves the element up by k > 0 levels then the element is
  smaller than its new parent, so the following hdn() terminates at its
  first comparison. If hup() moves nothing, hdn() may descend fully.
  Only one of the two can traverse.

      WCET(N) = C_fixed + C_level * 3*floor(log2 N)

WORST-CASE PATH
  Each level is charged the most expensive route through hdn(): both
  children present, BOTH float comparisons tie, so both integer
  tie-break branches execute, and a swap follows. Exact float ties in
  score[] are rare on real data, which is why the bound sits above the
  measured maxima -- that is what a bound is.

ASSUMPTIONS  (state these in the paper)
  A1. ART accelerator hit on all literal-pool loads. Both functions use
      ldr rN, [pc, #imm] to fetch constants from flash. With prefetch
      and instruction cache enabled (FLASH_ACR = 5 WS | ICEN | DCEN |
      PRFTEN, set in clock_init) these are single-cycle; a cold miss
      costs 5 wait states. A fully sound bound needs a flash model.
      This is the one remaining assumption.
  A2. No DMA contention on the SRAM bus (single master).
  A3. ISR overhead 160 cycles: exception entry/exit, lazy FPU context
      stacking, pipeline refill.
"""

import math
import os
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- 
# Instruction accounting. Each entry: (label, address range, cycles).
COUNTS = {
    "hdn_level": [
        ("loop head: lsls,adds,cmp,add.w,bls",        "0x8000742-0x800074c",  5),
        ("left child: 2 ldrh,2 add.w,2 vldr,vcmpe,vmrs,2 br",
                                                       "0x800074e-0x8000770", 14),
        ("left tie-break: 4 add.w,2 ldr.w,cmp,blt",   "0x8000772-0x800078c", 11),
        ("right child: cmp,bhi,2 ldrh,2 add.w,2 vldr,vcmpe,vmrs,2 br",
                                                       "0x800078e-0x80007b8", 16),
        ("right tie-break: mov,4 add.w,2 ldr.w,cmp,it+movge",
                                                       "0x80007ba-0x80007d6", 13),
        ("swap: cmp,bne,2 ldrh,4 strh,mov,b",         "0x80007d8-0x80007f6", 17),
    ],
    "hup_level": [
        ("loop: subs,mov,lsrs,ldrh,add.w,vldr,vcmpe,2 add.w,vmrs,2 br",
                                                       "0x80006f4-0x8000718", 15),
        ("tie-break: 2 ldr,cmp,bge",                   "0x80006d8-0x80006e0",  7),
        ("swap: 4 strh,cbz",                           "0x80006e2-0x80006f2", 11),
    ],
    "ie_of": [
        ("4 shifted add.w (address arithmetic)",       "0x8001078-0x8001084",  4),
        ("2 ldr (original_index fields)",              "0x8001088-0x800108c",  4),
        ("vmov,vcvt.f32.s32,vcmpe,vmrs (span>0 test)", "0x8001092-0x800109e",  4),
        ("4 vldr (node values)",                       "0x80010a4-0x80010b0",  8),
        ("vmov,vcvt,vsub",                             "0x80010b4-0x80010c0",  3),
        ("VDIV.F32  <-- dominant instruction",         "0x80010c4",           14),
        ("vfma.f32 (fused multiply-add)",              "0x80010c8",            3),
        ("vsub,vabs,vstr",                             "0x80010cc-0x80010d8",  4),
    ],
    "call_overhead": [
        ("hdn prologue: 6 literal ldr, stmdb{7}, ldr", "0x8000730-0x8000740", 20),
        ("hdn epilogue: ldmia{7}",                     "0x8000792",            9),
        ("hup prologue+epilogue: ldr, stmdb{6}, 3 ldr, ldmia{6}",
                                                       "0x80006b4-0x800071a", 24),
    ],
    "evict_fixed": [
        ("extract-min bookkeeping (hsize, hpos, root move)",
                                                       "0x8000e50-0x8000e72", 22),
        ("unlink_slot: 4 pointer writes + free-list splice",
                                                       "0x8000e78-0x8000ee4", 38),
        ("neighbour boundary tests and score[] stores","0x8000eec-0x8000f32", 32),
        ("register save/restore stmdb+ldmia {10}",     "0x8000e52,0x8000fe6", 26),
    ],
}

C_ISR = 160          # A3
N_IE_PER_EVICT = 2   # 2-neighbour invariant
N_CHAINS = 3         # extract-min sift-down + 2 x hfix


def total(key):
    return sum(c for _, _, c in COUNTS[key])


def main():
    print("=" * 78)
    print("WCET BOUND DERIVED FROM THE COMPILED BINARY")
    print("=" * 78)

    for key, title in [("hdn_level", "hdn(): cost per heap level (worst path)"),
                       ("hup_level", "hup(): cost per heap level (worst path)"),
                       ("ie_of",     "ie_of(): inlined interpolation error"),
                       ("call_overhead", "call prologue/epilogue"),
                       ("evict_fixed",   "ev_heap(): fixed work per eviction")]:
        print(f"\n{title}")
        print("-" * 78)
        for label, addr, cyc in COUNTS[key]:
            print(f"  {cyc:>3}  {addr:<24} {label}")
        print(f"  {'':>3}  {'':<24} {'TOTAL: ' + str(total(key)):>40}")

    c_level = total("hdn_level")          # hdn dominates hup (76 > 33)
    c_ie = total("ie_of")
    c_call = total("call_overhead")
    c_evict = total("evict_fixed")
    c_fixed = N_IE_PER_EVICT * c_ie + c_call + c_evict + C_ISR

    print("\n" + "=" * 78)
    print("THE BOUND")
    print("=" * 78)
    print(f"""
  C_level = {c_level} cycles   (hdn worst path; hup is {total('hup_level')}, dominated)
  C_fixed = {N_IE_PER_EVICT}x{c_ie} (ie_of) + {c_call} (calls) + {c_evict} (evict) + {C_ISR} (ISR)
          = {c_fixed} cycles

      WCET(N) = {c_fixed} + {c_level} * 3*floor(log2 N)   cycles

  The factor 3 is structural: the 2-neighbour invariant admits exactly
  three heap traversals per eviction, and hfix() traverses at most one
  of hup/hdn (see module docstring).
""")

    # Measured maxima from the v19 run (LOGN mode), 6 signals per buffer.
    static_max = {32: 992, 64: 1146, 128: 1301, 256: 1442,
                  512: 1701, 1024: 1815, 2048: 2194, 4096: 2360}
    static_avg = {32: 956, 64: 1112, 128: 1204, 256: 1339,
                  512: 1499, 1024: 1646, 2048: 1809, 4096: 1936}
    stream_max = {256: 1758, 1024: 2040, 4096: 2332}

    print("=" * 78)
    print("VALIDATION AGAINST MEASURED MAXIMA")
    print("=" * 78)
    print(f"\n{'buf':>6} {'3log2N':>7} {'bound':>7} {'static max':>11} "
          f"{'stream max':>11} {'avg':>7} {'tight':>7}  status")
    print("-" * 74)

    rows, ok = [], True
    for b in sorted(static_max):
        lv = 3 * math.floor(math.log2(b))
        bound = c_fixed + c_level * lv
        smax = static_max[b]
        tmax = stream_max.get(b)
        worst = max(smax, tmax) if tmax else smax
        good = bound >= worst
        ok &= good
        print(f"{b:>6} {lv:>7} {bound:>7} {smax:>11} "
              f"{(tmax if tmax else '-'):>11} {static_avg[b]:>7} "
              f"{bound/worst:>7.2f}  {'ok' if good else 'VIOLATED'}")
        rows.append((b, lv, bound, smax, tmax or "", static_avg[b],
                     round(bound / worst, 3), int(good), int(168e6 / bound)))

    print(f"\n  {'Bound holds on every configuration.' if ok else 'BOUND VIOLATED.'}")
    print(f"  Tightness {min(r[6] for r in rows):.2f}-{max(r[6] for r in rows):.2f}x. "
          f"Looser than a fitted bound by construction: every level is")
    print(f"  charged both tie-break branches and a swap, which real data "
          f"almost never triggers.")

    print("\n" + "=" * 78)
    print("MAXIMUM GUARANTEED SAMPLE RATE  (168 MHz / WCET)")
    print("=" * 78)
    print(f"\n{'buf':>6} {'WCET':>7} {'max rate':>12}  {'48 kHz headroom':>16}")
    print("-" * 46)
    dl48 = 168e6 / 48000
    for b, lv, bound, *_ in rows:
        print(f"{b:>6} {bound:>7} {168e6/bound/1000:>10.1f} kHz  "
              f"{100*(1-bound/dl48):>14.1f}%")

    out = os.path.join(R, "wcet_derived.csv")
    os.makedirs(R, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["buffer", "levels", "bound_cycles", "static_max",
                    "stream_max", "static_avg", "tightness", "holds",
                    "max_rate_hz"])
        w.writerows(rows)
    print(f"\nWrote {os.path.relpath(out, ROOT)}")

    print(f"""
{'=' * 78}
ASSUMPTIONS TO DECLARE
{'=' * 78}
  A1  ART accelerator hits on literal-pool loads (ldr rN,[pc,#imm]).
      Single-cycle when cached; 5 wait states on a cold miss. A fully
      sound bound needs a flash-access model. This is the one
      remaining assumption and should be stated as a limitation.
  A2  No DMA contention on the SRAM bus.
  A3  ISR overhead {C_ISR} cycles (exception entry/exit, lazy FPU
      stacking, pipeline refill).

WHAT IS AND IS NOT GUARANTEED
  Guaranteed: per-eviction cost logarithmic in N with the constants
  above; fixed memory, no dynamic allocation; decisions identical
  between the O(N) scan and the O(log N) heap (verified, zero SNR
  deviation across 144 configurations).
  Not guaranteed: bounded reconstruction error. The method is a greedy
  heuristic optimising average fidelity. SDT and LTC bound maximum
  deviation; this method does not, and that trade is the paper's.
""")


if __name__ == "__main__":
    main()
