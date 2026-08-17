#!/usr/bin/env python3
"""
M10 + M13 SOFTWARE CLEANUP

M10: Fix RDP framing — find all places where RDP is called "optimal" or
     "upper bound" and list them for correction.

M13 fixes (software-addressable):
  - Spectral correlation = 0.0000 for DROP: investigate and flag
  - Method naming consistency: identify which name is used where
  - Unequal n from saturation filtering: already fixed in Phase A
  - Memory arithmetic: document correct numbers

Usage:
    cd ~/DSA/adaptive-ring-buffer
    python3 m10_m13_cleanup.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
RESULTS = BASE / "results"


def main():
    print("=" * 76)
    print("M10 + M13 SOFTWARE CLEANUP")
    print("=" * 76)

    # ──────────────────────────────────────────────────────────────────
    # M10: RDP FRAMING
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("M10: RDP FRAMING CORRECTIONS")
    print("-" * 76)
    print("""
  WRONG phrases (find and replace in paper):
    "global optimum"        → "established offline heuristic"
    "offline upper bound"   → "offline reference (RDP)"
    "optimal"  (for RDP)    → "offline heuristic"
    "ground truth"          → "offline reference"

  CORRECT framing:
    "RDP (Ramer-Douglas-Peucker) is a widely-used offline line
    simplification heuristic. While it achieves high reconstruction
    quality, it is a greedy recursive algorithm with no formal
    optimality guarantee for the fixed-budget sample selection problem.
    The true optimal k-point piecewise-linear approximation is
    computable by dynamic programming in O(N²k) time [ref].
    We use RDP as an offline reference because it is the standard
    benchmark in the line simplification literature."

  NOTE: Our stress test shows Proposed (39.0 dB) beating RDP (36.7 dB)
  at 2x overload. Under "RDP = optimal" framing this is a contradiction.
  Under "RDP = heuristic" framing it's just two heuristics trading places
  at different operating points. Fix the framing and this becomes a
  FEATURE, not a bug.
""")

    # ──────────────────────────────────────────────────────────────────
    # M13a: SPECTRAL CORRELATION = 0.0000 INVESTIGATION
    # ──────────────────────────────────────────────────────────────────
    print("-" * 76)
    print("M13a: SPECTRAL CORRELATION = 0.0000 INVESTIGATION")
    print("-" * 76)

    # Check cross_domain for spec_corr = 0 or very close to 0
    cd_path = RESULTS / "cross_domain_results.csv"
    if cd_path.exists():
        cd = pd.read_csv(cd_path)
        if "spectral_correlation" in cd.columns:
            zero_spec = cd[cd["spectral_correlation"] == 0.0]
            neg_one_spec = cd[cd["spectral_correlation"] == -1.0]
            nan_spec = cd[cd["spectral_correlation"].isna()]

            print(f"  cross_domain_results.csv:")
            print(f"    spectral_correlation == 0.0:  {len(zero_spec)} rows")
            print(f"    spectral_correlation == -1.0: {len(neg_one_spec)} rows")
            print(f"    spectral_correlation is NaN:  {len(nan_spec)} rows")

            if len(neg_one_spec) > 0:
                print(f"\n    -1.0 appears to be a sentinel for 'not computed'")
                print(f"    Modes with -1.0: {neg_one_spec['mode'].unique()}")
                print(f"    Domains with -1.0: {neg_one_spec['domain'].unique()}")

            if len(zero_spec) > 0:
                print(f"\n    0.0 rows breakdown:")
                for mode in zero_spec["mode"].unique():
                    n = len(zero_spec[zero_spec["mode"] == mode])
                    print(f"      {mode}: {n} rows")

                print(f"\n    LIKELY CAUSE: Spectral correlation of exactly 0.0 for")
                print(f"    DROP at high compression is either:")
                print(f"    a) Genuine: FIFO destroys spectral content completely")
                print(f"    b) NaN→0 substitution: division by zero in correlation")
                print(f"    CHECK: If the reconstructed signal is flat (all same value),")
                print(f"    its FFT is zero, and correlation with any signal is undefined.")
                print(f"    RECOMMENDATION: Report as 'undefined (degenerate reconstruction)'")
                print(f"    instead of 0.0000.")

    # Check stress test
    st_path = RESULTS / "stress_test_results.csv"
    if st_path.exists():
        st = pd.read_csv(st_path)
        if "spectral_correlation" in st.columns:
            zero_st = st[st["spectral_correlation"] == 0.0]
            print(f"\n  stress_test_results.csv:")
            print(f"    spectral_correlation == 0.0: {len(zero_st)} rows")
            if len(zero_st) > 0:
                for mode in zero_st["mode"].unique():
                    n = len(zero_st[zero_st["mode"] == mode])
                    ovls = sorted(zero_st[zero_st["mode"] == mode]["overload"].unique())
                    print(f"      {mode}: {n} rows at overloads {ovls}")

    # ──────────────────────────────────────────────────────────────────
    # M13b: METHOD NAMING CONSISTENCY
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("M13b: METHOD NAMING CONSISTENCY")
    print("-" * 76)

    # Check which method names appear in which result files
    result_files = {
        "cross_domain": RESULTS / "cross_domain_results.csv",
        "novelty_eval": RESULTS / "novelty_eval_results.csv",
        "stress_test":  RESULTS / "stress_test_results.csv",
        "duration_eval": RESULTS / "duration_eval_results.csv",
        "sensitivity":  RESULTS / "sensitivity_results.csv",
    }

    method_sets = {}
    for name, path in result_files.items():
        if path.exists():
            df = pd.read_csv(path)
            if "mode" in df.columns:
                method_sets[name] = set(df["mode"].unique())

    if method_sets:
        # Find the proposed method variants across files
        proposed_variants = set()
        for name, modes in method_sets.items():
            for m in modes:
                if "INTERP" in m:
                    proposed_variants.add(m)

        print(f"\n  Proposed method variants across all result files:")
        for v in sorted(proposed_variants):
            files_with = [name for name, modes in method_sets.items() if v in modes]
            print(f"    {v:35s}  in: {', '.join(files_with)}")

        print(f"""
  RECOMMENDATION:
    PRIMARY method:   IMP_INTERP_ERROR
    Present as:       "Proposed" or "IE" in tables
    SPECTRAL variant: IMP_INTERP_SPECTRAL
    Present as:       "Proposed+S" or "IE-S" in tables (secondary)
    COMPOSITE:        IMP_INTERP_COMPOSITE
    Status:           DROP from paper. Marginal improvement, adds confusion.

    The paper should use ONE name consistently: IMP_INTERP_ERROR.
    SPECTRAL can appear as a variant in one table, not as the primary method.
""")

    # ──────────────────────────────────────────────────────────────────
    # M13c: MEMORY ARITHMETIC CORRECTION
    # ──────────────────────────────────────────────────────────────────
    print("-" * 76)
    print("M13c: MEMORY ARITHMETIC CORRECTION")
    print("-" * 76)
    print("""
  REVIEWER CAUGHT: "28 B/node but 512 × 28 = 14.0 KB, not 14.4 KB"

  The issue: The progress report mixed up the C++ and C implementations.

  C++ (ring_buffer.h) — O(N) scan:
    struct Node {
        double value;       // 8 bytes
        int original_index; // 4 bytes
        int prev;           // 4 bytes
        int next;           // 4 bytes
        bool active;        // 1 byte
        // padding           // 3 bytes (alignment to 4)
    };
    Total: 24 bytes/node (not 28)
    512 × 24 = 12,288 bytes = 12.0 KB

  C (STM32 main.c) — float:
    struct Node {
        float value;        // 4 bytes
        uint16_t orig_idx;  // 2 bytes
        uint16_t prev;      // 2 bytes
        uint16_t next;      // 2 bytes
        uint8_t active;     // 1 byte
        // padding           // 1 byte
    };
    Total: 12 bytes/node
    512 × 12 = 6,144 bytes = 6.0 KB

  O(log N) heap variant (C, STM32):
    Node: 12 bytes (same as above)
    Heap entry: { float key (4B) + uint16_t node_idx (2B) + uint16_t padding (2B) } = 8 bytes
    Index map: uint16_t per node = 2 bytes
    Total per entry: 12 + 8 + 2 = 22 bytes
    512 × 22 = 11,264 bytes = 11.0 KB = 8.6% of 128 KB SRAM

  CORRECTED TABLE FOR PAPER:
    Buffer   O(N) C/float   O(logN) C/float   % of 128 KB SRAM
    64       768 B           1,408 B           1.1%
    128      1,536 B         2,816 B           2.2%
    256      3,072 B         5,632 B           4.4%
    512      6,144 B        11,264 B           8.6%

  ACTION: Verify exact struct sizes by adding sizeof() prints to STM32 main.c.
""")

    # ──────────────────────────────────────────────────────────────────
    # M13d: DECIM/RSRVR "0 CYCLES" — HARDWARE FIX NEEDED
    # ──────────────────────────────────────────────────────────────────
    print("-" * 76)
    print("M13d: ITEMS REQUIRING HARDWARE FIXES (for STM32 session)")
    print("-" * 76)
    print("""
  1. DECIM/RSRVR report "0 cycles"
     CAUSE: These modes do decimation/reservoir sampling in the producer
     thread, not in the eviction path. DWT only measures eviction time.
     FIX: Report as "N/A (no eviction computation)" instead of 0.
     Or: add DWT measurement around the decimation logic.

  2. Energy measurement
     CAUSE: Estimated from datasheet, not measured.
     FIX: Use Nordic PPK2 or shunt resistor + oscilloscope.
     MINIMUM: Keep the estimate but label it clearly as "estimated from
     datasheet current draw, not measured."

  3. No consumer in streaming demo
     CAUSE: ISR pushes to buffer, nothing drains it.
     FIX: Add a main-loop consumer that pops from buffer every M samples.
     This tests contention and realistic deadline margin.
     MINIMUM: Acknowledge in paper: "The streaming demo measures ISR
     execution time without a concurrent consumer. Actual margin depends
     on consumer workload."
""")

    print("\n" + "=" * 76)
    print("M10 + M13 CLEANUP COMPLETE")
    print("Paste output to mentor.")
    print("=" * 76)


if __name__ == "__main__":
    main()
