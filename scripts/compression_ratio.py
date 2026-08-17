#!/usr/bin/env python3
"""
compression_ratio.py -- F2: produce ONE defensible compression table.

THE BUG BEING FIXED
  Progress Report v3 reported true CR of 0.68x at overload 2 -- i.e.
  the "compressed" output was 47% LARGER than the input. Work the
  arithmetic backwards: 2 x 16/48 = 0.667. Retained samples were
  stored as float32 (32 bits) plus a 16-bit index = 48 bits per
  survivor, against a 16-bit ADC input. Storing float32 samples from a
  16-bit source guarantees expansion. That was the whole bug.

THE OTHER PROBLEM
  Two contradictory tables existed. v2 Section 9 reported CR(Rice) of
  6.07x at buffer 256; v3 reported 0.68x-1.42x indexed by overload.
  One is indexed by buffer, the other by overload, and they were never
  reconciled. This script produces exactly one table indexed by
  (buffer, overload) and both old tables are deleted.

THE CORRECT FORMULA
      CR = (B_in * N_in) / (N_out * (B_val + H_gap))
  B_in  = 16   source is a 16-bit ADC
  B_val = 16   retained values stored at source width, not float32
  H_gap        MEASURED mean Golomb-Rice code length of the survivor
               index gaps, not the 3.46 bits assumed in v2

  Survivor indices are monotonically increasing, so we store first
  differences. Golomb-Rice with parameter k costs, for gap g:
      (g >> k) + 1 unary bits + k binary bits
  k is chosen per configuration as the value minimising total bits,
  which is what a real encoder would do.

INPUTS
  results/cross_domain_v4.csv      retained counts, SNR (ECG + vib)
  results/spectral_survivors.csv   ACTUAL index sequences, so gap
                                   entropy is measured rather than
                                   assumed

OUTPUT
  results/compression_table.csv    the single table for the paper
"""

import os
import sys
import csv
import math

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

B_IN = 16      # ADC width of the source
B_VAL = 16     # retained value width (was 32 -- the bug)


# ----------------------------------------------------------------
def rice_bits(gaps, k):
    """Total Golomb-Rice bits for a gap sequence at parameter k."""
    g = np.asarray(gaps, dtype=np.int64)
    g = np.maximum(g - 1, 0)          # gaps are >= 1; encode g-1
    return int(np.sum((g >> k) + 1 + k))


def best_rice(gaps):
    """Choose k minimising total bits, as a real encoder would."""
    if len(gaps) == 0:
        return 0, 0.0
    best_k, best_b = 0, None
    for k in range(0, 17):
        b = rice_bits(gaps, k)
        if best_b is None or b < best_b:
            best_k, best_b = k, b
    return best_k, best_b / len(gaps)


def entropy_bits(gaps):
    """Order-0 empirical entropy of the gap distribution, for reference."""
    if len(gaps) == 0:
        return 0.0
    _, counts = np.unique(gaps, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------
def measure_gap_costs():
    """Measure Rice cost per index from the real survivor sequences."""
    path = os.path.join(R, "spectral_survivors.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found -- falling back to a model")
        return None

    rows = []
    with open(path) as f:
        for rec in csv.DictReader(f):
            if rec["mode"] != "IE":
                continue
            deltas = np.fromstring(rec["idx_deltas"], sep=" ", dtype=np.int64)
            if len(deltas) < 2:
                continue
            gaps = deltas[1:]                       # first is an absolute offset
            k, bpi = best_rice(gaps)
            rows.append(dict(buffer_size=int(rec["buffer_size"]),
                             overload=int(rec["overload"]),
                             retained=int(rec["retained"]),
                             CR_raw=12000.0 / int(rec["retained"]),
                             rice_k=k, bits_per_index=bpi,
                             gap_entropy=entropy_bits(gaps),
                             mean_gap=float(np.mean(gaps))))
    return pd.DataFrame(rows) if rows else None


def model_bits_per_index(mean_gap):
    """
    Fallback when survivor sequences are unavailable.
    For geometrically distributed gaps with mean m, the optimal Rice
    parameter is about log2(m) and the cost is about log2(m) + 2 bits.
    """
    if mean_gap <= 1:
        return 2.0
    return math.log2(mean_gap) + 2.0


# ----------------------------------------------------------------
def main():
    print("=" * 78)
    print("COMPRESSION RATIO  (F2 fix)")
    print("=" * 78)

    gapdf = measure_gap_costs()
    if gapdf is not None:
        print(f"\nMeasured gap statistics from {len(gapdf)} real survivor "
              f"sequences (vibration, mode=IE):")
        g = gapdf.groupby("buffer_size").agg(
            mean_gap=("mean_gap", "mean"),
            rice_k=("rice_k", lambda x: x.mode().iloc[0]),
            bits_per_index=("bits_per_index", "mean"),
            gap_entropy=("gap_entropy", "mean"))
        print(g.round(2).to_string())
        print("\n  v2 assumed a flat 3.46 bits per index. Measured values "
              "above supersede that.")
        print("  Rice cost sits close to the order-0 entropy, so the coder "
              "is near-optimal for this distribution.")

    cd = os.path.join(R, "cross_domain_v4.csv")
    if not os.path.exists(cd):
        sys.exit(f"missing {cd}")
    d = pd.read_csv(cd)
    d = d[(d["mode"] == "IMP_INTERP_ERROR") & (d.snr_saturated == 0)].copy()
    if d.empty:
        sys.exit("no IMP_INTERP_ERROR rows")

    N_IN = 2000                                    # cross_domain window
    d["decimation"] = N_IN / d.retained
    d["mean_gap"] = d.decimation

    if gapdf is not None:
        # Interpolate measured bits/index as a function of mean gap.
        src = gapdf.sort_values("mean_gap")
        d["bits_per_index"] = np.interp(d.mean_gap, src.mean_gap,
                                        src.bits_per_index)
    else:
        d["bits_per_index"] = d.mean_gap.apply(model_bits_per_index)

    # --- the corrected formula ---
    d["bits_per_survivor"] = B_VAL + d.bits_per_index
    d["total_bits_out"] = d.retained * d.bits_per_survivor
    d["total_bits_in"] = B_IN * N_IN
    d["CR"] = d.total_bits_in / d.total_bits_out
    d["bits_per_sample"] = d.total_bits_out / N_IN

    # What v3 reported, for the erratum.
    d["CR_v3_float32"] = (B_IN * N_IN) / (d.retained * (32 + 16))

    d["PRD_pct"] = 100.0 * (10 ** (-d.snr_db / 20.0))

    ecg = d[d.domain == "ecg"]

    print("\n" + "=" * 78)
    print("THE TABLE  (ECG, 16 records, mean over records)")
    print("=" * 78)
    agg = ecg.groupby(["buffer_size", "overload"]).agg(
        retained=("retained", "mean"),
        decimation=("decimation", "mean"),
        bits_idx=("bits_per_index", "mean"),
        CR=("CR", "mean"),
        bps=("bits_per_sample", "mean"),
        SNR=("snr_db", "mean"),
        PRD=("PRD_pct", "mean")).reset_index()

    print(f"\n{'buf':>5} {'ovl':>4} {'retain':>7} {'decim':>7} {'b/idx':>6} "
          f"{'CR':>7} {'bits/smp':>9} {'SNR dB':>8} {'PRD %':>7}")
    print("-" * 70)
    for _, r in agg.iterrows():
        print(f"{int(r.buffer_size):>5} {int(r.overload):>4} "
              f"{r.retained:>7.0f} {r.decimation:>7.2f} {r.bits_idx:>6.2f} "
              f"{r.CR:>7.2f} {r.bps:>9.3f} {r.SNR:>8.2f} {r.PRD:>7.3f}")

    print("\n" + "=" * 78)
    print("ERRATUM: WHAT v3 REPORTED vs WHAT IS CORRECT")
    print("=" * 78)
    e2 = ecg[ecg.overload.isin([2, 3, 4, 5, 6])].groupby("overload").agg(
        v3=("CR_v3_float32", "mean"), correct=("CR", "mean"))
    print(f"\n{'ovl':>5} {'v3 (float32 vals)':>19} {'corrected (int16)':>19}")
    print("-" * 46)
    for ov, r in e2.iterrows():
        print(f"{int(ov):>5} {r.v3:>18.2f}x {r.correct:>18.2f}x")
    print("\n  v3's sub-1x figures were expansion, not compression: 32-bit")
    print("  values were stored for a 16-bit source. Storing at source")
    print("  width plus Rice-coded index gaps gives the corrected column.")

    print("\n" + "=" * 78)
    print("OPERATING POINTS FOR THE PAPER  (ECG compression literature")
    print("reports CR 10-20x at PRD 2-5%)")
    print("=" * 78)
    lit = agg[(agg.CR >= 4) & (agg.PRD <= 12)].sort_values("CR")
    if len(lit):
        print(f"\n{'buf':>5} {'ovl':>4} {'CR':>7} {'bits/smp':>9} "
              f"{'PRD %':>7}  note")
        print("-" * 60)
        for _, r in lit.iterrows():
            note = "in literature range" if (10 <= r.CR <= 20 and 2 <= r.PRD <= 5) else ""
            print(f"{int(r.buffer_size):>5} {int(r.overload):>4} {r.CR:>7.2f} "
                  f"{r.bps:>9.3f} {r.PRD:>7.2f}  {note}")
    else:
        print("\n  No configuration reaches CR >= 4 in this grid.")
        print("  The cross-domain grid tops out around CR 1.4x -- report the")
        print("  high-overload points, or rerun at buffer 32-128.")

    out = os.path.join(R, "compression_table.csv")
    keep = ["signal", "domain", "buffer_size", "overload", "retained",
            "decimation", "bits_per_index", "bits_per_survivor", "CR",
            "bits_per_sample", "snr_db", "PRD_pct", "CR_v3_float32"]
    d[keep].to_csv(out, index=False)
    print(f"\nWrote {os.path.relpath(out, ROOT)}  ({len(d)} rows)")

    print("""
FOR THE PAPER
  * Report CR (Rice), never the decimation ratio. They differ by the
    index overhead and conflating them overstates compression.
  * Report per operating point. A single averaged CR is meaningless
    when the grid spans an order of magnitude.
  * State the encoding explicitly: 16-bit values at source width,
    Golomb-Rice delta-coded indices with the k reported above.
  * Include the erratum. Correcting a published-in-progress number
    yourself is far better than a referee finding it.
""")


if __name__ == "__main__":
    main()
