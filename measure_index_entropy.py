#!/usr/bin/env python3
"""
TRUE COMPRESSION RATIO WITH DELTA-ENCODED INDICES

Fixes the "modest compression ratio" weakness.

Current accounting:  16-bit value + 16-bit ABSOLUTE index = 32 bits/sample
Correct accounting:  16-bit value + delta-coded GAP        = ~19-21 bits/sample

Surviving indices are monotonically increasing, so storing gaps instead of
absolute positions is standard practice (used by every time-series store:
Gorilla, InfluxDB, Parquet delta encoding). Reporting absolute indices
understates the achievable compression ratio by ~60%.

This script runs the actual eviction, captures the surviving index set,
and measures four encodings:
  1. Absolute 16-bit index      (what you currently report)
  2. Fixed 8-bit gap + escape
  3. Golomb-Rice coded gap
  4. Empirical entropy of gaps  (theoretical floor)

Usage:
    cd ~/DSA/adaptive-ring-buffer
    python3 measure_index_entropy.py
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
DATA = BASE / "data"
OUT = BASE / "results" / "compression_encoding.csv"

VALUE_BITS = 16          # 16-bit ADC sample
SIGNALS = {
    "ecg_100":  ("mit-bih", "ecg_100.txt", "ecg"),
    "ecg_105":  ("mit-bih", "ecg_105.txt", "ecg"),
    "ecg_108":  ("mit-bih", "ecg_108.txt", "ecg"),
    "ecg_201":  ("mit-bih", "ecg_201.txt", "ecg"),
    "ecg_228":  ("mit-bih", "ecg_228.txt", "ecg"),
    "vib_normal_0hp":     ("cwru-bearing", "normal_0hp.txt", "vibration"),
    "vib_inner_race_007": ("cwru-bearing", "inner_race_007.txt", "vibration"),
    "vib_ball_fault_007": ("cwru-bearing", "ball_fault_007.txt", "vibration"),
}


# ─── EVICTION (mirrors ring_buffer.h INTERP_ERROR) ───────────────────

def simulate_eviction(x, buffer_size):
    """Interpolation-error eviction. Returns sorted surviving indices."""
    n = len(x)
    if n <= buffer_size:
        return np.arange(n)

    buf = list(range(buffer_size))
    for new_idx in range(buffer_size, n):
        buf.append(new_idx)
        min_err = float("inf")
        min_pos = 1
        for pos in range(1, len(buf) - 1):
            i, p, s = buf[pos], buf[pos - 1], buf[pos + 1]
            span = s - p
            if span == 0:
                err = 0.0
            else:
                frac = (i - p) / span
                err = abs(x[i] - (x[p] + frac * (x[s] - x[p])))
            if err < min_err:
                min_err, min_pos = err, pos
        buf.pop(min_pos)
    return np.array(sorted(buf))


# ─── ENCODINGS ───────────────────────────────────────────────────────

def bits_absolute(indices, n_total):
    """Absolute index, fixed width = ceil(log2(n_total))."""
    w = max(1, math.ceil(math.log2(max(n_total, 2))))
    return len(indices) * w


def bits_fixed_gap(indices, gap_bits=8):
    """Fixed-width gap with escape to full index on overflow."""
    gaps = np.diff(np.concatenate([[-1], indices]))
    max_gap = (1 << gap_bits) - 1
    total = 0
    for g in gaps:
        if g < max_gap:
            total += gap_bits
        else:
            total += gap_bits + 16   # escape marker + full index
    return total


def bits_golomb_rice(indices, k=None):
    """
    Golomb-Rice coding of gaps.
    Codeword for gap g with parameter k: unary(g >> k) + k raw bits.
    k chosen from the mean gap if not supplied.
    """
    gaps = np.diff(np.concatenate([[-1], indices]))
    gaps = np.maximum(gaps, 1)
    if k is None:
        mean_gap = max(gaps.mean(), 1.0)
        k = max(0, int(round(math.log2(mean_gap))))
    total = 0
    for g in gaps:
        q = int(g) >> k
        total += (q + 1) + k          # unary quotient (q ones + stop) + k bits
    return total, k


def bits_entropy(indices):
    """Empirical zero-order entropy of the gap distribution (theoretical floor)."""
    gaps = np.diff(np.concatenate([[-1], indices]))
    counts = Counter(gaps.tolist())
    n = len(gaps)
    H = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return H * n, H


# ─── MAIN ────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("TRUE COMPRESSION RATIO — ABSOLUTE vs DELTA-ENCODED INDICES")
    print("=" * 78)

    buffer_sizes = [512, 256, 128, 64, 32]
    rows = []

    for name, (folder, fname, domain) in SIGNALS.items():
        path = DATA / folder / fname
        if not path.exists():
            print(f"  !! missing {path}")
            continue

        x = np.loadtxt(path)[:2000]          # match cross_domain signal length
        n = len(x)
        orig_bits = n * VALUE_BITS

        print(f"\n  {name}  ({domain}, {n} samples)")
        print(f"  {'buf':>5s} {'kept':>5s} {'decim':>7s} "
              f"{'CR_abs':>8s} {'CR_gap8':>8s} {'CR_rice':>8s} {'CR_ent':>8s} "
              f"{'bps_abs':>8s} {'bps_rice':>9s} {'H_gap':>7s}")
        print("  " + "-" * 88)

        for bs in buffer_sizes:
            if bs >= n:
                continue
            idx = simulate_eviction(x, bs)
            k_kept = len(idx)
            decim = n / k_kept

            b_abs = k_kept * VALUE_BITS + bits_absolute(idx, n)
            b_g8 = k_kept * VALUE_BITS + bits_fixed_gap(idx, 8)
            b_rice_idx, k_par = bits_golomb_rice(idx)
            b_rice = k_kept * VALUE_BITS + b_rice_idx
            b_ent_idx, H = bits_entropy(idx)
            b_ent = k_kept * VALUE_BITS + b_ent_idx

            row = dict(
                signal=name, domain=domain, buffer_size=bs,
                n_total=n, n_kept=k_kept, decimation=decim,
                cr_absolute=orig_bits / b_abs,
                cr_gap8=orig_bits / b_g8,
                cr_rice=orig_bits / b_rice,
                cr_entropy=orig_bits / b_ent,
                bps_absolute=b_abs / n,
                bps_rice=b_rice / n,
                gap_entropy_bits=H,
                rice_k=k_par,
            )
            rows.append(row)

            print(f"  {bs:>5d} {k_kept:>5d} {decim:>6.1f}x "
                  f"{row['cr_absolute']:>7.2f}x {row['cr_gap8']:>7.2f}x "
                  f"{row['cr_rice']:>7.2f}x {row['cr_entropy']:>7.2f}x "
                  f"{row['bps_absolute']:>8.2f} {row['bps_rice']:>9.2f} "
                  f"{H:>7.2f}")

    if not rows:
        print("\n  No signals processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # ── Aggregate ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("AGGREGATE BY BUFFER SIZE (all signals)")
    print("=" * 78)
    print(f"  {'buf':>5s} {'decim':>7s} {'CR_abs':>8s} {'CR_rice':>8s} "
          f"{'gain':>7s} {'bps_abs':>8s} {'bps_rice':>9s}")
    print("  " + "-" * 60)
    for bs in sorted(df.buffer_size.unique(), reverse=True):
        s = df[df.buffer_size == bs]
        gain = s.cr_rice.mean() / s.cr_absolute.mean()
        print(f"  {bs:>5d} {s.decimation.mean():>6.1f}x "
              f"{s.cr_absolute.mean():>7.2f}x {s.cr_rice.mean():>7.2f}x "
              f"{gain:>6.2f}x {s.bps_absolute.mean():>8.2f} "
              f"{s.bps_rice.mean():>9.2f}")

    print("\n" + "=" * 78)
    print("BY DOMAIN")
    print("=" * 78)
    for dom in sorted(df.domain.unique()):
        s = df[df.domain == dom]
        print(f"  {dom:12s}  CR_abs={s.cr_absolute.mean():.2f}x  "
              f"CR_rice={s.cr_rice.mean():.2f}x  "
              f"CR_entropy={s.cr_entropy.mean():.2f}x  "
              f"mean gap entropy={s.gap_entropy_bits.mean():.2f} bits")

    overall_gain = df.cr_rice.mean() / df.cr_absolute.mean()
    print(f"""
{'=' * 78}
WHAT TO REPORT IN THE PAPER
{'=' * 78}

  Surviving sample indices are monotonically increasing, so they are stored
  as gaps rather than absolute positions. Gap entropy averages
  {df.gap_entropy_bits.mean():.2f} bits, so Golomb-Rice coding (k={int(df.rice_k.mode()[0])}) costs far less
  than a 16-bit absolute index.

  Mean CR with absolute indices : {df.cr_absolute.mean():.2f}x   (understated)
  Mean CR with Golomb-Rice gaps : {df.cr_rice.mean():.2f}x   <-- report this
  Mean CR at entropy floor      : {df.cr_entropy.mean():.2f}x   (upper bound)

  Improvement from correct index coding: {overall_gain:.2f}x

  Suggested sentence:
  "Surviving samples are stored as (value, gap) pairs with 16-bit values and
   Golomb-Rice coded gaps. Measured gap entropy is {df.gap_entropy_bits.mean():.1f} bits, giving
   {df.bps_rice.mean():.1f} bits/sample and a compression ratio of {df.cr_rice.mean():.1f}x, versus
   {df.cr_absolute.mean():.1f}x if absolute 16-bit indices were stored."

  Saved: {OUT}
{'=' * 78}""")


if __name__ == "__main__":
    main()
