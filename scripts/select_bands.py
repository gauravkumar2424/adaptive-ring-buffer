#!/usr/bin/env python3
"""
select_bands.py -- S6b: choose the demodulation band per recording by
spectral kurtosis, instead of assuming a fixed 2-5 kHz resonance.

WHY
  Envelope analysis demodulates a structural resonance excited by the
  fault impacts. WHICH resonance depends on the machine, the sensor
  mount, and the fault type. A fixed 2-5 kHz band is a common default
  for CWRU drive-end data but it is a guess, and the evidence says it
  is a bad guess for ball faults: with the fixed band, ball recordings
  had median PNR 7.45 dB and only 8/12 exceeded the 6 dB detection
  threshold, versus 100% for inner race. That produced a
  fault-type-biased exclusion list, which is exactly the kind of
  discovered-rather-than-declared exclusion a reviewer punishes.

METHOD (kurtogram, Antoni 2006)
  Sweep a grid of (centre frequency, bandwidth). Band-pass, take the
  Hilbert envelope, measure kurtosis. Impulsive content -- which is
  what a bearing fault produces -- maximises kurtosis. Pick the
  argmax.

WHY THIS IS NOT TUNING  (state this in the paper)
  Kurtosis is computed WITHOUT reference to the fault frequency, so
  the selector cannot favour any particular defect.
  The band is selected ONCE per recording, on the UNCOMPRESSED signal,
  and then FROZEN across every method and every compression ratio. No
  compression method influences its own band. Selecting per method or
  per CR would let the metric adapt to the compression and would be
  indefensible.

Output: data/cwru-bearing/bands.csv
"""

import os
import sys
import csv

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, hilbert
    from scipy.stats import kurtosis
except ImportError:
    sys.exit("scipy required:  pip3 install scipy --break-system-packages")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "cwru-bearing")

WINDOW = 12000
FS = 12000.0
NYQ = FS / 2.0

# Grid. Kept coarse on purpose: a fine grid over 12,000 samples would
# overfit the noise realisation rather than find the resonance.
CENTRES = np.arange(500, 5501, 250)
WIDTHS = [500, 1000, 1500, 2000, 3000]

FIXED_BAND = (2000.0, 5000.0)   # the previous assumption, for comparison
TARGET_BY_COND = {"inner_race": "BPFI_hz",
                  "outer_race": "BPFO_hz",
                  "ball": "BSF_hz"}


def bandpass(x, lo, hi, fs=FS):
    """SOS form: numerically stable for narrow low-frequency bands."""
    lo = max(lo, 1.0)
    hi = min(hi, NYQ * 0.99)
    if hi <= lo:
        return None
    sos = butter(4, [lo / NYQ, hi / NYQ], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def env_of(x, lo, hi):
    y = bandpass(x, lo, hi)
    if y is None:
        return None
    return np.abs(hilbert(y))


def env_spectrum(env, fs=FS):
    e = env - env.mean()
    w = np.hanning(len(e))
    mag = np.abs(np.fft.rfft(e * w)) * (2.0 / w.sum())
    return np.fft.rfftfreq(len(e), 1.0 / fs), mag


def peak_pnr(freqs, mag, f0, tol=0.05):
    band = (freqs >= f0 * (1 - tol)) & (freqs <= f0 * (1 + tol))
    if not band.any():
        return np.nan, np.nan
    idx = np.where(band)[0]
    k = idx[np.argmax(mag[idx])]
    wide = (freqs >= f0 * 0.5) & (freqs <= f0 * 1.5)
    floor_bins = mag[wide & ~band]
    if floor_bins.size == 0:
        return freqs[k], np.nan
    floor = np.median(floor_bins)
    pnr = 20 * np.log10(mag[k] / floor) if floor > 0 else np.nan
    return freqs[k], pnr


def main():
    man = list(csv.DictReader(open(f"{D}/manifest.csv")))
    rows = []

    print("Kurtogram band selection (blind to fault frequency)\n")
    print(f"  {'recording':<14} {'cond':<11} {'band (Hz)':<16} "
          f"{'kurt':>6}  {'PNR fixed':>10} {'PNR kurt':>9}  delta")
    print("  " + "-" * 78)

    for m in man:
        name, cond = m["name"], m["condition"]
        path = f"{D}/vib_{name}.txt"
        if not os.path.exists(path):
            continue
        x = np.loadtxt(path, max_rows=WINDOW)
        if len(x) < WINDOW:
            print(f"  {name:<14} SKIP (only {len(x)} samples)")
            continue

        # ---- blind band search ----
        best = (-np.inf, None)
        for w in WIDTHS:
            for c in CENTRES:
                lo, hi = c - w / 2.0, c + w / 2.0
                if lo < 100 or hi > NYQ * 0.99:
                    continue
                e = env_of(x, lo, hi)
                if e is None:
                    continue
                k = kurtosis(e, fisher=True)
                if np.isfinite(k) and k > best[0]:
                    best = (k, (lo, hi))
        if best[1] is None:
            print(f"  {name:<14} SKIP (no valid band)")
            continue
        klo, khi = best[1]
        kurt = best[0]

        # ---- PNR under each band (normal has no defect frequency) ----
        key = TARGET_BY_COND.get(cond)
        if key is None:
            pnr_fix = pnr_kur = np.nan
            f0 = np.nan
        else:
            f0 = float(m[key])
            e = env_of(x, *FIXED_BAND)
            fr, mg = env_spectrum(e)
            _, pnr_fix = peak_pnr(fr, mg, f0)
            e = env_of(x, klo, khi)
            fr, mg = env_spectrum(e)
            _, pnr_kur = peak_pnr(fr, mg, f0)

        rows.append(dict(name=name, condition=cond, load_hp=m["load_hp"],
                         band_lo_hz=round(klo, 1), band_hi_hz=round(khi, 1),
                         kurtosis=round(kurt, 4),
                         f_target_hz=round(f0, 3) if np.isfinite(f0) else "",
                         pnr_fixed_db=round(pnr_fix, 3) if np.isfinite(pnr_fix) else "",
                         pnr_kurt_db=round(pnr_kur, 3) if np.isfinite(pnr_kur) else ""))

        d = (pnr_kur - pnr_fix) if np.isfinite(pnr_kur) and np.isfinite(pnr_fix) else np.nan
        print(f"  {name:<14} {cond:<11} {f'{klo:.0f}-{khi:.0f}':<16} "
              f"{kurt:>6.2f}  "
              f"{pnr_fix:>10.2f} {pnr_kur:>9.2f}  "
              f"{d:+6.2f}" if np.isfinite(d) else
              f"  {name:<14} {cond:<11} {f'{klo:.0f}-{khi:.0f}':<16} "
              f"{kurt:>6.2f}  {'n/a':>10} {'n/a':>9}")

    out = f"{D}/bands.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} bands -> {os.path.relpath(out, ROOT)}")

    # ---------------- did it help? ----------------
    fault = [r for r in rows if r["pnr_kurt_db"] != ""]
    print("\n" + "=" * 74)
    print("DETECTABILITY AT BASELINE  (uncompressed, threshold 6 dB)")
    print("=" * 74)
    for cname in ["inner_race", "ball", "outer_race"]:
        sub = [r for r in fault if r["condition"] == cname]
        if not sub:
            continue
        pf = np.array([r["pnr_fixed_db"] for r in sub], float)
        pk = np.array([r["pnr_kurt_db"] for r in sub], float)
        print(f"  {cname:<12} n={len(sub):2d}   "
              f"fixed: median {np.median(pf):6.2f} dB, {100*(pf>=6).mean():5.1f}% pass   "
              f"kurt: median {np.median(pk):6.2f} dB, {100*(pk>=6).mean():5.1f}% pass")

    pf = np.array([r["pnr_fixed_db"] for r in fault], float)
    pk = np.array([r["pnr_kurt_db"] for r in fault], float)
    print(f"\n  overall  fixed {int((pf>=6).sum())}/{len(pf)} pass   "
          f"kurt {int((pk>=6).sum())}/{len(pk)} pass")
    print(f"  recordings rescued (fail fixed, pass kurt): "
          f"{int(((pf<6)&(pk>=6)).sum())}")
    print(f"  recordings lost    (pass fixed, fail kurt): "
          f"{int(((pf>=6)&(pk<6)).sum())}")

    print("\n  Selected bands by condition (median centre):")
    for cname in ["normal", "inner_race", "ball", "outer_race"]:
        sub = [r for r in rows if r["condition"] == cname]
        if sub:
            ctr = np.median([(r["band_lo_hz"] + r["band_hi_hz"]) / 2 for r in sub])
            bw = np.median([r["band_hi_hz"] - r["band_lo_hz"] for r in sub])
            print(f"    {cname:<12} centre {ctr:6.0f} Hz, width {bw:5.0f} Hz")
    print("""
  If ball faults were rescued, the fixed 2-5 kHz band was the problem
  and the exclusion list shrinks. If they still fail, the ball-fault
  impulses are genuinely weak in these recordings and the exclusion is
  a property of the data -- declare it and move on.
""")


if __name__ == "__main__":
    main()
