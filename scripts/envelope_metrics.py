#!/usr/bin/env python3
"""
envelope_metrics.py -- S6: replace naive DFT magnitude correlation with
the metric a rotating-machinery engineer actually uses.

THE PROBLEM WITH THE OLD METRIC
  spectral_corr() in the C++ was a naive O(N^2) DFT over the FIRST 512
  of 2000 samples, no window (so leakage-dominated), correlated across
  all magnitude bins including DC. "0.979 vs 0.962" was not tied to any
  quantity a practitioner cares about, and the sentinel filtering
  silently dropped 37% of pairs.

THE REPLACEMENT
  Bearing faults produce periodic impacts that AMPLITUDE-MODULATE a
  high-frequency structural resonance. The diagnostic procedure is:

     band-pass around the resonance
       -> Hilbert envelope (demodulate)
         -> windowed FFT of the envelope
           -> look for a peak at the characteristic defect frequency

  We therefore ask a question with a right answer: after compression,
  is the fault still diagnosable?

     freq_error_hz   how far the detected peak moved
     amp_error_pct   how much the peak amplitude changed
     pnr_db          peak-to-noise ratio (is it still above the floor)
     diagnosable     freq_error <= 1 Hz AND pnr >= 6 dB

  A threshold that means something, rather than a correlation we chose
  and won.

GEOMETRY  (SKF 6205-2RS JEM, drive end)
  BPFI 5.4152 x shaft, BPFO 3.5848 x, BSF 2.3568 x, FTF 0.3983 x.
  Precomputed per recording in manifest.csv. Re-derive from the bearing
  spec before these go in the paper.

Usage:  python3 scripts/envelope_metrics.py
"""

import os
import sys
import csv

import numpy as np

try:
    from scipy.signal import hilbert, butter, filtfilt
except ImportError:
    sys.exit("scipy required:  pip3 install scipy --break-system-packages")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "cwru-bearing")
R = os.path.join(ROOT, "results")

WINDOW = 12000          # must match spectral_eval.cpp
FS = 12000.0
SEARCH_TOL = 0.05       # +-5% band around the nominal defect frequency
FREQ_OK_HZ = 1.0        # diagnosable if the peak moved <= this
PNR_OK_DB = 6.0         # ...and stands this far above the local floor
BAND = (2000.0, 5000.0) # resonance band for demodulation

TARGET_BY_COND = {
    "inner_race": "BPFI_hz",
    "outer_race": "BPFO_hz",
    "ball":       "BSF_hz",
}


def envelope_spectrum(x, fs=FS):
    """Band-pass -> Hilbert envelope -> Hann-windowed rFFT."""
    ny = fs / 2.0
    lo, hi = BAND[0] / ny, min(BAND[1] / ny, 0.99)
    b, a = butter(4, [lo, hi], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, x)))
    env = env - env.mean()                      # kill DC
    w = np.hanning(len(env))
    mag = np.abs(np.fft.rfft(env * w)) * (2.0 / w.sum())
    freqs = np.fft.rfftfreq(len(env), 1.0 / fs)
    return freqs, mag


def peak_near(freqs, mag, f0, tol=SEARCH_TOL):
    """Peak within +-tol of f0, plus its dB ratio to the local floor."""
    band = (freqs >= f0 * (1 - tol)) & (freqs <= f0 * (1 + tol))
    if not band.any():
        return np.nan, np.nan, np.nan
    idx = np.where(band)[0]
    k = idx[np.argmax(mag[idx])]

    # Local floor: median of a wider neighbourhood, excluding the peak band.
    wide = (freqs >= f0 * 0.5) & (freqs <= f0 * 1.5)
    floor_bins = mag[wide & ~band]
    floor = np.median(floor_bins) if floor_bins.size else np.nan
    pnr = 20 * np.log10(mag[k] / floor) if floor and floor > 0 else np.nan
    return freqs[k], mag[k], pnr


def reconstruct(idx, vals, n):
    """Linear interpolation between survivors; edges held flat."""
    out = np.empty(n)
    if len(idx) == 0:
        return np.zeros(n)
    if len(idx) == 1:
        out[:] = vals[0]
        return out
    out[: idx[0]] = vals[0]
    out[idx[-1]:] = vals[-1]
    out[idx[0]: idx[-1] + 1] = np.interp(
        np.arange(idx[0], idx[-1] + 1), idx, vals)
    return out


def main():
    man = {r["name"]: r for r in csv.DictReader(open(f"{D}/manifest.csv"))}
    surv_path = f"{R}/spectral_survivors.csv"
    if not os.path.exists(surv_path):
        sys.exit(f"missing {surv_path} -- run build/spectral_eval first")

    # Cache originals and their reference peaks (computed once per recording).
    sig_cache, ref_cache = {}, {}

    def get_sig(name):
        if name not in sig_cache:
            x = np.loadtxt(f"{D}/vib_{name}.txt", max_rows=WINDOW)
            sig_cache[name] = x
        return sig_cache[name]

    def get_ref(name):
        if name not in ref_cache:
            m = man[name]
            cond = m["condition"]
            key = TARGET_BY_COND.get(cond)
            if key is None:                      # normal: no defect frequency
                ref_cache[name] = None
            else:
                f0 = float(m[key])
                fr, mg = envelope_spectrum(get_sig(name))
                fpk, apk, pnr = peak_near(fr, mg, f0)
                ref_cache[name] = dict(f0=f0, f_orig=fpk, a_orig=apk,
                                       pnr_orig=pnr, cond=cond,
                                       load=int(m["load_hp"]),
                                       diam=float(m["diameter_in"]))
        return ref_cache[name]

    rows, skipped = [], 0
    with open(surv_path) as f:
        for rec in csv.DictReader(f):
            name = rec["signal"]
            ref = get_ref(name)
            if ref is None:                      # normal baseline
                skipped += 1
                continue

            deltas = np.fromstring(rec["idx_deltas"], sep=" ", dtype=np.int64)
            idx = np.cumsum(deltas)
            x = get_sig(name)
            vals = x[idx]
            recon = reconstruct(idx, vals, len(x))

            fr, mg = envelope_spectrum(recon)
            fpk, apk, pnr = peak_near(fr, mg, ref["f0"])

            ferr = abs(fpk - ref["f_orig"]) if np.isfinite(fpk) else np.nan
            aerr = (100 * abs(apk - ref["a_orig"]) / ref["a_orig"]
                    if ref["a_orig"] else np.nan)
            diag = bool(np.isfinite(ferr) and ferr <= FREQ_OK_HZ
                        and np.isfinite(pnr) and pnr >= PNR_OK_DB)

            rows.append(dict(
                signal=name, condition=ref["cond"], load=ref["load"],
                diameter=ref["diam"], mode=rec["mode"],
                buffer_size=int(rec["buffer_size"]),
                overload=int(rec["overload"]),
                retained=int(rec["retained"]),
                CR=round(len(x) / int(rec["retained"]), 3),
                f_nominal=round(ref["f0"], 3),
                f_orig=round(ref["f_orig"], 3) if np.isfinite(ref["f_orig"]) else "",
                f_recon=round(fpk, 3) if np.isfinite(fpk) else "",
                freq_error_hz=round(ferr, 4) if np.isfinite(ferr) else "",
                amp_error_pct=round(aerr, 3) if np.isfinite(aerr) else "",
                pnr_db=round(pnr, 3) if np.isfinite(pnr) else "",
                pnr_orig_db=round(ref["pnr_orig"], 3) if np.isfinite(ref["pnr_orig"]) else "",
                diagnosable=int(diag),
            ))

    if not rows:
        sys.exit("no rows produced")

    out = f"{R}/envelope_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {os.path.relpath(out, ROOT)}"
          f"   (skipped {skipped} normal-baseline runs: no defect frequency)")

    # ---------------- summary ----------------
    import collections
    print("\n" + "=" * 72)
    print("DIAGNOSABILITY BY MODE AND COMPRESSION RATIO")
    print("=" * 72)

    bands = [(1, 5), (5, 10), (10, 20), (20, 40)]
    modes = sorted({r["mode"] for r in rows})
    hdr = "  mode       " + "".join(f"{f'CR {lo}-{hi}x':>12}" for lo, hi in bands)
    print(hdr)
    for m in modes:
        line = f"  {m:<11}"
        for lo, hi in bands:
            s = [r for r in rows if r["mode"] == m and lo <= r["CR"] < hi]
            line += (f"{100*np.mean([r['diagnosable'] for r in s]):>11.1f}%"
                     if s else f"{'-':>12}")
        print(line)

    print("\n" + "=" * 72)
    print("MEDIAN FAULT-FREQUENCY ERROR (Hz)")
    print("=" * 72)
    print(hdr)
    for m in modes:
        line = f"  {m:<11}"
        for lo, hi in bands:
            e = [r["freq_error_hz"] for r in rows
                 if r["mode"] == m and lo <= r["CR"] < hi
                 and r["freq_error_hz"] != ""]
            line += f"{np.median(e):>12.2f}" if e else f"{'-':>12}"
        print(line)

    print("\n" + "=" * 72)
    print("PAIRED, PER-RECORDING (n = independent recordings)")
    print("=" * 72)
    key = lambda r: (r["signal"], r["buffer_size"], r["overload"])
    by = collections.defaultdict(dict)
    for r in rows:
        by[key(r)][r["mode"]] = r

    for a, b in [("IE", "IE_ORACLE"), ("IE", "VW"),
                 ("IE", "LTTB"), ("IE", "DROP")]:
        per = collections.defaultdict(list)
        for k, d in by.items():
            if a in d and b in d:
                per[k[0]].append(d[a]["diagnosable"] - d[b]["diagnosable"])
        if not per:
            continue
        s = np.array([np.mean(v) for v in per.values()])
        dz = s.mean() / s.std(ddof=1) if len(s) > 1 and s.std(ddof=1) > 0 else float("nan")
        print(f"  {a} vs {b:<10} diagnosability  n={len(s):3d} "
              f"mean={s.mean():+.4f} dz={dz:+.3f} win={100*(s>0).mean():5.1f}%")

    print(f"""
  Criterion: freq_error <= {FREQ_OK_HZ} Hz AND pnr >= {PNR_OK_DB} dB.
  This is an INDEPENDENT metric -- no method here optimises envelope
  fidelity. IE vs IE_ORACLE is the systems claim: does the bounded
  buffer beat unconstrained greedy optimisation of the same criterion?
""")


if __name__ == "__main__":
    main()
