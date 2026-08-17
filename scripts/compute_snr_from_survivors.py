#!/usr/bin/env python3
"""
compute_snr_from_survivors.py -- put every figure number in one file.

WHY THIS EXISTS
  The first version of the thesis figure took SNR for SDT from
  memory_contract.csv (4,000-sample windows), SNR for the other methods
  from cross_domain_v4.csv (2,000-sample windows), and envelope PNR from
  envelope_metrics.csv (12,000-sample windows). Three experiments, three
  window lengths, one scatter plot. That is not defensible, and a
  referee who checks would be right to ask.

  spectral_survivors.csv already contains the survivor INDICES for all
  seven methods -- including SDT -- on identical 12,000-sample vibration
  windows from all 40 CWRU recordings. Survivor values are exactly
  sig[idx], so SNR is recoverable exactly from the indices. Nothing new
  needs to be run.

  This script recomputes SNR from those indices and merges it with the
  envelope metrics already derived from the same file. Every number in
  the figure then comes from one experiment on one set of windows.

METHOD
  reconstruct: linear interpolation between consecutive survivors, edges
  held flat -- identical to reconstruct_signal() in metrics.h and to the
  reconstruction used by envelope_metrics.py.
  SNR = 10*log10(sum(x^2) / sum((x - x_hat)^2)).

OUTPUT
  results/figure_data.csv   one row per (signal, mode, buffer, overload)
                            with retained, CR, snr_db, pnr_loss,
                            amp_error_pct, diagnosable
"""

import os
import sys
import csv

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
D = os.path.join(ROOT, "data", "cwru-bearing")
WINDOW = 12000


def reconstruct(idx, vals, n):
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


def snr_db(x, r):
    sp = float(np.sum(x * x))
    npow = float(np.sum((x - r) ** 2))
    if sp < 1e-15:
        return 0.0
    if npow < 1e-15:
        return float("inf")
    return 10.0 * np.log10(sp / npow)


def main():
    surv = os.path.join(R, "spectral_survivors.csv")
    env = os.path.join(R, "envelope_metrics.csv")
    if not os.path.exists(surv):
        sys.exit(f"missing {surv} -- run build/spectral_eval_v2 first")

    # Envelope metrics, keyed for merging.
    envmap = {}
    if os.path.exists(env):
        with open(env) as f:
            for r in csv.DictReader(f):
                k = (r["signal"], r["mode"], int(r["buffer_size"]),
                     int(r["overload"]))
                envmap[k] = r
    else:
        print("  note: envelope_metrics.csv not found; PNR columns blank")

    cache = {}

    def sig_of(name):
        if name not in cache:
            p = os.path.join(D, f"vib_{name}.txt")
            cache[name] = np.loadtxt(p, max_rows=WINDOW)
        return cache[name]

    rows = []
    with open(surv) as f:
        for rec in csv.DictReader(f):
            name = rec["signal"]
            deltas = np.fromstring(rec["idx_deltas"], sep=" ", dtype=np.int64)
            if len(deltas) < 2:
                continue
            idx = np.cumsum(deltas)
            x = sig_of(name)
            idx = idx[idx < len(x)]
            r = reconstruct(idx, x[idx], len(x))
            s = snr_db(x, r)

            k = (name, rec["mode"], int(rec["buffer_size"]),
                 int(rec["overload"]))
            e = envmap.get(k, {})

            def num(key):
                v = e.get(key, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")

            pnr_orig, pnr = num("pnr_orig_db"), num("pnr_db")
            rows.append(dict(
                signal=name,
                condition=e.get("condition", ""),
                load=e.get("load", ""),
                mode=rec["mode"],
                buffer_size=int(rec["buffer_size"]),
                overload=int(rec["overload"]),
                retained=len(idx),
                CR=round(len(x) / max(len(idx), 1), 4),
                snr_db=round(s, 4) if np.isfinite(s) else "",
                snr_saturated=0 if np.isfinite(s) else 1,
                pnr_orig_db=e.get("pnr_orig_db", ""),
                pnr_db=e.get("pnr_db", ""),
                pnr_loss=(round(pnr_orig - pnr, 4)
                          if np.isfinite(pnr_orig) and np.isfinite(pnr) else ""),
                amp_error_pct=e.get("amp_error_pct", ""),
                diagnosable=e.get("diagnosable", ""),
            ))

    if not rows:
        sys.exit("no rows produced")

    out = os.path.join(R, "figure_data.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {os.path.relpath(out, ROOT)}  ({len(rows)} rows)\n")

    # ---------------- verification ----------------
    import pandas as pd
    d = pd.read_csv(out)
    d = d[d.snr_saturated == 0]

    print("=" * 70)
    print("MEAN SNR BY MODE  (all 40 recordings, 12,000-sample windows)")
    print("=" * 70)
    print(d.groupby("mode").snr_db.agg(["mean", "count"]).round(2).to_string())

    print("\n" + "=" * 70)
    print("PAIRED IE vs SDT, per recording -- THE DISSOCIATION")
    print("=" * 70)
    key = ["signal", "buffer_size", "overload"]
    p = d.pivot_table(index=key, columns="mode", values="snr_db")
    if "IE" in p and "SDT" in p:
        s = (p["IE"] - p["SDT"]).dropna().groupby(level=0).mean()
        ci = 1.96 * s.std() / np.sqrt(len(s))
        print(f"  SNR         IE - SDT = {s.mean():+.3f} dB  "
              f"95% CI [{s.mean()-ci:+.3f}, {s.mean()+ci:+.3f}]  "
              f"n={len(s)}  IE wins {100*(s>0).mean():.1f}%")

    dp = pd.read_csv(out)
    dp["pnr_loss"] = pd.to_numeric(dp.pnr_loss, errors="coerce")
    dp["pnr_orig_db"] = pd.to_numeric(dp.pnr_orig_db, errors="coerce")
    base = dp.groupby("signal").pnr_orig_db.first()
    ok = set(base[base >= 6].index)
    dv = dp[dp.signal.isin(ok)]
    q = dv.pivot_table(index=key, columns="mode", values="pnr_loss")
    if "IE" in q and "SDT" in q:
        s2 = (q["SDT"] - q["IE"]).dropna().groupby(level=0).mean()
        ci2 = 1.96 * s2.std() / np.sqrt(len(s2))
        print(f"  PNR loss    SDT - IE = {s2.mean():+.3f} dB  "
              f"95% CI [{s2.mean()-ci2:+.3f}, {s2.mean()+ci2:+.3f}]  "
              f"n={len(s2)}  IE better {100*(s2>0).mean():.1f}%")

    print("""
  If the first line is NEGATIVE and the second POSITIVE, the
  dissociation is confirmed on a single experiment: SDT reconstructs
  more accurately pointwise, yet loses more of the fault signature.
  Both numbers now come from the same file, the same windows, and the
  same recordings.
""")


if __name__ == "__main__":
    main()
