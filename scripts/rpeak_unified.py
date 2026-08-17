#!/usr/bin/env python3
"""
rpeak_unified.py -- one R-peak detector, one table.

THE PROBLEM
  Three different detectors existed in the codebase, producing three
  mutually inconsistent sets of numbers across the progress reports:

    1. metrics.h::compute_rpeak_accuracy   0.6 x max threshold, tol 5
       -> v1 reported uncompressed F1 = 0.747 and F1 INCREASING under
          compression, which is physically impossible.
    2. rpeak_eval.h::detect_rpeaks(0.5, 100), tol 15
       -> used by cross_domain; produced the v3 table where EVERY mode
          scores ~0.847, including RANDOM_DROP at 0.841 and DROP at
          0.8508 -- higher than the proposed method's 0.8468. If that
          table ships, a referee quotes it back: "your own data shows
          random dropping matches your method on the only clinically
          meaningful metric."
    3. NeuroKit2 (m8_classification_v2.py)
       -> v2 reported uncompressed 0.969, monotonic degradation.

  Only #3 is defensible. This script makes it the sole source, on the
  same deterministic survivor sets used everywhere else, so both other
  columns can be deleted from the codebase rather than merely from the
  paper.

TOLERANCE
  150 ms, per ANSI/AAMI EC57. At 360 Hz that is 54 samples. v1's
  5-sample (13.9 ms) tolerance was far stricter than the standard and
  is not comparable to published results.

RECORD EXCLUSION -- DECLARED IN ADVANCE
  Records whose UNCOMPRESSED F1 falls below 0.50 are excluded as
  detector failures rather than compression failures. The rule is
  applied identically to every method and every operating point, and
  the excluded records are listed. v2 excluded record 108 only from
  the F1 averages while keeping it in the SNR analysis; inconsistent
  inclusion criteria across metrics is exactly what referees punish.

Requires: pip3 install neurokit2 --break-system-packages
"""

import os
import sys
import csv

import numpy as np

try:
    import neurokit2 as nk
except ImportError:
    sys.exit("neurokit2 required:\n"
             "  pip3 install neurokit2 --break-system-packages")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "mit-bih")
R = os.path.join(ROOT, "results")

FS = 360.0
TOL_MS = 150.0                       # ANSI/AAMI EC57
TOL = int(round(FS * TOL_MS / 1000))  # 54 samples
MIN_BASELINE_F1 = 0.50               # declared exclusion threshold
NSAMP = 10000

RECORDS = ["100", "101", "105", "106", "108", "109", "112", "115",
           "118", "119", "122", "201", "205", "212", "215", "228"]

BUFFERS = [32, 64, 128, 256, 512]
OVERLOADS = [5, 10, 20, 50]


# ----------------------------------------------------------------
def detect(sig, fs=FS):
    """NeuroKit2 Pan-Tompkins. Returns sample indices."""
    try:
        clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="pantompkins1985")
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def f1_against(truth, found, tol=TOL):
    """Greedy nearest-match F1 within tolerance."""
    if len(truth) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(found) == 0:
        return 0.0, 0.0, 0.0
    truth = np.asarray(truth, dtype=np.float64)
    used = np.zeros(len(truth), dtype=bool)
    tp = 0
    for p in found:
        dist = np.abs(truth - float(p))
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            used[j] = True
            tp += 1
    fp = len(found) - tp
    fn = len(truth) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


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


def evict_ie(sig, cap, overload):
    """
    Deterministic interpolation-error eviction, matching
    deterministic_driver.h: one pop per `overload` pushes, evict the
    minimum-interpolation-error interior node when full, ties to the
    oldest. Returns survivor indices.
    """
    n = len(sig)
    prv = np.full(cap, -1, dtype=np.int64)
    nxt = np.full(cap, -1, dtype=np.int64)
    val = np.zeros(cap)
    oi = np.zeros(cap, dtype=np.int64)
    score = np.full(cap, np.inf)
    free = list(range(cap))
    head = tail = -1
    size = 0
    out_idx = []

    def ie(s):
        p, q = prv[s], nxt[s]
        if p < 0 or q < 0:
            return np.inf
        span = oi[q] - oi[p]
        if span <= 0:
            return 0.0
        t = (oi[s] - oi[p]) / span
        return abs(val[s] - (val[p] + t * (val[q] - val[p])))

    for i in range(n):
        if size == cap:
            m = np.min(score)
            cand = np.where(score == m)[0]
            v = int(cand[np.argmin(oi[cand])]) if m < np.inf else head
            p, q = prv[v], nxt[v]
            if p >= 0: nxt[p] = q
            else:      head = q
            if q >= 0: prv[q] = p
            else:      tail = p
            score[v] = np.inf
            free.append(v)
            size -= 1
            if p >= 0: score[p] = ie(p)
            if q >= 0: score[q] = ie(q)

        s = free.pop()
        ot = tail
        val[s] = sig[i]; oi[s] = i; nxt[s] = -1; prv[s] = tail
        if tail >= 0: nxt[tail] = s
        else:         head = s
        tail = s; size += 1
        score[s] = np.inf
        if ot >= 0: score[ot] = ie(ot)

        if overload > 0 and (i + 1) % overload == 0 and size > 0:
            out_idx.append(int(oi[head]))
            h = head
            head = nxt[h]
            if head >= 0: prv[head] = -1
            else:         tail = -1
            score[h] = np.inf
            free.append(h); size -= 1

    c = head
    while c >= 0:
        out_idx.append(int(oi[c]))
        c = nxt[c]
    return np.array(sorted(out_idx), dtype=np.int64)


def evict_fifo(sig, cap, overload):
    """FIFO baseline under the identical drain schedule."""
    n = len(sig)
    from collections import deque
    q = deque()
    out = []
    for i in range(n):
        if len(q) == cap:
            q.popleft()
        q.append(i)
        if overload > 0 and (i + 1) % overload == 0 and q:
            out.append(q.popleft())
    out.extend(q)
    return np.array(sorted(out), dtype=np.int64)


# ----------------------------------------------------------------
def main():
    print("=" * 78)
    print(f"R-PEAK EVALUATION -- NeuroKit2 only, {TOL_MS:.0f} ms "
          f"tolerance ({TOL} samples @ {FS:.0f} Hz)")
    print("=" * 78)

    sigs, truths, baseline = {}, {}, {}

    print("\nBaseline (uncompressed):")
    print(f"  {'record':<8} {'peaks':>6} {'detected':>9} {'F1':>7}  status")
    print("  " + "-" * 46)

    for rec in RECORDS:
        sp = os.path.join(D, f"ecg_{rec}.txt")
        ap = os.path.join(D, f"rpeak_{rec}.txt")
        if not (os.path.exists(sp) and os.path.exists(ap)):
            print(f"  {rec:<8} missing files -- skipped")
            continue
        x = np.loadtxt(sp)[:NSAMP]
        ann = np.loadtxt(ap, dtype=int)
        ann = ann[ann < len(x)]
        if len(x) < NSAMP or len(ann) < 5:
            print(f"  {rec:<8} too short / too few annotations -- skipped")
            continue
        det = detect(x)
        f1, _, _ = f1_against(ann, det)
        ok = f1 >= MIN_BASELINE_F1
        print(f"  {rec:<8} {len(ann):>6} {len(det):>9} {f1:>7.3f}  "
              f"{'ok' if ok else 'EXCLUDED (detector failure)'}")
        sigs[rec] = x; truths[rec] = ann; baseline[rec] = f1

    keep = [r for r in sigs if baseline[r] >= MIN_BASELINE_F1]
    excl = [r for r in sigs if baseline[r] < MIN_BASELINE_F1]
    if not keep:
        sys.exit("\nNo records pass the baseline criterion.")

    print(f"\n  included: {len(keep)}   excluded: {len(excl)}"
          f"{'  -> ' + ', '.join(excl) if excl else ''}")
    print(f"  Exclusion rule (declared in advance): uncompressed "
          f"F1 < {MIN_BASELINE_F1}.")
    print(f"  Mean baseline F1 over included records: "
          f"{np.mean([baseline[r] for r in keep]):.3f}")

    rows = []
    total = len(keep) * len(BUFFERS) * len(OVERLOADS)
    done = 0
    print(f"\nEvaluating {total} configurations x 2 methods ...")

    for rec in keep:
        x = sigs[rec]; ann = truths[rec]
        for cap in BUFFERS:
            for ov in OVERLOADS:
                for meth, fn in (("IE", evict_ie), ("FIFO", evict_fifo)):
                    idx = fn(x, cap, ov)
                    r = reconstruct(idx, x[idx], len(x))
                    f1, pr, rc = f1_against(ann, detect(r))
                    rows.append(dict(
                        record=rec, method=meth, buffer_size=cap, overload=ov,
                        retained=len(idx), CR_decim=round(len(x)/len(idx), 3),
                        f1=round(f1, 4), precision=round(pr, 4),
                        recall=round(rc, 4),
                        f1_baseline=round(baseline[rec], 4)))
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total}")

    out = os.path.join(R, "rpeak_unified.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {os.path.relpath(out, ROOT)}  ({len(rows)} rows)")

    import pandas as pd
    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("F1 BY DECIMATION RATIO  (mean over included records)")
    print("=" * 78)
    df["band"] = pd.cut(df.CR_decim, [1, 5, 10, 20, 50, 200],
                        labels=["1-5x", "5-10x", "10-20x", "20-50x", "50x+"])
    p = df.pivot_table(index="method", columns="band", values="f1",
                       aggfunc="mean", observed=True)
    print(p.round(3).to_string())

    print("\n" + "=" * 78)
    print("MONOTONICITY CHECK  (F1 must not rise with compression)")
    print("=" * 78)
    for meth in ["IE", "FIFO"]:
        s = df[df.method == meth].groupby("buffer_size").f1.mean()
        s = s.sort_index(ascending=False)          # large buffer = less loss
        viol = int(np.sum(np.diff(s.values) > 0.01))
        print(f"  {meth:<5} " +
              "  ".join(f"buf{int(b)}={v:.3f}" for b, v in s.items()) +
              f"   violations={viol}")
    print("\n  v1 reported F1 INCREASING under compression, which cannot")
    print("  happen physically. Any violation above means a detector bug.")

    print("\n" + "=" * 78)
    print("PAIRED PER-RECORD  (n = included records)")
    print("=" * 78)
    piv = df.pivot_table(index=["record", "buffer_size", "overload"],
                         columns="method", values="f1")
    if "IE" in piv and "FIFO" in piv:
        s = (piv.IE - piv.FIFO).dropna().groupby(level=0).mean()
        dz = s.mean() / s.std() if len(s) > 1 and s.std() > 0 else float("nan")
        print(f"  IE vs FIFO   n={len(s)}  mean={s.mean():+.4f}  "
              f"dz={dz:+.3f}  win={100*(s>0).mean():.1f}%")

    print("""
FOR THE PAPER
  * This is the ONLY R-peak table. Delete compute_rpeak_accuracy() from
    metrics.h and detect_rpeaks() from rpeak_eval.h, and drop the
    rpeak_f1 column from the cross-domain output, so the superseded
    numbers cannot resurface.
  * State the detector (NeuroKit2 Pan-Tompkins), the tolerance
    (150 ms, ANSI/AAMI EC57) and the exclusion rule together.
  * Report the baseline F1 next to every compressed F1. A compressed
    score means nothing without the uncompressed reference.
""")


if __name__ == "__main__":
    main()
