#!/usr/bin/env python3
"""
rpeak_sdt.py -- does the dissociation hold on ECG?

WHY THIS EXISTS
  rpeak_unified.py compared interpolation-error (IE) eviction against
  FIFO and found F1 held at 0.973 through 50x while FIFO fell to 0.486.
  That result does NOT test this paper's thesis. The thesis is that
  pointwise and diagnostic fidelity DIVERGE, and the divergence is
  between IE and SDT -- the method that wins on SNR. Showing IE beats
  FIFO shows only that importance-based eviction beats dropping the
  oldest sample, which nobody disputes.

  This script adds SDT to the same NeuroKit2 pipeline, at a matched
  sample budget, and reports:

    (a) SNR:  IE - SDT   (expect NEGATIVE, as on vibration)
    (b) F1:   IE - SDT   (POSITIVE => dissociation generalises to ECG;
                          ZERO     => the effect is vibration-specific
                                      and the paper must say so)

  Either outcome is publishable. Only one is currently claimed, and it
  is claimed without evidence.

MATCHING
  epsilon is binary-searched per (record, buffer, overload) so SDT emits
  the same number of samples the buffer retains. Both receive the same
  drain schedule: one pop per `overload` pushes.

OUTPUT
  results/rpeak_sdt.csv
"""

import os
import sys
import csv
from collections import deque

import numpy as np

try:
    import neurokit2 as nk
except ImportError:
    sys.exit("pip3 install neurokit2 --break-system-packages")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "mit-bih")
R = os.path.join(ROOT, "results")

FS = 360.0
TOL = int(round(FS * 0.150))     # 150 ms, ANSI/AAMI EC57 -> 54 samples
MIN_BASE_F1 = 0.50               # declared exclusion, as in rpeak_unified.py
NSAMP = 10000
RECORDS = ["100", "101", "105", "106", "108", "109", "112", "115",
           "118", "119", "122", "201", "205", "212", "215", "228"]
BUFFERS = [32, 64, 128, 256, 512]
OVERLOADS = [5, 10, 20, 50]


# ---------------- detection and scoring ----------------
def detect(sig):
    try:
        clean = nk.ecg_clean(sig, sampling_rate=FS, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=FS,
                               method="pantompkins1985")
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def f1_against(truth, found, tol=TOL):
    truth = np.asarray(truth, dtype=np.float64)
    if len(truth) == 0:
        return float("nan")
    if len(found) == 0:
        return 0.0
    used = np.zeros(len(truth), dtype=bool)
    tp = 0
    for p in found:
        dist = np.abs(truth - float(p))
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            used[j] = True
            tp += 1
    fp, fn = len(found) - tp, len(truth) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


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
    sp, npow = float(np.sum(x * x)), float(np.sum((x - r) ** 2))
    if sp < 1e-15:
        return 0.0
    if npow < 1e-15:
        return float("inf")
    return 10.0 * np.log10(sp / npow)


# ---------------- eviction policies ----------------
def evict_ie(sig, cap, overload):
    """Deterministic min-interpolation-error eviction, ties to oldest."""
    n = len(sig)
    prv = np.full(cap, -1, dtype=np.int64)
    nxt = np.full(cap, -1, dtype=np.int64)
    val = np.zeros(cap)
    oi = np.zeros(cap, dtype=np.int64)
    score = np.full(cap, np.inf)
    free = list(range(cap))
    head = tail = -1
    size = 0
    out = []

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
            free.append(v); size -= 1
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
            out.append(int(oi[head]))
            h = head
            head = nxt[h]
            if head >= 0: prv[head] = -1
            else:         tail = -1
            score[h] = np.inf
            free.append(h); size -= 1

    c = head
    while c >= 0:
        out.append(int(oi[c])); c = nxt[c]
    return np.array(sorted(out), dtype=np.int64)


def evict_fifo(sig, cap, overload):
    q, out = deque(), []
    for i in range(len(sig)):
        if len(q) == cap:
            q.popleft()
        q.append(i)
        if overload > 0 and (i + 1) % overload == 0 and q:
            out.append(q.popleft())
    out.extend(q)
    return np.array(sorted(out), dtype=np.int64)


def sdt_stream(sig, eps, cap, overload):
    """
    Swinging-door trending with an explicit output queue drained on the
    same schedule the buffer sees. cap=0 leaves the queue unbounded
    (the charitable case); otherwise it is capped and overflow drops the
    oldest queued point.
    """
    n = len(sig)
    q = deque()
    out = []
    emitted = 0
    last_i, last_v = -1, 0.0
    prev_i, prev_v = -1, 0.0
    shi, slo = 1e30, -1e30

    def push(i, v):
        nonlocal emitted
        if cap > 0 and len(q) >= cap:
            q.popleft()
        q.append(i)
        emitted += 1

    for i in range(n):
        v = sig[i]
        if last_i < 0:
            push(i, v); last_i, last_v = i, v
            shi, slo = 1e30, -1e30
        else:
            dt = float(i - last_i)
            dv = v - last_v
            su, sl = (dv + eps) / dt, (dv - eps) / dt
            shi = min(shi, su)
            slo = max(slo, sl)
            if slo > shi and prev_i >= 0:
                push(prev_i, prev_v)
                last_i, last_v = prev_i, prev_v
                d2 = float(i - last_i)
                if d2 > 0:
                    dv2 = v - last_v
                    shi, slo = (dv2 + eps) / d2, (dv2 - eps) / d2
                else:
                    shi, slo = 1e30, -1e30
        prev_i, prev_v = i, v

        if overload > 0 and (i + 1) % overload == 0 and q:
            out.append(q.popleft())

    if prev_i >= 0 and (not q or q[-1] != prev_i):
        push(prev_i, prev_v)
    out.extend(q)
    return np.array(sorted(set(out)), dtype=np.int64), emitted


def tune_eps(sig, target, overload):
    lo, hi = 1e-9, 1.0
    for _ in range(60):
        if sdt_stream(sig, hi, 0, overload)[1] <= target:
            break
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if sdt_stream(sig, mid, 0, overload)[1] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- main ----------------
def main():
    print("=" * 74)
    print("DOES THE DISSOCIATION HOLD ON ECG?  IE vs SDT, matched budget")
    print("=" * 74)

    sigs, truths, base = {}, {}, {}
    for rec in RECORDS:
        sp = os.path.join(D, f"ecg_{rec}.txt")
        ap = os.path.join(D, f"rpeak_{rec}.txt")
        if not (os.path.exists(sp) and os.path.exists(ap)):
            continue
        x = np.loadtxt(sp)[:NSAMP]
        ann = np.loadtxt(ap, dtype=int)
        ann = ann[ann < len(x)]
        if len(x) < NSAMP or len(ann) < 5:
            continue
        f1 = f1_against(ann, detect(x))
        base[rec] = f1
        if f1 >= MIN_BASE_F1:
            sigs[rec], truths[rec] = x, ann

    excl = [r for r in base if base[r] < MIN_BASE_F1]
    print(f"\nrecords: {len(base)}   included: {len(sigs)}   "
          f"excluded: {excl}   mean baseline F1 = "
          f"{np.mean([base[r] for r in sigs]):.3f}")

    rows = []
    total = len(sigs) * len(BUFFERS) * len(OVERLOADS)
    done = 0
    print(f"\nevaluating {total} configurations x 3 methods ...")

    for rec, x in sigs.items():
        ann = truths[rec]
        for cap in BUFFERS:
            for ov in OVERLOADS:
                ie_idx = evict_ie(x, cap, ov)
                target = len(ie_idx)
                eps = tune_eps(x, target, ov)
                sdt_idx, _ = sdt_stream(x, eps, 0, ov)
                fifo_idx = evict_fifo(x, cap, ov)

                for meth, idx in (("IE", ie_idx), ("SDT", sdt_idx),
                                  ("FIFO", fifo_idx)):
                    if len(idx) == 0:
                        continue
                    r = reconstruct(idx, x[idx], len(x))
                    rows.append(dict(
                        record=rec, method=meth, buffer_size=cap,
                        overload=ov, retained=len(idx),
                        CR=round(len(x) / len(idx), 3),
                        snr_db=round(snr_db(x, r), 4),
                        f1=round(f1_against(ann, detect(r)), 4),
                        f1_baseline=round(base[rec], 4)))
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total}")

    out = os.path.join(R, "rpeak_sdt.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {os.path.relpath(out, ROOT)}  ({len(rows)} rows)")

    import pandas as pd
    d = pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print("BUDGET MATCH  (median retained samples; must agree)")
    print("=" * 74)
    print(d.groupby("method").retained.median().round(1).to_string())

    print("\n" + "=" * 74)
    print("PAIRED PER RECORD, n = included records")
    print("=" * 74)
    key = ["record", "buffer_size", "overload"]
    for val, lab in [("snr_db", "SNR (dB)"), ("f1", "R-peak F1")]:
        p = d.pivot_table(index=key, columns="method", values=val)
        print(f"\n  -- {lab} --")
        for b in ["SDT", "FIFO"]:
            if "IE" in p and b in p:
                s = (p["IE"] - p[b]).dropna().groupby(level=0).mean()
                ci = 1.96 * s.std() / np.sqrt(len(s))
                print(f"    IE - {b:<5} n={len(s):3d} mean={s.mean():+8.4f} "
                      f"95% CI [{s.mean()-ci:+.4f}, {s.mean()+ci:+.4f}] "
                      f"IE better {100*(s>0).mean():5.1f}%")

    print("\n" + "=" * 74)
    print("F1 BY DECIMATION BAND")
    print("=" * 74)
    d["band"] = pd.cut(d.CR, [1, 5, 10, 20, 50, 200],
                       labels=["1-5x", "5-10x", "10-20x", "20-50x", "50x+"])
    print(d.pivot_table(index="method", columns="band", values="f1",
                        aggfunc="mean", observed=True).round(3).to_string())

    print("""
READ THIS
  If IE - SDT on SNR is NEGATIVE and on F1 is POSITIVE, the
  dissociation generalises to ECG and the paper's second-domain claim
  is earned.
  If IE - SDT on F1 is ~0, SDT preserves R-peaks as well as IE and the
  dissociation is specific to broadband vibration. Say so in the paper
  and reframe the R-peak result as evidence that the criterion
  preserves clinical features, not as evidence of divergence.
""")


if __name__ == "__main__":
    main()
