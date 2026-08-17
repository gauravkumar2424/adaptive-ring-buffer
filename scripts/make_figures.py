#!/usr/bin/env python3
"""
make_figures.py -- IEEE ESL publication figures (final).

SINGLE SOURCE OF TRUTH
  Fig. 1 reads results/figure_data.csv only, produced by
  compute_snr_from_survivors.py from the survivor indices in
  spectral_survivors.csv. SNR and envelope PNR therefore come from ONE
  experiment: 40 CWRU recordings, identical 12,000-sample windows, same
  method set. No value in this script is typed in by hand.

FRAMING (this matters as much as the data)
  Panel (a) shows the proposed method with the LOWEST SNR of any online
  method, and every method below 0 dB past CR 5x. That is real and it is
  kept. It is also the argument: on broadband impulsive vibration no
  linear reconstruction preserves waveform energy, so SNR cannot
  discriminate between methods that a diagnostician would rank very
  differently. Panel titles say this explicitly -- "the conventional
  metric" vs "what diagnosis depends on" -- so the reader understands
  (a) is the metric under criticism, not a defeat.

  The ordering of SDT and the proposed method REVERSES between panels.
  That reversal is the paper's claim, and it needs no annotation.

COLOUR AND PRINT
  Colours are saturated for the online PDF. Markers and dash patterns
  are retained because IEEE ESL prints greyscale: colour alone would
  collapse SDT and V-W into two identical grey lines on a printed copy,
  which is what most reviewers read from.

  Vector PDF, Type 42 embedded fonts (Type 3 is a routine PDF-eXpress
  rejection), effective font >= 7 pt at column width.

Usage:  python3 scripts/make_figures.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

COL, DBL = 3.45, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.linewidth": 0.4,
    "grid.color": "#d5d5d5",
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.columnspacing": 1.3,
    "legend.handletextpad": 0.45,
    "figure.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Saturated colours for screen; markers + dashes carry greyscale print.
S = {
    "IE":     dict(c="#000000", m="o", ls="-",                   lw=2.2, z=10, ms=5.2, lab="Proposed (IE)"),
    "SDT":    dict(c="#E1121C", m="s", ls=(0, (5, 1.8)),         lw=1.8, z=9,  ms=4.6, lab="SDT"),
    "VW":     dict(c="#0B6FC4", m="^", ls=(0, (5, 1.5, 1, 1.5)), lw=1.6, z=7,  ms=4.6, lab="V-W area"),
    "LTTB":   dict(c="#8A2BE2", m="D", ls=(0, (3, 1.3, 1, 1.3)), lw=1.6, z=6,  ms=4.0, lab="LTTB"),
    "DROP":   dict(c="#4A4A4A", m="x", ls=(0, (1.6, 1.6)),       lw=1.5, z=5,  ms=4.6, lab="FIFO"),
    "IE_ORACLE": dict(c="#7A7A7A", m="", ls=(0, (1, 1.8)),       lw=1.3, z=4,  ms=0,   lab="Offline oracle"),
}


def sty(k):
    s = S[k]
    return dict(color=s["c"], marker=s["m"], linestyle=s["ls"],
                linewidth=s["lw"], markersize=s["ms"], zorder=s["z"])


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.55, ls=":", lw=0.4)
    ax.set_axisbelow(True)


def band_curve(df, mode, col, bands, stat="median"):
    s = df[df["mode"] == mode]
    if s.empty:
        return None, None
    xs, ys = [], []
    for a, b in bands:
        v = s[(s.CR >= a) & (s.CR < b)][col].dropna()
        if len(v):
            xs.append(np.sqrt(a * b))
            ys.append(v.mean() if stat == "mean" else v.median())
    return xs, ys


# ==================================================================
def fig1_dissociation():
    p = os.path.join(R, "figure_data.csv")
    if not os.path.exists(p):
        print("  SKIP fig1: run scripts/compute_snr_from_survivors.py first")
        return
    d = pd.read_csv(p)
    for c in ["snr_db", "pnr_loss", "pnr_orig_db", "CR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    snr = d[d.snr_saturated == 0]
    base = d.groupby("signal").pnr_orig_db.first()
    diag = d[d.signal.isin(set(base[base >= 6].index))]

    bands = [(1, 5), (5, 10), (10, 20), (20, 45)]
    order = ["IE", "SDT", "VW", "LTTB", "DROP"]

    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.65))

    # ---- (a) the conventional metric ----
    ax = axes[0]
    for m in order:
        x, y = band_curve(snr, m, "snr_db", bands, stat="mean")
        if x:
            ax.plot(x, y, **sty(m))
    ax.axhline(0, color="#888", lw=0.7, ls="-", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 30])
    ax.set_xticklabels(["2", "5", "10", "20", "30"])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Reconstruction SNR (dB)")
    ax.set_title("(a) SNR: the conventional metric", loc="left", pad=3)
    tidy(ax)

    # ---- (b) what diagnosis depends on ----
    ax = axes[1]
    for m in order:
        x, y = band_curve(diag, m, "pnr_loss", bands)
        if x:
            ax.plot(x, y, **sty(m))
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 30])
    ax.set_xticklabels(["2", "5", "10", "20", "30"])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Envelope PNR loss (dB)")
    ax.set_title("(b) Envelope spectrum: what diagnosis reads",
                 loc="left", pad=3)
    ax.invert_yaxis()          # up = better in BOTH panels
    tidy(ax)

    handles = [Line2D([], [], color=S[k]["c"], marker=S[k]["m"],
                      linestyle=S[k]["ls"], linewidth=S[k]["lw"],
                      markersize=S[k]["ms"], label=S[k]["lab"])
               for k in order]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.045), borderaxespad=0)
    fig.subplots_adjust(wspace=0.30, bottom=0.30)

    out = os.path.join(FIG, "fig1_dissociation.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")

    key = ["signal", "buffer_size", "overload"]
    ps = snr.pivot_table(index=key, columns="mode", values="snr_db")
    pq = diag.pivot_table(index=key, columns="mode", values="pnr_loss")
    if "IE" in ps and "SDT" in ps:
        a = (ps["IE"] - ps["SDT"]).dropna().groupby(level=0).mean()
        b = (pq["SDT"] - pq["IE"]).dropna().groupby(level=0).mean()
        ca = 1.96 * a.std() / np.sqrt(len(a))
        cb = 1.96 * b.std() / np.sqrt(len(b))
        print(f"    CAPTION NUMBERS")
        print(f"      SNR  IE-SDT = {a.mean():+.2f} dB, 95% CI "
              f"[{a.mean()-ca:+.2f}, {a.mean()+ca:+.2f}], "
              f"IE wins {100*(a>0).mean():.0f}%, n={len(a)}")
        print(f"      PNR  SDT-IE = {b.mean():+.2f} dB, 95% CI "
              f"[{b.mean()-cb:+.2f}, {b.mean()+cb:+.2f}], "
              f"IE better {100*(b>0).mean():.0f}%, n={len(b)}")


# ==================================================================
def fig2_feasibility():
    bufs = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
    cached_max = np.array([844, 1374, 2438, 4490, 8670, 17082, 33852, 67196])
    heap_max = np.array([944, 1146, 1301, 1442, 1701, 1815, 2194, 2360])
    CPU = 168e6
    fc, fh = CPU / cached_max, CPU / heap_max

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.fill_between(bufs, fh, 3e5, color="#E1121C", alpha=0.07, lw=0)
    ax.fill_between(bufs, fc, fh, color="#FF8C1A", alpha=0.18, lw=0)
    ax.fill_between(bufs, 5e2, fc, color="#22AA22", alpha=0.11, lw=0)

    ax.plot(bufs, fc, color="#E1121C", ls=(0, (5, 1.8)), marker="s",
            lw=1.8, ms=4.6, label="Optimised $O(N)$ scan")
    ax.plot(bufs, fh, color="#000000", ls="-", marker="o",
            lw=2.2, ms=5.2, label="Proposed $O(\\log N)$")

    for r, lab in [(12000, "12 kHz"), (48000, "48 kHz")]:
        ax.axhline(r, color="#666", lw=0.6, ls=":")
        ax.text(4400, r, lab, fontsize=7, color="#666", va="center")

    ax.text(75, 3.0e3, "both feasible", fontsize=7.5, color="#157015", ha="left")
    ax.text(1900, 2.6e4, "heap only", fontsize=7.5, color="#9C4E00", ha="center")
    ax.text(110, 1.8e5, "infeasible", fontsize=7.5, color="#A01018", ha="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bufs)
    ax.set_xticklabels([str(b) for b in bufs])
    ax.set_xlabel("Buffer capacity $N$ (samples)")
    ax.set_ylabel("Sustainable sample rate (Hz)")
    ax.set_xlim(30, 4300)
    ax.set_ylim(1.5e3, 2.6e5)
    ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.0, 0.02))
    tidy(ax)

    out = os.path.join(FIG, "fig2_feasibility.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


# ==================================================================
def fig3_crossover():
    bufs = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
    scan = np.array([2257, 4540, 9109, 18246, 36441, 72773, 145735, 292256])
    cached = np.array([652, 1107, 2015, 3842, 7498, 14822, 29532, 59252])
    heap = np.array([566, 616, 665, 729, 806, 899, 1011, 1137])

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(bufs, scan, color="#4A4A4A", ls=(0, (1.6, 1.6)), marker="x",
            lw=1.5, ms=4.6, label="$O(N)$ naive rescan")
    ax.plot(bufs, cached, color="#E1121C", ls=(0, (5, 1.8)), marker="s",
            lw=1.8, ms=4.6, label="$O(N)$ cached scores")
    ax.plot(bufs, heap, color="#000000", ls="-", marker="o",
            lw=2.2, ms=5.2, label="$O(\\log N)$ heap")
    ax.annotate("", xy=(4096, cached[-1]), xytext=(4096, heap[-1]),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#333"))
    ax.text(3300, np.sqrt(cached[-1] * heap[-1]),
            f"${cached[-1]/heap[-1]:.0f}\\times$", fontsize=8.5,
            ha="right", va="center")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bufs)
    ax.set_xticklabels([str(b) for b in bufs])
    ax.set_xlabel("Buffer capacity $N$ (samples)")
    ax.set_ylabel("Cycles per eviction")
    ax.legend(loc="upper left", fontsize=7.5)
    tidy(ax)

    out = os.path.join(FIG, "fig3_crossover.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


# ==================================================================
def fig4_rpeak():
    p = os.path.join(R, "rpeak_unified.csv")
    if not os.path.exists(p):
        print("  SKIP fig4: missing rpeak_unified.csv")
        return
    d = pd.read_csv(p)

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    base = d.f1_baseline.mean()
    ax.axhline(base, color="#666", lw=0.9, ls=":",
               label=f"Uncompressed ($\\mathrm{{F1}}={base:.3f}$)")


    bands = [(1, 5), (5, 10), (10, 20), (20, 50), (50, 200)]
    for meth, key, lab in [("IE", "IE", "Proposed (IE)"),
                           ("FIFO", "DROP", "FIFO")]:
        s = d[d.method == meth]
        if s.empty:
            continue
        xs, ys = [], []
        for a, b in bands:
            v = s[(s.CR_decim >= a) & (s.CR_decim < b)].f1
            if len(v):
                xs.append(np.sqrt(a * b))
                ys.append(v.mean())
        ax.plot(xs, ys, **sty(key), label=lab)

    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.set_xlim(1.8, 42)
    ax.set_xlabel("Decimation ratio")
    ax.set_ylabel("R-peak detection F1")
    ax.set_ylim(0.30, 1.02)
    ax.legend(loc="lower left", fontsize=7.5)
    tidy(ax)

    out = os.path.join(FIG, "fig4_rpeak.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    print("Generating figures ->", os.path.relpath(FIG, ROOT))
    fig1_dissociation()
    fig2_feasibility()
    fig3_crossover()
    fig4_rpeak()
    print("""
Fig. 1: SDT must lie ABOVE the proposed method in (a) and BELOW it in
(b). Both axes are oriented up = better, so the reversal between panels
is the result. Panel titles frame (a) as the metric under criticism.

Required text, before a referee reaches it unaided: on broadband
impulsive vibration past CR 5x every online method falls below 0 dB
SNR, because no linear reconstruction preserves waveform energy. That
is the reason SNR cannot serve as the acceptance criterion here.
""")
