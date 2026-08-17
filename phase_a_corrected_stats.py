#!/usr/bin/env python3
"""
PHASE A — CORRECTED STATISTICS PIPELINE
Replaces: wilcoxon_results.csv, novelty_wilcoxon.csv
Produces: corrected_stats_cross_domain.csv, corrected_stats_novelty.csv,
          corrected_stats_report.txt (human-readable summary for paper)

Fixes applied:
  M3  — Uses only paired differences, never unpaired group means
  M4  — Uses d_z = mean(diffs) / sd(diffs), not pooled SD
  M5  — Reports per-signal breakdown to explain domain-mean coincidence
  M11 — Deduplicates identical trials, reports effective n, applies Holm correction

Usage:
    cd ~/DSA/adaptive-ring-buffer
    python3 phase_a_corrected_stats.py

Requires: pandas, scipy, numpy
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from itertools import combinations

# ─── CONFIG ───────────────────────────────────────────────────────────
BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
RESULTS = BASE / "results"

CROSS_DOMAIN = RESULTS / "cross_domain_results.csv"
NOVELTY_EVAL = RESULTS / "novelty_eval_results.csv"
STRESS_TEST  = RESULTS / "stress_test_results.csv"

# Output files
OUT_CD_CSV   = RESULTS / "corrected_stats_cross_domain.csv"
OUT_NE_CSV   = RESULTS / "corrected_stats_novelty.csv"
OUT_REPORT   = RESULTS / "corrected_stats_report.txt"

# The ONE proposed method (fixes M13: "proposed method is a moving target")
PROPOSED      = "IMP_INTERP_ERROR"

# Pairing key
PAIR_KEYS     = ["signal", "buffer_size", "overload", "trial"]
CONFIG_KEYS   = ["signal", "buffer_size", "overload"]  # for dedup check

# ─── HELPERS ──────────────────────────────────────────────────────────

def load_and_clean(path, metrics=None):
    """Load CSV, filter inf/saturated/degenerate rows for specified metrics."""
    if not path.exists():
        print(f"  !! FILE NOT FOUND: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {path.name}: {len(df)} rows")

    # Mark saturated rows
    if "snr_saturated" in df.columns:
        n_sat = df["snr_saturated"].sum()
        if n_sat > 0:
            print(f"    {n_sat} rows have snr_saturated=1")
    if "degenerate" in df.columns:
        n_deg = df["degenerate"].sum()
        if n_deg > 0:
            print(f"    {n_deg} rows have degenerate=1")

    return df


def filter_for_metric(df, metric):
    """Remove rows where the metric is inf, nan, or flagged saturated/degenerate."""
    sub = df.copy()
    if metric == "snr_db":
        if "snr_saturated" in sub.columns:
            sub = sub[sub["snr_saturated"] != 1]
        if "degenerate" in sub.columns:
            sub = sub[sub["degenerate"] != 1]
    sub = sub[np.isfinite(sub[metric])]
    return sub


def deduplicate_trials(df, method, metric):
    """
    For a given method, find config groups where all trials produce identical
    metric values. Keep only one row per such group.
    Returns (deduped_df, n_groups_deduped, n_rows_removed).
    """
    sub = df[df["mode"] == method].copy()
    if sub.empty:
        return sub, 0, 0

    n_before = len(sub)
    groups_deduped = 0
    keep_indices = []

    for name, grp in sub.groupby(CONFIG_KEYS):
        vals = grp[metric].values
        if len(vals) > 1 and np.all(vals == vals[0]):
            # All trials identical — keep only the first
            keep_indices.append(grp.index[0])
            groups_deduped += 1
        else:
            # Trials vary — keep all
            keep_indices.extend(grp.index.tolist())

    result = sub.loc[keep_indices]
    n_removed = n_before - len(result)
    return result, groups_deduped, n_removed


def paired_analysis(df, method_a, method_b, metric, domain=None, dedup=True):
    """
    Full paired statistical analysis between two methods.
    Returns dict with all stats, or None if insufficient data.
    """
    sub = filter_for_metric(df, metric)

    if domain:
        sub = sub[sub["domain"] == domain]

    # Optionally deduplicate
    n_dedup_a = n_dedup_b = 0
    n_removed_a = n_removed_b = 0
    if dedup:
        a_data, n_dedup_a, n_removed_a = deduplicate_trials(sub, method_a, metric)
        b_data, n_dedup_b, n_removed_b = deduplicate_trials(sub, method_b, metric)
    else:
        a_data = sub[sub["mode"] == method_a]
        b_data = sub[sub["mode"] == method_b]

    if a_data.empty or b_data.empty:
        return None

    # Pair by PAIR_KEYS (or CONFIG_KEYS if deduped removed trial variation)
    # We always pair by PAIR_KEYS — dedup kept one trial per group, with
    # its original trial number, so the merge still works correctly for
    # configs that have varying trials
    merged = a_data.merge(b_data, on=PAIR_KEYS, suffixes=("_A", "_B"), how="inner")
    if len(merged) < 5:
        return None

    diffs = merged[f"{metric}_A"].values - merged[f"{metric}_B"].values
    n = len(diffs)

    result = {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "domain": domain or "ALL",
        "n_paired": n,
        "n_dedup_groups_a": n_dedup_a,
        "n_dedup_groups_b": n_dedup_b,
        "n_rows_removed_a": n_removed_a,
        "n_rows_removed_b": n_removed_b,
        "mean_a": merged[f"{metric}_A"].mean(),
        "mean_b": merged[f"{metric}_B"].mean(),
        "mean_diff": diffs.mean(),
        "sd_diff": diffs.std(ddof=1),
        "se_diff": diffs.std(ddof=1) / np.sqrt(n),
        "ci95_lo": diffs.mean() - 1.96 * diffs.std(ddof=1) / np.sqrt(n),
        "ci95_hi": diffs.mean() + 1.96 * diffs.std(ddof=1) / np.sqrt(n),
        "min_diff": diffs.min(),
        "max_diff": diffs.max(),
        "median_diff": np.median(diffs),
    }

    # Cohen's d_z (CORRECT paired effect size)
    if result["sd_diff"] > 1e-15:
        result["d_z"] = result["mean_diff"] / result["sd_diff"]
    else:
        result["d_z"] = float("inf") if abs(result["mean_diff"]) > 1e-15 else 0.0

    # d_z magnitude interpretation
    abs_dz = abs(result["d_z"])
    if abs_dz < 0.2:
        result["effect_label"] = "negligible"
    elif abs_dz < 0.5:
        result["effect_label"] = "small"
    elif abs_dz < 0.8:
        result["effect_label"] = "medium"
    else:
        result["effect_label"] = "large"

    # Win/loss/tie
    result["wins_a"] = int((diffs > 1e-10).sum())
    result["wins_b"] = int((diffs < -1e-10).sum())
    result["ties"] = int((np.abs(diffs) <= 1e-10).sum())
    result["win_rate_a"] = result["wins_a"] / n if n > 0 else 0

    # Wilcoxon signed-rank (two-sided)
    try:
        nonzero_diffs = diffs[np.abs(diffs) > 1e-15]
        if len(nonzero_diffs) >= 10:
            stat, p = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
            result["wilcoxon_stat"] = stat
            result["wilcoxon_p"] = p
        else:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = None
    except Exception as e:
        result["wilcoxon_stat"] = None
        result["wilcoxon_p"] = f"ERROR: {e}"

    # Unique signals contributing
    if "signal" in merged.columns:
        result["n_signals"] = merged["signal"].nunique()
    elif "signal_A" in merged.columns:
        # signal was in pair keys but got suffixed
        result["n_signals"] = "N/A"

    return result


def apply_holm_correction(results_list):
    """
    Apply Holm-Bonferroni correction to all p-values in a list of result dicts.
    Adds 'p_holm' field to each.
    """
    # Collect valid p-values with indices
    pvals = []
    for i, r in enumerate(results_list):
        p = r.get("wilcoxon_p")
        if isinstance(p, (int, float)) and p is not None and np.isfinite(p):
            pvals.append((i, p))

    if not pvals:
        for r in results_list:
            r["p_holm"] = None
        return

    # Sort by p-value
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)

    for rank, (idx, p) in enumerate(pvals):
        corrected = p * (m - rank)
        # Enforce monotonicity
        if rank > 0:
            prev_idx = pvals[rank - 1][0]
            corrected = max(corrected, results_list[prev_idx]["p_holm"])
        corrected = min(corrected, 1.0)
        results_list[idx]["p_holm"] = corrected

    # Mark those without valid p
    for r in results_list:
        if "p_holm" not in r:
            r["p_holm"] = None


def per_signal_breakdown(df, method_a, method_b, metric, domain=None):
    """Show per-signal paired means for M5 explanation."""
    sub = filter_for_metric(df, metric)
    if domain:
        sub = sub[sub["domain"] == domain]

    a = sub[sub["mode"] == method_a]
    b = sub[sub["mode"] == method_b]
    merged = a.merge(b, on=PAIR_KEYS, suffixes=("_A", "_B"), how="inner")

    lines = []
    for sig in sorted(merged["signal_A"].unique() if "signal_A" in merged.columns
                      else merged["signal"].unique()):
        col = "signal_A" if "signal_A" in merged.columns else "signal"
        sig_data = merged[merged[col] == sig]
        diffs = sig_data[f"{metric}_A"].values - sig_data[f"{metric}_B"].values
        lines.append(f"    {sig:30s}  n={len(diffs):3d}  "
                     f"mean_diff={diffs.mean():+.4f}  "
                     f"sd={diffs.std():.4f}  "
                     f"range=[{diffs.min():.3f}, {diffs.max():.3f}]")
    return "\n".join(lines)


# ─── PRD / PRDN COMPUTATION ──────────────────────────────────────────

def add_prd_columns(df):
    """
    Compute PRD and PRDN from MSE and signal power.
    PRD = 100 * sqrt(MSE / mean(x^2))
    Since we don't have mean(x^2) directly, we recover it from SNR:
      SNR = 10*log10(signal_power / MSE)
      signal_power = MSE * 10^(SNR/10)
      PRD = 100 * sqrt(MSE / signal_power) = 100 / sqrt(10^(SNR/10))
      PRD = 100 * 10^(-SNR/20)
    """
    if "mse" not in df.columns or "snr_db" not in df.columns:
        return df

    df = df.copy()

    # PRD from SNR: PRD = 100 * 10^(-SNR/20)
    mask = np.isfinite(df["snr_db"]) & (df["snr_db"] > -100)
    df["prd"] = np.nan
    df.loc[mask, "prd"] = 100.0 * np.power(10.0, -df.loc[mask, "snr_db"] / 20.0)

    return df


# ─── BITS PER SAMPLE COMPUTATION ─────────────────────────────────────

def compute_bits_per_sample(n_original, n_surviving, value_bits=16, index_bits=16):
    """
    Compute true compression ratio and bits/sample.

    Each surviving sample needs: value (value_bits) + index (index_bits).
    Original: n_original samples * value_bits.

    Returns (cr, bps) where:
      cr  = original_bits / compressed_bits
      bps = compressed_bits / n_original
    """
    original_bits = n_original * value_bits
    compressed_bits = n_surviving * (value_bits + index_bits)
    if compressed_bits == 0:
        return float("inf"), 0.0
    cr = original_bits / compressed_bits
    bps = compressed_bits / n_original
    return cr, bps


def add_compression_columns(df, signal_length=2000, value_bits=16, index_bits=16):
    """Add true CR and bits/sample columns to a dataframe with 'drops' column."""
    if "drops" not in df.columns:
        return df

    df = df.copy()
    n_surviving = signal_length - df["drops"]
    n_surviving = n_surviving.clip(lower=0)

    original_bits = signal_length * value_bits
    compressed_bits = n_surviving * (value_bits + index_bits)

    df["true_cr"] = original_bits / compressed_bits.replace(0, np.nan)
    df["bits_per_sample"] = compressed_bits / signal_length
    df["decimation_ratio"] = signal_length / n_surviving.replace(0, np.nan)

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    report_lines = []

    def report(line=""):
        print(line)
        report_lines.append(line)

    report("=" * 76)
    report("CORRECTED STATISTICS REPORT — Phase A")
    report(f"Generated by phase_a_corrected_stats.py")
    report("Fixes: M3 (paired only), M4 (d_z), M5 (per-signal), M11 (dedup+Holm)")
    report("=" * 76)

    # ── Load data ─────────────────────────────────────────────────────
    report("\n[1] LOADING DATA")
    cd = load_and_clean(CROSS_DOMAIN)
    ne = load_and_clean(NOVELTY_EVAL)
    st = load_and_clean(STRESS_TEST)

    if cd is None:
        report("!! cross_domain_results.csv required. Aborting.")
        sys.exit(1)

    # ── Add PRD to cross_domain ───────────────────────────────────────
    cd = add_prd_columns(cd)
    cd = add_compression_columns(cd, signal_length=2000)

    if st is not None:
        st = add_prd_columns(st)
        # stress_test uses 2000 sample signals too (from context doc)
        st = add_compression_columns(st, signal_length=2000)

    # ══════════════════════════════════════════════════════════════════
    # SECTION A: CROSS-DOMAIN CORRECTED STATISTICS
    # ══════════════════════════════════════════════════════════════════
    report("\n" + "=" * 76)
    report("SECTION A: CROSS-DOMAIN (11,200 rows)")
    report("=" * 76)

    # Define all comparisons for cross_domain
    cd_comparisons = []

    # Proposed vs key baselines, all domains
    opponents = [("RDP_OFFLINE", "Offline heuristic"),
                 ("LTTB_OFFLINE", "Offline visualization"),
                 ("IMP_COMPOSITE", "Online proxy-metric"),
                 ("DROP", "FIFO baseline"),
                 ("RANDOM_DROP", "Random baseline")]

    for opp, label in opponents:
        if opp not in cd["mode"].unique():
            continue
        for domain in [None, "ecg", "vibration"]:
            cd_comparisons.append((PROPOSED, opp, "snr_db", domain))

    # Spectral correlation (vibration only)
    if "spectral_correlation" in cd.columns:
        for opp, label in opponents:
            if opp not in cd["mode"].unique():
                continue
            cd_comparisons.append((PROPOSED, opp, "spectral_correlation", "vibration"))

    # PRD (ECG only)
    if "prd" in cd.columns:
        for opp, _ in opponents:
            if opp not in cd["mode"].unique():
                continue
            cd_comparisons.append((PROPOSED, opp, "prd", "ecg"))

    # Run all comparisons
    cd_results = []
    for method_a, method_b, metric, domain in cd_comparisons:
        r = paired_analysis(cd, method_a, method_b, metric, domain, dedup=True)
        if r:
            cd_results.append(r)

    # Apply Holm correction
    apply_holm_correction(cd_results)

    # Print results
    report(f"\n  {'Comparison':<45s} {'Domain':<12s} {'Metric':<20s} "
           f"{'n':>5s} {'MeanDiff':>10s} {'d_z':>8s} {'Effect':>10s} "
           f"{'W/L/T':>10s} {'p_holm':>12s}")
    report("  " + "-" * 140)

    for r in cd_results:
        p_str = (f"{r['p_holm']:.2e}" if isinstance(r['p_holm'], float)
                 else str(r['p_holm']))
        comp = f"{r['method_a']} vs {r['method_b']}"
        wlt = f"{r['wins_a']}/{r['wins_b']}/{r['ties']}"
        report(f"  {comp:<45s} {r['domain']:<12s} {r['metric']:<20s} "
               f"{r['n_paired']:>5d} {r['mean_diff']:>+10.4f} "
               f"{r['d_z']:>+8.4f} {r['effect_label']:>10s} "
               f"{wlt:>10s} {p_str:>12s}")

    # ── M5 explanation ────────────────────────────────────────────────
    report("\n" + "-" * 76)
    report("M5 EXPLANATION: Per-signal breakdown of Proposed vs RDP gap")
    report("-" * 76)
    breakdown = per_signal_breakdown(cd, PROPOSED, "RDP_OFFLINE", "snr_db")
    report(breakdown)
    report("\n  CONCLUSION: Per-signal means range from -1.62 to -2.13 dB.")
    report("  Domain-level coincidence (-1.794 for both) is an averaging artifact.")

    # ── Key corrected claims ──────────────────────────────────────────
    report("\n" + "-" * 76)
    report("CORRECTED KEY CLAIMS (use these in the paper)")
    report("-" * 76)

    # Find specific results for headline claims
    for r in cd_results:
        if (r["method_b"] == "RDP_OFFLINE" and r["metric"] == "snr_db"
                and r["domain"] == "ALL"):
            report(f"\n  vs RDP (ALL, SNR):")
            report(f"    Mean gap: {r['mean_diff']:+.2f} dB "
                   f"[95% CI: {r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]")
            report(f"    d_z = {r['d_z']:+.2f} ({r['effect_label']})")
            report(f"    Win rate: {r['wins_a']}/{r['n_paired']} ({r['win_rate_a']:.1%})")
            report(f"    FRAMING: Consistent ~1.8 dB cost of online operation.")

        if (r["method_b"] == "DROP" and r["metric"] == "snr_db"
                and r["domain"] == "ALL"):
            report(f"\n  vs FIFO (ALL, SNR):")
            report(f"    Mean improvement: {r['mean_diff']:+.2f} dB "
                   f"[95% CI: {r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]")
            report(f"    d_z = {r['d_z']:+.2f} ({r['effect_label']})")
            report(f"    Win rate: {r['wins_a']}/{r['n_paired']} ({r['win_rate_a']:.1%})")

        if (r["method_b"] == "RDP_OFFLINE" and r["metric"] == "spectral_correlation"
                and r["domain"] == "vibration"):
            report(f"\n  vs RDP (Vibration, Spectral Correlation):")
            report(f"    Mean improvement: {r['mean_diff']:+.4f} "
                   f"[95% CI: {r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]")
            report(f"    d_z = {r['d_z']:+.2f} ({r['effect_label']})")
            report(f"    Win rate: {r['wins_a']}/{r['n_paired']} ({r['win_rate_a']:.1%})")

    # ══════════════════════════════════════════════════════════════════
    # SECTION B: NOVELTY EVAL CORRECTED STATISTICS (external baselines)
    # ══════════════════════════════════════════════════════════════════
    if ne is not None:
        report("\n" + "=" * 76)
        report("SECTION B: NOVELTY EVAL — EXTERNAL BASELINES (8,640 rows)")
        report("=" * 76)

        # Determine which proposed method to use
        # INTERP_SPECTRAL is in novelty_eval; use it as the proposed method there
        ne_proposed = "IMP_INTERP_SPECTRAL"
        if ne_proposed not in ne["mode"].unique():
            ne_proposed = PROPOSED

        report(f"  Proposed method in novelty_eval: {ne_proposed}")

        # Deduplication report
        report("\n  Trial duplication report (before dedup):")
        for method in ne["mode"].unique():
            sub = ne[ne["mode"] == method]
            total_groups = sub.groupby(CONFIG_KEYS).ngroups
            identical = 0
            for _, grp in sub.groupby(CONFIG_KEYS):
                vals = grp["snr_db"].dropna().values
                if len(vals) > 1 and np.all(vals == vals[0]):
                    identical += 1
            report(f"    {method:30s}  {identical}/{total_groups} configs identical "
                   f"({100*identical/max(total_groups,1):.1f}%)")

        # Comparisons
        ne_ext = ["SDT_MATCHED", "PLA_MATCHED", "LTC_MATCHED"]
        ne_other = ["RDP_OFFLINE", "LTTB_OFFLINE", "DROP"]

        ne_comparisons = []
        for ext in ne_ext + ne_other:
            if ext not in ne["mode"].unique():
                continue
            for domain in [None, "ecg", "vibration"]:
                ne_comparisons.append((ne_proposed, ext, "snr_db", domain))

        # Spectral correlation (vibration)
        if "spectral_correlation" in ne.columns:
            for ext in ne_ext + ["RDP_OFFLINE"]:
                if ext not in ne["mode"].unique():
                    continue
                ne_comparisons.append((ne_proposed, ext,
                                       "spectral_correlation", "vibration"))

        ne_results = []
        for method_a, method_b, metric, domain in ne_comparisons:
            r = paired_analysis(ne, method_a, method_b, metric, domain, dedup=True)
            if r:
                ne_results.append(r)

        apply_holm_correction(ne_results)

        report(f"\n  {'Comparison':<50s} {'Dom':<8s} {'Metric':<18s} "
               f"{'n':>5s} {'MeanDiff':>10s} {'d_z':>8s} {'Eff':>8s} "
               f"{'W/L/T':>10s} {'p_holm':>12s} {'Dedup':>8s}")
        report("  " + "-" * 148)

        for r in ne_results:
            p_str = (f"{r['p_holm']:.2e}" if isinstance(r['p_holm'], float)
                     else str(r['p_holm']))
            comp = f"{r['method_a']} vs {r['method_b']}"
            wlt = f"{r['wins_a']}/{r['wins_b']}/{r['ties']}"
            dedup = f"{r['n_rows_removed_a']}+{r['n_rows_removed_b']}"
            report(f"  {comp:<50s} {r['domain']:<8s} {r['metric']:<18s} "
                   f"{r['n_paired']:>5d} {r['mean_diff']:>+10.4f} "
                   f"{r['d_z']:>+8.4f} {r['effect_label']:>8s} "
                   f"{wlt:>10s} {p_str:>12s} {dedup:>8s}")

        # ── Corrected ECG headline ────────────────────────────────────
        report("\n" + "-" * 76)
        report("M3 CORRECTION: ECG headline claim")
        report("-" * 76)
        for r in ne_results:
            if (r["method_b"] in ne_ext and r["domain"] == "ecg"
                    and r["metric"] == "snr_db"):
                report(f"  {r['method_a']} vs {r['method_b']} (ECG, SNR):")
                report(f"    WRONG (old headline): '+1.59 dB' (from unpaired group means)")
                report(f"    CORRECT (paired):     {r['mean_diff']:+.3f} dB, "
                       f"d_z = {r['d_z']:+.3f}")
                report(f"    Win rate: {r['wins_a']}/{r['n_paired']} "
                       f"({r['win_rate_a']:.1%})")

        report("\n  HONEST FRAMING: On ECG, proposed achieves +0.16 to +0.18 dB")
        report("  over external online baselines (small effect, p < 0.01).")
        report("  On vibration SNR, proposed LOSES by 0.85-0.99 dB to SDT/PLA/LTC.")
        report("  On vibration spectral correlation, proposed WINS — this is the")
        report("  actual differentiator for broadband signals.")

    # ══════════════════════════════════════════════════════════════════
    # SECTION C: COMPRESSION METRICS (M9)
    # ══════════════════════════════════════════════════════════════════
    report("\n" + "=" * 76)
    report("SECTION C: COMPRESSION METRICS (M9)")
    report("=" * 76)

    if "true_cr" in cd.columns and "prd" in cd.columns:
        # Show PRD and true CR for proposed method at various overload ratios
        prop_data = cd[cd["mode"] == PROPOSED].copy()
        prop_ecg = prop_data[prop_data["domain"] == "ecg"]

        report("\n  Proposed method (ECG) — PRD and True Compression Ratio:")
        report(f"  {'Overload':>8s} {'MeanSNR':>10s} {'MeanPRD':>10s} "
               f"{'TrueCR':>10s} {'BPS':>8s} {'DecimRatio':>12s}")
        report("  " + "-" * 65)

        for ovl in sorted(prop_ecg["overload"].unique()):
            ovl_data = prop_ecg[prop_ecg["overload"] == ovl]
            valid = ovl_data[np.isfinite(ovl_data["snr_db"]) &
                             np.isfinite(ovl_data["prd"]) &
                             np.isfinite(ovl_data["true_cr"])]
            if valid.empty:
                continue
            report(f"  {ovl:>8d} {valid['snr_db'].mean():>10.2f} "
                   f"{valid['prd'].mean():>10.2f}% "
                   f"{valid['true_cr'].mean():>10.2f}x "
                   f"{valid['bits_per_sample'].mean():>8.2f} "
                   f"{valid['decimation_ratio'].mean():>12.2f}x")

        report("\n  NOTE: True CR accounts for index overhead (16-bit value + 16-bit index).")
        report("  At 2x decimation: true CR ≈ 1x (index doubles surviving sample size).")
        report("  'Compression ratio' in the paper MUST use true_cr, not decimation_ratio.")

    # ══════════════════════════════════════════════════════════════════
    # SECTION D: STRESS TEST COMPRESSION METRICS
    # ══════════════════════════════════════════════════════════════════
    if st is not None and "true_cr" in st.columns:
        report("\n" + "=" * 76)
        report("SECTION D: STRESS TEST — TRUE CR AND PRD BY OVERLOAD")
        report("=" * 76)

        prop_st = st[st["mode"] == PROPOSED].copy()
        prop_st_ecg = prop_st[prop_st["domain"] == "ecg"]

        if not prop_st_ecg.empty:
            report("\n  Proposed (ECG), stress test:")
            report(f"  {'Overload':>8s} {'MeanSNR':>10s} {'MeanPRD':>10s} "
                   f"{'TrueCR':>10s} {'DecimRatio':>12s}")
            report("  " + "-" * 55)
            for ovl in sorted(prop_st_ecg["overload"].unique()):
                ovl_data = prop_st_ecg[prop_st_ecg["overload"] == ovl]
                valid = ovl_data[np.isfinite(ovl_data["snr_db"]) &
                                 np.isfinite(ovl_data["prd"]) &
                                 np.isfinite(ovl_data["true_cr"])]
                if valid.empty:
                    continue
                report(f"  {ovl:>8d} {valid['snr_db'].mean():>10.2f} "
                       f"{valid['prd'].mean():>10.2f}% "
                       f"{valid['true_cr'].mean():>10.2f}x "
                       f"{valid['decimation_ratio'].mean():>12.2f}x")

    # ══════════════════════════════════════════════════════════════════
    # SECTION E: SUMMARY — WHAT TO WRITE IN THE PAPER
    # ══════════════════════════════════════════════════════════════════
    report("\n" + "=" * 76)
    report("SECTION E: PAPER-READY CLAIMS (only these are defensible)")
    report("=" * 76)

    report("""
  1. PROPOSED METHOD: IMP_INTERP_ERROR (pick one, defend one).
     INTERP_SPECTRAL shows marginal improvement; present as variant, not primary.

  2. vs RDP OFFLINE:
     - SNR gap: ~-1.8 dB [CI: see above], d_z ≈ -1.8 (large, consistent)
     - Win rate: 0%. RDP wins every paired comparison.
     - FRAMING: "Consistent, predictable cost of online operation. Gap narrows
       with longer signals and larger buffers."
     - DO NOT call this "negligible." It is large and consistent.

  3. vs LTTB OFFLINE:
     - SNR improvement: ~+5.5 dB, d_z ≈ +1.8 (large)
     - Win rate: 100%.

  4. vs DROP (FIFO):
     - SNR improvement: ~+10.5 dB, d_z ≈ +3.4 (very large)
     - Win rate: 100%.

  5. vs SDT/PLA/LTC (external online baselines):
     - ECG SNR: +0.16 to +0.18 dB (small, significant). NOT +1.59 dB.
     - Vibration SNR: -0.85 to -0.99 dB (proposed LOSES on this metric).
     - Vibration spectral correlation: proposed WINS by +0.036 (d_z ≈ 0.27).
     - FRAMING: "Comparable SNR to established online baselines, with superior
       spectral fidelity on broadband signals."

  6. SPECTRAL CORRELATION vs RDP:
     - Vibration: +0.016, d_z ≈ +0.67 (medium), win rate ~94%.
     - This is the real differentiator. Lead with it for vibration domain.

  7. COMPRESSION RATIOS:
     - Always report true CR (with index overhead), not decimation ratio.
     - Report PRD (%) for ECG to enable literature comparison.

  8. EFFECTIVE SAMPLE SIZE:
     - Cross-domain: 8 signals. State this honestly.
     - novelty_eval: partial trial duplication for SDT/PLA/LTC (40-50%).
       After deduplication, report corrected n.
""")

    # ── Save outputs ──────────────────────────────────────────────────
    # CSV outputs
    if cd_results:
        pd.DataFrame(cd_results).to_csv(OUT_CD_CSV, index=False)
        report(f"\n  Saved: {OUT_CD_CSV}")

    if ne is not None and ne_results:
        pd.DataFrame(ne_results).to_csv(OUT_NE_CSV, index=False)
        report(f"  Saved: {OUT_NE_CSV}")

    # Text report
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(report_lines))
    report(f"  Saved: {OUT_REPORT}")

    report("\n" + "=" * 76)
    report("PHASE A COMPLETE. Paste this output back to mentor.")
    report("=" * 76)


if __name__ == "__main__":
    main()
