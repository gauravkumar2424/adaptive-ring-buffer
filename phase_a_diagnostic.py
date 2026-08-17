
#!/usr/bin/env python3
"""
PHASE A DIAGNOSTIC — Run on Vinayak, paste output back to mentor.
Investigates M3, M4, M5, M11 from the mock review.

Usage:
    cd ~/DSA/adaptive-ring-buffer
    python3 phase_a_diagnostic.py

Requires: pandas, scipy, numpy (pip install pandas scipy numpy if missing)
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────
BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
CROSS_DOMAIN = BASE / "results" / "cross_domain_results.csv"
NOVELTY_EVAL = BASE / "results" / "novelty_eval_results.csv"
STRESS_TEST  = BASE / "results" / "stress_test_results.csv"

PROPOSED     = "IMP_INTERP_ERROR"      # primary proposed method
PROPOSED_S   = "IMP_INTERP_SPECTRAL"   # spectral variant
RDP          = "RDP_OFFLINE"
LTTB         = "LTTB_OFFLINE"
DROP_MODE    = "DROP"

# Pairing key columns
PAIR_KEYS    = ["signal", "buffer_size", "overload", "trial"]

# ─── HELPERS ──────────────────────────────────────────────────────────
def load_csv(path):
    if not path.exists():
        print(f"  !! FILE NOT FOUND: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {path.name}: {len(df)} rows, {len(df.columns)} cols")
    print(f"  Columns: {list(df.columns)}")
    return df


def paired_stats(df, method_a, method_b, metric="snr_db", domain_filter=None):
    """Compute paired differences between two methods matched by PAIR_KEYS."""
    sub = df.copy()
    if domain_filter:
        sub = sub[sub["domain"] == domain_filter]

    a = sub[sub["mode"] == method_a].copy()
    b = sub[sub["mode"] == method_b].copy()

    if a.empty or b.empty:
        return None

    # Filter out saturated/degenerate rows
    for col in ["snr_saturated", "degenerate"]:
        if col in a.columns:
            a = a[a[col] != 1]
            b = b[b[col] != 1]

    # Filter inf/nan in the metric
    a = a[np.isfinite(a[metric])]
    b = b[np.isfinite(b[metric])]

    merged = a.merge(b, on=PAIR_KEYS, suffixes=("_A", "_B"), how="inner")
    if merged.empty:
        return None

    diffs = merged[f"{metric}_A"] - merged[f"{metric}_B"]

    result = {
        "n_paired": len(diffs),
        "n_a_total": len(a),
        "n_b_total": len(b),
        "mean_diff": diffs.mean(),
        "sd_diff": diffs.std(ddof=1),
        "min_diff": diffs.min(),
        "max_diff": diffs.max(),
        "mean_a": merged[f"{metric}_A"].mean(),
        "mean_b": merged[f"{metric}_B"].mean(),
        "unpaired_diff": merged[f"{metric}_A"].mean() - merged[f"{metric}_B"].mean(),
    }

    # Cohen's d_z (paired)
    if result["sd_diff"] > 0:
        result["d_z"] = result["mean_diff"] / result["sd_diff"]
    else:
        result["d_z"] = float("inf") if result["mean_diff"] != 0 else 0.0

    # Cohen's d (unpaired, pooled SD — the WRONG way, for comparison)
    pooled_sd = np.sqrt(
        (merged[f"{metric}_A"].var(ddof=1) + merged[f"{metric}_B"].var(ddof=1)) / 2
    )
    if pooled_sd > 0:
        result["d_pooled_WRONG"] = result["mean_diff"] / pooled_sd
    else:
        result["d_pooled_WRONG"] = 0.0

    # Wilcoxon signed-rank
    try:
        stat, p = stats.wilcoxon(diffs, alternative="two-sided")
        result["wilcoxon_p"] = p
    except Exception as e:
        result["wilcoxon_p"] = f"ERROR: {e}"

    # Win/loss
    result["wins_a"] = int((diffs > 0).sum())
    result["wins_b"] = int((diffs < 0).sum())
    result["ties"] = int((diffs == 0).sum())

    return result


def check_trial_duplication(df, method, metric="snr_db"):
    """Check if 5 trials produce identical values for a deterministic method."""
    sub = df[df["mode"] == method].copy()
    if sub.empty:
        return None

    group_keys = ["signal", "buffer_size", "overload"]
    if "domain" in sub.columns:
        group_keys = ["signal", "domain", "buffer_size", "overload"]

    groups = sub.groupby(group_keys)
    total_groups = 0
    identical_groups = 0
    example_varying = None

    for name, grp in groups:
        total_groups += 1
        vals = grp[metric].dropna().values
        if len(vals) > 1 and np.all(vals == vals[0]):
            identical_groups += 1
        elif len(vals) > 1 and example_varying is None:
            example_varying = (name, vals.tolist())

    return {
        "method": method,
        "total_config_groups": total_groups,
        "identical_across_trials": identical_groups,
        "pct_identical": round(100 * identical_groups / max(total_groups, 1), 1),
        "example_varying": example_varying,
    }


# ─── MAIN DIAGNOSTIC ─────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("PHASE A DIAGNOSTIC — Investigating M3, M4, M5, M11")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1] LOADING FILES")
    cd = load_csv(CROSS_DOMAIN)
    ne = load_csv(NOVELTY_EVAL)
    st = load_csv(STRESS_TEST)

    if cd is None:
        print("\n!! cross_domain_results.csv is required. Cannot proceed.")
        sys.exit(1)

    # ── 2. Basic structure ────────────────────────────────────────────
    print("\n[2] DATA STRUCTURE")
    print(f"  Modes in cross_domain: {sorted(cd['mode'].unique())}")
    if "domain" in cd.columns:
        print(f"  Domains: {sorted(cd['domain'].unique())}")
    print(f"  Signals: {sorted(cd['signal'].unique())}")
    print(f"  Buffer sizes: {sorted(cd['buffer_size'].unique())}")
    print(f"  Overload ratios: {sorted(cd['overload'].unique())}")
    print(f"  Trials: {sorted(cd['trial'].unique())}")

    if ne is not None:
        print(f"\n  Modes in novelty_eval: {sorted(ne['mode'].unique())}")
        if "domain" in ne.columns:
            print(f"  Domains: {sorted(ne['domain'].unique())}")
        print(f"  Signals: {sorted(ne['signal'].unique())}")

    # ── 3. M5: The -1.7945 / -1.7943 coincidence ─────────────────────
    print("\n" + "=" * 72)
    print("[3] M5 INVESTIGATION: -1.7945 / -1.7943 coincidence")
    print("=" * 72)

    for domain in ["ecg", "ECG", "vibration", "Vibration", "vib", "Vib"]:
        r = paired_stats(cd, PROPOSED, RDP, metric="snr_db", domain_filter=domain)
        if r and r["n_paired"] > 0:
            print(f"\n  {PROPOSED} vs {RDP}, domain={domain}:")
            print(f"    n_paired      = {r['n_paired']}")
            print(f"    mean_diff     = {r['mean_diff']:.6f}")
            print(f"    sd_diff       = {r['sd_diff']:.6f}")
            print(f"    min_diff      = {r['min_diff']:.6f}")
            print(f"    max_diff      = {r['max_diff']:.6f}")
            print(f"    mean_A        = {r['mean_a']:.4f}")
            print(f"    mean_B(RDP)   = {r['mean_b']:.4f}")
            print(f"    unpaired_diff = {r['unpaired_diff']:.6f}")
            print(f"    d_z (CORRECT) = {r['d_z']:.4f}")
            print(f"    d_pooled(WRONG)= {r['d_pooled_WRONG']:.4f}")
            print(f"    wins/losses   = {r['wins_a']}/{r['wins_b']}/{r['ties']}")

    # Also show per-signal breakdown to trace the coincidence
    print("\n  Per-signal paired mean diff (Proposed - RDP):")
    for sig in sorted(cd["signal"].unique()):
        sub_a = cd[(cd["mode"] == PROPOSED) & (cd["signal"] == sig)]
        sub_b = cd[(cd["mode"] == RDP) & (cd["signal"] == sig)]
        # Filter saturated
        for col in ["snr_saturated", "degenerate"]:
            if col in sub_a.columns:
                sub_a = sub_a[sub_a[col] != 1]
                sub_b = sub_b[sub_b[col] != 1]
        sub_a = sub_a[np.isfinite(sub_a["snr_db"])]
        sub_b = sub_b[np.isfinite(sub_b["snr_db"])]
        m = sub_a.merge(sub_b, on=PAIR_KEYS, suffixes=("_A", "_B"), how="inner")
        if not m.empty:
            diffs = m["snr_db_A"] - m["snr_db_B"]
            print(f"    {sig:25s}  n={len(diffs):3d}  "
                  f"mean_diff={diffs.mean():+.6f}  "
                  f"sd={diffs.std():.6f}  "
                  f"range=[{diffs.min():.4f}, {diffs.max():.4f}]")

    # ── 4. M3: +1.59 dB vs +0.178 dB ─────────────────────────────────
    print("\n" + "=" * 72)
    print("[4] M3 INVESTIGATION: +1.59 dB (unpaired) vs +0.178 dB (paired)")
    print("=" * 72)

    if ne is not None:
        # Check which method names exist for proposed and external baselines
        ext_baselines = ["SDT_MATCHED", "PLA_MATCHED", "LTC_MATCHED",
                         "SDT", "PLA", "LTC"]
        proposed_names = [PROPOSED_S, PROPOSED]

        avail_ext = [m for m in ext_baselines if m in ne["mode"].unique()]
        avail_prop = [m for m in proposed_names if m in ne["mode"].unique()]
        print(f"  Available external baselines: {avail_ext}")
        print(f"  Available proposed methods: {avail_prop}")

        for prop in avail_prop:
            for ext in avail_ext:
                for dom in ne["domain"].unique() if "domain" in ne.columns else [None]:
                    r = paired_stats(ne, prop, ext, metric="snr_db", domain_filter=dom)
                    if r and r["n_paired"] > 0:
                        print(f"\n  {prop} vs {ext}, domain={dom}:")
                        print(f"    n_paired       = {r['n_paired']}")
                        print(f"    PAIRED mean    = {r['mean_diff']:+.6f} dB")
                        print(f"    UNPAIRED diff  = {r['unpaired_diff']:+.6f} dB")
                        print(f"    RATIO unp/pair = {r['unpaired_diff']/r['mean_diff']:.2f}x"
                              if r['mean_diff'] != 0 else "    RATIO: div by zero")
                        print(f"    d_z (CORRECT)  = {r['d_z']:.4f}")
                        print(f"    d_pooled(WRONG)= {r['d_pooled_WRONG']:.4f}")
                        print(f"    wilcoxon p     = {r['wilcoxon_p']}")
                        print(f"    wins/losses    = {r['wins_a']}/{r['wins_b']}/{r['ties']}")
    else:
        print("  !! novelty_eval_results.csv not found — cannot investigate M3")

    # ── 5. M4: Effect size comparison ─────────────────────────────────
    print("\n" + "=" * 72)
    print("[5] M4 INVESTIGATION: d_z vs d_pooled for all key comparisons")
    print("=" * 72)

    comparisons = [
        (PROPOSED, RDP, None),
        (PROPOSED, LTTB, None),
        (PROPOSED, DROP_MODE, None),
        (PROPOSED, "IMP_COMPOSITE", None),
    ]
    # Add per-domain if domain column exists
    if "domain" in cd.columns:
        for dom in cd["domain"].unique():
            comparisons.append((PROPOSED, RDP, dom))
            comparisons.append((PROPOSED, LTTB, dom))

    for method_a, method_b, dom in comparisons:
        if method_b not in cd["mode"].unique():
            continue
        r = paired_stats(cd, method_a, method_b, metric="snr_db", domain_filter=dom)
        if r and r["n_paired"] > 0:
            print(f"\n  {method_a} vs {method_b} [{dom or 'ALL'}]:")
            print(f"    n_paired       = {r['n_paired']}")
            print(f"    mean_diff      = {r['mean_diff']:+.6f}")
            print(f"    d_z (CORRECT)  = {r['d_z']:+.4f}")
            print(f"    d_pooled(WRONG)= {r['d_pooled_WRONG']:+.4f}")
            print(f"    wins/losses    = {r['wins_a']}/{r['wins_b']}/{r['ties']}")

    # ── 6. M11: Trial duplication check ───────────────────────────────
    print("\n" + "=" * 72)
    print("[6] M11 INVESTIGATION: Trial duplication for deterministic methods")
    print("=" * 72)

    det_methods = [RDP, LTTB, DROP_MODE, PROPOSED,
                   "IMP_COMPOSITE", "SDT_MATCHED", "PLA_MATCHED", "LTC_MATCHED"]

    for method in det_methods:
        if method in cd["mode"].unique():
            r = check_trial_duplication(cd, method)
            if r:
                print(f"  {r['method']:30s}  "
                      f"{r['identical_across_trials']}/{r['total_config_groups']} identical "
                      f"({r['pct_identical']}%)")
                if r["example_varying"]:
                    print(f"    Example varying: {r['example_varying'][0]} -> "
                          f"{r['example_varying'][1][:5]}")

    if ne is not None:
        print("\n  --- novelty_eval ---")
        for method in det_methods:
            if method in ne["mode"].unique():
                r = check_trial_duplication(ne, method)
                if r:
                    print(f"  {r['method']:30s}  "
                          f"{r['identical_across_trials']}/{r['total_config_groups']} identical "
                          f"({r['pct_identical']}%)")

    # ── 7. Spectral correlation check (vibration domain) ──────────────
    print("\n" + "=" * 72)
    print("[7] SPECTRAL CORRELATION: Proposed vs RDP (vibration)")
    print("=" * 72)

    if "spectral_correlation" in cd.columns:
        r = paired_stats(cd, PROPOSED, RDP, metric="spectral_correlation",
                         domain_filter="vibration")
        if r is None:
            # Try other domain labels
            for dom in cd["domain"].unique():
                if "vib" in dom.lower():
                    r = paired_stats(cd, PROPOSED, RDP,
                                     metric="spectral_correlation", domain_filter=dom)
                    break
        if r and r["n_paired"] > 0:
            print(f"  n_paired      = {r['n_paired']}")
            print(f"  mean_diff     = {r['mean_diff']:+.6f}")
            print(f"  d_z           = {r['d_z']:+.4f}")
            print(f"  wins/losses   = {r['wins_a']}/{r['wins_b']}/{r['ties']}")
    else:
        print("  !! spectral_correlation column not found")

    # ── 8. Sample rows for sanity ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("[8] SAMPLE ROWS (first 3 per method for cross_domain)")
    print("=" * 72)
    for mode in [PROPOSED, RDP, LTTB, DROP_MODE]:
        sub = cd[cd["mode"] == mode].head(3)
        if not sub.empty:
            print(f"\n  {mode}:")
            for _, row in sub.iterrows():
                cols_to_show = ["signal", "domain", "mode", "buffer_size",
                                "overload", "trial", "snr_db", "drops"]
                vals = {c: row[c] for c in cols_to_show if c in row.index}
                print(f"    {vals}")

    # ── 9. Quick PRD check (for M9) ───────────────────────────────────
    print("\n" + "=" * 72)
    print("[9] M9 CHECK: Is MSE available for PRD computation?")
    print("=" * 72)
    if "mse" in cd.columns:
        sample = cd[cd["mode"] == PROPOSED].head(3)[["signal", "snr_db", "mse"]].to_string()
        print(f"  MSE column exists. Sample:\n{sample}")
    else:
        print("  !! No MSE column — PRD will need recomputation from raw signals")

    # ── 10. Stress test quick look (for M9 bits/sample) ───────────────
    print("\n" + "=" * 72)
    print("[10] STRESS TEST: Column check + drop counts for bits/sample")
    print("=" * 72)
    if st is not None:
        print(f"  Columns: {list(st.columns)}")
        sample = st[st["mode"] == PROPOSED].head(5)
        if "drops" in st.columns:
            print(f"  Sample drop counts:")
            for _, row in sample.iterrows():
                cols = ["signal", "buffer_size", "overload", "drops", "snr_db"]
                vals = {c: row[c] for c in cols if c in row.index}
                print(f"    {vals}")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC COMPLETE — Copy everything above and paste to mentor")
    print("=" * 72)


if __name__ == "__main__":
    main()
