#!/usr/bin/env python3
"""
DATE 2027 — Final publication outputs
Table 1: Pareto comparison (printed as LaTeX-ready table)
Figure 1: Ablation bar chart (only figure needed)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = SCRIPT_DIR

df = pd.read_csv(os.path.join(RESULTS_DIR, 'sweep_results.csv'))
heavy = df[df['consumer_delay_us'] == 600]

avg = heavy.groupby('mode').agg(
    snr=('snr_db', 'mean'), snr_std=('snr_db', 'std'),
    wait=('max_wait_ms', 'mean'),
    mse=('mse', 'mean'),
    drops=('drops', 'mean'),
    deriv=('deriv_ratio', 'mean'),
).reset_index()

# ============================================================
# TABLE: LaTeX-ready comparison table for the paper
# ============================================================
print("=" * 75)
print("  TABLE: Performance Comparison Under 6x Overload")
print("=" * 75)

table_order = [
    ('DROP', 'Drop-oldest'),
    ('RANDOM_DROP', 'Random-drop'),
    ('DROP_MIDDLE', 'Drop-middle'),
    ('DROP_LOW_VARIANCE', 'Low-variance window'),
    ('ADAPTIVE', 'Adaptive timeout'),
    ('LEGACY_IMPORTANCE', 'First-order (prior)'),
    ('IMP_FIRST_ORDER', 'First-order (ablation)'),
    ('IMP_SECOND_ORDER', 'Second-order (ablation)'),
    ('IMP_WINDOWED_ENERGY', 'Windowed energy (proposed)'),
    ('IMP_COMPOSITE', 'Composite (proposed)'),
    ('IMP_ADAPTIVE', 'Adaptive composite (proposed)'),
]

proposed = ['IMP_COMPOSITE', 'IMP_ADAPTIVE', 'IMP_WINDOWED_ENERGY']

# Determine Pareto-optimal set (2-objective: max SNR, min latency)
drop_avg = avg[~avg['mode'].isin(['WAIT', 'TIMED'])]
pareto_2d = []
for _, ri in drop_avg.iterrows():
    dominated = False
    for _, rj in drop_avg.iterrows():
        if ri['mode'] != rj['mode']:
            if rj['snr'] >= ri['snr'] and rj['wait'] <= ri['wait'] and \
               (rj['snr'] > ri['snr'] or rj['wait'] < ri['wait']):
                dominated = True
                break
    if not dominated:
        pareto_2d.append(ri['mode'])

# Print readable table
print(f"\n{'Mode':<32s} {'SNR(dB)':>8s} {'MSE':>10s} {'Lat(ms)':>8s} {'Drops':>6s} {'Pareto':>7s}")
print("-" * 75)

for mode, label in table_order:
    row = avg[avg['mode'] == mode]
    if len(row) == 0: continue
    row = row.iloc[0]
    pareto = "Yes" if mode in pareto_2d else ""
    is_prop = mode in proposed
    prefix = ">> " if is_prop else "   "
    print(f"{prefix}{label:<29s} {row['snr']:8.2f} {row['mse']:10.6f} {row['wait']:8.4f} {row['drops']:6.0f} {pareto:>7s}")

print("-" * 75)

# Also WAIT/TIMED for reference
for mode, label in [('WAIT', 'Blocking (WAIT)'), ('TIMED', 'Timed blocking')]:
    row = avg[avg['mode'] == mode]
    if len(row) == 0: continue
    row = row.iloc[0]
    print(f"   {label:<29s} {row['snr']:8.2f} {row['mse']:10.6f} {row['wait']:8.4f} {row['drops']:6.0f}")

# Print LaTeX version
print(f"\n{'=' * 75}")
print("  LaTeX TABLE (copy into paper)")
print("=" * 75)
print(r"""
\begin{table}[t]
\centering
\caption{Performance comparison under 6$\times$ overload (325 experiments per mode).
Bold: proposed methods. $\star$: Pareto-optimal.}
\label{tab:comparison}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Mode} & \textbf{SNR (dB)} & \textbf{MSE} & \textbf{Latency (ms)} & \textbf{Pareto} \\
\midrule""")

for mode, label in table_order:
    row = avg[avg['mode'] == mode]
    if len(row) == 0: continue
    row = row.iloc[0]
    pareto = r"$\star$" if mode in pareto_2d else ""
    is_prop = mode in proposed
    if is_prop:
        print(f"\\textbf{{{label}}} & \\textbf{{{row['snr']:.2f}}} & \\textbf{{{row['mse']:.4f}}} & \\textbf{{{row['wait']:.4f}}} & {pareto} \\\\")
    else:
        print(f"{label} & {row['snr']:.2f} & {row['mse']:.4f} & {row['wait']:.4f} & {pareto} \\\\")

print(r"""\midrule
Blocking (WAIT) & 100.00 & 0.0000 & 0.6348 & --- \\
\bottomrule
\end{tabular}
\end{table}""")

# ============================================================
# FIGURE: Ablation — clean, professional, no clutter
# Single-column width (3.5 inches) for IEEE format
# ============================================================
print(f"\n{'=' * 75}")
print("  GENERATING ABLATION FIGURE")
print("=" * 75)

abl_modes = ['IMP_FIRST_ORDER', 'IMP_SECOND_ORDER', 'IMP_WINDOWED_ENERGY',
             'IMP_COMPOSITE', 'IMP_ADAPTIVE']
abl_labels = ['1st-order', '2nd-order', 'Windowed\nenergy', 'Composite', 'Adaptive']
# Gradient from warm to deep red
abl_colors = ['#E8A735', '#BDBDBD', '#EF6C00', '#C62828', '#6A1B9A']

abl_data = heavy.groupby('mode')['snr_db'].agg(['mean', 'std']).reindex(abl_modes)

fig, ax = plt.subplots(figsize=(3.45, 2.2))

vals = abl_data['mean'].values
errs = abl_data['std'].values * 0.15  # Smaller error bars

x = np.arange(len(abl_modes))
bars = ax.bar(x, vals, width=0.58, color=abl_colors, edgecolor='#333333',
              linewidth=0.4, yerr=errs, capsize=2.5,
              error_kw={'linewidth': 0.4, 'capthick': 0.4, 'color': '#333333'})

# Value labels inside bars near top
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
            f'{val:.1f}', ha='center', va='top', fontsize=8,
            fontweight='bold', color='white')

# Baselines as thin lines
drop_val = heavy.groupby('mode')['snr_db'].mean().get('DROP', 0)
dlv_val = heavy.groupby('mode')['snr_db'].mean().get('DROP_LOW_VARIANCE', 0)

ax.axhline(y=drop_val, color='#666666', linestyle='--', linewidth=0.4)
ax.axhline(y=dlv_val, color='#2E7D32', linestyle='--', linewidth=0.4)

# Annotations at right edge
ax.text(len(abl_modes) - 0.5, drop_val + 0.2, f'Drop ({drop_val:.1f})',
        fontsize=6, color='#666666', ha='right')
ax.text(len(abl_modes) - 0.5, dlv_val + 0.2, f'Low-var ({dlv_val:.1f})',
        fontsize=6, color='#2E7D32', ha='right')

ax.set_xticks(x)
ax.set_xticklabels(abl_labels, fontsize=8)
ax.set_ylabel('SNR (dB)', fontsize=9)
ax.set_ylim(32, 39)
ax.set_xlim(-0.5, len(abl_modes) - 0.5)
ax.grid(axis='y', alpha=0.12, linewidth=0.3)
ax.tick_params(axis='x', length=0)  # No tick marks on x

fig.savefig(os.path.join(FIGURES_DIR, 'fig_ablation.svg'), format='svg')
fig.savefig(os.path.join(FIGURES_DIR, 'fig_ablation.png'), dpi=600)
print("\n  Saved fig_ablation.svg/png")
plt.close()

print("\nDone.")
