import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        8,
    'axes.labelsize':   9,
    'axes.titlesize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'axes.linewidth':   0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'figure.dpi':       600,
    'savefig.dpi':      600,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})
abl_labels = [
    'Adaptive composite\n(proposed)',
    'Fixed composite\n(proposed)',
    'Windowed energy\n(proposed)',
    'First-order derivative',
    'Second-order derivative',
]
vals = np.array([36.89, 37.08, 37.00, 34.88, 33.26])
errs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
drop_val = 33.25
dlv_val  = 36.45
colors = ['#5C3566', '#C0392B', '#E67E22', '#D4AC0D', '#95A5A6']
EDGE   = '#1A1A1A'
fig, ax = plt.subplots(figsize=(3.5, 2.7))
y = np.arange(len(abl_labels))
bars = ax.barh(
    y, vals, height=0.52, color=colors, edgecolor=EDGE, linewidth=0.5,
    xerr=errs, capsize=2.5,
    error_kw={'linewidth': 0.55, 'capthick': 0.55, 'ecolor': EDGE, 'zorder': 5},
    zorder=3,
)
x_min, x_max = 29.5, 40.5
for bar, val, err in zip(bars, vals, errs):
    bar_w = val - x_min
    label = f'{val:.1f} dB'
    inside = bar_w > 3.5
    if inside:
        ax.text(val - err - 0.18, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='right',
                fontsize=7, fontweight='bold', color='white', zorder=6)
    else:
        ax.text(val + err + 0.18, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left',
                fontsize=7, fontweight='bold', color=EDGE, zorder=6)
l1, = ax.plot([], [], color='#555555', linestyle=(0,(4,2)), linewidth=0.9)
l2, = ax.plot([], [], color='#1A6B2A', linestyle=(0,(4,2)), linewidth=0.9)
ax.axvline(drop_val, color='#555555', linestyle=(0,(4,2)), linewidth=0.9, zorder=2)
ax.axvline(dlv_val,  color='#1A6B2A', linestyle=(0,(4,2)), linewidth=0.9, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(abl_labels, fontsize=7.5)
ax.set_xlabel('SNR (dB)', fontsize=9, labelpad=3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.55, len(abl_labels) - 0.45)
ax.invert_yaxis()
ax.grid(axis='x', color='#CCCCCC', linewidth=0.35, linestyle='--', zorder=1)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.tick_params(axis='y', length=0, pad=3)
ax.tick_params(axis='x', direction='out', length=3, width=0.6)
leg = ax.legend(
    [l1, l2],
    [f'DROP baseline ({drop_val:.1f} dB)', f'DROP-LV baseline ({dlv_val:.1f} dB)'],
    fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.22),
    ncol=2, frameon=True, framealpha=0.95, edgecolor='#AAAAAA',
    handlelength=1.8, handletextpad=0.4, borderpad=0.45, columnspacing=0.8,
)
leg.get_frame().set_linewidth(0.4)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(bottom=0.22)
fig.savefig('fig_ablation.png', dpi=600)
fig.savefig('fig_ablation.svg', format='svg')
print("Saved fig_ablation.png and fig_ablation.svg")
plt.close()
