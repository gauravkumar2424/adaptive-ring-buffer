#!/usr/bin/env python3
"""
analyze_span.py  --  S5 analysis.

Answers, in order:
  1. Is beta* (the best exponent) stable, or does it move with the
     operating point?
  2. Does beta* depend on compression ratio INDEPENDENTLY of buffer
     size? The v4 grid confounded the two -- CR 12-25x existed only at
     buffer 64 -- so any CR trend there was unattributable.
  3. Does the best beta on SNR also win on spectral correlation (a
     metric no beta directly optimises)?
  4. Honest per-signal effect sizes (n=19), not per-configuration.

Run from the project root after span_sweep.
"""

import pandas as pd
import numpy as np

d = pd.read_csv('results/span_sweep.csv')
d = d[d.snr_saturated == 0]
sp = d[d['mode'] == 'SPAN'].copy()

KEY = ['signal', 'buffer_size', 'overload']
betas = sorted(sp.beta.unique())

def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)

# ------------------------------------------------------------------
hdr("1. MEAN SNR BY BETA  (pooled, all operating points)")
g = sp.groupby('beta').snr_db.agg(['mean', 'std', 'count'])
best = g['mean'].idxmax()
for b, r in g.iterrows():
    mark = "  <== best" if b == best else ""
    print(f"  beta={b:<5} mean SNR={r['mean']:8.3f}  sd={r['std']:6.3f}"
          f"  n={int(r['count']):5d}{mark}")

# ------------------------------------------------------------------
hdr("2. BETA* BY COMPRESSION RATIO BAND  (buffer size held separate)")
sp['crband'] = pd.cut(sp.CR, [1, 2, 3, 5, 8, 12, 20, 100],
                      labels=['1-2x', '2-3x', '3-5x', '5-8x',
                              '8-12x', '12-20x', '20x+'])
piv = sp.pivot_table(index='crband', columns='beta',
                     values='snr_db', aggfunc='mean', observed=True)
print("\n  mean SNR (dB):")
print(piv.round(2).to_string())
print("\n  best beta per CR band:")
for band in piv.index:
    row = piv.loc[band].dropna()
    if len(row):
        print(f"    CR {str(band):<7} beta*={row.idxmax():<5} "
              f"SNR={row.max():7.3f}   "
              f"(beta=0: {row.get(0.0, float('nan')):7.3f}, "
              f"beta=1: {row.get(1.0, float('nan')):7.3f})")

# ------------------------------------------------------------------
hdr("3. IS IT CR OR BUFFER SIZE?  beta* in each (buffer, CR) cell")
print("\n  Same CR band now appears at several buffer sizes. If beta*")
print("  tracks CR down each column, CR is the driver. If it tracks")
print("  buffer size across each row, buffer size is the driver.\n")
cells = sp.pivot_table(index='buffer_size', columns='crband',
                       values='snr_db', aggfunc='mean', observed=True)
out = pd.DataFrame(index=cells.index, columns=cells.columns, dtype=object)
for bs in cells.index:
    for band in cells.columns:
        s = sp[(sp.buffer_size == bs) & (sp.crband == band)]
        if len(s) >= 10:
            m = s.groupby('beta').snr_db.mean()
            out.loc[bs, band] = f"{m.idxmax():.2f}"
        else:
            out.loc[bs, band] = "-"
print(out.to_string())

# ------------------------------------------------------------------
hdr("4. PAIRED: best beta vs beta=0 (IE) and beta=1 (V-W)")
p = sp.pivot_table(index=KEY, columns='beta', values='snr_db')
for b in betas:
    if b in (0.0, 1.0) or b not in p:
        continue
    for ref, lab in [(0.0, 'IE  '), (1.0, 'V-W ')]:
        x = (p[b] - p[ref]).dropna()
        if len(x) == 0:
            continue
        dz = x.mean() / x.std() if x.std() > 0 else float('nan')
        print(f"  beta={b:<5} vs {lab} n={len(x):4d} "
              f"mean={x.mean():+7.3f} dz={dz:+6.3f} "
              f"win={100*(x>0).mean():5.1f}%")

# ------------------------------------------------------------------
hdr("5. SPECTRAL CORRELATION BY BETA  (vibration; not optimised by any beta)")
v = sp[(sp.domain == 'vibration') & (sp.spectral_correlation >= 0)]
if len(v):
    gs = v.groupby('beta').spectral_correlation.agg(['mean', 'count'])
    bsp = gs['mean'].idxmax()
    for b, r in gs.iterrows():
        mark = "  <== best" if b == bsp else ""
        print(f"  beta={b:<5} spec_corr={r['mean']:.4f}  "
              f"n={int(r['count']):4d}{mark}")
    print(f"\n  best beta on SNR = {best};  best beta on spectral corr = {bsp}")
    if bsp != best:
        print("  -> the criterion trades pointwise for spectral fidelity.")
        print("     Report both; do not pick one and hide the other.")

# ------------------------------------------------------------------
hdr("6. PER-SIGNAL EFFECT SIZES (n=19 -- the number for the paper)")
pv = p.copy()
for b in betas:
    if b == 0.0 or b not in pv:
        continue
    s = (pv[b] - pv[0.0]).dropna().groupby(level=0).mean()
    dz = s.mean() / s.std() if s.std() > 0 else float('nan')
    print(f"  beta={b:<5} vs beta=0   n={len(s):3d} mean={s.mean():+7.3f} "
          f"dz={dz:+6.3f} win={100*(s>0).mean():5.1f}%")

print("\n  IE (beta=0) vs V-W (beta=1), per-signal:")
s = (pv[0.0] - pv[1.0]).dropna().groupby(level=0).mean()
dz = s.mean() / s.std() if s.std() > 0 else float('nan')
print(f"    n={len(s)} mean={s.mean():+.3f} dz={dz:+.3f} "
      f"win={100*(s>0).mean():.1f}%")

hdr("READ THIS")
print("""
  If beta*=0 everywhere      -> IE is simply the right criterion.
                                One clean sentence; V-W becomes a cited
                                baseline, not a contribution.

  If beta* moves with CR     -> criterion characterisation. Strongest
                                outcome: an operating-point-dependent
                                design rule, with V-W and IE as the two
                                endpoints of a continuum you identified.

  If beta*=0.5 everywhere    -> a better criterion than both, but then
                                the paper needs it justified, not fitted.
                                Hold it out on the 3 vibration signals.
""")
