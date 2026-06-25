import wfdb
import numpy as np
import os
import urllib.request

# ============================================================
# MIT-BIH Arrhythmia Database — 5 records with diverse pathologies
# ============================================================
mitbih_dir = os.path.expanduser('~/DSA/adaptive-ring-buffer/data/mit-bih')
os.makedirs(mitbih_dir, exist_ok=True)

records = {
    '100': 'Normal sinus rhythm',
    '105': 'Premature ventricular contractions',
    '108': 'Multiform PVCs, noise',
    '201': 'Supraventricular ectopy',
    '228': 'Ventricular bigeminy',
}

for rec_id, desc in records.items():
    print(f"Downloading MIT-BIH record {rec_id} ({desc})...")
    try:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb', channels=[0])
        signal = record.p_signal[:, 0]
        # Take first 10000 samples (at 360 Hz ~ 27.8 seconds)
        signal = signal[:10000]
        outpath = os.path.join(mitbih_dir, f'ecg_{rec_id}.txt')
        np.savetxt(outpath, signal, fmt='%.6f')
        print(f"  Saved {len(signal)} samples to {outpath}")
        
        # Also get annotations (R-peak locations) for accuracy measurement
        ann = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
        # Filter to R-peaks only (beat annotations) within our sample range
        rpeak_indices = [s for s in ann.sample if s < 10000]
        rpeak_path = os.path.join(mitbih_dir, f'rpeak_{rec_id}.txt')
        np.savetxt(rpeak_path, rpeak_indices, fmt='%d')
        print(f"  Saved {len(rpeak_indices)} R-peak annotations")
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================
# CWRU Bearing Dataset — 3 conditions
# Normal, Inner Race Fault, Ball Fault
# 12kHz drive end accelerometer data
# ============================================================
cwru_dir = os.path.expanduser('~/DSA/adaptive-ring-buffer/data/cwru-bearing')
os.makedirs(cwru_dir, exist_ok=True)

cwru_files = {
    'normal_0hp': 'https://engineering.case.edu/sites/default/files/97.mat',
    'inner_race_007': 'https://engineering.case.edu/sites/default/files/105.mat',
    'ball_fault_007': 'https://engineering.case.edu/sites/default/files/118.mat',
}

print("\nDownloading CWRU Bearing Dataset...")
for name, url in cwru_files.items():
    outpath = os.path.join(cwru_dir, f'{name}.mat')
    try:
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, outpath)
        print(f"  Saved to {outpath}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Extract CWRU .mat files to .txt
try:
    from scipy.io import loadmat
    for name in cwru_files:
        matpath = os.path.join(cwru_dir, f'{name}.mat')
        if not os.path.exists(matpath):
            continue
        mat = loadmat(matpath)
        # Find the drive end accelerometer data key
        data_key = [k for k in mat.keys() if 'DE' in k or 'X0' in k]
        if not data_key:
            data_key = [k for k in mat.keys() if not k.startswith('_')]
        if data_key:
            signal = mat[data_key[0]].flatten()[:10000]
            txtpath = os.path.join(cwru_dir, f'{name}.txt')
            np.savetxt(txtpath, signal, fmt='%.6f')
            print(f"  Extracted {name}: {len(signal)} samples to {txtpath}")
except ImportError:
    print("  scipy not installed — run: pip install scipy --break-system-packages")
    print("  Then rerun this script to extract CWRU .mat files")

print("\nDone. Check data/mit-bih/ and data/cwru-bearing/")
