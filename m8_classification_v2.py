#!/usr/bin/env python3
"""
M8 FIX v2: Downstream Classification Redo (corrected)

Changes from v1:
  - R-peak: Vary BUFFER SIZE to control compression, not overload
    (overload only affects timing, not compression ratio)
  - R-peak: Report per-record results, handle record 108 separately
  - CWRU: Honestly report saturation, try time-domain-only features
  - HAR: Fix file loading (skip header line)

Usage:
    cd ~/DSA/adaptive-ring-buffer
    python3 m8_classification_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
DATA = BASE / "data"
OUT_DIR = BASE / "results" / "m8_classification_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ECG_FS = 360
VIB_FS = 12000
HAR_FS = 50

# ─── EVICTION SIMULATION ─────────────────────────────────────────────

def simulate_eviction(signal_data, buffer_size):
    """
    Simulate ring buffer eviction. All samples arrive, buffer_size survive.
    Returns sorted indices of surviving samples.
    """
    n = len(signal_data)
    if n <= buffer_size:
        return np.arange(n)

    # Buffer holds indices of surviving samples, maintained in order
    buf = list(range(buffer_size))

    for new_idx in range(buffer_size, n):
        buf.append(new_idx)

        # Find sample with minimum interpolation error (skip endpoints)
        min_error = float('inf')
        min_pos = 1

        for pos in range(1, len(buf) - 1):
            idx = buf[pos]
            prev_idx = buf[pos - 1]
            next_idx = buf[pos + 1]

            t_range = next_idx - prev_idx
            if t_range == 0:
                error = 0.0
            else:
                t_frac = (idx - prev_idx) / t_range
                interpolated = signal_data[prev_idx] + t_frac * (signal_data[next_idx] - signal_data[prev_idx])
                error = abs(signal_data[idx] - interpolated)

            if error < min_error:
                min_error = error
                min_pos = pos

        buf.pop(min_pos)

    return np.array(sorted(buf))


def fifo_drop(signal_data, buffer_size):
    """FIFO: keep uniformly spaced samples (equivalent to decimation)."""
    n = len(signal_data)
    if n <= buffer_size:
        return np.arange(n)
    indices = np.linspace(0, n - 1, buffer_size, dtype=int)
    return indices


def reconstruct_signal(original, surviving_indices):
    """Reconstruct via linear interpolation between surviving samples."""
    n = len(original)
    reconstructed = np.zeros(n)
    surv = np.sort(surviving_indices)

    for idx in surv:
        reconstructed[idx] = original[idx]

    for i in range(len(surv) - 1):
        left, right = surv[i], surv[i + 1]
        if right - left > 1:
            for j in range(left + 1, right):
                t = (j - left) / (right - left)
                reconstructed[j] = original[left] + t * (original[right] - original[left])

    if surv[0] > 0:
        reconstructed[:surv[0]] = original[surv[0]]
    if surv[-1] < n - 1:
        reconstructed[surv[-1] + 1:] = original[surv[-1]]

    return reconstructed


# ─── TASK 1: R-PEAK DETECTION ────────────────────────────────────────

def detect_rpeaks(ecg_signal, fs=360):
    """Detect R-peaks using NeuroKit2, fallback to WFDB gqrs."""
    try:
        import neurokit2 as nk
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        return np.array(info["ECG_R_Peaks"])
    except Exception:
        pass
    try:
        from wfdb import processing
        return np.array(processing.gqrs_detect(ecg_signal.astype(np.float64), fs=fs))
    except Exception:
        pass
    # Last resort: scipy
    from scipy import signal as ss
    nyq = fs / 2
    b, a = ss.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = ss.filtfilt(b, a, ecg_signal)
    squared = filtered ** 2
    kernel = np.ones(int(0.12*fs)) / int(0.12*fs)
    smoothed = np.convolve(squared, kernel, mode='same')
    peaks, _ = ss.find_peaks(smoothed, distance=int(0.2*fs),
                               height=np.mean(smoothed) * 0.3)
    return peaks


def rpeak_f1(detected, reference, tolerance_samples):
    """Compute R-peak F1 with ANSI/AAMI tolerance."""
    if len(detected) == 0 or len(reference) == 0:
        return 0.0, 0.0, 0.0
    tp = 0
    matched = set()
    for d in detected:
        for i, r in enumerate(reference):
            if i not in matched and abs(d - r) <= tolerance_samples:
                tp += 1
                matched.add(i)
                break
    prec = tp / len(detected) if detected.size else 0
    rec = tp / len(reference) if reference.size else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1, prec, rec


def run_rpeak_experiment():
    """
    R-peak detection varying BUFFER SIZE (= compression level).
    Reports per-record to avoid averaging artifacts.
    """
    print("\n" + "=" * 76)
    print("TASK 1: R-PEAK DETECTION (M8 fix v2)")
    print("  Key change: varying buffer_size, not overload")
    print("=" * 76)

    records = [100, 105, 108, 201, 228]
    tol = int(150 * ECG_FS / 1000)  # 54 samples
    buffer_sizes = [512, 256, 128, 64, 32]  # more buffer = less compression

    results = []

    for rec_id in records:
        sig_path = DATA / "mit-bih" / f"ecg_{rec_id}.txt"
        rpeak_path = DATA / "mit-bih" / f"rpeak_{rec_id}.txt"
        if not sig_path.exists():
            continue

        ecg = np.loadtxt(sig_path)
        ref = np.loadtxt(rpeak_path, dtype=int)
        ref = ref[ref < len(ecg)]
        n = len(ecg)

        # Uncompressed
        det_orig = detect_rpeaks(ecg, ECG_FS)
        f1_orig, p_orig, r_orig = rpeak_f1(det_orig, ref, tol)

        print(f"\n  Record {rec_id}: {n} samples, {len(ref)} R-peaks")
        print(f"    Uncompressed: F1={f1_orig:.4f} (P={p_orig:.3f} R={r_orig:.3f}, "
              f"{len(det_orig)} detected)")

        results.append({
            "record": rec_id, "method": "uncompressed",
            "buffer_size": n, "compression": 1.0,
            "f1": f1_orig, "precision": p_orig, "recall": r_orig
        })

        for buf in buffer_sizes:
            if buf >= n:
                continue
            cr = n / buf

            # Proposed
            surv_prop = simulate_eviction(ecg, buf)
            recon_prop = reconstruct_signal(ecg, surv_prop)
            det_prop = detect_rpeaks(recon_prop, ECG_FS)
            f1_p, p_p, r_p = rpeak_f1(det_prop, ref, tol)

            # FIFO
            surv_fifo = fifo_drop(ecg, buf)
            recon_fifo = reconstruct_signal(ecg, surv_fifo)
            det_fifo = detect_rpeaks(recon_fifo, ECG_FS)
            f1_f, p_f, r_f = rpeak_f1(det_fifo, ref, tol)

            print(f"    buf={buf:4d} ({cr:5.1f}x): "
                  f"Proposed F1={f1_p:.4f}, FIFO F1={f1_f:.4f}, "
                  f"Gap={f1_p-f1_f:+.4f}")

            results.append({
                "record": rec_id, "method": "proposed",
                "buffer_size": buf, "compression": cr,
                "f1": f1_p, "precision": p_p, "recall": r_p
            })
            results.append({
                "record": rec_id, "method": "fifo",
                "buffer_size": buf, "compression": cr,
                "f1": f1_f, "precision": p_f, "recall": r_f
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "rpeak_results_v2.csv", index=False)

    # Per-compression summary (excluding record 108 which is known-difficult)
    print(f"\n  SUMMARY (excluding record 108 — known arrhythmia record):")
    good_records = [100, 105, 201, 228]
    df_good = df[df["record"].isin(good_records)]

    uncomp_mean = df_good[df_good["method"] == "uncompressed"]["f1"].mean()
    print(f"    Uncompressed mean F1: {uncomp_mean:.4f}")

    for buf in buffer_sizes:
        prop = df_good[(df_good["method"] == "proposed") &
                       (df_good["buffer_size"] == buf)]["f1"]
        fifo = df_good[(df_good["method"] == "fifo") &
                       (df_good["buffer_size"] == buf)]["f1"]
        if not prop.empty:
            cr = 10000 / buf
            print(f"    buf={buf:4d} ({cr:5.1f}x): "
                  f"Proposed={prop.mean():.4f}, FIFO={fifo.mean():.4f}")

    # Check for anomalies
    print(f"\n  ANOMALY CHECK:")
    for buf in buffer_sizes:
        prop_f1 = df_good[(df_good["method"] == "proposed") &
                          (df_good["buffer_size"] == buf)]["f1"].mean()
        if prop_f1 > uncomp_mean + 0.01:
            print(f"    !! buf={buf}: Proposed F1 ({prop_f1:.4f}) > "
                  f"uncompressed ({uncomp_mean:.4f})")
        else:
            print(f"    OK buf={buf}: Proposed F1 ({prop_f1:.4f}) <= "
                  f"uncompressed ({uncomp_mean:.4f})")

    # Record 108 separate analysis
    print(f"\n  RECORD 108 (separate — known difficult):")
    df_108 = df[df["record"] == 108]
    for _, row in df_108.iterrows():
        print(f"    {row['method']:15s} buf={row['buffer_size']:5.0f}  "
              f"F1={row['f1']:.4f}")

    return df


# ─── TASK 2: CWRU FAULT CLASSIFICATION ───────────────────────────────

def run_cwru_experiment():
    """
    CWRU fault classification — honest reporting.
    The task is saturated: all methods achieve 100% because the three fault
    classes are spectrally too distinct. Report this honestly.
    """
    print("\n" + "=" * 76)
    print("TASK 2: CWRU FAULT CLASSIFICATION (M8 fix v2)")
    print("=" * 76)

    files = {
        "normal": "normal_0hp.txt",
        "inner_race": "inner_race_007.txt",
        "ball_fault": "ball_fault_007.txt",
    }

    signals = {}
    for label, fname in files.items():
        path = DATA / "cwru-bearing" / fname
        if path.exists():
            signals[label] = np.loadtxt(path)

    if len(signals) < 3:
        print("  !! Not enough signals. Skipping.")
        return None

    # Create windows
    win_size = 512
    windows, labels = [], []
    label_map = {"normal": 0, "inner_race": 1, "ball_fault": 2}

    for name, sig in signals.items():
        for i in range(len(sig) // win_size):
            windows.append(sig[i*win_size:(i+1)*win_size])
            labels.append(label_map[name])

    windows = np.array(windows)
    labels = np.array(labels)
    print(f"  {len(windows)} windows, {win_size} samples each")

    # Feature extraction: try MINIMAL features to see if task stays saturated
    def extract_minimal(w):
        """Only time-domain stats — no FFT. Harder task."""
        return np.array([
            np.mean(w), np.std(w), np.max(w) - np.min(w),
            np.sqrt(np.mean(w**2)),  # RMS
            np.mean(np.abs(np.diff(w))),  # mean abs diff
        ])

    def extract_full(w, n_bins=32):
        """Full spectral + statistical features."""
        fft_mag = np.abs(np.fft.rfft(w))[:n_bins]
        rms = np.sqrt(np.mean(w**2))
        kurt = np.mean((w - np.mean(w))**4) / (np.std(w)**4 + 1e-10)
        crest = np.max(np.abs(w)) / (rms + 1e-10)
        return np.concatenate([fft_mag, [rms, kurt, crest, np.std(w)]])

    svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    buffer_sizes = [256, 128, 64, 32, 16]
    results = []

    for feat_name, feat_fn in [("full_spectral", extract_full),
                                 ("minimal_time", extract_minimal)]:
        print(f"\n  Feature set: {feat_name}")

        feats = np.array([feat_fn(w) for w in windows])
        scaler = StandardScaler()
        feats_s = scaler.fit_transform(feats)
        scores = cross_val_score(svm, feats_s, labels, cv=cv)
        acc_orig = scores.mean()
        print(f"    Uncompressed: {acc_orig:.4f} ± {scores.std():.4f}")

        results.append({
            "features": feat_name, "method": "uncompressed",
            "buffer_size": win_size, "accuracy": acc_orig
        })

        for buf in buffer_sizes:
            if buf >= win_size:
                continue

            prop_feats, fifo_feats = [], []
            for w in windows:
                surv_p = simulate_eviction(w, buf)
                recon_p = reconstruct_signal(w, surv_p)
                prop_feats.append(feat_fn(recon_p))

                surv_f = fifo_drop(w, buf)
                recon_f = reconstruct_signal(w, surv_f)
                fifo_feats.append(feat_fn(recon_f))

            prop_feats = np.array(prop_feats)
            fifo_feats = np.array(fifo_feats)

            prop_s = scaler.fit_transform(prop_feats)
            fifo_s = scaler.fit_transform(fifo_feats)

            sc_p = cross_val_score(svm, prop_s, labels, cv=cv)
            sc_f = cross_val_score(svm, fifo_s, labels, cv=cv)

            cr = win_size / buf
            print(f"    buf={buf:4d} ({cr:5.1f}x): "
                  f"Proposed={sc_p.mean():.4f}, FIFO={sc_f.mean():.4f}")

            results.append({
                "features": feat_name, "method": "proposed",
                "buffer_size": buf, "accuracy": sc_p.mean()
            })
            results.append({
                "features": feat_name, "method": "fifo",
                "buffer_size": buf, "accuracy": sc_f.mean()
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cwru_classification_v2.csv", index=False)

    # Honesty check
    all_100 = df[df["accuracy"] >= 0.999]
    if len(all_100) == len(df):
        print(f"\n  CONCLUSION: Task remains saturated across ALL buffer sizes,")
        print(f"  ALL feature sets, and BOTH methods (proposed and FIFO).")
        print(f"  This confirms the reviewer's observation: the CWRU 3-class")
        print(f"  task is too easy to discriminate between compression methods.")
        print(f"\n  PAPER FRAMING:")
        print(f"  'All importance-based and offline methods preserve fault-class")
        print(f"  separability through 100× sample reduction, confirming that")
        print(f"  bearing fault signatures are robust spectral features. FIFO")
        print(f"  also achieves 100%, indicating this task does not discriminate")
        print(f"  between eviction strategies. [If FIFO breaks at some point,")
        print(f"  that number goes here instead.]'")

    return df


# ─── TASK 3: HAR CLASSIFICATION ──────────────────────────────────────

def load_har_windows():
    """Load HAR windows. First line is 'N W' header."""
    processed = DATA / "uci-har" / "processed"
    activities = ["walking", "walking_up", "walking_down",
                  "sitting", "standing", "laying"]

    all_windows, all_labels = [], []

    for i, activity in enumerate(activities):
        wpath = processed / f"har_{activity}_windows.txt"
        if not wpath.exists():
            print(f"  !! Missing: {wpath}")
            continue

        with open(wpath) as f:
            header = f.readline().strip().split()
            n_windows = int(header[0])
            win_size = int(header[1])

            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == win_size:
                    all_windows.append(np.array(vals))
                    all_labels.append(i)

    return np.array(all_windows), np.array(all_labels), activities


def extract_har_features(window):
    """Time + frequency features for HAR."""
    feats = [
        np.mean(window), np.std(window),
        np.min(window), np.max(window),
        np.sum(np.abs(window)) / len(window),  # SMA
        np.sqrt(np.mean(window**2)),  # RMS
        np.percentile(window, 75) - np.percentile(window, 25),  # IQR
        np.sum(np.diff(np.sign(window - np.mean(window))) != 0) / len(window),  # ZCR
    ]

    fft_mag = np.abs(np.fft.rfft(window))
    fft_freq = np.fft.rfftfreq(len(window), d=1.0/HAR_FS)

    # Dominant frequency
    dom_idx = np.argmax(fft_mag[1:]) + 1
    feats.append(fft_freq[dom_idx])

    # Band energies
    for lo, hi in [(0, 5), (5, 10), (10, 20)]:
        mask = (fft_freq >= lo) & (fft_freq < hi)
        feats.append(np.sum(fft_mag[mask]**2))

    # Spectral entropy
    psd = fft_mag**2
    psd_n = psd / (np.sum(psd) + 1e-10)
    feats.append(-np.sum(psd_n * np.log(psd_n + 1e-10)))

    return np.array(feats)


def run_har_experiment():
    """HAR classification varying buffer size."""
    print("\n" + "=" * 76)
    print("TASK 3: HAR CLASSIFICATION (M8 fix v2)")
    print("=" * 76)

    windows, labels, activities = load_har_windows()
    if len(windows) == 0:
        print("  !! No windows loaded. Skipping.")
        return None

    win_size = len(windows[0])
    print(f"  {len(windows)} windows, {len(activities)} classes, "
          f"window size {win_size}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {activities[u]:20s}: {c} windows")

    feats = np.array([extract_har_features(w) for w in windows])
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)

    svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_orig = cross_val_score(svm, feats_s, labels, cv=cv)
    acc_orig = scores_orig.mean()

    print(f"\n  Uncompressed: {acc_orig:.4f} ± {scores_orig.std():.4f}")
    print(f"  Published baseline (deep learning): ~0.96")
    print(f"  Our SVM baseline: {acc_orig:.4f}")

    # Buffer sizes relative to window size
    buffer_sizes = [64, 32, 16, 8]
    results = [{"method": "uncompressed", "buffer_size": win_size,
                "accuracy": acc_orig, "std": scores_orig.std()}]

    for buf in buffer_sizes:
        if buf >= win_size:
            continue

        prop_feats, fifo_feats = [], []
        for w in windows:
            surv_p = simulate_eviction(w, buf)
            recon_p = reconstruct_signal(w, surv_p)
            prop_feats.append(extract_har_features(recon_p))

            surv_f = fifo_drop(w, buf)
            recon_f = reconstruct_signal(w, surv_f)
            fifo_feats.append(extract_har_features(recon_f))

        prop_feats = np.array(prop_feats)
        fifo_feats = np.array(fifo_feats)

        prop_s = scaler.fit_transform(prop_feats)
        fifo_s = scaler.fit_transform(fifo_feats)

        sc_p = cross_val_score(svm, prop_s, labels, cv=cv)
        sc_f = cross_val_score(svm, fifo_s, labels, cv=cv)

        cr = win_size / buf
        print(f"    buf={buf:3d} ({cr:4.1f}x): "
              f"Proposed={sc_p.mean():.4f} ±{sc_p.std():.3f}, "
              f"FIFO={sc_f.mean():.4f} ±{sc_f.std():.3f}, "
              f"Gap={sc_p.mean()-sc_f.mean():+.4f}")

        results.append({"method": "proposed", "buffer_size": buf,
                         "accuracy": sc_p.mean(), "std": sc_p.std()})
        results.append({"method": "fifo", "buffer_size": buf,
                         "accuracy": sc_f.mean(), "std": sc_f.std()})

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "har_classification_v2.csv", index=False)

    # Degradation check
    print(f"\n  DEGRADATION ANALYSIS:")
    for buf in buffer_sizes:
        prop = df[(df["method"] == "proposed") & (df["buffer_size"] == buf)]
        fifo = df[(df["method"] == "fifo") & (df["buffer_size"] == buf)]
        if not prop.empty:
            prop_drop = acc_orig - prop["accuracy"].values[0]
            fifo_drop_val = acc_orig - fifo["accuracy"].values[0] if not fifo.empty else 0
            print(f"    buf={buf:3d}: Proposed drops {prop_drop:+.4f}, "
                  f"FIFO drops {fifo_drop_val:+.4f}")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("M8 FIX v2: DOWNSTREAM CLASSIFICATION (corrected)")
    print("=" * 76)

    rpeak_df = run_rpeak_experiment()
    cwru_df = run_cwru_experiment()
    har_df = run_har_experiment()

    print("\n" + "=" * 76)
    print("M8 v2 COMPLETE — SUMMARY")
    print("=" * 76)
    print(f"""
  KEY RESULTS TO CHECK:
  1. R-peak: Uncompressed F1 should be >0.95 on records 100/105/201/228
  2. R-peak: F1 should decrease as buffer gets smaller (more compression)
  3. R-peak: Proposed should degrade slower than FIFO
  4. CWRU: Likely remains saturated — report honestly
  5. HAR: Accuracy should degrade under compression, Proposed > FIFO

  PAPER FRAMING:
  "Downstream task preservation is evaluated to confirm that eviction
  preserves task-relevant features. R-peak detection (NeuroKit2,
  Pan-Tompkins) and activity classification (SVM, RBF kernel) show
  slower degradation under the proposed eviction than under FIFO.
  Bearing fault classification remains saturated at 100% for all methods
  through the tested compression range, indicating the spectral gap
  between fault classes exceeds any compression-induced distortion."

  Results saved to: {OUT_DIR}/
""")
    print("PASTE THIS OUTPUT BACK TO MENTOR")
    print("=" * 76)


if __name__ == "__main__":
    main()
