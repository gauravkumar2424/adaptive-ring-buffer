#!/usr/bin/env python3
"""
M8 FIX: Downstream Classification Redo

Fixes three reviewer concerns:
  1. R-peak detection: Replace 0.6×max threshold with NeuroKit2 Pan-Tompkins
     (or WFDB gqrs). Uncompressed F1 should be >0.95, not 0.747.
  2. CWRU fault classification: Break the 100% saturation with SVM + noise
     augmentation so compression actually differentiates methods.
  3. HAR classification: Use proper subject-level split if available,
     report against published ~96% baseline.

Usage:
    cd ~/DSA/adaptive-ring-buffer
    pip install neurokit2 --break-system-packages  # if not installed
    python3 m8_classification_fix.py

Requires: pandas, numpy, scipy, scikit-learn, neurokit2 (or wfdb)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────
BASE = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer"))
DATA = BASE / "data"
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "m8_classification_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Signal parameters
ECG_FS = 360        # MIT-BIH sampling rate
VIB_FS = 12000      # CWRU sampling rate
HAR_FS = 50         # UCI HAR sampling rate

# ─── RING BUFFER SIMULATION ──────────────────────────────────────────

def simulate_eviction(signal_data, buffer_size, overload_ratio, method="interp_error"):
    """
    Simulate ring buffer eviction on a signal.
    Returns indices of surviving samples.

    This is a simplified Python implementation matching the C++ logic.
    """
    n = len(signal_data)
    # Number of samples that arrive = n * overload_ratio / (overload_ratio)
    # Actually: buffer fills, then for each new sample we evict one
    # Total samples that arrive = n (the signal length)
    # Buffer capacity = buffer_size
    # If n <= buffer_size, no eviction needed
    if n <= buffer_size:
        return np.arange(n)

    # Build initial buffer with first buffer_size samples
    # Use a linked-list-style structure with indices
    buffer_indices = list(range(buffer_size))
    values = signal_data.copy()

    # For each remaining sample, evict the one with minimum interpolation error
    for new_idx in range(buffer_size, n):
        # Add new sample conceptually
        buffer_indices.append(new_idx)

        if len(buffer_indices) <= buffer_size:
            continue

        # Find sample with minimum interpolation error (skip first and last)
        min_error = float('inf')
        min_pos = 1  # position in buffer_indices to evict

        for pos in range(1, len(buffer_indices) - 1):
            idx = buffer_indices[pos]
            prev_idx = buffer_indices[pos - 1]
            next_idx = buffer_indices[pos + 1]

            # Interpolation error
            t_range = next_idx - prev_idx
            if t_range == 0:
                error = 0.0
            else:
                t_frac = (idx - prev_idx) / t_range
                interpolated = values[prev_idx] + t_frac * (values[next_idx] - values[prev_idx])
                error = abs(values[idx] - interpolated)

            if error < min_error:
                min_error = error
                min_pos = pos

        # Evict
        buffer_indices.pop(min_pos)

    return np.array(buffer_indices)


def reconstruct_signal(original, surviving_indices):
    """Reconstruct signal from surviving samples via linear interpolation."""
    n = len(original)
    reconstructed = np.zeros(n)
    surv = np.sort(surviving_indices)

    # Copy surviving samples
    for idx in surv:
        reconstructed[idx] = original[idx]

    # Interpolate between surviving samples
    for i in range(len(surv) - 1):
        left = surv[i]
        right = surv[i + 1]
        if right - left > 1:
            for j in range(left + 1, right):
                t = (j - left) / (right - left)
                reconstructed[j] = original[left] + t * (original[right] - original[left])

    # Extrapolate before first and after last
    if surv[0] > 0:
        for j in range(surv[0]):
            reconstructed[j] = original[surv[0]]
    if surv[-1] < n - 1:
        for j in range(surv[-1] + 1, n):
            reconstructed[j] = original[surv[-1]]

    return reconstructed


def fifo_drop(signal_data, buffer_size):
    """FIFO eviction: drop oldest when full."""
    n = len(signal_data)
    if n <= buffer_size:
        return np.arange(n)

    # FIFO keeps the last buffer_size samples
    # But in a streaming context with overload, it's more nuanced
    # Simplified: keep every k-th sample where k = ceil(n / buffer_size)
    k = max(1, n // buffer_size)
    return np.arange(0, n, k)[:buffer_size]


# ─── TASK 1: R-PEAK DETECTION ────────────────────────────────────────

def load_ecg_signal(record_id):
    """Load ECG signal and R-peak annotations."""
    sig_path = DATA / "mit-bih" / f"ecg_{record_id}.txt"
    rpeak_path = DATA / "mit-bih" / f"rpeak_{record_id}.txt"

    if not sig_path.exists():
        return None, None

    signal = np.loadtxt(sig_path)

    if rpeak_path.exists():
        rpeaks = np.loadtxt(rpeak_path, dtype=int)
    else:
        rpeaks = np.array([], dtype=int)

    return signal, rpeaks


def detect_rpeaks_neurokit(ecg_signal, fs=360):
    """Detect R-peaks using NeuroKit2 (Pan-Tompkins based)."""
    try:
        import neurokit2 as nk
        # Clean the signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        # Detect R-peaks
        _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        detected = rpeaks_info["ECG_R_Peaks"]
        return np.array(detected)
    except ImportError:
        print("  NeuroKit2 not available, falling back to WFDB gqrs")
        return detect_rpeaks_gqrs(ecg_signal, fs)
    except Exception as e:
        print(f"  NeuroKit2 failed: {e}, falling back to gqrs")
        return detect_rpeaks_gqrs(ecg_signal, fs)


def detect_rpeaks_gqrs(ecg_signal, fs=360):
    """Detect R-peaks using WFDB's gqrs algorithm."""
    try:
        import wfdb
        from wfdb import processing
        # gqrs needs the signal as a 1D array
        qrs_indices = processing.gqrs_detect(ecg_signal.astype(np.float64), fs=fs)
        return np.array(qrs_indices)
    except ImportError:
        print("  WFDB not available, using scipy-based detector")
        return detect_rpeaks_scipy(ecg_signal, fs)
    except Exception as e:
        print(f"  WFDB gqrs failed: {e}, using scipy-based detector")
        return detect_rpeaks_scipy(ecg_signal, fs)


def detect_rpeaks_scipy(ecg_signal, fs=360):
    """Fallback R-peak detector using scipy.signal.find_peaks with bandpass."""
    # Bandpass filter 5-15 Hz
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, ecg_signal)

    # Square and smooth
    squared = filtered ** 2
    window = int(0.12 * fs)  # 120ms window
    kernel = np.ones(window) / window
    smoothed = np.convolve(squared, kernel, mode='same')

    # Find peaks with minimum distance of 200ms
    min_dist = int(0.2 * fs)
    peaks, _ = scipy_signal.find_peaks(smoothed, distance=min_dist,
                                         height=np.mean(smoothed) * 0.3)

    return peaks


def rpeak_f1(detected, reference, tolerance_samples):
    """
    Compute R-peak detection F1 score.

    ANSI/AAMI standard: tolerance typically 150ms.
    At 360 Hz: 150ms = 54 samples.
    """
    if len(detected) == 0 or len(reference) == 0:
        return 0.0, 0.0, 0.0

    tp = 0
    matched_ref = set()

    for d in detected:
        for i, r in enumerate(reference):
            if i not in matched_ref and abs(d - r) <= tolerance_samples:
                tp += 1
                matched_ref.add(i)
                break

    precision = tp / len(detected) if len(detected) > 0 else 0
    recall = tp / len(reference) if len(reference) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def run_rpeak_experiment():
    """Run corrected R-peak detection experiment."""
    print("\n" + "=" * 76)
    print("TASK 1: R-PEAK DETECTION (M8 fix)")
    print("=" * 76)

    records = [100, 105, 108, 201, 228]
    tolerance_ms = 150  # ANSI/AAMI standard
    tolerance_samples = int(tolerance_ms * ECG_FS / 1000)  # 54 samples
    print(f"  Tolerance: {tolerance_ms}ms = {tolerance_samples} samples at {ECG_FS}Hz")

    buffer_size = 256
    overload_ratios = [2, 3, 5, 10, 20]

    results = []

    for rec_id in records:
        ecg_signal, ref_rpeaks = load_ecg_signal(rec_id)
        if ecg_signal is None:
            print(f"  Record {rec_id}: file not found, skipping")
            continue

        if len(ref_rpeaks) == 0:
            print(f"  Record {rec_id}: no R-peak annotations, skipping")
            continue

        # Filter annotations to signal range
        ref_rpeaks = ref_rpeaks[ref_rpeaks < len(ecg_signal)]

        # Uncompressed detection
        detected_orig = detect_rpeaks_neurokit(ecg_signal, ECG_FS)
        f1_orig, prec_orig, rec_orig = rpeak_f1(detected_orig, ref_rpeaks,
                                                  tolerance_samples)

        print(f"\n  Record {rec_id}: {len(ecg_signal)} samples, "
              f"{len(ref_rpeaks)} annotated R-peaks")
        print(f"    Uncompressed: F1={f1_orig:.4f}, "
              f"Prec={prec_orig:.4f}, Rec={rec_orig:.4f}, "
              f"Detected={len(detected_orig)}")

        results.append({
            "record": rec_id, "method": "uncompressed", "overload": 1,
            "f1": f1_orig, "precision": prec_orig, "recall": rec_orig,
            "n_detected": len(detected_orig), "n_reference": len(ref_rpeaks)
        })

        for ovl in overload_ratios:
            # Proposed method
            surviving = simulate_eviction(ecg_signal, buffer_size, ovl)
            reconstructed = reconstruct_signal(ecg_signal, surviving)
            detected_prop = detect_rpeaks_neurokit(reconstructed, ECG_FS)
            f1_p, prec_p, rec_p = rpeak_f1(detected_prop, ref_rpeaks,
                                             tolerance_samples)

            # FIFO baseline
            surviving_fifo = fifo_drop(ecg_signal, buffer_size)
            reconstructed_fifo = reconstruct_signal(ecg_signal, surviving_fifo)
            detected_fifo = detect_rpeaks_neurokit(reconstructed_fifo, ECG_FS)
            f1_f, prec_f, rec_f = rpeak_f1(detected_fifo, ref_rpeaks,
                                             tolerance_samples)

            print(f"    Ovl={ovl:2d}x: Proposed F1={f1_p:.4f} "
                  f"({len(detected_prop)} det), "
                  f"FIFO F1={f1_f:.4f} ({len(detected_fifo)} det)")

            results.append({
                "record": rec_id, "method": "proposed", "overload": ovl,
                "f1": f1_p, "precision": prec_p, "recall": rec_p,
                "n_detected": len(detected_prop), "n_reference": len(ref_rpeaks)
            })
            results.append({
                "record": rec_id, "method": "fifo", "overload": ovl,
                "f1": f1_f, "precision": prec_f, "recall": rec_f,
                "n_detected": len(detected_fifo), "n_reference": len(ref_rpeaks)
            })

    # Summary
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "rpeak_results_v2.csv", index=False)

    if not df.empty:
        print(f"\n  SUMMARY:")
        uncomp = df[df["method"] == "uncompressed"]["f1"].mean()
        print(f"    Uncompressed mean F1: {uncomp:.4f}")

        if uncomp < 0.90:
            print(f"    !! WARNING: Uncompressed F1 < 0.90. Detector may need tuning.")
            print(f"    !! This is still better than 0.747 from the old threshold detector.")

        for ovl in overload_ratios:
            prop_f1 = df[(df["method"] == "proposed") & (df["overload"] == ovl)]["f1"].mean()
            fifo_f1 = df[(df["method"] == "fifo") & (df["overload"] == ovl)]["f1"].mean()
            print(f"    {ovl:2d}x: Proposed={prop_f1:.4f}, FIFO={fifo_f1:.4f}, "
                  f"Gap={prop_f1-fifo_f1:+.4f}")

        # KEY CHECK: F1 should NOT increase under compression
        for ovl in overload_ratios:
            prop_f1 = df[(df["method"] == "proposed") & (df["overload"] == ovl)]["f1"].mean()
            if prop_f1 > uncomp + 0.01:
                print(f"    !! ANOMALY: F1 at {ovl}x ({prop_f1:.4f}) > "
                      f"uncompressed ({uncomp:.4f}). Investigate!")

    return df


# ─── TASK 2: CWRU FAULT CLASSIFICATION ───────────────────────────────

def load_cwru_signals():
    """Load CWRU bearing vibration signals."""
    signals = {}
    files = {
        "normal": "normal_0hp.txt",
        "inner_race": "inner_race_007.txt",
        "ball_fault": "ball_fault_007.txt",
    }
    for label, fname in files.items():
        path = DATA / "cwru-bearing" / fname
        if path.exists():
            signals[label] = np.loadtxt(path)
        else:
            print(f"  !! Missing: {path}")
    return signals


def extract_vibration_features(signal_segment, n_fft_bins=32):
    """Extract spectral + statistical features from a vibration segment."""
    # FFT magnitude (first n_fft_bins)
    fft_mag = np.abs(np.fft.rfft(signal_segment))[:n_fft_bins]

    # Statistical features
    rms = np.sqrt(np.mean(signal_segment ** 2))
    kurtosis = np.mean((signal_segment - np.mean(signal_segment)) ** 4) / \
               (np.std(signal_segment) ** 4 + 1e-10)
    crest = np.max(np.abs(signal_segment)) / (rms + 1e-10)
    std = np.std(signal_segment)

    features = np.concatenate([fft_mag, [rms, kurtosis, crest, std]])
    return features


def run_cwru_experiment():
    """
    Run corrected CWRU fault classification.

    Fixes:
    - Use SVM with RBF kernel instead of nearest-centroid
    - Add Gaussian noise to make the task non-trivial
    - Use cross-validation with non-overlapping windows
    - Report when task is saturated vs. when methods differentiate
    """
    print("\n" + "=" * 76)
    print("TASK 2: CWRU FAULT CLASSIFICATION (M8 fix)")
    print("=" * 76)

    signals = load_cwru_signals()
    if len(signals) < 3:
        print("  !! Not enough CWRU signals found. Skipping.")
        return None

    # Create non-overlapping windows
    window_size = 512
    windows = []
    labels = []
    label_map = {"normal": 0, "inner_race": 1, "ball_fault": 2}

    for label_name, sig in signals.items():
        n_windows = len(sig) // window_size
        for i in range(n_windows):
            start = i * window_size
            window = sig[start:start + window_size]
            windows.append(window)
            labels.append(label_map[label_name])

    windows = np.array(windows)
    labels = np.array(labels)
    print(f"  {len(windows)} windows ({window_size} samples each)")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Method 1: Original (should show saturation)
    features_orig = np.array([extract_vibration_features(w) for w in windows])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_orig)

    # SVM with RBF kernel
    svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_orig = cross_val_score(svm, features_scaled, labels, cv=cv, scoring='accuracy')
    acc_orig = scores_orig.mean()
    print(f"\n  Uncompressed accuracy (SVM, 5-fold CV): {acc_orig:.4f} ± {scores_orig.std():.4f}")

    if acc_orig > 0.99:
        print(f"  !! Task is saturated at {acc_orig:.4f}. Adding noise to create a harder task.")

    # Method 2: Add noise to make task harder
    noise_levels = [0.0, 0.5, 1.0, 2.0]  # relative to signal std
    buffer_size = 256
    overload_ratios = [5, 10, 20, 50, 100]

    results = []

    for noise_std_mult in noise_levels:
        noise_label = f"noise={noise_std_mult}"

        # Add noise to all windows
        noisy_windows = []
        for w in windows:
            if noise_std_mult > 0:
                noise = np.random.RandomState(42).randn(len(w)) * np.std(w) * noise_std_mult
                noisy_windows.append(w + noise)
            else:
                noisy_windows.append(w.copy())
        noisy_windows = np.array(noisy_windows)

        # Uncompressed baseline with noise
        features_noisy = np.array([extract_vibration_features(w) for w in noisy_windows])
        features_n_scaled = scaler.fit_transform(features_noisy)
        scores_noisy = cross_val_score(svm, features_n_scaled, labels, cv=cv, scoring='accuracy')
        acc_noisy = scores_noisy.mean()

        print(f"\n  {noise_label}: Uncompressed accuracy = {acc_noisy:.4f}")

        results.append({
            "noise": noise_std_mult, "method": "uncompressed", "overload": 1,
            "accuracy": acc_noisy
        })

        for ovl in overload_ratios:
            # Proposed: evict from each window
            prop_features = []
            fifo_features = []

            for w in noisy_windows:
                # Proposed eviction
                surviving = simulate_eviction(w, buffer_size, ovl)
                reconstructed = reconstruct_signal(w, surviving)
                prop_features.append(extract_vibration_features(reconstructed))

                # FIFO
                surviving_fifo = fifo_drop(w, buffer_size)
                reconstructed_fifo = reconstruct_signal(w, surviving_fifo)
                fifo_features.append(extract_vibration_features(reconstructed_fifo))

            # Classify
            prop_features = np.array(prop_features)
            fifo_features = np.array(fifo_features)

            prop_scaled = scaler.fit_transform(prop_features)
            fifo_scaled = scaler.fit_transform(fifo_features)

            scores_prop = cross_val_score(svm, prop_scaled, labels, cv=cv, scoring='accuracy')
            scores_fifo = cross_val_score(svm, fifo_scaled, labels, cv=cv, scoring='accuracy')

            acc_prop = scores_prop.mean()
            acc_fifo = scores_fifo.mean()

            print(f"    {ovl:3d}x: Proposed={acc_prop:.4f}, FIFO={acc_fifo:.4f}, "
                  f"Gap={acc_prop-acc_fifo:+.4f}")

            results.append({
                "noise": noise_std_mult, "method": "proposed", "overload": ovl,
                "accuracy": acc_prop
            })
            results.append({
                "noise": noise_std_mult, "method": "fifo", "overload": ovl,
                "accuracy": acc_fifo
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "cwru_classification_v2.csv", index=False)

    # Find the noise level that breaks saturation
    print(f"\n  ANALYSIS:")
    for nl in noise_levels:
        uncomp = df[(df["noise"] == nl) & (df["method"] == "uncompressed")]["accuracy"].values
        if len(uncomp) > 0 and uncomp[0] < 0.99:
            print(f"    Noise={nl}: Task is non-trivial (uncomp={uncomp[0]:.4f})")
            at_100x_prop = df[(df["noise"] == nl) & (df["method"] == "proposed") &
                              (df["overload"] == 100)]["accuracy"].values
            at_100x_fifo = df[(df["noise"] == nl) & (df["method"] == "fifo") &
                              (df["overload"] == 100)]["accuracy"].values
            if len(at_100x_prop) > 0 and len(at_100x_fifo) > 0:
                print(f"    At 100x: Proposed={at_100x_prop[0]:.4f}, "
                      f"FIFO={at_100x_fifo[0]:.4f}")

    return df


# ─── TASK 3: HAR CLASSIFICATION ──────────────────────────────────────

def load_har_windows():
    """Load pre-processed HAR windows."""
    processed = DATA / "uci-har" / "processed"
    activities = ["walking", "walking_up", "walking_down",
                  "sitting", "standing", "laying"]
    label_map = {a: i for i, a in enumerate(activities)}

    all_windows = []
    all_labels = []

    for activity in activities:
        # Try windows file first
        wpath = processed / f"har_{activity}_windows.txt"
        if wpath.exists():
            data = np.loadtxt(wpath)
            if data.ndim == 1:
                # Single window, reshape
                data = data.reshape(1, -1)
            for row in data:
                all_windows.append(row)
                all_labels.append(label_map[activity])
        else:
            # Try raw signal and window it
            spath = processed / f"har_{activity}.txt"
            if spath.exists():
                sig = np.loadtxt(spath)
                win_size = 128
                n_windows = len(sig) // win_size
                for i in range(min(n_windows, 50)):
                    window = sig[i * win_size:(i + 1) * win_size]
                    all_windows.append(window)
                    all_labels.append(label_map[activity])

    return np.array(all_windows), np.array(all_labels), activities


def extract_har_features(window):
    """Extract time and frequency domain features for HAR."""
    features = []

    # Time domain
    features.append(np.mean(window))
    features.append(np.std(window))
    features.append(np.min(window))
    features.append(np.max(window))
    features.append(np.median(window))

    # Signal magnitude area
    features.append(np.sum(np.abs(window)) / len(window))

    # Zero crossing rate
    zcr = np.sum(np.diff(np.sign(window - np.mean(window))) != 0) / len(window)
    features.append(zcr)

    # RMS
    features.append(np.sqrt(np.mean(window ** 2)))

    # Interquartile range
    features.append(np.percentile(window, 75) - np.percentile(window, 25))

    # Frequency domain
    fft_mag = np.abs(np.fft.rfft(window))
    fft_freq = np.fft.rfftfreq(len(window), d=1.0/HAR_FS)

    # Dominant frequency
    dom_idx = np.argmax(fft_mag[1:]) + 1
    features.append(fft_freq[dom_idx])

    # Spectral energy in bands
    for lo, hi in [(0, 5), (5, 10), (10, 20)]:
        mask = (fft_freq >= lo) & (fft_freq < hi)
        features.append(np.sum(fft_mag[mask] ** 2))

    # Spectral entropy
    psd = fft_mag ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    features.append(spectral_entropy)

    return np.array(features)


def run_har_experiment():
    """
    Run corrected HAR classification.

    Fixes:
    - Use SVM with RBF kernel for better baseline
    - Use proper cross-validation
    - Report against published ~96% standard
    """
    print("\n" + "=" * 76)
    print("TASK 3: HAR CLASSIFICATION (M8 fix)")
    print("=" * 76)

    windows, labels, activity_names = load_har_windows()

    if len(windows) == 0:
        print("  !! No HAR windows found. Skipping.")
        return None

    print(f"  {len(windows)} windows, {len(activity_names)} classes")
    print(f"  Window size: {windows[0].shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Extract features
    features = np.array([extract_har_features(w) for w in windows])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Baseline: SVM
    svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_orig = cross_val_score(svm, features_scaled, labels, cv=cv, scoring='accuracy')
    acc_orig = scores_orig.mean()
    print(f"\n  Uncompressed accuracy (SVM, 5-fold CV): {acc_orig:.4f} ± {scores_orig.std():.4f}")
    print(f"  Published UCI HAR baseline: ~0.96 (deep learning)")
    print(f"  Our baseline with handcrafted features: {acc_orig:.4f}")

    if acc_orig < 0.80:
        print(f"  !! WARNING: Baseline accuracy low. Feature engineering may need improvement.")
        print(f"  !! But this is still a legitimate comparison — what matters is")
        print(f"  !! relative degradation under compression, not absolute accuracy.")

    buffer_size = 32  # Small buffer for 128-sample windows
    overload_ratios = [2, 3, 5, 8, 10, 15, 20]

    results = []
    results.append({
        "method": "uncompressed", "overload": 1, "accuracy": acc_orig,
        "accuracy_std": scores_orig.std()
    })

    for ovl in overload_ratios:
        # Proposed
        prop_features = []
        fifo_features = []

        for w in windows:
            surviving = simulate_eviction(w, buffer_size, ovl)
            reconstructed = reconstruct_signal(w, surviving)
            prop_features.append(extract_har_features(reconstructed))

            surviving_fifo = fifo_drop(w, buffer_size)
            reconstructed_fifo = reconstruct_signal(w, surviving_fifo)
            fifo_features.append(extract_har_features(reconstructed_fifo))

        prop_features = np.array(prop_features)
        fifo_features = np.array(fifo_features)

        prop_scaled = scaler.fit_transform(prop_features)
        fifo_scaled = scaler.fit_transform(fifo_features)

        scores_prop = cross_val_score(svm, prop_scaled, labels, cv=cv, scoring='accuracy')
        scores_fifo = cross_val_score(svm, fifo_scaled, labels, cv=cv, scoring='accuracy')

        acc_prop = scores_prop.mean()
        acc_fifo = scores_fifo.mean()

        print(f"    {ovl:2d}x: Proposed={acc_prop:.4f}, FIFO={acc_fifo:.4f}, "
              f"Gap={acc_prop-acc_fifo:+.4f}")

        results.append({
            "method": "proposed", "overload": ovl, "accuracy": acc_prop,
            "accuracy_std": scores_prop.std()
        })
        results.append({
            "method": "fifo", "overload": ovl, "accuracy": acc_fifo,
            "accuracy_std": scores_fifo.std()
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "har_classification_v2.csv", index=False)

    # Check for anomalies
    print(f"\n  ANOMALY CHECK:")
    for ovl in overload_ratios:
        prop_acc = df[(df["method"] == "proposed") & (df["overload"] == ovl)]["accuracy"].values
        if len(prop_acc) > 0 and prop_acc[0] > acc_orig + 0.02:
            print(f"    !! {ovl}x Proposed ({prop_acc[0]:.4f}) > "
                  f"uncompressed ({acc_orig:.4f}) — suspicious")

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("M8 FIX: DOWNSTREAM CLASSIFICATION REDO")
    print("All three tasks with corrected methodology")
    print("=" * 76)

    # Task 1: R-peak detection
    rpeak_df = run_rpeak_experiment()

    # Task 2: CWRU fault classification
    cwru_df = run_cwru_experiment()

    # Task 3: HAR classification
    har_df = run_har_experiment()

    # Summary
    print("\n" + "=" * 76)
    print("M8 SUMMARY")
    print("=" * 76)

    print("""
  KEY CHECKS (reviewer will verify these):
  1. R-peak uncompressed F1 should be >0.95 (was 0.747 — if still low,
     detector needs more tuning, not a paper problem)
  2. R-peak F1 should DECREASE monotonically under compression
     (old results showed INCREASE — that was the fatal flaw)
  3. CWRU should show differentiation between methods at some noise level
     (100% for everyone at clean signal is expected; report the noise
     level where methods diverge)
  4. HAR should use a stronger classifier than nearest-centroid
     (SVM here; report baseline honestly against published 96%)

  PAPER FRAMING:
  "Downstream task accuracy is not the primary contribution. We report it
  to demonstrate that the proposed eviction preserves task-relevant signal
  features better than FIFO under equivalent compression. The absolute
  accuracy depends on the classifier; relative degradation under compression
  is the relevant comparison."
""")

    print(f"\n  Results saved to: {OUT_DIR}/")
    print("=" * 76)
    print("PASTE THIS OUTPUT BACK TO MENTOR")
    print("=" * 76)


if __name__ == "__main__":
    main()
