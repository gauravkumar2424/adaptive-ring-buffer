#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

// ============================================================
// Corrected Evaluation Metrics with Signal Reconstruction
// ============================================================

struct EvalMetrics {
    double mse;
    double snr_db;          // can be +inf — see snr_saturated below
    double max_error;
    double deriv_ratio;
    size_t drop_count;
    double max_wait_ms;
    double r_peak_accuracy;
    double spectral_fidelity;
    bool   degenerate;      // true when drop_count == 0 — NO EVICTION occurred at all
                             // (e.g. WAIT/TIMED_WAIT blocking, or buffer_size >= stream length)
    bool   snr_saturated;   // true when snr_db is non-finite — reconstruction error
                             // underflowed below the noise floor. This can ALSO happen
                             // with drop_count > 0 (e.g. an offline method like RDP found
                             // a near-perfectly-interpolable selection on a signal with
                             // long near-linear segments). degenerate and snr_saturated
                             // are DIFFERENT conditions — check both, don't assume one
                             // implies the other.
};

// ============================================================
// Reconstruct full-length signal from surviving samples
// using linear interpolation between known points.
// ============================================================

inline std::vector<double> reconstruct_signal(
    const std::vector<int>& surviving_indices,
    const std::vector<double>& surviving_values,
    int original_length)
{
    std::vector<double> reconstructed(original_length, 0.0);

    if (surviving_indices.empty()) return reconstructed;

    if (surviving_indices.size() == 1) {
        std::fill(reconstructed.begin(), reconstructed.end(), surviving_values[0]);
        return reconstructed;
    }

    size_t n = surviving_indices.size();

    for (int i = 0; i < surviving_indices[0] && i < original_length; ++i) {
        reconstructed[i] = surviving_values[0];
    }

    for (size_t k = 0; k < n - 1; ++k) {
        int idx0 = surviving_indices[k];
        int idx1 = surviving_indices[k + 1];
        double val0 = surviving_values[k];
        double val1 = surviving_values[k + 1];

        for (int i = idx0; i <= idx1 && i < original_length; ++i) {
            if (idx1 == idx0) {
                reconstructed[i] = val0;
            } else {
                double t = static_cast<double>(i - idx0) / (idx1 - idx0);
                reconstructed[i] = val0 + t * (val1 - val0);
            }
        }
    }

    for (int i = surviving_indices[n - 1]; i < original_length; ++i) {
        reconstructed[i] = surviving_values[n - 1];
    }

    return reconstructed;
}

inline double compute_mse_aligned(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed)
{
    size_t n = std::min(original.size(), reconstructed.size());
    if (n == 0) return std::numeric_limits<double>::max();

    double sum_sq_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double err = reconstructed[i] - original[i];
        sum_sq_err += err * err;
    }
    return sum_sq_err / n;
}

// ============================================================
// SNR: 10 * log10(signal_power / error_power)
//
// Returns +infinity (not a hardcoded 100.0) when noise_power
// underflows below 1e-15. This can occur for two distinct
// reasons — see EvalMetrics.degenerate vs .snr_saturated:
//   1. No eviction occurred (drop_count == 0) — reconstruction
//      is trivially exact because nothing was removed.
//   2. Eviction occurred, but the evicted points happened to lie
//      on (or extremely close to) the line between their
//      surviving neighbors, so linear interpolation reconstructs
//      them almost exactly anyway. This is rarer but real —
//      observed with RDP_OFFLINE on ECG baseline segments.
// ============================================================
inline double compute_snr(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed)
{
    size_t n = std::min(original.size(), reconstructed.size());
    if (n == 0) return -std::numeric_limits<double>::infinity();

    double signal_power = 0.0;
    double noise_power = 0.0;

    for (size_t i = 0; i < n; ++i) {
        signal_power += original[i] * original[i];
        double err = original[i] - reconstructed[i];
        noise_power += err * err;
    }

    if (signal_power < 1e-15) return 0.0;
    if (noise_power < 1e-15) return std::numeric_limits<double>::infinity();

    return 10.0 * std::log10(signal_power / noise_power);
}

inline double compute_max_error(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed)
{
    size_t n = std::min(original.size(), reconstructed.size());
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(original[i] - reconstructed[i]));
    }
    return max_err;
}

inline double compute_deriv_ratio(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed)
{
    auto avg_deriv = [](const std::vector<double>& data) -> double {
        if (data.size() < 2) return 0.0;
        double sum = 0.0;
        for (size_t i = 1; i < data.size(); ++i) {
            sum += std::abs(data[i] - data[i - 1]);
        }
        return sum / (data.size() - 1);
    };

    double orig_deriv = avg_deriv(original);
    double recon_deriv = avg_deriv(reconstructed);

    return (orig_deriv > 1e-15) ? recon_deriv / orig_deriv : 0.0;
}

inline double compute_rpeak_accuracy(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed,
    double threshold_fraction = 0.6,
    int tolerance_samples = 5)
{
    auto detect_peaks = [&](const std::vector<double>& signal) -> std::vector<int> {
        std::vector<int> peaks;
        if (signal.size() < 3) return peaks;

        double max_val = *std::max_element(signal.begin(), signal.end());
        double threshold = max_val * threshold_fraction;

        for (size_t i = 1; i < signal.size() - 1; ++i) {
            if (signal[i] > threshold &&
                signal[i] >= signal[i - 1] &&
                signal[i] >= signal[i + 1]) {
                if (!peaks.empty() && (int)i - peaks.back() < tolerance_samples * 2) {
                    if (signal[i] > signal[peaks.back()]) {
                        peaks.back() = (int)i;
                    }
                } else {
                    peaks.push_back((int)i);
                }
            }
        }
        return peaks;
    };

    auto true_peaks = detect_peaks(original);
    auto detected_peaks = detect_peaks(reconstructed);

    if (true_peaks.empty()) return 1.0;

    int tp = 0;
    std::vector<bool> matched(true_peaks.size(), false);
    for (int dp : detected_peaks) {
        for (size_t j = 0; j < true_peaks.size(); ++j) {
            if (!matched[j] && std::abs(dp - true_peaks[j]) <= tolerance_samples) {
                ++tp;
                matched[j] = true;
                break;
            }
        }
    }

    int fn = (int)true_peaks.size() - tp;
    int fp = (int)detected_peaks.size() - tp;

    double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;

    return (precision + recall > 0)
        ? 2.0 * precision * recall / (precision + recall)
        : 0.0;
}

inline EvalMetrics compute_all_metrics(
    const std::vector<double>& original,
    const std::vector<int>& surviving_indices,
    const std::vector<double>& surviving_values,
    size_t drop_count,
    double max_wait_ms,
    bool is_ecg = false)
{
    std::vector<double> reconstructed = reconstruct_signal(
        surviving_indices, surviving_values, (int)original.size());

    EvalMetrics m;
    m.mse = compute_mse_aligned(original, reconstructed);
    m.snr_db = compute_snr(original, reconstructed);
    m.max_error = compute_max_error(original, reconstructed);
    m.deriv_ratio = compute_deriv_ratio(original, reconstructed);
    m.drop_count = drop_count;
    m.max_wait_ms = max_wait_ms;
    m.degenerate = (drop_count == 0);
    m.snr_saturated = !std::isfinite(m.snr_db);

    if (is_ecg) {
        m.r_peak_accuracy = compute_rpeak_accuracy(original, reconstructed);
    } else {
        m.r_peak_accuracy = -1.0;
    }

    m.spectral_fidelity = -1.0;

    return m;
}

#endif // METRICS_H
