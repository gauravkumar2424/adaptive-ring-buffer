#ifndef RPEAK_EVAL_H
#define RPEAK_EVAL_H

#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================
// Ground-Truth R-Peak Evaluation
// Compares R-peaks detected in the reconstructed signal against
// cardiologist-annotated ground truth from MIT-BIH database.
// This is the gold standard for ECG buffer evaluation.
// ============================================================

struct RPeakResult {
    int true_positives;
    int false_positives;
    int false_negatives;
    double precision;
    double recall;
    double f1_score;
    double timing_error_ms;  // Average timing error of matched peaks
};

// Simple R-peak detector: local maximum above adaptive threshold
inline std::vector<int> detect_rpeaks(const std::vector<double>& signal,
                                       double threshold_frac = 0.5,
                                       int min_distance = 100) {
    std::vector<int> peaks;
    if (signal.size() < 5) return peaks;

    // Find signal range for adaptive threshold
    double sig_max = *std::max_element(signal.begin(), signal.end());
    double sig_min = *std::min_element(signal.begin(), signal.end());
    double threshold = sig_min + threshold_frac * (sig_max - sig_min);

    for (int i = 2; i < (int)signal.size() - 2; ++i) {
        if (signal[i] > threshold &&
            signal[i] >= signal[i-1] && signal[i] >= signal[i-2] &&
            signal[i] >= signal[i+1] && signal[i] >= signal[i+2]) {
            // Enforce minimum distance between peaks
            if (!peaks.empty() && (i - peaks.back()) < min_distance) {
                // Keep the taller peak
                if (signal[i] > signal[peaks.back()]) {
                    peaks.back() = i;
                }
            } else {
                peaks.push_back(i);
            }
        }
    }
    return peaks;
}

// Evaluate detected peaks against ground truth
// tolerance: how many samples a detected peak can be from the true peak
inline RPeakResult evaluate_rpeaks(
    const std::vector<int>& ground_truth,
    const std::vector<int>& detected,
    int tolerance = 15)  // ~42ms at 360Hz MIT-BIH sampling rate
{
    RPeakResult result;
    result.true_positives = 0;
    result.false_positives = 0;
    result.false_negatives = 0;
    result.timing_error_ms = 0.0;

    if (ground_truth.empty()) {
        result.precision = 1.0;
        result.recall = 1.0;
        result.f1_score = 1.0;
        return result;
    }

    std::vector<bool> gt_matched(ground_truth.size(), false);
    double total_timing_error = 0.0;

    for (int det : detected) {
        bool matched = false;
        int best_gt = -1;
        int best_dist = tolerance + 1;

        for (int j = 0; j < (int)ground_truth.size(); ++j) {
            if (!gt_matched[j]) {
                int dist = std::abs(det - ground_truth[j]);
                if (dist <= tolerance && dist < best_dist) {
                    best_dist = dist;
                    best_gt = j;
                    matched = true;
                }
            }
        }

        if (matched) {
            result.true_positives++;
            gt_matched[best_gt] = true;
            total_timing_error += best_dist;  // In samples
        } else {
            result.false_positives++;
        }
    }

    result.false_negatives = (int)ground_truth.size() - result.true_positives;

    result.precision = (result.true_positives + result.false_positives > 0)
        ? (double)result.true_positives / (result.true_positives + result.false_positives)
        : 0.0;

    result.recall = (result.true_positives + result.false_negatives > 0)
        ? (double)result.true_positives / (result.true_positives + result.false_negatives)
        : 0.0;

    result.f1_score = (result.precision + result.recall > 0)
        ? 2.0 * result.precision * result.recall / (result.precision + result.recall)
        : 0.0;

    // Convert timing error from samples to ms (MIT-BIH: 360 Hz)
    result.timing_error_ms = (result.true_positives > 0)
        ? (total_timing_error / result.true_positives) * (1000.0 / 360.0)
        : 0.0;

    return result;
}

#endif // RPEAK_EVAL_H
