#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "rpeak_eval.h"

using namespace std;

void producer_fn(RingBuffer<double>& buf, const vector<double>& sig, int delay_us) {
    for (size_t i = 0; i < sig.size(); ++i) {
        buf.push(sig[i], (int)i);
        this_thread::sleep_for(chrono::microseconds(delay_us));
    }
}

void consumer_fn(RingBuffer<double>& buf, vector<double>& vals,
                 vector<int>& idx, int delay_us, int n) {
    try {
        for (int i = 0; i < n; ++i) {
            auto s = buf.pop_indexed();
            vals.push_back(s.value);
            idx.push_back(s.original_index);
            this_thread::sleep_for(chrono::microseconds(delay_us));
        }
    } catch (...) {}
}

// ============================================================
// FIX: run_with_weights previously returned a bare double (just
// the SNR value), discarding all information about whether that
// SNR came from a genuine, non-saturated measurement. This made
// it structurally impossible for any caller to filter out
// saturated/degenerate trials before averaging -- the same root
// cause as the original compute_snr bug, just one level removed.
//
// Now returns a small struct carrying both the SNR and an
// explicit snr_saturated flag, computed the same way
// metrics.h's compute_all_metrics does it
// (!std::isfinite(snr)). Every caller below has been updated to
// check this flag before accumulating into any average.
// ============================================================

struct WeightedRunResult {
    double snr;
    bool snr_saturated;
};

WeightedRunResult run_with_weights(const vector<double>& signal, double alpha, double beta,
                        double gamma, size_t wh, size_t buf_size, int overload) {
    ImportanceConfig cfg;
    cfg.alpha = alpha; cfg.beta = beta; cfg.gamma = gamma; cfg.window_half = wh;

    RingBuffer<double> buffer(buf_size, BufferMode::IMPORTANCE_COMPOSITE,
                               chrono::milliseconds(2), cfg);
    vector<double> sv; vector<int> si;
    sv.reserve(signal.size()); si.reserve(signal.size());

    thread p(producer_fn, ref(buffer), cref(signal), 100);
    thread c(consumer_fn, ref(buffer), ref(sv), ref(si), 100 * overload, (int)signal.size());
    p.join(); buffer.finish(); c.join();

    auto recon = reconstruct_signal(si, sv, (int)signal.size());
    double snr = compute_snr(signal, recon);
    return { snr, !std::isfinite(snr) };
}

// ============================================================
// Signal characterizer: detects signal type from statistics
// Returns a feature vector that drives weight selection
// ============================================================
struct SignalProfile {
    double peak_ratio;      // max/mean — high for impulsive (ECG)
    double zero_crossing_rate; // high for oscillatory (vibration)
    double kurtosis;        // high for impulsive signals
    string predicted_type;  // "impulsive" or "oscillatory"
};

SignalProfile characterize_signal(const vector<double>& signal, int analysis_window = 500) {
    SignalProfile prof;
    int n = min((int)signal.size(), analysis_window);

    // Basic stats
    double sum = 0, sum2 = 0;
    double maxval = -1e30, minval = 1e30;
    for (int i = 0; i < n; ++i) {
        sum += signal[i]; sum2 += signal[i] * signal[i];
        maxval = max(maxval, signal[i]);
        minval = min(minval, signal[i]);
    }
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    double std_dev = sqrt(max(var, 1e-15));

    // Peak ratio: max amplitude / RMS
    double rms = sqrt(sum2 / n);
    prof.peak_ratio = (rms > 1e-15) ? maxval / rms : 1.0;

    // Zero crossing rate
    int zc = 0;
    for (int i = 1; i < n; ++i) {
        if ((signal[i] - mean) * (signal[i-1] - mean) < 0) zc++;
    }
    prof.zero_crossing_rate = (double)zc / n;

    // Kurtosis (excess)
    double sum4 = 0;
    for (int i = 0; i < n; ++i) {
        double d = (signal[i] - mean) / std_dev;
        sum4 += d * d * d * d;
    }
    prof.kurtosis = sum4 / n - 3.0;  // Excess kurtosis

    // Classification: high kurtosis + high peak ratio = impulsive (ECG)
    // Low kurtosis + high ZCR = oscillatory (vibration)
    if (prof.kurtosis > 2.0 || prof.peak_ratio > 3.0) {
        prof.predicted_type = "impulsive";
    } else {
        prof.predicted_type = "oscillatory";
    }

    return prof;
}

// Domain-adaptive weight selection based on signal profile
ImportanceConfig select_weights(const SignalProfile& prof) {
    ImportanceConfig cfg;
    if (prof.predicted_type == "impulsive") {
        // ECG-like: boost first-order (preserve peaks)
        cfg.alpha = 0.3;
        cfg.beta = 0.3;
        cfg.gamma = 0.4;
    } else {
        // Vibration-like: boost windowed energy (preserve oscillations)
        cfg.alpha = 0.05;
        cfg.beta = 0.25;
        cfg.gamma = 0.7;
    }
    cfg.window_half = 3;
    return cfg;
}

int main() {
    cout << "=== Domain-Adaptive Weight Optimization (Novelty 5) ===" << endl;

    string data_dir = "../data";
    auto signals = load_all_real_signals(data_dir, 2000);
    cout << "Loaded " << signals.size() << " signals\n" << endl;

    size_t buf_size = 256;
    int overload = 5;
    int trials = 3;

    // ============================================================
    // Part 1: Per-domain weight optimization
    // ============================================================
    cout << "=== Part 1: Per-Domain Weight Grid Search ===" << endl;

    map<string, vector<RealSignal*>> by_domain;
    for (auto& s : signals) by_domain[s.domain].push_back(&s);

    map<string, tuple<double,double,double,double>> best_weights; // domain -> (a,b,g,snr)

    for (auto& [domain, sigs] : by_domain) {
        cout << "\nOptimizing for " << domain << " (" << sigs.size() << " signals)..." << endl;

        double best_snr = -1000;
        double ba = 0.1, bb = 0.3, bg = 0.6;

        for (int a = 0; a <= 8; ++a) {
            for (int b = 0; b <= (10 - a); ++b) {
                int g = 10 - a - b;
                if (g < 0) continue;
                double alpha = max(0.05, a / 10.0);
                double beta = max(0.0, b / 10.0);
                double gamma = max(0.05, g / 10.0);
                double total = alpha + beta + gamma;
                alpha /= total; beta /= total; gamma /= total;

                double avg_snr = 0;
                int count = 0;
                int saturated_skipped = 0;
                for (auto* sig : sigs) {
                    for (int t = 0; t < trials; ++t) {
                        auto result = run_with_weights(sig->data, alpha, beta, gamma, 3, buf_size, overload);
                        if (!result.snr_saturated) {
                            avg_snr += result.snr;
                            count++;
                        } else {
                            ++saturated_skipped;
                        }
                    }
                }
                // FIX: guard against count==0 (all trials saturated for this
                // weight combo) instead of dividing by zero, and only
                // increment count for genuine, non-saturated measurements.
                avg_snr = (count > 0) ? avg_snr / count : -1000.0;

                if (avg_snr > best_snr) {
                    best_snr = avg_snr;
                    ba = alpha; bb = beta; bg = gamma;
                }
            }
        }

        best_weights[domain] = {ba, bb, bg, best_snr};
        cout << "  Best weights for " << domain << ": alpha=" << fixed << setprecision(2)
             << ba << " beta=" << bb << " gamma=" << bg
             << " -> SNR=" << setprecision(2) << best_snr << " dB" << endl;
    }

    // ============================================================
    // Part 2: Signal characterization and auto-detection
    // ============================================================
    cout << "\n=== Part 2: Automatic Signal Characterization ===" << endl;

    for (auto& sig : signals) {
        auto prof = characterize_signal(sig.data);
        cout << "  " << left << setw(20) << sig.name
             << " domain=" << setw(10) << sig.domain
             << " kurtosis=" << fixed << setprecision(2) << setw(8) << prof.kurtosis
             << " peak_ratio=" << setw(6) << prof.peak_ratio
             << " ZCR=" << setw(6) << prof.zero_crossing_rate
             << " -> " << prof.predicted_type;
        if ((prof.predicted_type == "impulsive" && sig.domain == "ecg") ||
            (prof.predicted_type == "oscillatory" && sig.domain == "vibration")) {
            cout << " CORRECT";
        } else {
            cout << " MISMATCH";
        }
        cout << endl;
    }

    // ============================================================
    // Part 3: Compare universal vs domain-specific vs auto-tuned
    //
    // FIX: each of the three SNR averages below (universal,
    // domain-specific, auto-tuned) now only accumulates over
    // trials where snr_saturated is false, and divides by the
    // actual count of non-saturated trials rather than the
    // nominal trial count. If ALL trials for a given signal
    // happen to saturate, the value is reported as NaN-safe
    // -1000.0 sentinel and printed with a flag, rather than
    // silently dividing by trials and producing a misleadingly
    // "normal-looking" number from zero real measurements.
    // ============================================================
    cout << "\n=== Part 3: Universal vs Domain-Specific vs Auto-Tuned ===" << endl;

    for (auto& sig : signals) {
        // Universal weights (from original optimization)
        double snr_universal = 0;
        int count_universal = 0;
        for (int t = 0; t < trials; ++t) {
            auto result = run_with_weights(sig.data, 0.1, 0.3, 0.6, 3, buf_size, overload);
            if (!result.snr_saturated) { snr_universal += result.snr; ++count_universal; }
        }
        snr_universal = (count_universal > 0) ? snr_universal / count_universal : -1000.0;

        // Domain-specific weights (from Part 1)
        auto [da, db, dg, _] = best_weights[sig.domain];
        double snr_domain = 0;
        int count_domain = 0;
        for (int t = 0; t < trials; ++t) {
            auto result = run_with_weights(sig.data, da, db, dg, 3, buf_size, overload);
            if (!result.snr_saturated) { snr_domain += result.snr; ++count_domain; }
        }
        snr_domain = (count_domain > 0) ? snr_domain / count_domain : -1000.0;

        // Auto-tuned weights (from signal characterization)
        auto prof = characterize_signal(sig.data);
        auto auto_cfg = select_weights(prof);
        double snr_auto = 0;
        int count_auto = 0;
        for (int t = 0; t < trials; ++t) {
            auto result = run_with_weights(sig.data, auto_cfg.alpha, auto_cfg.beta, auto_cfg.gamma,
                                          3, buf_size, overload);
            if (!result.snr_saturated) { snr_auto += result.snr; ++count_auto; }
        }
        snr_auto = (count_auto > 0) ? snr_auto / count_auto : -1000.0;

        double gain_domain = snr_domain - snr_universal;
        double gain_auto = snr_auto - snr_universal;

        cout << "  " << left << setw(20) << sig.name << right << fixed << setprecision(2)
             << "  Universal=" << setw(7) << snr_universal << "(n=" << count_universal << ")"
             << "  Domain=" << setw(7) << snr_domain << "(n=" << count_domain << ")"
             << " (" << showpos << gain_domain << noshowpos << ")"
             << "  Auto=" << setw(7) << snr_auto << "(n=" << count_auto << ")"
             << " (" << showpos << gain_auto << noshowpos << ")"
             << endl;
    }

    cout << "\nDone." << endl;
    return 0;
}
