#include <set>
// ============================================================
// stress_test.cpp — High-Compression Stress Test
// DATE 2027: Adaptive Ring Buffer
//
// PURPOSE: Push compression harder (overload 8/10/15/20, buffer
// 32/64) to:
//   1. Differentiate R-peak F1 across methods (currently flat)
//   2. Expose method breakdown points
//   3. Produce publication figure: "graceful degradation" curve
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/stress_test stress_test.cpp
// Run:   cd ../build && ./stress_test
// ============================================================

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
#include <queue>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "rpeak_eval.h"

using namespace std;

// ============================================================
// Offline baselines (identical to cross_domain.cpp)
// ============================================================

vector<int> offline_rdp(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n);
        iota(all.begin(), all.end(), 0);
        return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<bool> kept(n, true);
    vector<int> prev_kept(n), next_kept(n);
    for (int i = 0; i < n; ++i) { prev_kept[i] = i - 1; next_kept[i] = i + 1; }
    next_kept[n - 1] = -1;

    auto compute_error = [&](int i) -> double {
        int p = prev_kept[i], s = next_kept[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p);
        if (span <= 0) return 0.0;
        double t = (double)(i - p) / span;
        return abs(signal[i] - (signal[p] + t * (signal[s] - signal[p])));
    };

    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({compute_error(i), i});

    int current_count = n;
    while (current_count > target_points && !pq.empty()) {
        auto [err, idx] = pq.top(); pq.pop();
        if (!kept[idx]) continue;
        double actual = compute_error(idx);
        if (abs(actual - err) > 1e-12) { pq.push({actual, idx}); continue; }
        kept[idx] = false;
        int p = prev_kept[idx], s = next_kept[idx];
        if (p >= 0) next_kept[p] = s;
        if (s >= 0 && s < n) prev_kept[s] = p;
        --current_count;
        if (p > 0 && kept[p]) pq.push({compute_error(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({compute_error(s), s});
    }

    vector<int> result;
    for (int i = 0; i < n; ++i) if (kept[i]) result.push_back(i);
    return result;
}

vector<int> offline_lttb(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n); iota(all.begin(), all.end(), 0); return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<int> result;
    result.reserve(target_points);
    result.push_back(0);
    double bucket_size = (double)(n - 2) / (target_points - 2);
    int prev_selected = 0;

    for (int bucket = 0; bucket < target_points - 2; ++bucket) {
        int b_start = (int)(bucket * bucket_size) + 1;
        int b_end   = min((int)((bucket + 1) * bucket_size) + 1, n - 1);
        int nb_start = (int)((bucket + 1) * bucket_size) + 1;
        int nb_end   = min((int)((bucket + 2) * bucket_size) + 1, n);
        double avg_x = 0, avg_y = 0; int nb_count = 0;
        for (int i = nb_start; i < nb_end; ++i) { avg_x += i; avg_y += signal[i]; ++nb_count; }
        if (nb_count > 0) { avg_x /= nb_count; avg_y /= nb_count; }
        double max_area = -1; int best_idx = b_start;
        for (int i = b_start; i < b_end; ++i) {
            double area = abs((prev_selected - avg_x) * (signal[i] - signal[prev_selected])
                            - (prev_selected - i) * (avg_y - signal[prev_selected])) * 0.5;
            if (area > max_area) { max_area = area; best_idx = i; }
        }
        result.push_back(best_idx);
        prev_selected = best_idx;
    }
    result.push_back(n - 1);
    return result;
}

// ============================================================
// Producer / Consumer
// ============================================================

void producer_thread(RingBuffer<double>& buffer, const vector<double>& signal, int delay_us) {
    for (size_t i = 0; i < signal.size(); ++i) {
        buffer.push(signal[i], static_cast<int>(i));
        this_thread::sleep_for(chrono::microseconds(delay_us));
    }
}

void consumer_thread(RingBuffer<double>& buffer,
                     vector<double>& values, vector<int>& indices,
                     int delay_us, int expected) {
    try {
        for (int i = 0; i < expected; ++i) {
            IndexedSample s = buffer.pop_indexed();
            values.push_back(s.value);
            indices.push_back(s.original_index);
            this_thread::sleep_for(chrono::microseconds(delay_us));
        }
    } catch (const runtime_error&) {}
}

double compute_spectral_correlation(const vector<double>& orig,
                                     const vector<double>& recon) {
    if (orig.size() < 64 || recon.size() < 64) return -1.0;
    int N = min((int)min(orig.size(), recon.size()), 512);
    vector<double> mag_orig(N/2), mag_recon(N/2);
    for (int k = 0; k < N/2; ++k) {
        double re_o = 0, im_o = 0, re_r = 0, im_r = 0;
        for (int n = 0; n < N; ++n) {
            double angle = 2.0 * M_PI * k * n / N;
            re_o += orig[n] * cos(angle); im_o -= orig[n] * sin(angle);
            re_r += recon[n] * cos(angle); im_r -= recon[n] * sin(angle);
        }
        mag_orig[k] = sqrt(re_o*re_o + im_o*im_o);
        mag_recon[k] = sqrt(re_r*re_r + im_r*im_r);
    }
    double mean_o = accumulate(mag_orig.begin(), mag_orig.end(), 0.0) / mag_orig.size();
    double mean_r = accumulate(mag_recon.begin(), mag_recon.end(), 0.0) / mag_recon.size();
    double num = 0, den_o = 0, den_r = 0;
    for (int k = 0; k < N/2; ++k) {
        double d_o = mag_orig[k] - mean_o, d_r = mag_recon[k] - mean_r;
        num += d_o * d_r; den_o += d_o * d_o; den_r += d_r * d_r;
    }
    double den = sqrt(den_o * den_r);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================
// Result
// ============================================================

struct Result {
    string signal_name, domain, mode_name;
    size_t buffer_size;
    int overload_ratio, trial;
    double snr, mse;
    size_t drops;
    bool snr_saturated;
    double rpeak_f1, rpeak_precision, rpeak_recall;
    double spectral_correlation;
    double compression_ratio; // original_size / surviving_size
};

Result run_online_experiment(const RealSignal& sig, const string& mode_name,
                             BufferMode mode, size_t buf_size,
                             int overload_ratio, int trial) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload_ratio;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());

    vector<double> sv; vector<int> si;
    sv.reserve(sig.data.size()); si.reserve(sig.data.size());

    thread prod(producer_thread, ref(buffer), cref(sig.data), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(sv), ref(si),
                cons_delay, (int)sig.data.size());
    prod.join(); buffer.finish(); cons.join();

    vector<double> reconstructed = reconstruct_signal(si, sv, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name; r.domain = sig.domain; r.mode_name = mode_name;
    r.buffer_size = buf_size; r.overload_ratio = overload_ratio; r.trial = trial;
    r.snr = compute_snr(sig.data, reconstructed);
    r.mse = compute_mse_aligned(sig.data, reconstructed);
    r.drops = buffer.getDropCount();
    r.snr_saturated = !isfinite(r.snr);
    int surviving = (int)sig.data.size() - (int)r.drops;
    r.compression_ratio = (surviving > 0) ? (double)sig.data.size() / surviving : 999.0;

    r.rpeak_f1 = -1; r.rpeak_precision = -1; r.rpeak_recall = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto detected = detect_rpeaks(reconstructed, 0.5, 100);
        auto eval = evaluate_rpeaks(sig.rpeak_indices, detected, 15);
        r.rpeak_f1 = eval.f1_score;
        r.rpeak_precision = eval.precision;
        r.rpeak_recall = eval.recall;
    }

    r.spectral_correlation = -1;
    if (sig.domain == "vibration")
        r.spectral_correlation = compute_spectral_correlation(sig.data, reconstructed);

    return r;
}

Result run_offline_experiment(const RealSignal& sig, const string& method,
                              size_t buf_size, int overload_ratio, int trial,
                              int target_surviving) {
    vector<int> selected;
    if (method == "RDP_OFFLINE") selected = offline_rdp(sig.data, target_surviving);
    else selected = offline_lttb(sig.data, target_surviving);

    vector<double> sv; vector<int> si;
    for (int idx : selected) { si.push_back(idx); sv.push_back(sig.data[idx]); }
    vector<double> reconstructed = reconstruct_signal(si, sv, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name; r.domain = sig.domain; r.mode_name = method;
    r.buffer_size = buf_size; r.overload_ratio = overload_ratio; r.trial = trial;
    r.snr = compute_snr(sig.data, reconstructed);
    r.mse = compute_mse_aligned(sig.data, reconstructed);
    r.drops = sig.data.size() - selected.size();
    r.snr_saturated = !isfinite(r.snr);
    r.compression_ratio = (selected.size() > 0) ? (double)sig.data.size() / selected.size() : 999.0;

    r.rpeak_f1 = -1; r.rpeak_precision = -1; r.rpeak_recall = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto detected = detect_rpeaks(reconstructed, 0.5, 100);
        auto eval = evaluate_rpeaks(sig.rpeak_indices, detected, 15);
        r.rpeak_f1 = eval.f1_score;
        r.rpeak_precision = eval.precision;
        r.rpeak_recall = eval.recall;
    }

    r.spectral_correlation = -1;
    if (sig.domain == "vibration")
        r.spectral_correlation = compute_spectral_correlation(sig.data, reconstructed);

    return r;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    cout << "=== HIGH-COMPRESSION STRESS TEST ===" << endl;
    cout << "Overload ratios: 2-20x, Buffer sizes: 32-512" << endl << endl;

    string data_dir = "../data";
    int max_samples = 2000;
    bool quick = (argc > 1 && string(argv[1]) == "--quick");

    auto signals = load_all_real_signals(data_dir, max_samples);
    cout << "Loaded " << signals.size() << " signals\n" << endl;

    if (signals.empty()) { cerr << "ERROR: No signals." << endl; return 1; }

    // KEY: extended ranges beyond original cross_domain.cpp
    vector<size_t> buf_sizes = quick ?
        vector<size_t>{64, 256} :
        vector<size_t>{32, 64, 128, 256, 512};
    vector<int> overloads = quick ?
        vector<int>{3, 10, 20} :
        vector<int>{2, 3, 5, 8, 10, 15, 20};
    int num_trials = quick ? 2 : 5;

    // Focused mode set: only the ones that matter for the paper
    struct ModeEntry { string name; BufferMode mode; };
    vector<ModeEntry> online_modes = {
        {"DROP",                 BufferMode::DROP},
        {"RANDOM_DROP",          BufferMode::RANDOM_DROP},
        {"IMP_COMPOSITE",        BufferMode::IMPORTANCE_COMPOSITE},
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
    };
    vector<string> offline_methods = {"RDP_OFFLINE", "LTTB_OFFLINE"};

    int total_modes = online_modes.size() + offline_methods.size();
    int total = signals.size() * buf_sizes.size() * overloads.size() * num_trials * total_modes;
    cout << "Running " << total << " experiments..." << endl;

    ofstream csv("../results/stress_test_results.csv");
    csv << "signal,domain,mode,buffer_size,overload,trial,"
        << "snr_db,mse,drops,snr_saturated,compression_ratio,"
        << "rpeak_f1,rpeak_precision,rpeak_recall,"
        << "spectral_correlation" << endl;

    auto write_row = [&](const Result& r) {
        csv << r.signal_name << "," << r.domain << ","
            << r.mode_name << "," << r.buffer_size << ","
            << r.overload_ratio << "," << r.trial << ","
            << fixed << setprecision(4) << r.snr << ","
            << scientific << setprecision(6) << r.mse << ","
            << r.drops << "," << r.snr_saturated << ","
            << fixed << setprecision(2) << r.compression_ratio << ","
            << setprecision(4) << r.rpeak_f1 << ","
            << r.rpeak_precision << "," << r.rpeak_recall << ","
            << r.spectral_correlation << endl;
    };

    int done = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& sig : signals) {
        for (size_t bs : buf_sizes) {
            for (int ol : overloads) {
                for (int trial = 0; trial < num_trials; ++trial) {
                    int interp_drops = -1;
                    for (auto& m : online_modes) {
                        auto r = run_online_experiment(sig, m.name, m.mode, bs, ol, trial);
                        write_row(r);
                        if (m.name == "IMP_INTERP_ERROR")
                            interp_drops = (int)r.drops;
                        ++done;
                    }

                    if (interp_drops >= 0) {
                        int target = max(2, (int)sig.data.size() - interp_drops);
                        for (auto& method : offline_methods) {
                            auto r = run_offline_experiment(sig, method, bs, ol, trial, target);
                            write_row(r);
                            ++done;
                        }
                    }

                    if (done % 200 == 0) {
                        auto now = chrono::high_resolution_clock::now();
                        double elapsed = chrono::duration<double>(now - t0).count();
                        double remaining = (total - done) * (elapsed / max(done, 1));
                        cout << "  " << done << "/" << total
                             << " (" << (int)remaining << "s remaining)" << endl;
                    }
                }
            }
        }
    }
    csv.close();

    auto t1 = chrono::high_resolution_clock::now();
    double total_sec = chrono::duration<double>(t1 - t0).count();

    // ============================================================
    // Summary: degradation curves by overload ratio
    // ============================================================
    cout << "\n=== DEGRADATION CURVES (Mean SNR by Overload) ===" << endl;

    // Read back results for summary
    map<string, map<int, vector<double>>> mode_overload_snr;
    map<string, map<int, vector<double>>> mode_overload_f1;
    map<string, map<int, vector<double>>> mode_overload_spec;

    ifstream results_file("../results/stress_test_results.csv");
    string line;
    getline(results_file, line); // header

    while (getline(results_file, line)) {
        stringstream ss(line);
        string sig_name, dom, mode;
        int bs, ol, trial;
        double snr, mse;
        int drops, sat;
        double comp_ratio, f1, prec, rec, spec;

        getline(ss, sig_name, ','); getline(ss, dom, ','); getline(ss, mode, ',');
        ss >> bs; ss.ignore(); ss >> ol; ss.ignore(); ss >> trial; ss.ignore();
        ss >> snr; ss.ignore(); ss >> mse; ss.ignore();
        ss >> drops; ss.ignore(); ss >> sat; ss.ignore();
        ss >> comp_ratio; ss.ignore();
        ss >> f1; ss.ignore(); ss >> prec; ss.ignore(); ss >> rec; ss.ignore();
        ss >> spec;

        if (!sat && isfinite(snr))
            mode_overload_snr[mode][ol].push_back(snr);
        if (f1 >= 0)
            mode_overload_f1[mode][ol].push_back(f1);
        if (spec >= 0 && !sat)
            mode_overload_spec[mode][ol].push_back(spec);
    }

    // Print SNR degradation
    cout << "\n--- SNR (dB) by Overload Ratio ---" << endl;
    cout << left << setw(22) << "Mode";
    set<int> all_overloads;
    for (auto& [m, omap] : mode_overload_snr)
        for (auto& [ol, _] : omap) all_overloads.insert(ol);
    for (int ol : all_overloads) cout << right << setw(8) << (to_string(ol) + "x");
    cout << endl << string(22 + 8 * all_overloads.size(), '-') << endl;

    for (auto& [mode, omap] : mode_overload_snr) {
        cout << left << setw(22) << mode;
        for (int ol : all_overloads) {
            auto it = omap.find(ol);
            if (it != omap.end() && !it->second.empty()) {
                double avg = accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
                cout << right << fixed << setprecision(1) << setw(8) << avg;
            } else {
                cout << right << setw(8) << "-";
            }
        }
        cout << endl;
    }

    // Print R-peak F1 degradation (ECG only)
    cout << "\n--- R-peak F1 by Overload Ratio (ECG only) ---" << endl;
    cout << left << setw(22) << "Mode";
    for (int ol : all_overloads) cout << right << setw(8) << (to_string(ol) + "x");
    cout << endl << string(22 + 8 * all_overloads.size(), '-') << endl;

    for (auto& [mode, omap] : mode_overload_f1) {
        cout << left << setw(22) << mode;
        for (int ol : all_overloads) {
            auto it = omap.find(ol);
            if (it != omap.end() && !it->second.empty()) {
                double avg = accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
                cout << right << fixed << setprecision(3) << setw(8) << avg;
            } else {
                cout << right << setw(8) << "-";
            }
        }
        cout << endl;
    }

    // Print spectral correlation degradation (vibration only)
    cout << "\n--- Spectral Correlation by Overload (Vibration only) ---" << endl;
    cout << left << setw(22) << "Mode";
    for (int ol : all_overloads) cout << right << setw(8) << (to_string(ol) + "x");
    cout << endl << string(22 + 8 * all_overloads.size(), '-') << endl;

    for (auto& [mode, omap] : mode_overload_spec) {
        cout << left << setw(22) << mode;
        for (int ol : all_overloads) {
            auto it = omap.find(ol);
            if (it != omap.end() && !it->second.empty()) {
                double avg = accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
                cout << right << fixed << setprecision(4) << setw(8) << avg;
            } else {
                cout << right << setw(8) << "-";
            }
        }
        cout << endl;
    }

    cout << "\nTotal experiments: " << done << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_sec << "s ("
         << setprecision(1) << total_sec/60 << " min)" << endl;
    cout << "Results: results/stress_test_results.csv" << endl;

    return 0;
}
