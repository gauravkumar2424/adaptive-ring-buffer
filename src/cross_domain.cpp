#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <random>
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
// Offline baselines (same as main.cpp)
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
    for (int i = 0; i < n; ++i) {
        prev_kept[i] = i - 1;
        next_kept[i] = i + 1;
    }
    next_kept[n - 1] = -1;

    auto compute_error = [&](int i) -> double {
        int p = prev_kept[i];
        int s = next_kept[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p);
        if (span <= 0) return 0.0;
        double t = (double)(i - p) / span;
        double x_hat = signal[p] + t * (signal[s] - signal[p]);
        return abs(signal[i] - x_hat);
    };

    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i)
        pq.push({compute_error(i), i});

    int current_count = n;
    while (current_count > target_points && !pq.empty()) {
        auto [err, idx] = pq.top();
        pq.pop();
        if (!kept[idx]) continue;
        double actual = compute_error(idx);
        if (abs(actual - err) > 1e-12) {
            pq.push({actual, idx});
            continue;
        }
        kept[idx] = false;
        int p = prev_kept[idx], s = next_kept[idx];
        if (p >= 0) next_kept[p] = s;
        if (s >= 0 && s < n) prev_kept[s] = p;
        --current_count;
        if (p > 0 && kept[p]) pq.push({compute_error(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({compute_error(s), s});
    }

    vector<int> result;
    for (int i = 0; i < n; ++i)
        if (kept[i]) result.push_back(i);
    return result;
}

vector<int> offline_lttb(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n);
        iota(all.begin(), all.end(), 0);
        return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<int> result;
    result.reserve(target_points);
    result.push_back(0);

    double bucket_size = (double)(n - 2) / (target_points - 2);
    int prev_selected = 0;

    for (int bucket = 0; bucket < target_points - 2; ++bucket) {
        int b_start = (int)(bucket * bucket_size) + 1;
        int b_end   = (int)((bucket + 1) * bucket_size) + 1;
        b_end = min(b_end, n - 1);
        int nb_start = (int)((bucket + 1) * bucket_size) + 1;
        int nb_end   = (int)((bucket + 2) * bucket_size) + 1;
        nb_end = min(nb_end, n);

        double avg_x = 0, avg_y = 0;
        int nb_count = 0;
        for (int i = nb_start; i < nb_end; ++i) {
            avg_x += i; avg_y += signal[i]; ++nb_count;
        }
        if (nb_count > 0) { avg_x /= nb_count; avg_y /= nb_count; }

        double max_area = -1;
        int best_idx = b_start;
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
// Spectral correlation
// ============================================================

double compute_spectral_correlation(const vector<double>& orig,
                                     const vector<double>& recon) {
    if (orig.size() < 64 || recon.size() < 64) return -1.0;
    int N = min((int)min(orig.size(), recon.size()), 512);
    vector<double> mag_orig(N/2), mag_recon(N/2);

    for (int k = 0; k < N/2; ++k) {
        double re_o = 0, im_o = 0, re_r = 0, im_r = 0;
        for (int n = 0; n < N; ++n) {
            double angle = 2.0 * M_PI * k * n / N;
            re_o += orig[n] * cos(angle);
            im_o -= orig[n] * sin(angle);
            re_r += recon[n] * cos(angle);
            im_r -= recon[n] * sin(angle);
        }
        mag_orig[k] = sqrt(re_o*re_o + im_o*im_o);
        mag_recon[k] = sqrt(re_r*re_r + im_r*im_r);
    }

    double mean_o = accumulate(mag_orig.begin(), mag_orig.end(), 0.0) / mag_orig.size();
    double mean_r = accumulate(mag_recon.begin(), mag_recon.end(), 0.0) / mag_recon.size();
    double num = 0, den_o = 0, den_r = 0;
    for (int k = 0; k < N/2; ++k) {
        double do_ = mag_orig[k] - mean_o;
        double dr = mag_recon[k] - mean_r;
        num += do_ * dr;
        den_o += do_ * do_;
        den_r += dr * dr;
    }
    double den = sqrt(den_o * den_r);
    return (den > 1e-15) ? num / den : 0.0;
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

// ============================================================
// Result struct
//
// FIX: added degenerate and snr_saturated, mirroring EvalMetrics
// in metrics.h. degenerate = no eviction occurred (drops == 0).
// snr_saturated = snr is non-finite (can occur with or without
// eviction). Both must be checked before aggregating SNR.
// ============================================================

struct Result {
    string signal_name, domain, mode_name;
    size_t buffer_size;
    int overload_ratio, trial;
    double snr, mse, max_wait, deriv_ratio;
    size_t drops;
    bool degenerate;
    bool snr_saturated;
    double rpeak_f1, rpeak_precision, rpeak_recall, rpeak_timing_ms;
    double spectral_correlation;
};

// ============================================================
// Run online experiment
// ============================================================

Result run_online_experiment(const RealSignal& sig, const string& mode_name,
                             BufferMode mode, size_t buf_size,
                             int overload_ratio, int trial) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload_ratio;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());

    vector<double> surv_vals;
    vector<int> surv_idx;
    surv_vals.reserve(sig.data.size());
    surv_idx.reserve(sig.data.size());

    thread prod(producer_thread, ref(buffer), cref(sig.data), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(surv_vals), ref(surv_idx),
                cons_delay, (int)sig.data.size());
    prod.join();
    buffer.finish();
    cons.join();

    vector<double> reconstructed = reconstruct_signal(surv_idx, surv_vals, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name;
    r.domain = sig.domain;
    r.mode_name = mode_name;
    r.buffer_size = buf_size;
    r.overload_ratio = overload_ratio;
    r.trial = trial;
    r.snr = compute_snr(sig.data, reconstructed);
    r.mse = compute_mse_aligned(sig.data, reconstructed);
    r.max_wait = buffer.getMaxProducerWaitMs();
    r.deriv_ratio = compute_deriv_ratio(sig.data, reconstructed);
    r.drops = buffer.getDropCount();
    r.degenerate = (r.drops == 0);
    r.snr_saturated = !std::isfinite(r.snr);

    r.rpeak_f1 = -1; r.rpeak_precision = -1;
    r.rpeak_recall = -1; r.rpeak_timing_ms = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto detected = detect_rpeaks(reconstructed, 0.5, 100);
        auto eval = evaluate_rpeaks(sig.rpeak_indices, detected, 15);
        r.rpeak_f1 = eval.f1_score;
        r.rpeak_precision = eval.precision;
        r.rpeak_recall = eval.recall;
        r.rpeak_timing_ms = eval.timing_error_ms;
    }

    r.spectral_correlation = -1;
    if (sig.domain == "vibration")
        r.spectral_correlation = compute_spectral_correlation(sig.data, reconstructed);

    return r;
}

// ============================================================
// Run offline baseline with matched drop count
// ============================================================

Result run_offline_experiment(const RealSignal& sig, const string& method,
                              size_t buf_size, int overload_ratio, int trial,
                              int target_surviving) {
    vector<int> selected;
    if (method == "RDP_OFFLINE")
        selected = offline_rdp(sig.data, target_surviving);
    else
        selected = offline_lttb(sig.data, target_surviving);

    vector<double> surv_vals;
    vector<int> surv_idx;
    for (int idx : selected) {
        surv_idx.push_back(idx);
        surv_vals.push_back(sig.data[idx]);
    }

    vector<double> reconstructed = reconstruct_signal(surv_idx, surv_vals, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name;
    r.domain = sig.domain;
    r.mode_name = method;
    r.buffer_size = buf_size;
    r.overload_ratio = overload_ratio;
    r.trial = trial;
    r.snr = compute_snr(sig.data, reconstructed);
    r.mse = compute_mse_aligned(sig.data, reconstructed);
    r.max_wait = 0;
    r.deriv_ratio = compute_deriv_ratio(sig.data, reconstructed);
    r.drops = sig.data.size() - selected.size();
    r.degenerate = (r.drops == 0);
    r.snr_saturated = !std::isfinite(r.snr);

    r.rpeak_f1 = -1; r.rpeak_precision = -1;
    r.rpeak_recall = -1; r.rpeak_timing_ms = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto detected = detect_rpeaks(reconstructed, 0.5, 100);
        auto eval = evaluate_rpeaks(sig.rpeak_indices, detected, 15);
        r.rpeak_f1 = eval.f1_score;
        r.rpeak_precision = eval.precision;
        r.rpeak_recall = eval.recall;
        r.rpeak_timing_ms = eval.timing_error_ms;
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
    cout << "=== Cross-Domain Signal Validation (N5) ===" << endl;

    string data_dir = "../data";
    int max_samples = 2000;
    bool quick = (argc > 1 && string(argv[1]) == "--quick");

    cout << "\nLoading signals..." << endl;
    auto signals = load_all_real_signals(data_dir, max_samples);
    cout << "Loaded " << signals.size() << " signals total\n" << endl;

    if (signals.empty()) {
        cerr << "ERROR: No signals loaded. Check data directory." << endl;
        return 1;
    }

    vector<size_t> buf_sizes = quick ? vector<size_t>{128, 256} : vector<size_t>{64, 128, 256, 512};
    vector<int> overloads = quick ? vector<int>{3, 6} : vector<int>{2, 3, 4, 5, 6};
    int num_trials = quick ? 2 : 5;

    struct ModeEntry { string name; BufferMode mode; };
    vector<ModeEntry> online_modes = {
        {"DROP",              BufferMode::DROP},
        {"RANDOM_DROP",       BufferMode::RANDOM_DROP},
        {"DROP_MIDDLE",       BufferMode::DROP_MIDDLE},
        {"DROP_LOW_VARIANCE", BufferMode::DROP_LOW_VARIANCE},
        {"ADAPTIVE",          BufferMode::ADAPTIVE_TIMED_WAIT},
        {"LEGACY_IMPORTANCE", BufferMode::ADAPTIVE_IMPORTANCE},
        {"IMP_FIRST_ORDER",   BufferMode::IMPORTANCE_FIRST_ORDER},
        {"IMP_WINDOWED_ENERGY", BufferMode::IMPORTANCE_WINDOWED_ENERGY},
        {"IMP_COMPOSITE",     BufferMode::IMPORTANCE_COMPOSITE},
        {"IMP_ADAPTIVE",      BufferMode::IMPORTANCE_ADAPTIVE},
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
    };
    vector<string> offline_methods = {"RDP_OFFLINE", "LTTB_OFFLINE"};

    int total_modes = online_modes.size() + offline_methods.size();
    int total = signals.size() * buf_sizes.size() * overloads.size() * num_trials * total_modes;
    cout << "Running " << total << " experiments..." << endl;

    ofstream csv("../results/cross_domain_results.csv");
    csv << "signal,domain,mode,buffer_size,overload,trial,"
        << "snr_db,mse,max_wait_ms,deriv_ratio,drops,degenerate,snr_saturated,"
        << "rpeak_f1,rpeak_precision,rpeak_recall,rpeak_timing_ms,"
        << "spectral_correlation" << endl;

    auto write_row = [&](const Result& r) {
        csv << r.signal_name << "," << r.domain << ","
            << r.mode_name << "," << r.buffer_size << ","
            << r.overload_ratio << "," << r.trial << ","
            << fixed << setprecision(4) << r.snr << ","
            << scientific << setprecision(6) << r.mse << ","
            << fixed << setprecision(4) << r.max_wait << ","
            << r.deriv_ratio << "," << r.drops << ","
            << r.degenerate << "," << r.snr_saturated << ","
            << r.rpeak_f1 << "," << r.rpeak_precision << ","
            << r.rpeak_recall << "," << r.rpeak_timing_ms << ","
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
                        int target = (int)sig.data.size() - interp_drops;
                        target = max(2, target);
                        for (auto& method : offline_methods) {
                            auto r = run_offline_experiment(sig, method, bs, ol, trial, target);
                            write_row(r);
                            ++done;
                        }
                    }

                    if (done % 100 == 0) {
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
    // Summary by domain — now excludes degenerate/saturated rows
    // ============================================================
    cout << "\n=== RESULTS SUMMARY ===" << endl;

    ifstream results_file("../results/cross_domain_results.csv");
    string line;
    getline(results_file, line); // skip header

    struct Stats {
        double snr_sum = 0, f1_sum = 0, spec_sum = 0;
        int snr_count = 0, f1_count = 0, spec_count = 0;
    };
    map<string, map<string, Stats>> domain_mode_stats;

    while (getline(results_file, line)) {
        stringstream ss(line);
        string sig_name, dom, mode;
        int bs, ol, trial;
        double snr, mse, wait, deriv;
        int drops;
        int degenerate_flag, snr_saturated_flag;
        double f1, prec, rec, timing, spec;

        getline(ss, sig_name, ','); getline(ss, dom, ','); getline(ss, mode, ',');
        ss >> bs; ss.ignore(); ss >> ol; ss.ignore(); ss >> trial; ss.ignore();
        ss >> snr; ss.ignore(); ss >> mse; ss.ignore();
        ss >> wait; ss.ignore(); ss >> deriv; ss.ignore(); ss >> drops; ss.ignore();
        ss >> degenerate_flag; ss.ignore(); ss >> snr_saturated_flag; ss.ignore();
        ss >> f1; ss.ignore(); ss >> prec; ss.ignore();
        ss >> rec; ss.ignore(); ss >> timing; ss.ignore(); ss >> spec;

        // FIX: skip rows where SNR is non-finite when accumulating
        // the SNR average — this is the same leak that produced
        // the fake 100 dB readings before. Reading "inf" via >>
        // into a double will fail/zero in many implementations,
        // so explicitly check snr_saturated_flag rather than
        // trusting the parsed snr value for saturated rows.
        if (!snr_saturated_flag) {
            auto& s = domain_mode_stats[dom][mode];
            s.snr_sum += snr; s.snr_count++;
        }
        if (f1 >= 0) {
            auto& s = domain_mode_stats[dom][mode];
            s.f1_sum += f1; s.f1_count++;
        }
        if (spec >= 0) {
            auto& s = domain_mode_stats[dom][mode];
            s.spec_sum += spec; s.spec_count++;
        }
    }

    for (auto& [domain, mode_stats] : domain_mode_stats) {
        cout << "\n--- " << (domain == "ecg" ? "ECG (MIT-BIH)" : "Vibration (CWRU Bearing)") << " ---" << endl;

        if (domain == "ecg") {
            cout << left << setw(22) << "Mode" << right
                 << setw(10) << "SNR(dB)" << setw(12) << "R-peak F1" << setw(8) << "n" << endl;
            cout << string(52, '-') << endl;
        } else {
            cout << left << setw(22) << "Mode" << right
                 << setw(10) << "SNR(dB)" << setw(12) << "Spectral r" << setw(8) << "n" << endl;
            cout << string(52, '-') << endl;
        }

        vector<pair<string, Stats>> sorted_modes(mode_stats.begin(), mode_stats.end());
        sort(sorted_modes.begin(), sorted_modes.end(),
             [](auto& a, auto& b) {
                 double sa = (a.second.snr_count > 0) ? a.second.snr_sum/a.second.snr_count : -1000;
                 double sb = (b.second.snr_count > 0) ? b.second.snr_sum/b.second.snr_count : -1000;
                 return sa > sb;
             });

        for (auto& [mode, s] : sorted_modes) {
            double avg_snr = (s.snr_count > 0) ? s.snr_sum / s.snr_count : -1000;
            string tag = "";
            if (mode == "IMP_INTERP_ERROR" || mode == "IMP_INTERP_COMPOSITE") tag = " <<<";

            if (domain == "ecg") {
                double avg_f1 = (s.f1_count > 0) ? s.f1_sum / s.f1_count : -1;
                cout << left << setw(22) << mode << right << fixed
                     << setprecision(2) << setw(10) << avg_snr
                     << setprecision(4) << setw(12) << avg_f1
                     << setw(8) << s.snr_count << tag << endl;
            } else {
                double avg_spec = (s.spec_count > 0) ? s.spec_sum / s.spec_count : -1;
                cout << left << setw(22) << mode << right << fixed
                     << setprecision(2) << setw(10) << avg_snr
                     << setprecision(4) << setw(12) << avg_spec
                     << setw(8) << s.snr_count << tag << endl;
            }
        }
    }

    cout << "\nTotal experiments: " << done << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_sec << "s ("
         << setprecision(1) << total_sec/60 << " min)" << endl;
    cout << "Results: results/cross_domain_results.csv" << endl;

    return 0;
}
