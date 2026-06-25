#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <random>
#include <mutex>
#include <functional>
#include <iomanip>
#include <queue>
#include <numeric>

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

// ============================================================
// Signal Generators
// ============================================================

vector<double> generate_sine(int count, double freq_hz, double noise_std, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> noise(0.0, noise_std);
    vector<double> signal(count);
    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / 1000.0;
        signal[i] = sin(2.0 * M_PI * freq_hz * t) + noise(gen);
    }
    return signal;
}

vector<double> generate_chirp(int count, double f0, double f1, double noise_std, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> noise(0.0, noise_std);
    vector<double> signal(count);
    double T = count / 1000.0;
    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / 1000.0;
        double freq = f0 + (f1 - f0) * t / T;
        signal[i] = sin(2.0 * M_PI * freq * t) + noise(gen);
    }
    return signal;
}

vector<double> generate_spikes(int count, double spike_interval_ms, double noise_std, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> noise(0.0, noise_std);
    vector<double> signal(count);
    int interval = static_cast<int>(spike_interval_ms);
    for (int i = 0; i < count; ++i) {
        signal[i] = noise(gen);
        if (interval > 0 && i % interval == 0) {
            signal[i] += 2.0;
            if (i + 1 < count) signal[i + 1] += 1.5;
            if (i + 2 < count) signal[i + 2] += 0.5;
        }
    }
    return signal;
}

vector<double> load_ecg(const string& filepath) {
    vector<double> data;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "WARNING: Cannot open " << filepath << endl;
        return data;
    }
    double val;
    while (file >> val) data.push_back(val);
    cout << "  Loaded " << data.size() << " ECG samples from " << filepath << endl;
    return data;
}

// ============================================================
// OFFLINE BASELINE 1: Ramer-Douglas-Peucker (iterative removal)
// Gold standard. O(N log N) with min-heap + lazy deletion.
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

// ============================================================
// OFFLINE BASELINE 2: Largest-Triangle-Three-Buckets (LTTB)
// Industry standard (Grafana, TimescaleDB). O(N).
// ============================================================

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
        int b_start = (int)((bucket) * bucket_size) + 1;
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
// Run offline baseline with a specific target surviving count
// ============================================================

EvalMetrics run_offline_baseline(const string& method,
                                 const vector<double>& signal,
                                 int target_surviving,
                                 bool is_ecg) {
    vector<int> selected;
    if (method == "RDP_OFFLINE")
        selected = offline_rdp(signal, target_surviving);
    else
        selected = offline_lttb(signal, target_surviving);

    vector<double> surviving_values;
    vector<int> surviving_indices;
    for (int idx : selected) {
        surviving_indices.push_back(idx);
        surviving_values.push_back(signal[idx]);
    }

    size_t drops = signal.size() - selected.size();
    return compute_all_metrics(signal, surviving_indices, surviving_values,
                               drops, 0.0, is_ecg);
}

// ============================================================
// Online ring buffer experiment
// ============================================================

struct ExperimentConfig {
    string signal_name;
    string mode_name;
    BufferMode mode;
    size_t buffer_size;
    int consumer_delay_us;
    int producer_delay_us;
    double noise_std;
    int trial;
    ImportanceConfig imp_config;
};

struct ExperimentResult {
    ExperimentConfig config;
    EvalMetrics metrics;
};

void producer_thread(RingBuffer<double>& buffer, const vector<double>& signal, int delay_us) {
    for (size_t i = 0; i < signal.size(); ++i) {
        buffer.push(signal[i], static_cast<int>(i));
        this_thread::sleep_for(chrono::microseconds(delay_us));
    }
}

void consumer_thread(RingBuffer<double>& buffer,
                     vector<double>& surviving_values,
                     vector<int>& surviving_indices,
                     int delay_us, int expected_count) {
    try {
        for (int i = 0; i < expected_count; ++i) {
            IndexedSample sample = buffer.pop_indexed();
            surviving_values.push_back(sample.value);
            surviving_indices.push_back(sample.original_index);
            this_thread::sleep_for(chrono::microseconds(delay_us));
        }
    } catch (const runtime_error&) {}
}

ExperimentResult run_online_experiment(
    const ExperimentConfig& config,
    const vector<double>& original_signal)
{
    RingBuffer<double> buffer(config.buffer_size, config.mode,
                               chrono::milliseconds(2), config.imp_config);

    vector<double> surviving_values;
    vector<int> surviving_indices;
    surviving_values.reserve(original_signal.size());
    surviving_indices.reserve(original_signal.size());

    thread prod(producer_thread, ref(buffer), cref(original_signal), config.producer_delay_us);
    thread cons(consumer_thread, ref(buffer), ref(surviving_values), ref(surviving_indices),
                config.consumer_delay_us, (int)original_signal.size());
    prod.join();
    buffer.finish();
    cons.join();

    bool is_ecg = (config.signal_name.find("ecg") != string::npos);
    EvalMetrics metrics = compute_all_metrics(
        original_signal, surviving_indices, surviving_values,
        buffer.getDropCount(), buffer.getMaxProducerWaitMs(), is_ecg);
    return {config, metrics};
}

// ============================================================
// CSV Output
//
// Two distinct flags accompany every row:
//   degenerate     -> drop_count == 0 (no eviction occurred at all)
//   snr_saturated  -> snr_db is non-finite (reconstruction error
//                      underflowed below the noise floor). This can
//                      happen even when drop_count > 0, e.g. an
//                      offline method finding a near-perfectly-
//                      interpolable selection on a smooth signal
//                      segment. Do NOT assume one implies the other —
//                      filter aggregate SNR statistics on
//                      snr_saturated, and use degenerate only to
//                      explain *why* a row should or should not be
//                      compared at all (blocking mode vs. real
//                      eviction with exact reconstruction).
// ============================================================

void write_csv_header(ofstream& file) {
    file << "signal,mode,buffer_size,consumer_delay_us,producer_delay_us,"
         << "noise_std,trial,drops,max_wait_ms,mse,snr_db,"
         << "max_error,deriv_ratio,rpeak_accuracy,degenerate,snr_saturated" << endl;
}

void write_csv_row(ofstream& file, const string& sig_name,
                   const string& mode_name, size_t buf_size,
                   int cons_delay, int prod_delay, double noise,
                   int trial, const EvalMetrics& m) {
    file << sig_name << ","
         << mode_name << ","
         << buf_size << ","
         << cons_delay << ","
         << prod_delay << ","
         << fixed << setprecision(4) << noise << ","
         << trial << ","
         << m.drop_count << ","
         << fixed << setprecision(4) << m.max_wait_ms << ","
         << scientific << setprecision(6) << m.mse << ","
         << fixed << setprecision(4) << m.snr_db << ","
         << scientific << setprecision(6) << m.max_error << ","
         << fixed << setprecision(4) << m.deriv_ratio << ","
         << fixed << setprecision(4) << m.r_peak_accuracy << ","
         << m.degenerate << ","
         << m.snr_saturated
         << endl;
}

// ============================================================
// Main Sweep — Two-pass: online first, then offline matched
// ============================================================

int main(int argc, char* argv[]) {
    cout << "=== Adaptive Ring Buffer — Experiment Sweep ===" << endl;
    cout << "Target: DATE 2027" << endl << endl;

    int producer_delay_us = 100;

    vector<size_t> buffer_sizes = {32, 64, 128, 256, 512};
    vector<int> consumer_delay_ratios = {2, 3, 4, 5, 6};
    vector<double> noise_levels = {0.01, 0.05, 0.1, 0.2};
    int sample_count = 1000;
    int num_trials = 5;

    if (argc > 1 && string(argv[1]) == "--quick") {
        buffer_sizes = {128, 256};
        consumer_delay_ratios = {2, 4};
        noise_levels = {0.05};
        num_trials = 2;
        cout << "[QUICK MODE]" << endl << endl;
    }

    struct ModeEntry { string name; BufferMode mode; };
    vector<ModeEntry> online_modes = {
        {"DROP",                BufferMode::DROP},
        {"WAIT",               BufferMode::WAIT},
        {"TIMED",              BufferMode::TIMED_WAIT},
        {"ADAPTIVE",           BufferMode::ADAPTIVE_TIMED_WAIT},
        {"LEGACY_IMPORTANCE",  BufferMode::ADAPTIVE_IMPORTANCE},
        {"RANDOM_DROP",        BufferMode::RANDOM_DROP},
        {"DROP_MIDDLE",        BufferMode::DROP_MIDDLE},
        {"DROP_LOW_VARIANCE",  BufferMode::DROP_LOW_VARIANCE},
        {"IMP_FIRST_ORDER",    BufferMode::IMPORTANCE_FIRST_ORDER},
        {"IMP_SECOND_ORDER",   BufferMode::IMPORTANCE_SECOND_ORDER},
        {"IMP_WINDOWED_ENERGY",BufferMode::IMPORTANCE_WINDOWED_ENERGY},
        {"IMP_COMPOSITE",      BufferMode::IMPORTANCE_COMPOSITE},
        {"IMP_ADAPTIVE",       BufferMode::IMPORTANCE_ADAPTIVE},
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
    };

    vector<string> offline_methods = {"RDP_OFFLINE", "LTTB_OFFLINE"};

    int total_modes = online_modes.size() + offline_methods.size();
    int total_est = buffer_sizes.size() * consumer_delay_ratios.size() *
                    noise_levels.size() * total_modes * num_trials * 3;
    cout << "Estimated experiments: " << total_est << endl;
    cout << "Two-pass: online first, offline matched to same drop count" << endl << endl;

    ofstream csv("../results/sweep_results.csv");
    write_csv_header(csv);

    int completed = 0;
    auto start_time = chrono::high_resolution_clock::now();

    auto run_signal_set = [&](const string& sig_name, const vector<double>& sig_data,
                              double noise, size_t buf_size, int ratio, int trial) {
        int cons_delay_us = producer_delay_us * ratio;
        bool is_ecg = (sig_name.find("ecg") != string::npos);

        int interp_drops = -1;

        for (auto& m : online_modes) {
            ExperimentConfig config;
            config.signal_name = sig_name;
            config.mode_name = m.name;
            config.mode = m.mode;
            config.buffer_size = buf_size;
            config.consumer_delay_us = cons_delay_us;
            config.producer_delay_us = producer_delay_us;
            config.noise_std = noise;
            config.trial = trial;
            config.imp_config = ImportanceConfig();

            auto result = run_online_experiment(config, sig_data);
            write_csv_row(csv, sig_name, m.name, buf_size,
                         cons_delay_us, producer_delay_us, noise,
                         trial, result.metrics);

            if (m.name == "IMP_INTERP_ERROR")
                interp_drops = (int)result.metrics.drop_count;

            ++completed;
        }

        if (interp_drops >= 0) {
            int target_surviving = (int)sig_data.size() - interp_drops;
            target_surviving = max(2, target_surviving);

            for (auto& method : offline_methods) {
                auto metrics = run_offline_baseline(method, sig_data,
                                                     target_surviving, is_ecg);

                write_csv_row(csv, sig_name, method, buf_size,
                             cons_delay_us, producer_delay_us, noise,
                             trial, metrics);
                ++completed;
            }
        }

        if (completed % 200 == 0) {
            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            double rate = completed / elapsed;
            double remaining = (total_est - completed) / rate;
            cout << "  Progress: " << completed << "/" << total_est
                 << " (" << fixed << setprecision(0) << remaining << "s remaining)" << endl;
        }
    };

    for (size_t buf_size : buffer_sizes) {
        for (int ratio : consumer_delay_ratios) {
            for (double noise : noise_levels) {
                for (int trial = 0; trial < num_trials; ++trial) {
                    unsigned seed = 42 + trial * 1000;

                    run_signal_set("sine_1hz",
                        generate_sine(sample_count, 1.0, noise, seed),
                        noise, buf_size, ratio, trial);
                    run_signal_set("chirp",
                        generate_chirp(sample_count, 0.5, 5.0, noise, seed),
                        noise, buf_size, ratio, trial);
                    run_signal_set("spikes",
                        generate_spikes(sample_count, 200, noise, seed),
                        noise, buf_size, ratio, trial);
                }
            }
        }
    }

    vector<double> ecg = load_ecg("../data/mit-bih/ecg_signal.txt");
    if (!ecg.empty()) {
        if ((int)ecg.size() > sample_count) ecg.resize(sample_count);
        cout << endl << "Running ECG experiments..." << endl;

        for (size_t buf_size : buffer_sizes) {
            for (int ratio : consumer_delay_ratios) {
                for (int trial = 0; trial < num_trials; ++trial) {
                    run_signal_set("ecg_mitbih_100", ecg, 0.0,
                                   buf_size, ratio, trial);
                }
            }
        }
    }

    csv.close();

    auto end_time = chrono::high_resolution_clock::now();
    double total_secs = chrono::duration<double>(end_time - start_time).count();

    cout << endl << "=== SWEEP COMPLETE ===" << endl;
    cout << "Total experiments: " << completed << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_secs << " seconds ("
         << setprecision(1) << total_secs/60.0 << " minutes)" << endl;
    cout << "Results: results/sweep_results.csv" << endl;

    return 0;
}
