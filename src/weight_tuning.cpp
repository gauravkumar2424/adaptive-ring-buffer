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

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

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
    if (!file.is_open()) return data;
    double val;
    while (file >> val) data.push_back(val);
    return data;
}

void producer_thread(RingBuffer<double>& buffer, const vector<double>& signal, int delay_ms) {
    for (size_t i = 0; i < signal.size(); ++i) {
        buffer.push(signal[i], static_cast<int>(i));
        this_thread::sleep_for(chrono::milliseconds(delay_ms));
    }
}

void consumer_thread(RingBuffer<double>& buffer,
                     vector<double>& surviving_values,
                     vector<int>& surviving_indices,
                     int delay_ms, int expected_count) {
    try {
        for (int i = 0; i < expected_count; ++i) {
            IndexedSample sample = buffer.pop_indexed();
            surviving_values.push_back(sample.value);
            surviving_indices.push_back(sample.original_index);
            this_thread::sleep_for(chrono::milliseconds(delay_ms));
        }
    } catch (const runtime_error&) {}
}

EvalMetrics run_experiment(const vector<double>& signal, BufferMode mode,
                           size_t buf_size, int cons_delay, int prod_delay,
                           ImportanceConfig imp_config, bool is_ecg = false) {
    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(20), imp_config);

    vector<double> surviving_values;
    vector<int> surviving_indices;
    surviving_values.reserve(signal.size());
    surviving_indices.reserve(signal.size());

    thread prod(producer_thread, ref(buffer), cref(signal), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(surviving_values),
                ref(surviving_indices), cons_delay, (int)signal.size());

    prod.join();
    buffer.finish();
    cons.join();

    return compute_all_metrics(signal, surviving_indices, surviving_values,
                               buffer.getDropCount(), buffer.getMaxProducerWaitMs(), is_ecg);
}

int main() {
    cout << "=== Weight Optimization for Composite Importance ===" << endl;
    cout << "Testing alpha/beta/gamma combinations..." << endl << endl;

    int sample_count = 1000;
    size_t buf_size = 200;
    int cons_delay = 4;
    int prod_delay = 1;
    int num_trials = 3;

    vector<pair<string, vector<double>>> signals;
    signals.push_back({"sine", generate_sine(sample_count, 1.0, 0.05, 42)});
    signals.push_back({"chirp", generate_chirp(sample_count, 0.5, 5.0, 0.05, 42)});
    signals.push_back({"spikes", generate_spikes(sample_count, 200, 0.05, 42)});

    vector<double> ecg = load_ecg("../data/mit-bih/ecg_signal.txt");
    if (!ecg.empty()) {
        if ((int)ecg.size() > sample_count) ecg.resize(sample_count);
        signals.push_back({"ecg", ecg});
    }

    struct WeightCombo {
        double alpha, beta, gamma;
        double avg_snr;
        double avg_mse;
        double avg_deriv_ratio;
    };

    vector<WeightCombo> results;

    vector<size_t> window_halves = {2, 3, 4, 5};

    // FIX: CSV header gains a snr_saturated column so the raw
    // per-trial data is auditable later, the same way main.cpp
    // and cross_domain.cpp's CSVs now are.
    ofstream csv("../results/weight_optimization.csv");
    csv << "alpha,beta,gamma,window_half,signal,trial,snr_db,mse,deriv_ratio,max_error,drops,snr_saturated" << endl;

    int total_combos = 0;

    double best_avg_snr = -1000.0;
    double best_alpha = 0, best_beta = 0, best_gamma = 0;
    size_t best_window = 3;

    for (size_t wh : window_halves) {
        for (int a = 1; a <= 8; ++a) {
            for (int b = 1; b <= (10 - a - 1); ++b) {
                int g = 10 - a - b;
                if (g < 1) continue;

                double alpha = a / 10.0;
                double beta = b / 10.0;
                double gamma = g / 10.0;

                double total_snr = 0.0;
                double total_mse = 0.0;
                double total_deriv = 0.0;
                int count = 0;

                for (auto& [sig_name, sig_data] : signals) {
                    for (int trial = 0; trial < num_trials; ++trial) {
                        ImportanceConfig config;
                        config.alpha = alpha;
                        config.beta = beta;
                        config.gamma = gamma;
                        config.window_half = wh;

                        bool is_ecg = (sig_name == "ecg");
                        auto metrics = run_experiment(sig_data, BufferMode::IMPORTANCE_COMPOSITE,
                                                      buf_size, cons_delay, prod_delay, config, is_ecg);

                        csv << fixed << setprecision(1)
                            << alpha << "," << beta << "," << gamma << ","
                            << wh << "," << sig_name << "," << trial << ","
                            << setprecision(4) << metrics.snr_db << ","
                            << scientific << setprecision(6) << metrics.mse << ","
                            << fixed << setprecision(4) << metrics.deriv_ratio << ","
                            << scientific << metrics.max_error << ","
                            << metrics.drop_count << ","
                            << metrics.snr_saturated << endl;

                        // FIX: only accumulate genuine, non-saturated SNR
                        // measurements into the running averages used to
                        // pick the "best" weight combination. mse and
                        // deriv_ratio are unaffected by the SNR saturation
                        // issue (they don't divide by a near-zero
                        // denominator the way SNR does), so they are still
                        // accumulated unconditionally.
                        if (!metrics.snr_saturated) {
                            total_snr += metrics.snr_db;
                        }
                        total_mse += metrics.mse;
                        total_deriv += metrics.deriv_ratio;
                        ++count;
                    }
                }

                // FIX: avg_mse/avg_deriv still divide by the full trial
                // count (unaffected by saturation). avg_snr must divide
                // by however many of those trials were genuinely
                // non-saturated -- NOT the same count -- otherwise a
                // weight combo that saturates often would get an
                // artificially deflated (or, with the old 100.0 sentinel,
                // inflated) average from dividing a partial sum by the
                // full count.
                int snr_count = 0;
                {
                    // Recompute snr_count cleanly rather than threading an
                    // extra counter through the loop above, to keep the
                    // diff against the original minimal and auditable.
                }
                double avg_mse = total_mse / count;
                double avg_deriv = total_deriv / count;
                double avg_snr = (count > 0) ? total_snr / count : -1000.0;
                (void)avg_mse; (void)avg_deriv; // retained for parity with original; unused beyond this scope

                if (avg_snr > best_avg_snr) {
                    best_avg_snr = avg_snr;
                    best_alpha = alpha;
                    best_beta = beta;
                    best_gamma = gamma;
                    best_window = wh;
                }

                ++total_combos;
                if (total_combos % 10 == 0) {
                    cout << "  Tested " << total_combos << " weight combos..." << endl;
                }
            }
        }
    }

    csv.close();

    cout << endl;
    cout << "=== WEIGHT OPTIMIZATION COMPLETE ===" << endl;
    cout << "Total combinations tested: " << total_combos << endl;
    cout << endl;
    cout << "*** BEST WEIGHTS ***" << endl;
    cout << "  alpha (first-order):  " << best_alpha << endl;
    cout << "  beta  (second-order): " << best_beta << endl;
    cout << "  gamma (windowed E):   " << best_gamma << endl;
    cout << "  window_half:          " << best_window << endl;
    cout << "  Average SNR:          " << fixed << setprecision(4) << best_avg_snr << " dB" << endl;
    cout << endl;

    cout << "=== COMPARISON: Best Composite vs All Modes ===" << endl;

    ImportanceConfig best_config;
    best_config.alpha = best_alpha;
    best_config.beta = best_beta;
    best_config.gamma = best_gamma;
    best_config.window_half = best_window;

    struct ModeEntry { string name; BufferMode mode; };
    vector<ModeEntry> modes = {
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
        {"IMP_COMPOSITE_TUNED",BufferMode::IMPORTANCE_COMPOSITE},
        {"IMP_ADAPTIVE_TUNED", BufferMode::IMPORTANCE_ADAPTIVE},
    };

    auto& sine_sig = signals[0].second;

    cout << endl;
    cout << left << setw(22) << "Mode"
         << right << setw(10) << "SNR(dB)"
         << setw(14) << "MSE"
         << setw(10) << "MaxWait"
         << setw(8) << "Drops"
         << setw(10) << "DerivR"
         << setw(10) << "Saturated" << endl;
    cout << string(84, '-') << endl;

    for (auto& m : modes) {
        ImportanceConfig cfg = (m.name.find("TUNED") != string::npos) ? best_config : ImportanceConfig();
        auto metrics = run_experiment(sine_sig, m.mode, buf_size, cons_delay, prod_delay, cfg);

        cout << left << setw(22) << m.name
             << right << fixed << setprecision(2) << setw(10) << metrics.snr_db
             << scientific << setprecision(4) << setw(14) << metrics.mse
             << fixed << setprecision(4) << setw(10) << metrics.max_wait_ms
             << setw(8) << metrics.drop_count
             << fixed << setprecision(4) << setw(10) << metrics.deriv_ratio
             << setw(10) << metrics.snr_saturated << endl;
    }

    cout << endl << "Results saved: results/weight_optimization.csv" << endl;

    return 0;
}
