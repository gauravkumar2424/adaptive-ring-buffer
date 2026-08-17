// ============================================================
// spectral_theory_experiment.cpp — Phase 1: Theoretical Foundation
// DAC 2027: Spectral Error Distribution in Online Signal Compression
//
// PURPOSE: Controlled experiments on synthetic signals with KNOWN
// spectral content to:
//   (1) Demonstrate the spectral error concentration phenomenon
//   (2) Validate the theoretical bound on per-bin spectral error
//   (3) Show the mechanism generalizes beyond vibration signals
//
// KEY IDEA (Theorem sketch):
//   Global optimization (RDP) minimizes total time-domain error.
//   By Parseval's theorem, this equals total frequency-domain error.
//   But minimizing the TOTAL does not minimize error at EACH bin.
//   Global optimization concentrates surviving samples near transients
//   (high per-sample error), under-sampling smooth oscillatory regions.
//   This creates CORRELATED gaps in oscillatory segments, causing
//   coherent spectral distortion at the frequencies of those oscillations.
//
//   Local (online) eviction, constrained to a buffer window of size N,
//   cannot create arbitrarily long sample-free gaps. The maximum gap
//   between surviving samples is bounded by a function of N and the
//   overload ratio. This acts as an implicit Nyquist constraint:
//   frequencies below f_Nyquist_local = 1/(2 * max_gap) are preserved.
//
// EXPERIMENTS:
//   Exp A: Two-tone signal (f1=low,large + f2=high,small)
//          Show RDP sacrifices f2, proposed preserves both
//   Exp B: Sweep amplitude ratio (A2/A1) to find crossover
//   Exp C: Sweep frequency ratio (f2/f1) 
//   Exp D: Multi-tone signal (3+ components) — generalization
//   Exp E: Chirp signal — continuous frequency variation
//   Exp F: Measure max-gap distribution vs buffer size (bound validation)
//
// Build: g++ -std=c++17 -O2 -Wall -o ../build/spectral_theory spectral_theory_experiment.cpp
// Run:   cd ../build && ./spectral_theory
// ============================================================

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
#include <map>
#include <cassert>
#include <sstream>

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

// ============================================================
// Offline RDP (from spectral_analysis.cpp)
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
// DFT (same as spectral_analysis.cpp, but with phase)
// ============================================================
struct SpectrumFull {
    vector<double> magnitude;
    vector<double> phase;
    vector<double> freq_bins;
};

SpectrumFull compute_spectrum_full(const vector<double>& signal) {
    int n = (int)signal.size();
    int half = n / 2;
    SpectrumFull spec;
    spec.magnitude.resize(half);
    spec.phase.resize(half);
    spec.freq_bins.resize(half);

    for (int k = 0; k < half; ++k) {
        double re = 0, im = 0;
        for (int i = 0; i < n; ++i) {
            double angle = 2.0 * M_PI * k * i / n;
            re += signal[i] * cos(angle);
            im -= signal[i] * sin(angle);
        }
        spec.magnitude[k] = sqrt(re * re + im * im);
        spec.phase[k] = atan2(im, re);
        spec.freq_bins[k] = (double)k / n;
    }
    return spec;
}

double spectral_correlation(const vector<double>& a, const vector<double>& b) {
    int n = min(a.size(), b.size());
    if (n < 4) return -1.0;
    double ma = accumulate(a.begin(), a.begin()+n, 0.0) / n;
    double mb = accumulate(b.begin(), b.begin()+n, 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < n; ++k) {
        double xa = a[k] - ma;
        double xb = b[k] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    double den = sqrt(da * db);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================
// Spacing analysis
// ============================================================
struct GapStats {
    double mean_gap, max_gap, cv;
    int n_samples;
    vector<int> gaps;
};

GapStats analyze_gaps(const vector<int>& indices) {
    GapStats s{};
    s.n_samples = (int)indices.size();
    if (indices.size() < 2) return s;
    for (size_t i = 1; i < indices.size(); ++i)
        s.gaps.push_back(indices[i] - indices[i-1]);
    s.max_gap = *max_element(s.gaps.begin(), s.gaps.end());
    s.mean_gap = accumulate(s.gaps.begin(), s.gaps.end(), 0.0) / s.gaps.size();
    double var = 0;
    for (int g : s.gaps) var += (g - s.mean_gap) * (g - s.mean_gap);
    double std_gap = sqrt(var / s.gaps.size());
    s.cv = (s.mean_gap > 1e-15) ? std_gap / s.mean_gap : 0.0;
    return s;
}

// ============================================================
// Producer/Consumer for online method
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

struct Result {
    string method;
    vector<int> surviving_indices;
    vector<double> reconstructed;
    double snr_db;
    double spec_corr;
    GapStats gaps;
    // Per-component error (for multi-tone)
    vector<double> component_mag_error;  // |orig_mag[k] - recon_mag[k]| at each tone bin
    vector<double> component_rel_error;  // relative error at each tone bin
};

Result run_online_method(const vector<double>& signal, BufferMode mode,
                         const string& name, size_t buf_size, int overload) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());
    vector<double> surv_vals;
    vector<int> surv_idx;
    surv_vals.reserve(signal.size());
    surv_idx.reserve(signal.size());

    thread prod(producer_thread, ref(buffer), cref(signal), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(surv_vals), ref(surv_idx),
                cons_delay, (int)signal.size());
    prod.join();
    buffer.finish();
    cons.join();

    Result r;
    r.method = name;
    r.surviving_indices = surv_idx;
    r.reconstructed = reconstruct_signal(surv_idx, surv_vals, (int)signal.size());
    r.snr_db = compute_snr(signal, r.reconstructed);
    r.gaps = analyze_gaps(surv_idx);
    return r;
}

Result run_rdp(const vector<double>& signal, int target_points) {
    auto selected = offline_rdp(signal, target_points);
    vector<double> surv_vals;
    for (int idx : selected) surv_vals.push_back(signal[idx]);

    Result r;
    r.method = "RDP";
    r.surviving_indices = selected;
    r.reconstructed = reconstruct_signal(selected, surv_vals, (int)signal.size());
    r.snr_db = compute_snr(signal, r.reconstructed);
    r.gaps = analyze_gaps(selected);
    return r;
}

// ============================================================
// Signal generators
// ============================================================
vector<double> make_two_tone(int N, double f1, double A1, double f2, double A2) {
    // f1, f2 in normalized frequency (cycles per sample)
    vector<double> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = A1 * sin(2.0 * M_PI * f1 * i) + A2 * sin(2.0 * M_PI * f2 * i);
    return sig;
}

vector<double> make_multi_tone(int N, const vector<double>& freqs,
                                const vector<double>& amps) {
    vector<double> sig(N, 0.0);
    for (size_t t = 0; t < freqs.size(); ++t)
        for (int i = 0; i < N; ++i)
            sig[i] += amps[t] * sin(2.0 * M_PI * freqs[t] * i);
    return sig;
}

vector<double> make_chirp(int N, double f_start, double f_end) {
    // Linear chirp from f_start to f_end (normalized freq)
    vector<double> sig(N);
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double f = f_start + (f_end - f_start) * t;
        double phase = 2.0 * M_PI * (f_start * i + 0.5 * (f_end - f_start) * i * i / N);
        sig[i] = sin(phase);
    }
    return sig;
}

// Tone with transient: smooth oscillation + occasional sharp spike
// This models the RDP failure case: RDP will concentrate samples
// at the spike, starving the oscillatory region
vector<double> make_tone_plus_transient(int N, double f_tone, double A_tone,
                                         int spike_pos, double spike_amp) {
    vector<double> sig(N, 0.0);
    for (int i = 0; i < N; ++i)
        sig[i] = A_tone * sin(2.0 * M_PI * f_tone * i);
    // Add Gaussian spike
    for (int i = 0; i < N; ++i) {
        double d = (double)(i - spike_pos);
        sig[i] += spike_amp * exp(-d*d / 50.0);  // sigma ~ 5 samples
    }
    return sig;
}

// ============================================================
// Per-bin spectral error at specific frequency bins
// ============================================================
void compute_component_errors(Result& r, const vector<double>& orig_signal,
                               const vector<int>& tone_bins) {
    auto orig_spec = compute_spectrum_full(orig_signal);
    auto recon_spec = compute_spectrum_full(r.reconstructed);

    r.component_mag_error.clear();
    r.component_rel_error.clear();

    for (int k : tone_bins) {
        if (k < (int)orig_spec.magnitude.size() && k < (int)recon_spec.magnitude.size()) {
            double abs_err = fabs(orig_spec.magnitude[k] - recon_spec.magnitude[k]);
            double rel_err = (orig_spec.magnitude[k] > 1e-10) ?
                abs_err / orig_spec.magnitude[k] : 0.0;
            r.component_mag_error.push_back(abs_err);
            r.component_rel_error.push_back(rel_err);
        }
    }

    // Also compute spectral correlation
    r.spec_corr = spectral_correlation(orig_spec.magnitude, recon_spec.magnitude);
}

// ============================================================
// EXPERIMENT A: Two-tone demonstration
// The core demonstration for the paper
// ============================================================
void experiment_A(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT A: Two-Tone Spectral Error Concentration" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f1 = 0.02;   // low freq: 20 cycles in 1000 samples
    double A1 = 1.0;     // large amplitude
    double f2 = 0.15;    // high freq: 150 cycles in 1000 samples
    double A2 = 0.3;     // small amplitude

    // The DFT bin for each tone: bin_k = f * N
    int bin_f1 = (int)(f1 * N);  // bin 40
    int bin_f2 = (int)(f2 * N);  // bin 300

    cout << "Signal: " << N << " samples" << endl;
    cout << "  Tone 1: f=" << f1 << " (bin " << bin_f1 << "), A=" << A1 << endl;
    cout << "  Tone 2: f=" << f2 << " (bin " << bin_f2 << "), A=" << A2 << endl;

    auto signal = make_two_tone(N, f1, A1, f2, A2);

    // Multiple buffer sizes and overload ratios
    vector<size_t> buf_sizes = {64, 128, 256, 512};
    vector<int> overloads = {3, 5, 8, 10};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_A_two_tone.csv");
    csv << "buf_size,overload,trial,method,snr_db,spec_corr,"
        << "err_f1_rel,err_f2_rel,err_ratio_f2_f1,"
        << "mean_gap,max_gap,cv_gap,n_surviving" << endl;

    ofstream detail_csv(out_dir + "/exp_A_per_bin.csv");
    detail_csv << "buf_size,overload,trial,method,bin,norm_freq,"
               << "orig_mag,recon_mag,abs_error,rel_error" << endl;

    vector<int> tone_bins = {bin_f1, bin_f2};

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                // Run proposed
                auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                                   "PROPOSED", buf, ovl);
                int drops = (int)signal.size() - (int)proposed.surviving_indices.size();
                int target = max(2, (int)signal.size() - drops);

                // Run RDP matched to same drop count
                auto rdp = run_rdp(signal, target);

                // Compute per-component errors
                compute_component_errors(proposed, signal, tone_bins);
                compute_component_errors(rdp, signal, tone_bins);

                // Write results
                for (auto* r : {&proposed, &rdp}) {
                    double err_f1 = r->component_rel_error.size() > 0 ? r->component_rel_error[0] : -1;
                    double err_f2 = r->component_rel_error.size() > 1 ? r->component_rel_error[1] : -1;
                    double ratio = (err_f1 > 1e-10) ? err_f2 / err_f1 : -1;

                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr << ","
                        << err_f1 << ","
                        << err_f2 << ","
                        << ratio << ","
                        << r->gaps.mean_gap << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.cv << ","
                        << r->gaps.n_samples << endl;
                }

                // Full per-bin detail (first trial only per config)
                if (trial == 0) {
                    auto orig_spec = compute_spectrum_full(signal);
                    for (auto* r : {&proposed, &rdp}) {
                        auto recon_spec = compute_spectrum_full(r->reconstructed);
                        int nbins = min(orig_spec.magnitude.size(), recon_spec.magnitude.size());
                        for (int k = 0; k < nbins; ++k) {
                            double ae = fabs(orig_spec.magnitude[k] - recon_spec.magnitude[k]);
                            double re = (orig_spec.magnitude[k] > 1e-10) ?
                                ae / orig_spec.magnitude[k] : 0.0;
                            detail_csv << buf << "," << ovl << "," << trial << ","
                                       << r->method << "," << k << ","
                                       << orig_spec.freq_bins[k] << ","
                                       << orig_spec.magnitude[k] << ","
                                       << recon_spec.magnitude[k] << ","
                                       << ae << "," << re << endl;
                        }
                    }
                }
            }

            // Print summary for this config
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    detail_csv.close();
    cout << "  Output: exp_A_two_tone.csv, exp_A_per_bin.csv" << endl;
}

// ============================================================
// EXPERIMENT B: Amplitude ratio sweep
// At what A2/A1 ratio does RDP start sacrificing the weaker tone?
// ============================================================
void experiment_B(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT B: Amplitude Ratio Sweep" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f1 = 0.02, f2 = 0.15;
    double A1 = 1.0;
    vector<double> A2_values = {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0};
    size_t buf = 256;
    int ovl = 5;
    int num_trials = 5;

    int bin_f1 = (int)(f1 * N);
    int bin_f2 = (int)(f2 * N);
    vector<int> tone_bins = {bin_f1, bin_f2};

    ofstream csv(out_dir + "/exp_B_amplitude_sweep.csv");
    csv << "A2,A2_A1_ratio,trial,method,snr_db,spec_corr,"
        << "err_f1_rel,err_f2_rel,err_ratio_f2_f1,max_gap" << endl;

    for (double A2 : A2_values) {
        auto signal = make_two_tone(N, f1, A1, f2, A2);
        for (int trial = 0; trial < num_trials; ++trial) {
            auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                               "PROPOSED", buf, ovl);
            int target = max(2, (int)signal.size() -
                         (int)(signal.size() - proposed.surviving_indices.size()));
            auto rdp = run_rdp(signal, target);

            compute_component_errors(proposed, signal, tone_bins);
            compute_component_errors(rdp, signal, tone_bins);

            for (auto* r : {&proposed, &rdp}) {
                double ef1 = r->component_rel_error.size() > 0 ? r->component_rel_error[0] : -1;
                double ef2 = r->component_rel_error.size() > 1 ? r->component_rel_error[1] : -1;
                csv << A2 << "," << A2/A1 << "," << trial << ","
                    << r->method << ","
                    << fixed << setprecision(4)
                    << r->snr_db << ","
                    << r->spec_corr << ","
                    << ef1 << "," << ef2 << ","
                    << ((ef1 > 1e-10) ? ef2/ef1 : -1.0) << ","
                    << r->gaps.max_gap << endl;
            }
        }
        cout << "  A2/A1=" << A2/A1 << ": done" << endl;
    }
    csv.close();
    cout << "  Output: exp_B_amplitude_sweep.csv" << endl;
}

// ============================================================
// EXPERIMENT C: Frequency ratio sweep
// How does the gap between f1 and f2 affect the phenomenon?
// ============================================================
void experiment_C(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT C: Frequency Ratio Sweep" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f1 = 0.02;
    double A1 = 1.0, A2 = 0.3;
    vector<double> f2_values = {0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40};
    size_t buf = 256;
    int ovl = 5;
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_C_frequency_sweep.csv");
    csv << "f2,f2_f1_ratio,trial,method,snr_db,spec_corr,"
        << "err_f1_rel,err_f2_rel,max_gap" << endl;

    for (double f2 : f2_values) {
        int bin_f1 = (int)(f1 * N);
        int bin_f2 = (int)(f2 * N);
        vector<int> tone_bins = {bin_f1, bin_f2};

        auto signal = make_two_tone(N, f1, A1, f2, A2);
        for (int trial = 0; trial < num_trials; ++trial) {
            auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                               "PROPOSED", buf, ovl);
            int target = max(2, (int)signal.size() -
                         (int)(signal.size() - proposed.surviving_indices.size()));
            auto rdp = run_rdp(signal, target);

            compute_component_errors(proposed, signal, tone_bins);
            compute_component_errors(rdp, signal, tone_bins);

            for (auto* r : {&proposed, &rdp}) {
                double ef1 = r->component_rel_error.size() > 0 ? r->component_rel_error[0] : -1;
                double ef2 = r->component_rel_error.size() > 1 ? r->component_rel_error[1] : -1;
                csv << f2 << "," << f2/f1 << "," << trial << ","
                    << r->method << ","
                    << fixed << setprecision(6)
                    << r->snr_db << ","
                    << r->spec_corr << ","
                    << ef1 << "," << ef2 << ","
                    << r->gaps.max_gap << endl;
            }
        }
        cout << "  f2/f1=" << f2/f1 << ": done" << endl;
    }
    csv.close();
    cout << "  Output: exp_C_frequency_sweep.csv" << endl;
}

// ============================================================
// EXPERIMENT D: Multi-tone signal (3+ components)
// Does the mechanism generalize beyond two tones?
// ============================================================
void experiment_D(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT D: Multi-Tone Generalization" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    // 5-tone signal with decreasing amplitudes
    vector<double> freqs = {0.01, 0.04, 0.10, 0.20, 0.35};
    vector<double> amps  = {1.0,  0.5,  0.3,  0.15, 0.08};

    vector<int> tone_bins;
    for (double f : freqs) tone_bins.push_back((int)(f * N));

    auto signal = make_multi_tone(N, freqs, amps);

    vector<size_t> buf_sizes = {128, 256, 512};
    vector<int> overloads = {3, 5, 10};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_D_multi_tone.csv");
    csv << "buf_size,overload,trial,method,snr_db,spec_corr";
    for (size_t t = 0; t < freqs.size(); ++t)
        csv << ",err_f" << t+1 << "_rel";
    csv << ",max_gap,cv_gap" << endl;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                                   "PROPOSED", buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                compute_component_errors(proposed, signal, tone_bins);
                compute_component_errors(rdp, signal, tone_bins);

                for (auto* r : {&proposed, &rdp}) {
                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr;
                    for (size_t t = 0; t < freqs.size(); ++t) {
                        csv << "," << (t < r->component_rel_error.size() ?
                                       r->component_rel_error[t] : -1.0);
                    }
                    csv << "," << r->gaps.max_gap << "," << r->gaps.cv << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_D_multi_tone.csv" << endl;
}

// ============================================================
// EXPERIMENT E: Tone + Transient
// The critical case: RDP concentrates at the spike, online doesn't
// This is the clearest demonstration of the mechanism
// ============================================================
void experiment_E(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT E: Tone + Transient (Mechanism Isolation)" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f_tone = 0.08;
    double A_tone = 1.0;
    double spike_amp = 5.0;  // spike is 5x the tone amplitude
    int spike_pos = N / 2;   // spike in the middle

    int bin_tone = (int)(f_tone * N);
    vector<int> tone_bins = {bin_tone};

    auto signal = make_tone_plus_transient(N, f_tone, A_tone, spike_pos, spike_amp);

    vector<size_t> buf_sizes = {64, 128, 256, 512};
    vector<int> overloads = {3, 5, 8, 10, 15};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_E_tone_transient.csv");
    csv << "buf_size,overload,trial,method,snr_db,spec_corr,"
        << "err_tone_rel,max_gap,cv_gap,"
        << "n_samples_near_spike,n_samples_total" << endl;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                                   "PROPOSED", buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                compute_component_errors(proposed, signal, tone_bins);
                compute_component_errors(rdp, signal, tone_bins);

                // Count samples near spike (within +/- 20 of spike_pos)
                auto count_near_spike = [&](const vector<int>& idx) -> int {
                    int count = 0;
                    for (int i : idx)
                        if (abs(i - spike_pos) <= 20) count++;
                    return count;
                };

                for (auto* r : {&proposed, &rdp}) {
                    double et = r->component_rel_error.size() > 0 ? r->component_rel_error[0] : -1;
                    int near_spike = count_near_spike(r->surviving_indices);
                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr << ","
                        << et << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.cv << ","
                        << near_spike << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_E_tone_transient.csv" << endl;
}

// ============================================================
// EXPERIMENT F: Max-gap bound validation
// Measure empirical max gap vs buffer size to validate the
// theoretical bound: max_gap <= f(N, overload)
// ============================================================
void experiment_F(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT F: Max-Gap Bound vs Buffer Size" << endl;
    cout << "============================================================\n" << endl;

    int N = 5000;  // longer signal for better statistics
    // Use vibration-like signal (multiple frequencies)
    vector<double> freqs = {0.01, 0.05, 0.12, 0.25};
    vector<double> amps  = {1.0, 0.4, 0.2, 0.1};
    auto signal = make_multi_tone(N, freqs, amps);

    vector<size_t> buf_sizes = {32, 64, 128, 256, 512, 1024};
    vector<int> overloads = {2, 3, 5, 8, 10, 15, 20};
    int num_trials = 10;

    ofstream csv(out_dir + "/exp_F_max_gap_bound.csv");
    csv << "buf_size,overload,trial,method,max_gap,mean_gap,cv_gap,"
        << "p95_gap,p99_gap,n_surviving,n_total" << endl;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                                   "PROPOSED", buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    auto& g = r->gaps;
                    // Compute percentiles
                    vector<int> sorted_gaps = g.gaps;
                    sort(sorted_gaps.begin(), sorted_gaps.end());
                    int p95_idx = (int)(sorted_gaps.size() * 0.95);
                    int p99_idx = (int)(sorted_gaps.size() * 0.99);
                    double p95 = sorted_gaps.empty() ? 0 : sorted_gaps[min(p95_idx, (int)sorted_gaps.size()-1)];
                    double p99 = sorted_gaps.empty() ? 0 : sorted_gaps[min(p99_idx, (int)sorted_gaps.size()-1)];

                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << g.max_gap << ","
                        << fixed << setprecision(2) << g.mean_gap << ","
                        << setprecision(4) << g.cv << ","
                        << p95 << "," << p99 << ","
                        << g.n_samples << "," << N << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_F_max_gap_bound.csv" << endl;
}

// ============================================================
// EXPERIMENT G: Waveform + spectrum dump for paper figures
// Detailed output for one representative configuration
// ============================================================
void experiment_G(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT G: Paper Figure Data (representative config)" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;

    // Tone + transient: the cleanest demonstration
    double f_tone = 0.08;
    auto signal = make_tone_plus_transient(N, f_tone, 1.0, N/2, 5.0);

    size_t buf = 256;
    int ovl = 5;

    auto proposed = run_online_method(signal, BufferMode::IMPORTANCE_INTERP_ERROR,
                                       "PROPOSED", buf, ovl);
    int target = max(2, (int)signal.size() -
                 (int)(signal.size() - proposed.surviving_indices.size()));
    auto rdp = run_rdp(signal, target);

    // Dump waveforms
    ofstream wav_csv(out_dir + "/exp_G_waveforms.csv");
    wav_csv << "sample,original,proposed_recon,rdp_recon" << endl;
    for (int i = 0; i < N; ++i) {
        wav_csv << i << "," << signal[i] << ","
                << proposed.reconstructed[i] << ","
                << rdp.reconstructed[i] << endl;
    }
    wav_csv.close();

    // Dump surviving sample positions
    ofstream pos_csv(out_dir + "/exp_G_surviving_positions.csv");
    pos_csv << "method,sample_idx,value" << endl;
    for (int i : proposed.surviving_indices)
        pos_csv << "PROPOSED," << i << "," << signal[i] << endl;
    for (int i : rdp.surviving_indices)
        pos_csv << "RDP," << i << "," << signal[i] << endl;
    pos_csv.close();

    // Dump spectra
    auto orig_spec = compute_spectrum_full(signal);
    auto prop_spec = compute_spectrum_full(proposed.reconstructed);
    auto rdp_spec = compute_spectrum_full(rdp.reconstructed);

    ofstream spec_csv(out_dir + "/exp_G_spectra.csv");
    spec_csv << "bin,norm_freq,orig_mag,proposed_mag,rdp_mag,"
             << "proposed_err,rdp_err,proposed_rel_err,rdp_rel_err" << endl;
    int nbins = min({(int)orig_spec.magnitude.size(),
                     (int)prop_spec.magnitude.size(),
                     (int)rdp_spec.magnitude.size()});
    for (int k = 0; k < nbins; ++k) {
        double pe = fabs(orig_spec.magnitude[k] - prop_spec.magnitude[k]);
        double re = fabs(orig_spec.magnitude[k] - rdp_spec.magnitude[k]);
        double pre = (orig_spec.magnitude[k] > 1e-10) ? pe / orig_spec.magnitude[k] : 0;
        double rre = (orig_spec.magnitude[k] > 1e-10) ? re / orig_spec.magnitude[k] : 0;

        spec_csv << k << "," << orig_spec.freq_bins[k] << ","
                 << orig_spec.magnitude[k] << ","
                 << prop_spec.magnitude[k] << ","
                 << rdp_spec.magnitude[k] << ","
                 << pe << "," << re << ","
                 << pre << "," << rre << endl;
    }
    spec_csv.close();

    // Dump gap histograms
    ofstream gap_csv(out_dir + "/exp_G_gaps.csv");
    gap_csv << "method,gap_size" << endl;
    for (int g : proposed.gaps.gaps) gap_csv << "PROPOSED," << g << endl;
    for (int g : rdp.gaps.gaps) gap_csv << "RDP," << g << endl;
    gap_csv.close();

    // Print summary
    cout << "  PROPOSED: SNR=" << fixed << setprecision(2) << proposed.snr_db
         << " dB, spec_corr=" << setprecision(4) << proposed.spec_corr
         << ", max_gap=" << proposed.gaps.max_gap
         << ", surviving=" << proposed.gaps.n_samples << endl;
    cout << "  RDP:      SNR=" << fixed << setprecision(2) << rdp.snr_db
         << " dB, spec_corr=" << setprecision(4) << rdp.spec_corr
         << ", max_gap=" << rdp.gaps.max_gap
         << ", surviving=" << rdp.gaps.n_samples << endl;

    cout << "  Output: exp_G_waveforms.csv, exp_G_surviving_positions.csv," << endl;
    cout << "          exp_G_spectra.csv, exp_G_gaps.csv" << endl;
}


// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  SPECTRAL THEORY EXPERIMENTS — Phase 1" << endl;
    cout << "  DAC 2027: Why Greedy Beats Optimal in Spectral Fidelity" << endl;
    cout << "============================================================\n" << endl;

    string out_dir = "../results/spectral_theory";
    system(("mkdir -p " + out_dir).c_str());

    auto t0 = chrono::high_resolution_clock::now();

    experiment_A(out_dir);  // Two-tone demonstration
    experiment_B(out_dir);  // Amplitude ratio sweep
    experiment_C(out_dir);  // Frequency ratio sweep
    experiment_D(out_dir);  // Multi-tone generalization
    experiment_E(out_dir);  // Tone + transient (mechanism isolation)
    experiment_F(out_dir);  // Max-gap bound validation
    experiment_G(out_dir);  // Paper figure data

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    cout << "\n============================================================" << endl;
    cout << "  ALL EXPERIMENTS COMPLETE" << endl;
    cout << "  Wall-clock: " << fixed << setprecision(1) << elapsed << " s" << endl;
    cout << "  Output directory: " << out_dir << endl;
    cout << "============================================================" << endl;

    return 0;
}
