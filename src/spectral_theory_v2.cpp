// ============================================================
// spectral_theory_v2.cpp — Phase 1b: Non-Stationarity Hypothesis
// DAC 2027
//
// BACKGROUND:
//   Phase 1a (spectral_theory_experiment.cpp) showed that on STATIONARY
//   synthetic signals (two-tone, tone+transient), RDP beats proposed
//   on ALL metrics including spectral correlation. The spectral advantage
//   observed on real vibration signals cannot be explained by gap
//   uniformity or implicit Nyquist constraints on stationary signals.
//
// CORRECTED HYPOTHESIS:
//   The spectral advantage of online eviction appears specifically on
//   NON-STATIONARY signals where different temporal regions have
//   different spectral content. RDP, optimizing globally, misallocates
//   samples by concentrating at high-error features (transients)
//   regardless of what spectral content each region carries. Online
//   eviction, constrained to a local buffer window, allocates samples
//   proportionally to each region's local content, preserving per-
//   region spectral structure.
//
//   Formally: for a signal x[n] = Σ_k s_k[n] where s_k are locally-
//   supported components with different spectral content, global
//   optimization treats the signal as a single entity and allocates
//   samples by global interpolation error. Local optimization treats
//   each windowed segment semi-independently, preserving each s_k's
//   spectral content in proportion to its local prominence.
//
// EXPERIMENTS:
//   H: Piecewise-stationary signal (two segments, different frequencies)
//   I: Piecewise + transient (isolation of the misallocation mechanism)
//   J: Gradual chirp (continuous non-stationarity)
//   K: Stationarity control — confirm NO advantage on stationary signals
//   L: Multi-segment (4 segments, different frequencies)
//   M: Sweep non-stationarity degree (mixing ratio)
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/spectral_v2 spectral_theory_v2.cpp
// Run:   cd ../build && ./spectral_v2
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

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

// ============================================================
// Offline RDP
// ============================================================
vector<int> offline_rdp(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n); iota(all.begin(), all.end(), 0); return all;
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

// ============================================================
// DFT + spectral analysis
// ============================================================
struct Spectrum {
    vector<double> magnitude;
    vector<double> freq_bins;
};

Spectrum compute_spectrum(const vector<double>& signal) {
    int n = (int)signal.size();
    int half = n / 2;
    Spectrum spec;
    spec.magnitude.resize(half);
    spec.freq_bins.resize(half);
    for (int k = 0; k < half; ++k) {
        double re = 0, im = 0;
        for (int i = 0; i < n; ++i) {
            double angle = 2.0 * M_PI * k * i / n;
            re += signal[i] * cos(angle);
            im -= signal[i] * sin(angle);
        }
        spec.magnitude[k] = sqrt(re * re + im * im);
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
        double xa = a[k] - ma, xb = b[k] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    double den = sqrt(da * db);
    return (den > 1e-15) ? num / den : 0.0;
}

// Per-segment spectral correlation: compute spectrum on a segment only
double segment_spectral_corr(const vector<double>& orig,
                              const vector<double>& recon,
                              int seg_start, int seg_end) {
    int len = seg_end - seg_start;
    if (len < 8) return -1.0;

    // Extract segments
    vector<double> orig_seg(orig.begin() + seg_start, orig.begin() + seg_end);
    vector<double> recon_seg(recon.begin() + seg_start, recon.begin() + seg_end);

    auto s1 = compute_spectrum(orig_seg);
    auto s2 = compute_spectrum(recon_seg);
    return spectral_correlation(s1.magnitude, s2.magnitude);
}

// Count surviving samples in a segment
int count_in_segment(const vector<int>& indices, int seg_start, int seg_end) {
    int c = 0;
    for (int i : indices)
        if (i >= seg_start && i < seg_end) c++;
    return c;
}

// ============================================================
// Spacing analysis
// ============================================================
struct GapStats {
    double mean_gap, max_gap, cv;
    int n_samples;
};

GapStats analyze_gaps(const vector<int>& indices) {
    GapStats s{};
    s.n_samples = (int)indices.size();
    if (indices.size() < 2) return s;
    vector<int> gaps;
    for (size_t i = 1; i < indices.size(); ++i)
        gaps.push_back(indices[i] - indices[i-1]);
    s.max_gap = *max_element(gaps.begin(), gaps.end());
    s.mean_gap = accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
    double var = 0;
    for (int g : gaps) var += (g - s.mean_gap) * (g - s.mean_gap);
    s.cv = (s.mean_gap > 1e-15) ? sqrt(var / gaps.size()) / s.mean_gap : 0.0;
    return s;
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

struct Result {
    string method;
    vector<int> surviving_indices;
    vector<double> reconstructed;
    double snr_db;
    double spec_corr_global;
    GapStats gaps;
};

Result run_online(const vector<double>& signal, size_t buf, int ovl) {
    int prod_delay = 100;
    int cons_delay = prod_delay * ovl;
    RingBuffer<double> buffer(buf, BufferMode::IMPORTANCE_INTERP_ERROR,
                               chrono::milliseconds(2), ImportanceConfig());
    vector<double> sv; vector<int> si;
    sv.reserve(signal.size()); si.reserve(signal.size());

    thread prod(producer_thread, ref(buffer), cref(signal), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(sv), ref(si),
                cons_delay, (int)signal.size());
    prod.join(); buffer.finish(); cons.join();

    Result r;
    r.method = "PROPOSED";
    r.surviving_indices = si;
    r.reconstructed = reconstruct_signal(si, sv, (int)signal.size());
    r.snr_db = compute_snr(signal, r.reconstructed);
    r.gaps = analyze_gaps(si);
    auto sp_o = compute_spectrum(signal);
    auto sp_r = compute_spectrum(r.reconstructed);
    r.spec_corr_global = spectral_correlation(sp_o.magnitude, sp_r.magnitude);
    return r;
}

Result run_rdp(const vector<double>& signal, int target) {
    auto sel = offline_rdp(signal, target);
    vector<double> sv;
    for (int i : sel) sv.push_back(signal[i]);

    Result r;
    r.method = "RDP";
    r.surviving_indices = sel;
    r.reconstructed = reconstruct_signal(sel, sv, (int)signal.size());
    r.snr_db = compute_snr(signal, r.reconstructed);
    r.gaps = analyze_gaps(sel);
    auto sp_o = compute_spectrum(signal);
    auto sp_r = compute_spectrum(r.reconstructed);
    r.spec_corr_global = spectral_correlation(sp_o.magnitude, sp_r.magnitude);
    return r;
}

// ============================================================
// Signal generators
// ============================================================

// Piecewise-stationary: two halves with different frequencies
vector<double> make_piecewise_two_freq(int N, double f1, double f2,
                                        double A1, double A2) {
    vector<double> sig(N);
    int half = N / 2;
    for (int i = 0; i < half; ++i)
        sig[i] = A1 * sin(2.0 * M_PI * f1 * i);
    for (int i = half; i < N; ++i)
        sig[i] = A2 * sin(2.0 * M_PI * f2 * i);
    return sig;
}

// Piecewise + transient spikes at segment boundaries
vector<double> make_piecewise_transient(int N, double f1, double f2,
                                         double A, double spike_amp) {
    auto sig = make_piecewise_two_freq(N, f1, f2, A, A);
    // Add spikes at 25% and 75% positions
    int spike1 = N / 4, spike2 = 3 * N / 4;
    for (int i = 0; i < N; ++i) {
        double d1 = (double)(i - spike1);
        double d2 = (double)(i - spike2);
        sig[i] += spike_amp * exp(-d1*d1 / 50.0);
        sig[i] += spike_amp * exp(-d2*d2 / 50.0);
    }
    return sig;
}

// Multi-segment: 4 segments with 4 different frequencies
vector<double> make_multi_segment(int N, const vector<double>& freqs,
                                   double A) {
    vector<double> sig(N);
    int seg_len = N / (int)freqs.size();
    for (size_t s = 0; s < freqs.size(); ++s) {
        int start = s * seg_len;
        int end = (s == freqs.size() - 1) ? N : (s + 1) * seg_len;
        for (int i = start; i < end; ++i)
            sig[i] = A * sin(2.0 * M_PI * freqs[s] * i);
    }
    return sig;
}

// Chirp: continuously varying frequency
vector<double> make_chirp(int N, double f_start, double f_end, double A) {
    vector<double> sig(N);
    for (int i = 0; i < N; ++i) {
        double t = (double)i / N;
        double phase = 2.0 * M_PI * (f_start * i +
                       0.5 * (f_end - f_start) * (double)i * i / N);
        sig[i] = A * sin(phase);
    }
    return sig;
}

// Stationary control: single frequency throughout
vector<double> make_stationary(int N, double f, double A) {
    vector<double> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = A * sin(2.0 * M_PI * f * i);
    return sig;
}

// Stationary with transients (control: non-stationarity from amplitude, not frequency)
vector<double> make_stationary_transient(int N, double f, double A,
                                          double spike_amp) {
    auto sig = make_stationary(N, f, A);
    int spike1 = N / 4, spike2 = 3 * N / 4;
    for (int i = 0; i < N; ++i) {
        double d1 = (double)(i - spike1);
        double d2 = (double)(i - spike2);
        sig[i] += spike_amp * exp(-d1*d1 / 50.0);
        sig[i] += spike_amp * exp(-d2*d2 / 50.0);
    }
    return sig;
}

// Mixing: interpolate between stationary and piecewise
// alpha=0: fully stationary (same freq everywhere)
// alpha=1: fully piecewise (different freq in each half)
vector<double> make_mixed_stationarity(int N, double f_base, double f_alt,
                                        double A, double alpha) {
    vector<double> sig(N);
    int half = N / 2;
    // First half: f_base always
    for (int i = 0; i < half; ++i)
        sig[i] = A * sin(2.0 * M_PI * f_base * i);
    // Second half: blend f_base and f_alt
    double f_mix = f_base * (1.0 - alpha) + f_alt * alpha;
    for (int i = half; i < N; ++i)
        sig[i] = A * sin(2.0 * M_PI * f_mix * i);
    return sig;
}

// ============================================================
// EXPERIMENT H: Piecewise-stationary (two segments)
// The core test of the non-stationarity hypothesis
// ============================================================
void experiment_H(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT H: Piecewise-Stationary (Two Segments)" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f1 = 0.03;   // segment 1 frequency
    double f2 = 0.12;   // segment 2 frequency (4x higher)
    double A = 1.0;

    auto signal = make_piecewise_two_freq(N, f1, f2, A, A);

    vector<size_t> buf_sizes = {64, 128, 256, 512};
    vector<int> overloads = {3, 5, 8, 10, 15};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_H_piecewise.csv");
    csv << "buf_size,overload,trial,method,"
        << "snr_db,spec_corr_global,"
        << "spec_corr_seg1,spec_corr_seg2,"
        << "n_seg1,n_seg2,frac_seg1,"
        << "max_gap,cv_gap,n_surviving" << endl;

    int seg_boundary = N / 2;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(signal, buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    double sc1 = segment_spectral_corr(signal, r->reconstructed,
                                                        0, seg_boundary);
                    double sc2 = segment_spectral_corr(signal, r->reconstructed,
                                                        seg_boundary, N);
                    int n1 = count_in_segment(r->surviving_indices, 0, seg_boundary);
                    int n2 = count_in_segment(r->surviving_indices, seg_boundary, N);
                    double frac1 = (double)n1 / (n1 + n2);

                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << sc1 << "," << sc2 << ","
                        << n1 << "," << n2 << ","
                        << frac1 << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.cv << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_H_piecewise.csv" << endl;
}

// ============================================================
// EXPERIMENT I: Piecewise + Transients
// Transients create sample-stealing pressure on RDP
// ============================================================
void experiment_I(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT I: Piecewise + Transients" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f1 = 0.03, f2 = 0.12, A = 1.0;

    // Sweep spike amplitudes to control transient severity
    vector<double> spike_amps = {0.0, 1.0, 3.0, 5.0, 10.0};
    size_t buf = 256;
    vector<int> overloads = {5, 8, 10, 15};
    int num_trials = 5;

    int seg_boundary = N / 2;

    ofstream csv(out_dir + "/exp_I_piecewise_transient.csv");
    csv << "spike_amp,overload,trial,method,"
        << "snr_db,spec_corr_global,"
        << "spec_corr_seg1,spec_corr_seg2,"
        << "n_seg1,n_seg2,frac_seg1,"
        << "max_gap,n_surviving" << endl;

    for (double sa : spike_amps) {
        auto signal = make_piecewise_transient(N, f1, f2, A, sa);
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(signal, buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    double sc1 = segment_spectral_corr(signal, r->reconstructed,
                                                        0, seg_boundary);
                    double sc2 = segment_spectral_corr(signal, r->reconstructed,
                                                        seg_boundary, N);
                    int n1 = count_in_segment(r->surviving_indices, 0, seg_boundary);
                    int n2 = count_in_segment(r->surviving_indices, seg_boundary, N);
                    double frac1 = (double)n1 / (n1 + n2);

                    csv << sa << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << sc1 << "," << sc2 << ","
                        << n1 << "," << n2 << ","
                        << frac1 << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  spike=" << sa << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_I_piecewise_transient.csv" << endl;
}

// ============================================================
// EXPERIMENT J: Chirp (continuous non-stationarity)
// ============================================================
void experiment_J(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT J: Chirp (Continuous Non-Stationarity)" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    // Chirp from low to high frequency
    double f_start = 0.02, f_end = 0.20;

    auto signal = make_chirp(N, f_start, f_end, 1.0);

    vector<size_t> buf_sizes = {64, 128, 256, 512};
    vector<int> overloads = {3, 5, 8, 10, 15};
    int num_trials = 5;

    // 4 segments for per-segment analysis
    int seg_len = N / 4;

    ofstream csv(out_dir + "/exp_J_chirp.csv");
    csv << "buf_size,overload,trial,method,"
        << "snr_db,spec_corr_global,"
        << "sc_q1,sc_q2,sc_q3,sc_q4,"
        << "n_q1,n_q2,n_q3,n_q4,"
        << "max_gap,n_surviving" << endl;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(signal, buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    vector<double> sc(4);
                    vector<int> nq(4);
                    for (int q = 0; q < 4; ++q) {
                        int qs = q * seg_len;
                        int qe = (q == 3) ? N : (q + 1) * seg_len;
                        sc[q] = segment_spectral_corr(signal, r->reconstructed, qs, qe);
                        nq[q] = count_in_segment(r->surviving_indices, qs, qe);
                    }

                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << sc[0] << "," << sc[1] << ","
                        << sc[2] << "," << sc[3] << ","
                        << nq[0] << "," << nq[1] << ","
                        << nq[2] << "," << nq[3] << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_J_chirp.csv" << endl;
}

// ============================================================
// EXPERIMENT K: Stationarity Control
// Confirm that on stationary signals, RDP wins (no advantage)
// ============================================================
void experiment_K(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT K: Stationarity Control" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;

    struct TestSignal {
        string name;
        vector<double> data;
    };

    vector<TestSignal> signals = {
        {"stationary_f03",        make_stationary(N, 0.03, 1.0)},
        {"stationary_f12",        make_stationary(N, 0.12, 1.0)},
        {"stationary_transient",  make_stationary_transient(N, 0.08, 1.0, 5.0)},
        {"piecewise_f03_f12",     make_piecewise_two_freq(N, 0.03, 0.12, 1.0, 1.0)},
        {"piecewise_transient",   make_piecewise_transient(N, 0.03, 0.12, 1.0, 5.0)},
    };

    size_t buf = 256;
    vector<int> overloads = {3, 5, 8, 10, 15};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_K_stationarity_control.csv");
    csv << "signal,overload,trial,method,"
        << "snr_db,spec_corr_global,max_gap,n_surviving,"
        << "is_stationary" << endl;

    for (auto& ts : signals) {
        bool is_stat = ts.name.find("piecewise") == string::npos;
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(ts.data, buf, ovl);
                int target = max(2, (int)ts.data.size() -
                             (int)(ts.data.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(ts.data, target);

                for (auto* r : {&proposed, &rdp}) {
                    csv << ts.name << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.n_samples << ","
                        << (is_stat ? 1 : 0) << endl;
                }
            }
            cout << "  " << ts.name << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_K_stationarity_control.csv" << endl;
}

// ============================================================
// EXPERIMENT L: Multi-segment (4 segments, 4 frequencies)
// ============================================================
void experiment_L(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT L: Multi-Segment (4 Frequencies)" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    vector<double> freqs = {0.02, 0.06, 0.12, 0.25};
    double A = 1.0;

    auto signal = make_multi_segment(N, freqs, A);

    vector<size_t> buf_sizes = {128, 256, 512};
    vector<int> overloads = {3, 5, 8, 10, 15};
    int num_trials = 5;
    int seg_len = N / 4;

    ofstream csv(out_dir + "/exp_L_multi_segment.csv");
    csv << "buf_size,overload,trial,method,"
        << "snr_db,spec_corr_global,"
        << "sc_seg1,sc_seg2,sc_seg3,sc_seg4,"
        << "n_seg1,n_seg2,n_seg3,n_seg4,"
        << "max_gap,n_surviving" << endl;

    for (size_t buf : buf_sizes) {
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(signal, buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    vector<double> sc(4);
                    vector<int> ns(4);
                    for (int s = 0; s < 4; ++s) {
                        int ss = s * seg_len;
                        int se = (s == 3) ? N : (s + 1) * seg_len;
                        sc[s] = segment_spectral_corr(signal, r->reconstructed, ss, se);
                        ns[s] = count_in_segment(r->surviving_indices, ss, se);
                    }

                    csv << buf << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << sc[0] << "," << sc[1] << ","
                        << sc[2] << "," << sc[3] << ","
                        << ns[0] << "," << ns[1] << ","
                        << ns[2] << "," << ns[3] << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  buf=" << buf << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_L_multi_segment.csv" << endl;
}

// ============================================================
// EXPERIMENT M: Non-stationarity degree sweep
// Gradually increase non-stationarity from 0 (fully stationary)
// to 1 (fully piecewise). Prediction: spectral advantage of
// proposed method grows monotonically with non-stationarity.
// ============================================================
void experiment_M(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "EXPERIMENT M: Non-Stationarity Degree Sweep" << endl;
    cout << "============================================================\n" << endl;

    int N = 2000;
    double f_base = 0.03, f_alt = 0.12;
    double A = 1.0;

    vector<double> alphas = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    size_t buf = 256;
    vector<int> overloads = {5, 10, 15};
    int num_trials = 5;

    ofstream csv(out_dir + "/exp_M_nonstationarity_sweep.csv");
    csv << "alpha,overload,trial,method,"
        << "snr_db,spec_corr_global,max_gap,n_surviving" << endl;

    for (double alpha : alphas) {
        auto signal = make_mixed_stationarity(N, f_base, f_alt, A, alpha);
        for (int ovl : overloads) {
            for (int trial = 0; trial < num_trials; ++trial) {
                auto proposed = run_online(signal, buf, ovl);
                int target = max(2, (int)signal.size() -
                             (int)(signal.size() - proposed.surviving_indices.size()));
                auto rdp = run_rdp(signal, target);

                for (auto* r : {&proposed, &rdp}) {
                    csv << alpha << "," << ovl << "," << trial << ","
                        << r->method << ","
                        << fixed << setprecision(4)
                        << r->snr_db << ","
                        << r->spec_corr_global << ","
                        << r->gaps.max_gap << ","
                        << r->gaps.n_samples << endl;
                }
            }
            cout << "  alpha=" << alpha << " ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "  Output: exp_M_nonstationarity_sweep.csv" << endl;
}


// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  SPECTRAL THEORY v2 — Non-Stationarity Hypothesis" << endl;
    cout << "  DAC 2027" << endl;
    cout << "============================================================\n" << endl;

    string out_dir = "../results/spectral_theory_v2";
    system(("mkdir -p " + out_dir).c_str());

    auto t0 = chrono::high_resolution_clock::now();

    experiment_H(out_dir);  // Piecewise-stationary (core test)
    experiment_I(out_dir);  // Piecewise + transients
    experiment_J(out_dir);  // Chirp (continuous non-stationarity)
    experiment_K(out_dir);  // Stationarity control
    experiment_L(out_dir);  // Multi-segment
    experiment_M(out_dir);  // Non-stationarity degree sweep

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    cout << "\n============================================================" << endl;
    cout << "  ALL v2 EXPERIMENTS COMPLETE" << endl;
    cout << "  Wall-clock: " << fixed << setprecision(1) << elapsed << " s" << endl;
    cout << "  Output directory: " << out_dir << endl;
    cout << "============================================================" << endl;

    return 0;
}
