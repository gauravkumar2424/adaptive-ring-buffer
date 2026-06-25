// ============================================================
// N2 Validation Benchmark — DATE 2027
// ============================================================
// Phase 1: Eviction latency scaling (V1 O(N) shift vs V2 O(1) unlink)
// Phase 2: Combination rule comparison (5 strategies)
// Phase 3: Full mode comparison using V2 architecture
// ============================================================

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
#include <map>
#include <sstream>

#include "ring_buffer_v2.h"

using namespace std;

// ============================================================
// Signal reconstruction and metrics (self-contained)
// ============================================================

vector<double> reconstruct_signal(const vector<int>& idx,
                                   const vector<double>& vals,
                                   int orig_len) {
    vector<double> recon(orig_len, 0.0);
    if (idx.empty()) return recon;
    if (idx.size() == 1) {
        fill(recon.begin(), recon.end(), vals[0]);
        return recon;
    }
    size_t n = idx.size();
    for (int i = 0; i < idx[0] && i < orig_len; ++i)
        recon[i] = vals[0];
    for (size_t k = 0; k < n - 1; ++k) {
        int i0 = idx[k], i1 = idx[k + 1];
        for (int i = i0; i <= i1 && i < orig_len; ++i) {
            if (i1 == i0) recon[i] = vals[k];
            else recon[i] = vals[k] + (double)(i - i0) / (i1 - i0) * (vals[k+1] - vals[k]);
        }
    }
    for (int i = idx[n-1]; i < orig_len; ++i)
        recon[i] = vals[n-1];
    return recon;
}

double compute_snr(const vector<double>& orig, const vector<double>& recon) {
    size_t n = min(orig.size(), recon.size());
    if (n == 0) return -1e10;
    double sig_pow = 0, noise_pow = 0;
    for (size_t i = 0; i < n; ++i) {
        sig_pow += orig[i] * orig[i];
        double e = orig[i] - recon[i];
        noise_pow += e * e;
    }
    if (noise_pow < 1e-15) return 100.0;
    if (sig_pow < 1e-15) return 0.0;
    return 10.0 * log10(sig_pow / noise_pow);
}

double compute_mse(const vector<double>& orig, const vector<double>& recon) {
    size_t n = min(orig.size(), recon.size());
    if (n == 0) return 1e10;
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = orig[i] - recon[i];
        sum += e * e;
    }
    return sum / n;
}

double compute_deriv_ratio(const vector<double>& orig, const vector<double>& recon) {
    auto avg_d = [](const vector<double>& d) -> double {
        if (d.size() < 2) return 0.0;
        double s = 0;
        for (size_t i = 1; i < d.size(); ++i) s += abs(d[i] - d[i-1]);
        return s / (d.size() - 1);
    };
    double od = avg_d(orig), rd = avg_d(recon);
    return (od > 1e-15) ? rd / od : 0.0;
}

double compute_max_error(const vector<double>& orig, const vector<double>& recon) {
    double mx = 0;
    for (size_t i = 0; i < min(orig.size(), recon.size()); ++i)
        mx = max(mx, abs(orig[i] - recon[i]));
    return mx;
}

// ============================================================
// Signal generators
// ============================================================

vector<double> gen_sine(int n, double freq, double noise, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> nd(0, noise);
    vector<double> s(n);
    for (int i = 0; i < n; ++i)
        s[i] = sin(2.0 * M_PI * freq * i / 1000.0) + nd(gen);
    return s;
}

vector<double> gen_chirp(int n, double f0, double f1, double noise, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> nd(0, noise);
    double T = n / 1000.0;
    vector<double> s(n);
    for (int i = 0; i < n; ++i) {
        double t = i / 1000.0;
        s[i] = sin(2.0 * M_PI * (f0 + (f1 - f0) * t / T) * t) + nd(gen);
    }
    return s;
}

vector<double> gen_spikes(int n, int interval, double noise, unsigned seed) {
    mt19937 gen(seed);
    normal_distribution<double> nd(0, noise);
    vector<double> s(n);
    for (int i = 0; i < n; ++i) {
        s[i] = nd(gen);
        if (interval > 0 && i % interval == 0) {
            s[i] += 2.0;
            if (i+1 < n) s[i+1] += 1.5;
            if (i+2 < n) s[i+2] += 0.5;
        }
    }
    return s;
}

vector<double> gen_ecg_synthetic(int n, unsigned seed) {
    // Synthetic ECG-like signal: periodic QRS complexes
    mt19937 gen(seed);
    normal_distribution<double> nd(0, 0.02);
    vector<double> s(n);
    int rr_interval = 280;  // ~360Hz * 0.8s
    for (int i = 0; i < n; ++i) {
        s[i] = nd(gen);
        int phase = i % rr_interval;
        // P wave
        if (phase >= 50 && phase <= 80) {
            double t = (phase - 65.0) / 10.0;
            s[i] += 0.15 * exp(-t * t);
        }
        // QRS complex
        if (phase >= 90 && phase <= 110) {
            double t = (phase - 100.0) / 5.0;
            s[i] += 1.2 * exp(-t * t);
            if (phase == 95) s[i] -= 0.3;
        }
        // T wave
        if (phase >= 130 && phase <= 180) {
            double t = (phase - 155.0) / 15.0;
            s[i] += 0.3 * exp(-t * t);
        }
    }
    return s;
}

// ============================================================
// V1 Ring Buffer — O(N) shift eviction (for latency comparison)
// Minimal implementation, just enough for benchmarking.
// ============================================================

class RingBufferV1 {
public:
    RingBufferV1(size_t cap) : cap_(cap), head_(0), tail_(0), size_(0),
        drop_count_(0), total_ev_ns_(0), ev_count_(0), max_ev_ns_(0) {
        buf_.resize(cap);
        idx_.resize(cap);
    }

    void push(double value, int orig_idx) {
        if (size_ == cap_) performEviction();
        buf_[tail_] = value;
        idx_[tail_] = orig_idx;
        tail_ = (tail_ + 1) % cap_;
        ++size_;
    }

    struct Sample { double value; int index; };

    Sample pop() {
        Sample s{buf_[head_], idx_[head_]};
        head_ = (head_ + 1) % cap_;
        --size_;
        return s;
    }

    size_t getDropCount() const { return drop_count_; }
    uint64_t getTotalEvNs() const { return total_ev_ns_; }
    uint64_t getEvCount() const { return ev_count_; }
    uint64_t getMaxEvNs() const { return max_ev_ns_; }
    size_t getSize() const { return size_; }

private:
    size_t cap_, head_, tail_, size_;
    vector<double> buf_;
    vector<int> idx_;
    size_t drop_count_;
    uint64_t total_ev_ns_, ev_count_, max_ev_ns_;

    // Composite importance scan + O(N) shift removal
    void performEviction() {
        if (size_ == 0) return;
        auto t0 = chrono::high_resolution_clock::now();

        // Same composite importance scan as V2
        double alpha = 0.1, beta = 0.3, gamma = 0.6;
        size_t wh = 3;
        double min_score = numeric_limits<double>::max();
        size_t min_idx = 1;

        size_t start = 1, end = (size_ > 1) ? size_ - 1 : size_;
        for (size_t i = start; i < end; ++i) {
            // d1
            double d1 = abs(buf_[(head_+i)%cap_] - buf_[(head_+i-1)%cap_]);
            // d2
            double d2 = 0;
            if (i >= 2) {
                double d1c = abs(buf_[(head_+i)%cap_] - buf_[(head_+i-1)%cap_]);
                double d1p = abs(buf_[(head_+i-1)%cap_] - buf_[(head_+i-2)%cap_]);
                d2 = abs(d1c - d1p);
            }
            // Ew
            double ew = 0; int cnt = 0;
            size_t ws = (i > wh) ? i - wh : 0;
            size_t we = (i + wh < size_) ? i + wh : size_ - 1;
            for (size_t j = ws + 1; j <= we; ++j) {
                double diff = buf_[(head_+j)%cap_] - buf_[(head_+j-1)%cap_];
                ew += diff * diff; ++cnt;
            }
            if (cnt > 0) ew /= cnt;

            double score = alpha * d1 + beta * d2 + gamma * ew;
            if (score < min_score) { min_score = score; min_idx = i; }
        }

        // O(N) SHIFT — this is the bottleneck we're eliminating
        for (size_t i = min_idx; i < size_ - 1; ++i) {
            size_t src = (head_ + i + 1) % cap_;
            size_t dst = (head_ + i) % cap_;
            buf_[dst] = buf_[src];
            idx_[dst] = idx_[src];
        }
        tail_ = (head_ + size_ - 1) % cap_;
        --size_;
        ++drop_count_;

        auto t1 = chrono::high_resolution_clock::now();
        uint64_t ns = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
        total_ev_ns_ += ns;
        ++ev_count_;
        if (ns > max_ev_ns_) max_ev_ns_ = ns;
    }
};

// ============================================================
// Producer/Consumer threads for V2
// ============================================================

void producer_v2(RingBufferV2<double>& buf, const vector<double>& sig, int delay_us) {
    for (size_t i = 0; i < sig.size(); ++i) {
        buf.push(sig[i], (int)i);
        this_thread::sleep_for(chrono::microseconds(delay_us));
    }
}

void consumer_v2(RingBufferV2<double>& buf, vector<double>& vals,
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
// PHASE 1: Eviction Latency Scaling
// Proves O(1) removal in V2 vs O(N) shift in V1
// ============================================================

void phase1_eviction_latency(const string& results_dir) {
    cout << "=====================================================" << endl;
    cout << "PHASE 1: Eviction Latency — V1 O(N) vs V2 O(1)" << endl;
    cout << "=====================================================" << endl;

    ofstream csv(results_dir + "/n2_eviction_latency.csv");
    csv << "buffer_size,version,num_evictions,avg_eviction_ns,max_eviction_ns,"
        << "avg_scan_ns,avg_removal_ns,total_ms" << endl;

    vector<size_t> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_evictions = 2000;
    mt19937 rng(42);
    normal_distribution<double> noise(0, 0.1);

    for (size_t buf_size : sizes) {
        cout << "  Buffer size: " << buf_size << " ... " << flush;

        // Generate signal large enough to cause many evictions
        int sig_len = buf_size + num_evictions;
        vector<double> signal(sig_len);
        for (int i = 0; i < sig_len; ++i)
            signal[i] = sin(2.0 * M_PI * i / 200.0) + noise(rng);

        // --- V1: O(N) shift ---
        {
            RingBufferV1 v1(buf_size);
            // Fill buffer
            for (size_t i = 0; i < buf_size; ++i)
                v1.push(signal[i], (int)i);
            // Force evictions
            for (int i = 0; i < num_evictions; ++i)
                v1.push(signal[buf_size + i], (int)(buf_size + i));

            double avg_ns = (v1.getEvCount() > 0) ?
                (double)v1.getTotalEvNs() / v1.getEvCount() : 0;
            double total_ms = v1.getTotalEvNs() / 1e6;

            csv << buf_size << ",V1_shift," << v1.getEvCount() << ","
                << fixed << setprecision(1) << avg_ns << ","
                << v1.getMaxEvNs() << ","
                << avg_ns << ",0," // V1 doesn't split scan/removal
                << setprecision(3) << total_ms << endl;

            cout << "V1=" << (int)avg_ns << "ns ";
        }

        // --- V2: O(1) unlink ---
        {
            ImportanceConfigV2 cfg;
            cfg.combination = CombinationRule::WEIGHTED_LINEAR;
            RingBufferV2<double> v2(buf_size, BufferModeV2::IMPORTANCE_COMPOSITE,
                                    chrono::milliseconds(2), cfg);
            // Fill buffer (no threading needed for pure push benchmark)
            // We use push which triggers eviction when full
            // But push uses mutex, so for clean timing we push sequentially
            for (int i = 0; i < sig_len; ++i)
                v2.push(signal[i], i);

            auto prof = v2.getProfile();
            double avg_ns = prof.count > 0 ? (double)prof.total_ns / prof.count : 0;
            double avg_scan = prof.count > 0 ? (double)prof.scan_ns / prof.count : 0;
            double avg_removal = prof.count > 0 ? (double)prof.removal_ns / prof.count : 0;
            double total_ms = prof.total_ns / 1e6;

            csv << buf_size << ",V2_unlink," << prof.count << ","
                << fixed << setprecision(1) << avg_ns << ","
                << prof.max_ns << ","
                << setprecision(1) << avg_scan << ","
                << setprecision(1) << avg_removal << ","
                << setprecision(3) << total_ms << endl;

            cout << "V2=" << (int)avg_ns << "ns "
                 << "(scan=" << (int)avg_scan << " removal=" << (int)avg_removal << ")"
                 << endl;
        }
    }

    csv.close();
    cout << "\nPhase 1 results: " << results_dir << "/n2_eviction_latency.csv" << endl;
}

// ============================================================
// PHASE 2: Combination Rule Comparison
// Tests 5 strategies to justify weighted linear choice
// ============================================================

struct Phase2Result {
    string signal_name, rule_name;
    double snr, mse, deriv_ratio, max_error;
    size_t drops;
    double avg_eviction_us;
};

Phase2Result run_combination_test(const vector<double>& signal,
                                   const string& sig_name,
                                   CombinationRule rule,
                                   const string& rule_name,
                                   size_t buf_size, int overload, int prod_delay) {
    ImportanceConfigV2 cfg;
    cfg.combination = rule;
    cfg.alpha = 0.1; cfg.beta = 0.3; cfg.gamma = 0.6;

    int cons_delay = prod_delay * overload;
    RingBufferV2<double> buffer(buf_size, BufferModeV2::IMPORTANCE_COMPOSITE,
                                 chrono::milliseconds(2), cfg);

    vector<double> sv; vector<int> si;
    sv.reserve(signal.size()); si.reserve(signal.size());

    thread p(producer_v2, ref(buffer), cref(signal), prod_delay);
    thread c(consumer_v2, ref(buffer), ref(sv), ref(si), cons_delay, (int)signal.size());
    p.join(); buffer.finish(); c.join();

    auto recon = reconstruct_signal(si, sv, (int)signal.size());

    Phase2Result r;
    r.signal_name = sig_name;
    r.rule_name = rule_name;
    r.snr = compute_snr(signal, recon);
    r.mse = compute_mse(signal, recon);
    r.deriv_ratio = compute_deriv_ratio(signal, recon);
    r.max_error = compute_max_error(signal, recon);
    r.drops = buffer.getDropCount();
    r.avg_eviction_us = buffer.getProfile().avg_us();
    return r;
}

void phase2_combination_rules(const string& results_dir) {
    cout << "\n=====================================================" << endl;
    cout << "PHASE 2: Combination Rule Comparison" << endl;
    cout << "=====================================================" << endl;

    ofstream csv(results_dir + "/n2_combination_rules.csv");
    csv << "signal,rule,trial,snr_db,mse,deriv_ratio,max_error,drops,avg_eviction_us" << endl;

    struct RuleEntry { CombinationRule rule; string name; };
    vector<RuleEntry> rules = {
        {CombinationRule::WEIGHTED_LINEAR,    "WEIGHTED_LINEAR"},
        {CombinationRule::WEIGHTED_GEOMETRIC, "WEIGHTED_GEOMETRIC"},
        {CombinationRule::RANK_FUSION,        "RANK_FUSION"},
        {CombinationRule::HARMONIC_WEIGHTED,  "HARMONIC_WEIGHTED"},
        {CombinationRule::WORST_CASE_MIN,     "WORST_CASE_MIN"},
    };

    int sample_count = 2000;
    size_t buf_size = 256;
    int overload = 5;
    int prod_delay = 100;  // microseconds
    int num_trials = 5;

    struct SignalEntry { string name; vector<double> data; };
    vector<SignalEntry> signals = {
        {"sine_1hz",      gen_sine(sample_count, 1.0, 0.05, 42)},
        {"chirp_0.5_5hz", gen_chirp(sample_count, 0.5, 5.0, 0.05, 42)},
        {"spikes_200",    gen_spikes(sample_count, 200, 0.05, 42)},
        {"ecg_synthetic", gen_ecg_synthetic(sample_count, 42)},
    };

    // Per-rule aggregation
    map<string, vector<double>> rule_snrs;

    for (auto& sig : signals) {
        cout << "  Signal: " << sig.name << endl;
        for (auto& rule : rules) {
            double total_snr = 0;
            for (int t = 0; t < num_trials; ++t) {
                auto r = run_combination_test(sig.data, sig.name, rule.rule,
                                              rule.name, buf_size, overload, prod_delay);
                csv << r.signal_name << "," << r.rule_name << "," << t << ","
                    << fixed << setprecision(4) << r.snr << ","
                    << scientific << setprecision(6) << r.mse << ","
                    << fixed << setprecision(4) << r.deriv_ratio << ","
                    << scientific << r.max_error << ","
                    << r.drops << ","
                    << fixed << setprecision(3) << r.avg_eviction_us << endl;
                total_snr += r.snr;
                rule_snrs[rule.name].push_back(r.snr);
            }
            double avg = total_snr / num_trials;
            cout << "    " << left << setw(22) << rule.name
                 << right << fixed << setprecision(2) << avg << " dB" << endl;
        }
    }

    csv.close();

    // Summary
    cout << "\n  --- COMBINATION RULE RANKING (avg SNR across all signals) ---" << endl;
    vector<pair<string, double>> ranking;
    for (auto& [name, snrs] : rule_snrs) {
        double avg = accumulate(snrs.begin(), snrs.end(), 0.0) / snrs.size();
        ranking.push_back({name, avg});
    }
    sort(ranking.begin(), ranking.end(),
         [](auto& a, auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < ranking.size(); ++i) {
        string marker = (i == 0) ? " <<< BEST" : "";
        cout << "    " << (i+1) << ". " << left << setw(22) << ranking[i].first
             << right << fixed << setprecision(2) << ranking[i].second
             << " dB" << marker << endl;
    }

    cout << "\nPhase 2 results: " << results_dir << "/n2_combination_rules.csv" << endl;
}

// ============================================================
// PHASE 3: Full Mode Comparison (V2 architecture)
// ============================================================

struct Phase3Result {
    string signal_name, mode_name;
    size_t buf_size;
    int overload, trial;
    double snr, mse, deriv_ratio, max_error, max_wait_ms;
    size_t drops;
    double avg_eviction_us, max_eviction_us;
};

Phase3Result run_mode_test(const vector<double>& signal,
                            const string& sig_name,
                            BufferModeV2 mode, const string& mode_name,
                            size_t buf_size, int overload, int trial,
                            int prod_delay) {
    int cons_delay = prod_delay * overload;
    ImportanceConfigV2 cfg;
    cfg.combination = CombinationRule::WEIGHTED_LINEAR;

    RingBufferV2<double> buffer(buf_size, mode, chrono::milliseconds(2), cfg);

    vector<double> sv; vector<int> si;
    sv.reserve(signal.size()); si.reserve(signal.size());

    thread p(producer_v2, ref(buffer), cref(signal), prod_delay);
    thread c(consumer_v2, ref(buffer), ref(sv), ref(si), cons_delay, (int)signal.size());
    p.join(); buffer.finish(); c.join();

    auto recon = reconstruct_signal(si, sv, (int)signal.size());

    Phase3Result r;
    r.signal_name = sig_name;
    r.mode_name = mode_name;
    r.buf_size = buf_size;
    r.overload = overload;
    r.trial = trial;
    r.snr = compute_snr(signal, recon);
    r.mse = compute_mse(signal, recon);
    r.deriv_ratio = compute_deriv_ratio(signal, recon);
    r.max_error = compute_max_error(signal, recon);
    r.max_wait_ms = buffer.getMaxProducerWaitMs();
    r.drops = buffer.getDropCount();
    r.avg_eviction_us = buffer.getProfile().avg_us();
    r.max_eviction_us = buffer.getProfile().max_us();
    return r;
}

void phase3_full_comparison(const string& results_dir, bool quick) {
    cout << "\n=====================================================" << endl;
    cout << "PHASE 3: Full Mode Comparison (V2 Architecture)" << endl;
    cout << "=====================================================" << endl;

    ofstream csv(results_dir + "/n2_mode_comparison.csv");
    csv << "signal,mode,buffer_size,overload,trial,"
        << "snr_db,mse,deriv_ratio,max_error,max_wait_ms,drops,"
        << "avg_eviction_us,max_eviction_us" << endl;

    struct ModeEntry { string name; BufferModeV2 mode; };
    vector<ModeEntry> modes = {
        {"DROP",              BufferModeV2::DROP},
        {"RANDOM_DROP",       BufferModeV2::RANDOM_DROP},
        {"DROP_MIDDLE",       BufferModeV2::DROP_MIDDLE},
        {"DROP_LOW_VARIANCE", BufferModeV2::DROP_LOW_VARIANCE},
        {"ADAPTIVE",          BufferModeV2::ADAPTIVE_TIMED_WAIT},
        {"LEGACY_IMPORTANCE", BufferModeV2::ADAPTIVE_IMPORTANCE},
        {"IMP_FIRST_ORDER",   BufferModeV2::IMPORTANCE_FIRST_ORDER},
        {"IMP_SECOND_ORDER",  BufferModeV2::IMPORTANCE_SECOND_ORDER},
        {"IMP_WINDOWED_E",    BufferModeV2::IMPORTANCE_WINDOWED_ENERGY},
        {"IMP_COMPOSITE",     BufferModeV2::IMPORTANCE_COMPOSITE},
        {"IMP_ADAPTIVE",      BufferModeV2::IMPORTANCE_ADAPTIVE},
    };

    int sample_count = 2000;
    int prod_delay = 100;
    vector<size_t> buf_sizes = quick ? vector<size_t>{128, 256} : vector<size_t>{64, 128, 256, 512};
    vector<int> overloads = quick ? vector<int>{3, 5} : vector<int>{2, 3, 4, 5, 6};
    int num_trials = quick ? 2 : 5;

    struct SignalEntry { string name; vector<double> data; };
    vector<SignalEntry> signals = {
        {"sine_1hz",      gen_sine(sample_count, 1.0, 0.05, 42)},
        {"chirp_0.5_5hz", gen_chirp(sample_count, 0.5, 5.0, 0.05, 42)},
        {"spikes_200",    gen_spikes(sample_count, 200, 0.05, 42)},
        {"ecg_synthetic", gen_ecg_synthetic(sample_count, 42)},
    };

    int total = signals.size() * buf_sizes.size() * overloads.size() * modes.size() * num_trials;
    int done = 0;
    auto t0 = chrono::high_resolution_clock::now();

    cout << "  Running " << total << " experiments..." << endl;

    // Accumulate for summary
    map<string, vector<double>> mode_snrs_heavy;  // 5x+ overload

    for (auto& sig : signals) {
        for (size_t bs : buf_sizes) {
            for (int ol : overloads) {
                for (int trial = 0; trial < num_trials; ++trial) {
                    for (auto& m : modes) {
                        auto r = run_mode_test(sig.data, sig.name, m.mode, m.name,
                                               bs, ol, trial, prod_delay);

                        csv << r.signal_name << "," << r.mode_name << ","
                            << r.buf_size << "," << r.overload << "," << r.trial << ","
                            << fixed << setprecision(4) << r.snr << ","
                            << scientific << setprecision(6) << r.mse << ","
                            << fixed << setprecision(4) << r.deriv_ratio << ","
                            << scientific << r.max_error << ","
                            << fixed << setprecision(4) << r.max_wait_ms << ","
                            << r.drops << ","
                            << setprecision(3) << r.avg_eviction_us << ","
                            << r.max_eviction_us << endl;

                        if (ol >= 5) mode_snrs_heavy[m.name].push_back(r.snr);

                        ++done;
                        if (done % 200 == 0) {
                            auto now = chrono::high_resolution_clock::now();
                            double elapsed = chrono::duration<double>(now - t0).count();
                            double remain = (total - done) * elapsed / done;
                            cout << "    " << done << "/" << total
                                 << " (" << (int)remain << "s remaining)" << endl;
                        }
                    }
                }
            }
        }
    }

    csv.close();

    auto t1 = chrono::high_resolution_clock::now();
    double total_sec = chrono::duration<double>(t1 - t0).count();

    // Summary table: heavy overload (5x+)
    cout << "\n  --- MODE RANKING (avg SNR under heavy overload 5x+) ---" << endl;
    cout << "  " << left << setw(22) << "Mode" << right
         << setw(10) << "SNR(dB)" << setw(10) << "N" << endl;
    cout << "  " << string(42, '-') << endl;

    vector<pair<string, double>> ranking;
    for (auto& [name, snrs] : mode_snrs_heavy) {
        double avg = accumulate(snrs.begin(), snrs.end(), 0.0) / snrs.size();
        ranking.push_back({name, avg});
    }
    sort(ranking.begin(), ranking.end(),
         [](auto& a, auto& b) { return a.second > b.second; });

    for (auto& [name, avg] : ranking) {
        string tag = "";
        if (name.find("IMP_COMPOSITE") != string::npos) tag = " <<< PROPOSED";
        else if (name.find("IMP_") != string::npos) tag = " (proposed)";
        cout << "  " << left << setw(22) << name
             << right << fixed << setprecision(2) << setw(10) << avg
             << setw(10) << mode_snrs_heavy[name].size()
             << tag << endl;
    }

    cout << "\n  Total experiments: " << done << endl;
    cout << "  Total time: " << fixed << setprecision(1) << total_sec << "s ("
         << setprecision(1) << total_sec / 60.0 << " min)" << endl;
    cout << "  Phase 3 results: " << results_dir << "/n2_mode_comparison.csv" << endl;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    cout << "============================================================" << endl;
    cout << "  N2 VALIDATION BENCHMARK — Adaptive Ring Buffer DATE 2027" << endl;
    cout << "  RingBufferV2: O(1) eviction + 5 combination strategies" << endl;
    cout << "============================================================" << endl;

    string results_dir = "../results";
    bool quick = false;

    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--quick") quick = true;
        if (string(argv[i]) == "--results" && i + 1 < argc)
            results_dir = argv[++i];
    }

    if (quick) cout << "\n[QUICK MODE — reduced parameter sweep]\n" << endl;

    // Ensure results directory exists
    system(("mkdir -p " + results_dir).c_str());

    phase1_eviction_latency(results_dir);
    phase2_combination_rules(results_dir);
    phase3_full_comparison(results_dir, quick);

    cout << "\n============================================================" << endl;
    cout << "  ALL PHASES COMPLETE" << endl;
    cout << "  CSV outputs in: " << results_dir << "/" << endl;
    cout << "    n2_eviction_latency.csv   — V1 vs V2 latency scaling" << endl;
    cout << "    n2_combination_rules.csv  — 5 combination strategies" << endl;
    cout << "    n2_mode_comparison.csv    — Full 11-mode comparison" << endl;
    cout << "============================================================" << endl;

    return 0;
}
