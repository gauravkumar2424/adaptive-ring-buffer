// ============================================================
// novelty_eval.cpp — External Baselines + Spectral Mode Test
// DATE 2027: Adaptive Ring Buffer
//
// Implements and evaluates:
//   1. Swing-Door Trending (SDT) — industrial standard (PI/OSIsoft)
//   2. Piecewise Linear Approximation (PLA) — classic online compression
//   3. Lightweight Temporal Compression (LTC) — IoT-oriented
//   4. IMPORTANCE_INTERP_SPECTRAL — new frequency-aware eviction
//
// All external baselines are implemented as offline-matched to the same
// drop count as the proposed method, giving a fair comparison. They are
// also implemented in their natural online form for an honest test.
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/novelty_eval novelty_eval.cpp
// Run:   cd ../build && ./novelty_eval
// ============================================================

#include <set>
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
#include <deque>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "rpeak_eval.h"

using namespace std;

// ============================================================
// EXTERNAL BASELINE 1: Swing-Door Trending (SDT)
//
// Classic industrial compression (OSIsoft PI system).
// Maintains a "door" (tolerance band) around each stored point.
// A new point is stored only when the current point falls outside
// the tolerance band projected from the last stored point.
//
// Online, causal, O(1) per sample.
// ============================================================

vector<int> online_sdt(const vector<double>& signal, double tolerance) {
    int n = (int)signal.size();
    if (n < 2) return {0};

    vector<int> selected;
    selected.push_back(0);

    int last_stored = 0;
    double upper_slope = numeric_limits<double>::max();
    double lower_slope = -numeric_limits<double>::max();

    for (int i = 1; i < n; ++i) {
        double dt = (double)(i - last_stored);
        if (dt < 1e-15) continue;

        double new_upper = (signal[i] - signal[last_stored] + tolerance) / dt;
        double new_lower = (signal[i] - signal[last_stored] - tolerance) / dt;

        if (new_upper < lower_slope || new_lower > upper_slope) {
            // Door broken — store previous point
            selected.push_back(i - 1);
            last_stored = i - 1;
            // Reset doors from newly stored point
            dt = (double)(i - last_stored);
            if (dt > 0) {
                upper_slope = (signal[i] - signal[last_stored] + tolerance) / dt;
                lower_slope = (signal[i] - signal[last_stored] - tolerance) / dt;
            }
        } else {
            upper_slope = min(upper_slope, new_upper);
            lower_slope = max(lower_slope, new_lower);
        }
    }
    selected.push_back(n - 1);

    // Deduplicate
    sort(selected.begin(), selected.end());
    selected.erase(unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

// Adaptive SDT: binary search for tolerance that gives target surviving count
vector<int> sdt_matched(const vector<double>& signal, int target_surviving) {
    if (target_surviving >= (int)signal.size()) {
        vector<int> all(signal.size());
        iota(all.begin(), all.end(), 0);
        return all;
    }

    double lo = 0.0, hi = 1.0;
    // Find upper bound
    while ((int)online_sdt(signal, hi).size() > target_surviving) hi *= 2.0;

    // Binary search
    for (int iter = 0; iter < 50; ++iter) {
        double mid = (lo + hi) / 2.0;
        int count = (int)online_sdt(signal, mid).size();
        if (count > target_surviving) lo = mid;
        else hi = mid;
    }
    return online_sdt(signal, hi);
}

// ============================================================
// EXTERNAL BASELINE 2: Piecewise Linear Approximation (PLA)
//
// Online segmentation: extend current linear segment as long as
// max perpendicular error stays within tolerance. When it exceeds,
// start a new segment. Classic signal compression (Keogh et al.).
//
// Online, causal, O(1) amortized per sample.
// ============================================================

vector<int> online_pla(const vector<double>& signal, double tolerance) {
    int n = (int)signal.size();
    if (n < 2) return {0};

    vector<int> selected;
    selected.push_back(0);

    int seg_start = 0;
    for (int i = 2; i < n; ++i) {
        // Check if point i can be added to current segment [seg_start, i]
        double max_err = 0.0;
        double span = (double)(i - seg_start);
        for (int j = seg_start + 1; j < i; ++j) {
            double t = (double)(j - seg_start) / span;
            double interp = signal[seg_start] + t * (signal[i] - signal[seg_start]);
            max_err = max(max_err, abs(signal[j] - interp));
        }
        if (max_err > tolerance) {
            // Store the point before the break
            selected.push_back(i - 1);
            seg_start = i - 1;
        }
    }
    selected.push_back(n - 1);

    sort(selected.begin(), selected.end());
    selected.erase(unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

// Adaptive PLA: binary search for tolerance matching target count
vector<int> pla_matched(const vector<double>& signal, int target_surviving) {
    if (target_surviving >= (int)signal.size()) {
        vector<int> all(signal.size());
        iota(all.begin(), all.end(), 0);
        return all;
    }

    double lo = 0.0, hi = 1.0;
    while ((int)online_pla(signal, hi).size() > target_surviving) hi *= 2.0;

    for (int iter = 0; iter < 50; ++iter) {
        double mid = (lo + hi) / 2.0;
        int count = (int)online_pla(signal, mid).size();
        if (count > target_surviving) lo = mid;
        else hi = mid;
    }
    return online_pla(signal, hi);
}

// ============================================================
// EXTERNAL BASELINE 3: Lightweight Temporal Compression (LTC)
//
// Schoellhammer et al. (SenSys 2004). Designed for sensor networks.
// Maintains upper/lower bound lines from last transmitted point.
// Transmits only when current value exits the feasibility cone.
// Similar to SDT but with different cone update rule.
//
// Online, causal, O(1) per sample.
// ============================================================

vector<int> online_ltc(const vector<double>& signal, double tolerance) {
    int n = (int)signal.size();
    if (n < 2) return {0};

    vector<int> selected;
    selected.push_back(0);

    int last = 0;
    double hi_slope =  numeric_limits<double>::max();
    double lo_slope = -numeric_limits<double>::max();

    for (int i = 1; i < n; ++i) {
        double dt = (double)(i - last);
        if (dt < 1e-15) continue;

        double up = (signal[i] + tolerance - signal[last]) / dt;
        double dn = (signal[i] - tolerance - signal[last]) / dt;

        if (up < lo_slope || dn > hi_slope) {
            // Cone broken — record previous point, restart
            selected.push_back(i - 1);
            last = i - 1;
            dt = (double)(i - last);
            if (dt > 0) {
                hi_slope = (signal[i] + tolerance - signal[last]) / dt;
                lo_slope = (signal[i] - tolerance - signal[last]) / dt;
            } else {
                hi_slope =  numeric_limits<double>::max();
                lo_slope = -numeric_limits<double>::max();
            }
        } else {
            hi_slope = min(hi_slope, up);
            lo_slope = max(lo_slope, dn);
        }
    }
    selected.push_back(n - 1);

    sort(selected.begin(), selected.end());
    selected.erase(unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

vector<int> ltc_matched(const vector<double>& signal, int target_surviving) {
    if (target_surviving >= (int)signal.size()) {
        vector<int> all(signal.size());
        iota(all.begin(), all.end(), 0);
        return all;
    }

    double lo = 0.0, hi = 1.0;
    while ((int)online_ltc(signal, hi).size() > target_surviving) hi *= 2.0;

    for (int iter = 0; iter < 50; ++iter) {
        double mid = (lo + hi) / 2.0;
        int count = (int)online_ltc(signal, mid).size();
        if (count > target_surviving) lo = mid;
        else hi = mid;
    }
    return online_ltc(signal, hi);
}

// ============================================================
// Offline RDP and LTTB (from cross_domain.cpp)
// ============================================================

vector<int> offline_rdp(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) { vector<int> all(n); iota(all.begin(), all.end(), 0); return all; }
    if (target_points < 2) return {0, n - 1};
    vector<bool> kept(n, true);
    vector<int> prev_kept(n), next_kept(n);
    for (int i = 0; i < n; ++i) { prev_kept[i] = i - 1; next_kept[i] = i + 1; }
    next_kept[n - 1] = -1;
    auto compute_error = [&](int i) -> double {
        int p = prev_kept[i], s = next_kept[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p); if (span <= 0) return 0.0;
        return abs(signal[i] - (signal[p] + (double)(i-p)/span*(signal[s]-signal[p])));
    };
    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({compute_error(i), i});
    int cc = n;
    while (cc > target_points && !pq.empty()) {
        auto [err, idx] = pq.top(); pq.pop();
        if (!kept[idx]) continue;
        double actual = compute_error(idx);
        if (abs(actual - err) > 1e-12) { pq.push({actual, idx}); continue; }
        kept[idx] = false;
        int p = prev_kept[idx], s = next_kept[idx];
        if (p >= 0) next_kept[p] = s; if (s >= 0 && s < n) prev_kept[s] = p;
        --cc;
        if (p > 0 && kept[p]) pq.push({compute_error(p), p});
        if (s > 0 && s < n-1 && kept[s]) pq.push({compute_error(s), s});
    }
    vector<int> result; for (int i = 0; i < n; ++i) if (kept[i]) result.push_back(i);
    return result;
}

vector<int> offline_lttb(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) { vector<int> all(n); iota(all.begin(), all.end(), 0); return all; }
    if (target_points < 2) return {0, n - 1};
    vector<int> result; result.reserve(target_points); result.push_back(0);
    double bs = (double)(n-2)/(target_points-2); int ps = 0;
    for (int b = 0; b < target_points-2; ++b) {
        int bst = (int)(b*bs)+1, ben = min((int)((b+1)*bs)+1, n-1);
        int nbs = (int)((b+1)*bs)+1, nbe = min((int)((b+2)*bs)+1, n);
        double ax=0,ay=0; int nc=0;
        for (int i=nbs;i<nbe;++i){ax+=i;ay+=signal[i];++nc;}
        if(nc>0){ax/=nc;ay/=nc;}
        double ma=-1; int bi2=bst;
        for(int i=bst;i<ben;++i){
            double a=abs((ps-ax)*(signal[i]-signal[ps])-(ps-i)*(ay-signal[ps]))*0.5;
            if(a>ma){ma=a;bi2=i;}
        }
        result.push_back(bi2); ps=bi2;
    }
    result.push_back(n-1); return result;
}

// ============================================================
// Spectral correlation
// ============================================================

double compute_spectral_correlation(const vector<double>& orig, const vector<double>& recon) {
    if (orig.size() < 64 || recon.size() < 64) return -1.0;
    int N = min((int)min(orig.size(), recon.size()), 512);
    vector<double> mo(N/2), mr(N/2);
    for (int k = 0; k < N/2; ++k) {
        double reo=0,imo=0,rer=0,imr=0;
        for (int n = 0; n < N; ++n) {
            double a = 2.0*M_PI*k*n/N;
            reo+=orig[n]*cos(a); imo-=orig[n]*sin(a);
            rer+=recon[n]*cos(a); imr-=recon[n]*sin(a);
        }
        mo[k]=sqrt(reo*reo+imo*imo); mr[k]=sqrt(rer*rer+imr*imr);
    }
    double mmo=accumulate(mo.begin(),mo.end(),0.0)/mo.size();
    double mmr=accumulate(mr.begin(),mr.end(),0.0)/mr.size();
    double num=0,do2=0,dr2=0;
    for(int k=0;k<N/2;++k){double a=mo[k]-mmo,b=mr[k]-mmr;num+=a*b;do2+=a*a;dr2+=b*b;}
    double den=sqrt(do2*dr2);
    return (den>1e-15)?num/den:0.0;
}

// ============================================================
// Producer / Consumer for online modes
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
// Unified result struct
// ============================================================

struct Result {
    string signal_name, domain, mode_name;
    size_t buffer_size;
    int overload_ratio, trial;
    double snr;
    bool snr_saturated;
    size_t drops;
    double rpeak_f1;
    double spectral_correlation;
};

Result eval_selection(const RealSignal& sig, const vector<int>& selected,
                      const string& mode_name, size_t buf_size, int overload, int trial) {
    vector<double> sv; vector<int> si;
    for (int idx : selected) { si.push_back(idx); sv.push_back(sig.data[idx]); }
    auto recon = reconstruct_signal(si, sv, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name; r.domain = sig.domain; r.mode_name = mode_name;
    r.buffer_size = buf_size; r.overload_ratio = overload; r.trial = trial;
    r.snr = compute_snr(sig.data, recon);
    r.snr_saturated = !isfinite(r.snr);
    r.drops = sig.data.size() - selected.size();

    r.rpeak_f1 = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto det = detect_rpeaks(recon, 0.5, 100);
        auto ev = evaluate_rpeaks(sig.rpeak_indices, det, 15);
        r.rpeak_f1 = ev.f1_score;
    }

    r.spectral_correlation = -1;
    if (sig.domain == "vibration")
        r.spectral_correlation = compute_spectral_correlation(sig.data, recon);

    return r;
}

Result run_online_mode(const RealSignal& sig, BufferMode mode, const string& name,
                       size_t buf_size, int overload, int trial) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload;
    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());

    vector<double> sv; vector<int> si;
    sv.reserve(sig.data.size()); si.reserve(sig.data.size());

    thread prod(producer_thread, ref(buffer), cref(sig.data), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(sv), ref(si), cons_delay, (int)sig.data.size());
    prod.join(); buffer.finish(); cons.join();

    auto recon = reconstruct_signal(si, sv, (int)sig.data.size());

    Result r;
    r.signal_name = sig.name; r.domain = sig.domain; r.mode_name = name;
    r.buffer_size = buf_size; r.overload_ratio = overload; r.trial = trial;
    r.snr = compute_snr(sig.data, recon);
    r.snr_saturated = !isfinite(r.snr);
    r.drops = buffer.getDropCount();

    r.rpeak_f1 = -1;
    if (sig.domain == "ecg" && sig.has_rpeaks) {
        auto det = detect_rpeaks(recon, 0.5, 100);
        auto ev = evaluate_rpeaks(sig.rpeak_indices, det, 15);
        r.rpeak_f1 = ev.f1_score;
    }

    r.spectral_correlation = -1;
    if (sig.domain == "vibration")
        r.spectral_correlation = compute_spectral_correlation(sig.data, recon);

    return r;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    cout << "============================================================" << endl;
    cout << "  NOVELTY EVALUATION" << endl;
    cout << "  External Baselines (SDT, PLA, LTC) + Spectral Mode" << endl;
    cout << "============================================================\n" << endl;

    string data_dir = "../data";
    bool quick = (argc > 1 && string(argv[1]) == "--quick");

    auto signals = load_all_real_signals(data_dir, 2000);
    cout << "Loaded " << signals.size() << " signals\n" << endl;
    if (signals.empty()) { cerr << "ERROR: No signals." << endl; return 1; }

    vector<size_t> buf_sizes = quick ? vector<size_t>{128, 256} : vector<size_t>{64, 128, 256, 512};
    vector<int> overloads = quick ? vector<int>{3, 5, 10} : vector<int>{2, 3, 5, 8, 10, 15};
    int num_trials = quick ? 2 : 5;

    // Online modes to test
    struct OnlineMode { string name; BufferMode mode; };
    vector<OnlineMode> online_modes = {
        {"DROP",                 BufferMode::DROP},
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
        {"IMP_INTERP_SPECTRAL",  BufferMode::IMPORTANCE_INTERP_SPECTRAL},
    };

    // Offline/external methods (matched to proposed method's drop count)
    vector<string> offline_methods = {"RDP_OFFLINE", "LTTB_OFFLINE", "SDT_MATCHED", "PLA_MATCHED", "LTC_MATCHED"};

    int total_online = signals.size() * buf_sizes.size() * overloads.size() * num_trials * online_modes.size();
    int total_offline = signals.size() * buf_sizes.size() * overloads.size() * num_trials * offline_methods.size();
    cout << "Running " << total_online << " online + " << total_offline << " offline experiments..." << endl;

    ofstream csv("../results/novelty_eval_results.csv");
    csv << "signal,domain,mode,buffer_size,overload,trial,"
        << "snr_db,snr_saturated,drops,"
        << "rpeak_f1,spectral_correlation" << endl;

    auto write_row = [&](const Result& r) {
        csv << r.signal_name << "," << r.domain << ","
            << r.mode_name << "," << r.buffer_size << ","
            << r.overload_ratio << "," << r.trial << ","
            << fixed << setprecision(4) << r.snr << ","
            << r.snr_saturated << "," << r.drops << ","
            << r.rpeak_f1 << "," << r.spectral_correlation << endl;
    };

    int done = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& sig : signals) {
        for (size_t bs : buf_sizes) {
            for (int ol : overloads) {
                for (int trial = 0; trial < num_trials; ++trial) {
                    int proposed_drops = -1;

                    // Online modes
                    for (auto& m : online_modes) {
                        auto r = run_online_mode(sig, m.mode, m.name, bs, ol, trial);
                        write_row(r);
                        if (m.name == "IMP_INTERP_ERROR")
                            proposed_drops = (int)r.drops;
                        ++done;
                    }

                    // Offline/external baselines matched to proposed drop count
                    if (proposed_drops >= 0) {
                        int target = max(2, (int)sig.data.size() - proposed_drops);

                        for (auto& method : offline_methods) {
                            vector<int> selected;
                            if (method == "RDP_OFFLINE")
                                selected = offline_rdp(sig.data, target);
                            else if (method == "LTTB_OFFLINE")
                                selected = offline_lttb(sig.data, target);
                            else if (method == "SDT_MATCHED")
                                selected = sdt_matched(sig.data, target);
                            else if (method == "PLA_MATCHED")
                                selected = pla_matched(sig.data, target);
                            else if (method == "LTC_MATCHED")
                                selected = ltc_matched(sig.data, target);

                            auto r = eval_selection(sig, selected, method, bs, ol, trial);
                            write_row(r);
                            ++done;
                        }
                    }

                    if (done % 200 == 0) {
                        auto now = chrono::high_resolution_clock::now();
                        double elapsed = chrono::duration<double>(now - t0).count();
                        double remaining = (done > 0) ? (total_online + total_offline - done) * elapsed / done : 0;
                        cout << "  " << done << " done (" << (int)remaining << "s remaining)" << endl;
                    }
                }
            }
        }
    }
    csv.close();

    auto t1 = chrono::high_resolution_clock::now();
    double total_sec = chrono::duration<double>(t1 - t0).count();

    // ============================================================
    // Summary tables
    // ============================================================
    cout << "\n============================================================" << endl;
    cout << "  RESULTS SUMMARY" << endl;
    cout << "============================================================\n" << endl;

    // Read back and aggregate
    ifstream rf("../results/novelty_eval_results.csv");
    string line; getline(rf, line); // header

    struct Stats {
        vector<double> snrs, specs, f1s;
    };
    map<string, map<string, Stats>> domain_mode_stats; // domain -> mode -> stats

    while (getline(rf, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string sig_name, dom, mode;
        int bs, ol, trial;
        double snr;
        int sat;
        int drops;
        double f1, spec;

        getline(ss, sig_name, ','); getline(ss, dom, ','); getline(ss, mode, ',');
        ss >> bs; ss.ignore(); ss >> ol; ss.ignore(); ss >> trial; ss.ignore();
        ss >> snr; ss.ignore(); ss >> sat; ss.ignore(); ss >> drops; ss.ignore();
        ss >> f1; ss.ignore(); ss >> spec;

        if (!sat && isfinite(snr)) {
            domain_mode_stats[dom][mode].snrs.push_back(snr);
            domain_mode_stats["ALL"][mode].snrs.push_back(snr);
        }
        if (f1 >= 0) domain_mode_stats[dom][mode].f1s.push_back(f1);
        if (spec >= 0 && !sat) domain_mode_stats[dom][mode].specs.push_back(spec);
    }

    auto avg = [](const vector<double>& v) -> double {
        return v.empty() ? -999.0 : accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stddev = [&](const vector<double>& v) -> double {
        if (v.size() < 2) return 0.0;
        double m = avg(v), s = 0;
        for (double x : v) s += (x-m)*(x-m);
        return sqrt(s / (v.size()-1));
    };

    for (auto& domain_label : {"ALL", "ecg", "vibration"}) {
        string dl = domain_label;
        if (domain_mode_stats.find(dl) == domain_mode_stats.end()) continue;

        cout << "--- " << dl << " (SNR dB) ---" << endl;
        cout << left << setw(22) << "Mode"
             << right << setw(6) << "n"
             << setw(10) << "Mean"
             << setw(10) << "StdDev" << endl;
        cout << string(48, '-') << endl;

        vector<pair<string, Stats*>> sorted;
        for (auto& [mode, stats] : domain_mode_stats[dl])
            sorted.push_back({mode, &stats});
        sort(sorted.begin(), sorted.end(),
             [&](auto& a, auto& b) { return avg(a.second->snrs) > avg(b.second->snrs); });

        for (auto& [mode, stats] : sorted) {
            string tag = "";
            if (mode.find("INTERP") != string::npos) tag = " <<<";
            if (mode.find("SDT") != string::npos || mode.find("PLA") != string::npos ||
                mode.find("LTC") != string::npos) tag = " [EXT]";

            cout << left << setw(22) << mode
                 << right << setw(6) << stats->snrs.size()
                 << fixed << setprecision(2)
                 << setw(10) << avg(stats->snrs)
                 << setw(10) << stddev(stats->snrs)
                 << tag << endl;
        }
        cout << endl;

        // Spectral correlation for vibration
        if (dl == "vibration") {
            cout << "--- " << dl << " (Spectral Correlation) ---" << endl;
            cout << left << setw(22) << "Mode"
                 << right << setw(6) << "n"
                 << setw(10) << "Mean"
                 << setw(10) << "StdDev" << endl;
            cout << string(48, '-') << endl;

            vector<pair<string, Stats*>> ssorted;
            for (auto& [mode, stats] : domain_mode_stats[dl])
                if (!stats.specs.empty()) ssorted.push_back({mode, &stats});
            sort(ssorted.begin(), ssorted.end(),
                 [&](auto& a, auto& b) { return avg(a.second->specs) > avg(b.second->specs); });

            for (auto& [mode, stats] : ssorted) {
                string tag = "";
                if (mode.find("INTERP") != string::npos) tag = " <<<";
                if (mode.find("SDT") != string::npos || mode.find("PLA") != string::npos ||
                    mode.find("LTC") != string::npos) tag = " [EXT]";
                cout << left << setw(22) << mode
                     << right << setw(6) << stats->specs.size()
                     << fixed << setprecision(4)
                     << setw(10) << avg(stats->specs)
                     << setw(10) << stddev(stats->specs)
                     << tag << endl;
            }
            cout << endl;
        }

        // R-peak F1 for ECG
        if (dl == "ecg") {
            cout << "--- " << dl << " (R-peak F1) ---" << endl;
            cout << left << setw(22) << "Mode"
                 << right << setw(6) << "n"
                 << setw(10) << "Mean" << endl;
            cout << string(38, '-') << endl;

            vector<pair<string, Stats*>> fsorted;
            for (auto& [mode, stats] : domain_mode_stats[dl])
                if (!stats.f1s.empty()) fsorted.push_back({mode, &stats});
            sort(fsorted.begin(), fsorted.end(),
                 [&](auto& a, auto& b) { return avg(a.second->f1s) > avg(b.second->f1s); });

            for (auto& [mode, stats] : fsorted) {
                string tag = "";
                if (mode.find("INTERP") != string::npos) tag = " <<<";
                if (mode.find("SDT") != string::npos || mode.find("PLA") != string::npos ||
                    mode.find("LTC") != string::npos) tag = " [EXT]";
                cout << left << setw(22) << mode
                     << right << setw(6) << stats->f1s.size()
                     << fixed << setprecision(4)
                     << setw(10) << avg(stats->f1s)
                     << tag << endl;
            }
            cout << endl;
        }
    }

    cout << "Total experiments: " << done << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_sec << "s ("
         << setprecision(1) << total_sec/60 << " min)" << endl;
    cout << "Results: results/novelty_eval_results.csv" << endl;

    return 0;
}
