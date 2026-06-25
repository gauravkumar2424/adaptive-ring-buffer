// ============================================================
// lookahead_eval.cpp — Bounded Lookahead Eviction Evaluation
// DATE 2027: Adaptive Ring Buffer
//
// PURPOSE: Evaluate the new IMPORTANCE_INTERP_LOOKAHEAD mode
// against INTERP_ERROR and RDP_OFFLINE to determine whether
// the one-step cascade penalty closes the greedy-vs-optimal gap.
//
// Compares: INTERP_ERROR, INTERP_LOOKAHEAD, INTERP_SPECTRAL,
//           RDP_OFFLINE, LTTB_OFFLINE, DROP
//
// Signals: 5 MIT-BIH ECG + 3 CWRU vibration (all 8)
// Buffer sizes: {64, 128, 256, 512}
// Overload ratios: {2, 3, 5, 8, 10}
// Trials: 5
// Also: α sensitivity sweep {0.0, 0.25, 0.5, 0.75, 1.0}
//
// Outputs:
//   results/lookahead_eval_results.csv     — main comparison
//   results/lookahead_alpha_sweep.csv      — α sensitivity
//   results/lookahead_summary.csv          — per-domain aggregates
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/lookahead_eval lookahead_eval.cpp
// Run:   cd ../build && ./lookahead_eval
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
#include <tuple>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"

using namespace std;

// ============================================================
// Offline baselines (same proven implementations)
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
// DFT spectral correlation (vibration domain only)
// ============================================================
double compute_spectral_correlation(const vector<double>& original,
                                     const vector<double>& reconstructed,
                                     int fft_N = 512) {
    int n_orig = min((int)original.size(), fft_N);
    int n_recon = min((int)reconstructed.size(), fft_N);
    int n = min(n_orig, n_recon);
    int half = n / 2;
    if (half < 4) return -1.0;

    // Compute magnitude spectra
    vector<double> mag_orig(half), mag_recon(half);
    for (int k = 0; k < half; ++k) {
        double re_o = 0, im_o = 0, re_r = 0, im_r = 0;
        for (int i = 0; i < n; ++i) {
            double angle = 2.0 * M_PI * k * i / n;
            double c = cos(angle), s = sin(angle);
            re_o += original[i] * c;
            im_o -= original[i] * s;
            re_r += reconstructed[i] * c;
            im_r -= reconstructed[i] * s;
        }
        mag_orig[k] = sqrt(re_o * re_o + im_o * im_o);
        mag_recon[k] = sqrt(re_r * re_r + im_r * im_r);
    }

    // Pearson correlation of magnitude spectra
    double mo = accumulate(mag_orig.begin(), mag_orig.end(), 0.0) / half;
    double mr = accumulate(mag_recon.begin(), mag_recon.end(), 0.0) / half;
    double num = 0, do2 = 0, dr2 = 0;
    for (int k = 0; k < half; ++k) {
        double a = mag_orig[k] - mo;
        double b = mag_recon[k] - mr;
        num += a * b;
        do2 += a * a;
        dr2 += b * b;
    }
    double den = sqrt(do2 * dr2);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================
// Gap statistics
// ============================================================
struct GapStats {
    double mean_gap, cv, max_gap;
    int n_surviving;
};

GapStats compute_gap_stats(const vector<int>& indices) {
    GapStats g{0, 0, 0, (int)indices.size()};
    if (indices.size() < 2) return g;

    vector<int> gaps;
    for (size_t i = 1; i < indices.size(); ++i)
        gaps.push_back(indices[i] - indices[i-1]);

    g.max_gap = *max_element(gaps.begin(), gaps.end());
    g.mean_gap = accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
    double var = 0;
    for (int x : gaps) var += (x - g.mean_gap) * (x - g.mean_gap);
    double sd = sqrt(var / gaps.size());
    g.cv = (g.mean_gap > 1e-9) ? sd / g.mean_gap : 0.0;
    return g;
}

// ============================================================
// Producer / Consumer
// ============================================================
void producer_thread(RingBuffer<double>& buffer,
                     const vector<double>& signal, int delay_us) {
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

struct RunResult {
    string method;
    string signal_name;
    string domain;
    int buf_size;
    int overload;
    int trial;
    double snr_db;
    double spec_corr;
    int n_surviving;
    int n_dropped;
    double mean_gap;
    double cv_gap;
    double max_gap;
    bool snr_saturated;
};

RunResult run_online_method(const vector<double>& signal,
                            const string& sig_name, const string& domain,
                            BufferMode mode, const string& method_name,
                            size_t buf_size, int overload, int trial,
                            ImportanceConfig cfg = ImportanceConfig()) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), cfg);

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

    auto reconstructed = reconstruct_signal(surv_idx, surv_vals, (int)signal.size());
    double snr = compute_snr(signal, reconstructed);
    double sc = (domain == "vibration") ?
        compute_spectral_correlation(signal, reconstructed) : -1.0;
    auto gs = compute_gap_stats(surv_idx);

    RunResult r;
    r.method = method_name;
    r.signal_name = sig_name;
    r.domain = domain;
    r.buf_size = (int)buf_size;
    r.overload = overload;
    r.trial = trial;
    r.snr_db = snr;
    r.spec_corr = sc;
    r.n_surviving = gs.n_surviving;
    r.n_dropped = (int)signal.size() - gs.n_surviving;
    r.mean_gap = gs.mean_gap;
    r.cv_gap = gs.cv;
    r.max_gap = gs.max_gap;
    r.snr_saturated = !isfinite(snr);
    return r;
}

RunResult run_offline_method(const vector<double>& signal,
                             const string& sig_name, const string& domain,
                             const string& method, int target_surviving,
                             int buf_size, int overload, int trial) {
    vector<int> selected;
    if (method == "RDP_OFFLINE")
        selected = offline_rdp(signal, target_surviving);
    else
        selected = offline_lttb(signal, target_surviving);

    vector<double> surv_vals;
    for (int idx : selected) surv_vals.push_back(signal[idx]);

    auto reconstructed = reconstruct_signal(selected, surv_vals, (int)signal.size());
    double snr = compute_snr(signal, reconstructed);
    double sc = (domain == "vibration") ?
        compute_spectral_correlation(signal, reconstructed) : -1.0;
    auto gs = compute_gap_stats(selected);

    RunResult r;
    r.method = method;
    r.signal_name = sig_name;
    r.domain = domain;
    r.buf_size = buf_size;
    r.overload = overload;
    r.trial = trial;
    r.snr_db = snr;
    r.spec_corr = sc;
    r.n_surviving = gs.n_surviving;
    r.n_dropped = (int)signal.size() - gs.n_surviving;
    r.mean_gap = gs.mean_gap;
    r.cv_gap = gs.cv;
    r.max_gap = gs.max_gap;
    r.snr_saturated = !isfinite(snr);
    return r;
}

void write_result(ofstream& csv, const RunResult& r) {
    csv << r.signal_name << "," << r.domain << "," << r.method << ","
        << r.buf_size << "," << r.overload << "," << r.trial << ","
        << r.n_surviving << "," << r.n_dropped << ",";
    if (r.snr_saturated)
        csv << "inf";
    else
        csv << fixed << setprecision(4) << r.snr_db;
    csv << "," << r.snr_saturated << ","
        << fixed << setprecision(6) << r.spec_corr << ","
        << setprecision(4) << r.mean_gap << ","
        << r.cv_gap << "," << r.max_gap << endl;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  LOOKAHEAD EVICTION EVALUATION" << endl;
    cout << "  Comparing INTERP_LOOKAHEAD vs INTERP_ERROR vs RDP" << endl;
    cout << "============================================================\n" << endl;

    auto wall_start = chrono::high_resolution_clock::now();

    string data_dir = "../data";
    string out_dir = "../results";

    auto all_signals = load_all_real_signals(data_dir, 2000);
    if (all_signals.empty()) {
        cerr << "ERROR: no signals loaded." << endl;
        return 1;
    }

    // Configuration
    vector<size_t> buf_sizes = {64, 128, 256, 512};
    vector<int> overloads = {2, 3, 5, 8, 10};
    int num_trials = 5;

    int total_configs = (int)(all_signals.size() * buf_sizes.size() *
                              overloads.size() * num_trials);
    cout << "Signals: " << all_signals.size() << endl;
    cout << "Configs per method: " << total_configs << endl;
    cout << "Methods: INTERP_ERROR, INTERP_LOOKAHEAD, INTERP_SPECTRAL, "
         << "RDP_OFFLINE, LTTB_OFFLINE, DROP" << endl;
    cout << "Total experiments: ~" << total_configs * 6 << endl;
    cout << endl;

    // Main results CSV
    ofstream main_csv(out_dir + "/lookahead_eval_results.csv");
    main_csv << "signal,domain,method,buf_size,overload,trial,"
             << "n_surviving,n_dropped,snr_db,snr_saturated,"
             << "spec_corr,mean_gap,cv_gap,max_gap" << endl;

    int done = 0;
    int total_est = total_configs * 6;

    for (auto& sig : all_signals) {
        for (size_t bs : buf_sizes) {
            for (int ol : overloads) {
                for (int t = 0; t < num_trials; ++t) {

                    // --- Online methods ---

                    // 1. INTERP_ERROR (existing baseline)
                    auto r_ie = run_online_method(sig.data, sig.name, sig.domain,
                        BufferMode::IMPORTANCE_INTERP_ERROR, "IMP_INTERP_ERROR",
                        bs, ol, t);
                    write_result(main_csv, r_ie);
                    ++done;

                    // 2. INTERP_LOOKAHEAD (NEW — α=0.5 default)
                    ImportanceConfig la_cfg;
                    la_cfg.lookahead_alpha = 0.5;
                    auto r_la = run_online_method(sig.data, sig.name, sig.domain,
                        BufferMode::IMPORTANCE_INTERP_LOOKAHEAD, "IMP_INTERP_LOOKAHEAD",
                        bs, ol, t, la_cfg);
                    write_result(main_csv, r_la);
                    ++done;

                    // 3. INTERP_SPECTRAL (existing)
                    auto r_sp = run_online_method(sig.data, sig.name, sig.domain,
                        BufferMode::IMPORTANCE_INTERP_SPECTRAL, "IMP_INTERP_SPECTRAL",
                        bs, ol, t);
                    write_result(main_csv, r_sp);
                    ++done;

                    // 4. DROP (sanity baseline)
                    auto r_drop = run_online_method(sig.data, sig.name, sig.domain,
                        BufferMode::DROP, "DROP",
                        bs, ol, t);
                    write_result(main_csv, r_drop);
                    ++done;

                    // Use INTERP_ERROR drop count to match offline methods
                    int target = max(2, (int)sig.data.size() - r_ie.n_dropped);

                    // 5. RDP_OFFLINE
                    auto r_rdp = run_offline_method(sig.data, sig.name, sig.domain,
                        "RDP_OFFLINE", target, (int)bs, ol, t);
                    write_result(main_csv, r_rdp);
                    ++done;

                    // 6. LTTB_OFFLINE
                    auto r_lttb = run_offline_method(sig.data, sig.name, sig.domain,
                        "LTTB_OFFLINE", target, (int)bs, ol, t);
                    write_result(main_csv, r_lttb);
                    ++done;

                    // Progress
                    if (done % 120 == 0 || done == total_est) {
                        cout << "\r  Progress: " << done << " / ~" << total_est
                             << " (" << (100*done/max(1,total_est)) << "%)" << flush;
                    }
                }
            }
        }
    }
    cout << "\n\n  Main evaluation complete." << endl;
    main_csv.close();

    // ============================================================
    // Alpha sensitivity sweep (smaller config set)
    // ============================================================
    cout << "\n  Running alpha sensitivity sweep..." << endl;

    ofstream alpha_csv(out_dir + "/lookahead_alpha_sweep.csv");
    alpha_csv << "signal,domain,alpha,buf_size,overload,trial,"
              << "snr_db,snr_saturated,spec_corr,n_dropped" << endl;

    vector<double> alphas = {0.0, 0.25, 0.5, 0.75, 1.0};
    // Reduced config for sweep: 2 buffer sizes, 3 overloads, 3 trials
    vector<size_t> sweep_bufs = {128, 256};
    vector<int> sweep_ols = {3, 5, 10};
    int sweep_trials = 3;

    int sweep_done = 0;
    int sweep_total = (int)(all_signals.size() * alphas.size() * sweep_bufs.size() *
                            sweep_ols.size() * sweep_trials);

    for (auto& sig : all_signals) {
        for (double alpha : alphas) {
            for (size_t bs : sweep_bufs) {
                for (int ol : sweep_ols) {
                    for (int t = 0; t < sweep_trials; ++t) {
                        ImportanceConfig cfg;
                        cfg.lookahead_alpha = alpha;

                        auto r = run_online_method(sig.data, sig.name, sig.domain,
                            BufferMode::IMPORTANCE_INTERP_LOOKAHEAD,
                            "LOOKAHEAD_a" + to_string(alpha).substr(0,4),
                            bs, ol, t, cfg);

                        alpha_csv << sig.name << "," << sig.domain << ","
                                  << fixed << setprecision(2) << alpha << ","
                                  << bs << "," << ol << "," << t << ",";
                        if (r.snr_saturated)
                            alpha_csv << "inf";
                        else
                            alpha_csv << fixed << setprecision(4) << r.snr_db;
                        alpha_csv << "," << r.snr_saturated << ","
                                  << fixed << setprecision(6) << r.spec_corr << ","
                                  << r.n_dropped << endl;

                        ++sweep_done;
                        if (sweep_done % 60 == 0) {
                            cout << "\r  Alpha sweep: " << sweep_done << " / "
                                 << sweep_total << " (" << (100*sweep_done/max(1,sweep_total))
                                 << "%)" << flush;
                        }
                    }
                }
            }
        }
    }
    cout << "\n  Alpha sweep complete." << endl;
    alpha_csv.close();

    // ============================================================
    // Aggregate summary
    // ============================================================
    cout << "\n  Computing summary statistics..." << endl;

    // Re-read main results for aggregation
    ifstream read_csv(out_dir + "/lookahead_eval_results.csv");
    string header_line;
    getline(read_csv, header_line);

    struct Accum {
        vector<double> snrs;
        vector<double> spec_corrs;
        vector<double> cv_gaps;
    };
    // key: (domain, method)
    map<pair<string,string>, Accum> agg;

    string line;
    while (getline(read_csv, line)) {
        if (line.empty()) continue;
        // Parse: signal,domain,method,buf_size,overload,trial,
        //        n_surviving,n_dropped,snr_db,snr_saturated,
        //        spec_corr,mean_gap,cv_gap,max_gap
        stringstream ss(line);
        string sig_name, domain, method, bs_s, ol_s, t_s;
        string nsurv_s, ndrop_s, snr_s, sat_s, sc_s, mg_s, cv_s, xg_s;
        getline(ss, sig_name, ',');
        getline(ss, domain, ',');
        getline(ss, method, ',');
        getline(ss, bs_s, ',');
        getline(ss, ol_s, ',');
        getline(ss, t_s, ',');
        getline(ss, nsurv_s, ',');
        getline(ss, ndrop_s, ',');
        getline(ss, snr_s, ',');
        getline(ss, sat_s, ',');
        getline(ss, sc_s, ',');
        getline(ss, mg_s, ',');
        getline(ss, cv_s, ',');
        getline(ss, xg_s, ',');

        bool saturated = (sat_s == "1");
        if (saturated) continue;

        double snr = stod(snr_s);
        double sc = stod(sc_s);
        double cv = stod(cv_s);

        auto& a = agg[{domain, method}];
        a.snrs.push_back(snr);
        if (sc >= 0) a.spec_corrs.push_back(sc);
        a.cv_gaps.push_back(cv);

        // Also aggregate as "all"
        auto& a2 = agg[{"all", method}];
        a2.snrs.push_back(snr);
        if (sc >= 0) a2.spec_corrs.push_back(sc);
        a2.cv_gaps.push_back(cv);
    }
    read_csv.close();

    auto mean_v = [](const vector<double>& v) -> double {
        return v.empty() ? 0.0 : accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto std_v = [](const vector<double>& v, double m) -> double {
        if (v.size() < 2) return 0.0;
        double s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return sqrt(s / v.size());
    };

    // Write summary
    ofstream sum_csv(out_dir + "/lookahead_summary.csv");
    sum_csv << "domain,method,n,mean_snr,std_snr,mean_spec_corr,std_spec_corr,mean_cv_gap" << endl;

    // Print to console too
    cout << "\n============================================================" << endl;
    cout << "  RESULTS SUMMARY" << endl;
    cout << "============================================================\n" << endl;

    vector<string> domains_to_show = {"ecg", "vibration", "all"};
    vector<string> methods_order = {"RDP_OFFLINE", "IMP_INTERP_LOOKAHEAD",
                                     "IMP_INTERP_ERROR", "IMP_INTERP_SPECTRAL",
                                     "LTTB_OFFLINE", "DROP"};

    for (auto& dom : domains_to_show) {
        cout << "  --- " << dom << " ---" << endl;
        cout << "  " << left << setw(24) << "Method"
             << right << setw(6) << "n"
             << setw(10) << "SNR(dB)"
             << setw(10) << "Std"
             << setw(12) << "Spec.Corr"
             << setw(10) << "CV(gap)" << endl;
        cout << "  " << string(72, '-') << endl;

        for (auto& meth : methods_order) {
            auto key = make_pair(dom, meth);
            if (!agg.count(key)) continue;
            auto& a = agg[key];
            double ms = mean_v(a.snrs);
            double ss = std_v(a.snrs, ms);
            double msc = mean_v(a.spec_corrs);
            double ssc = std_v(a.spec_corrs, msc);
            double mcv = mean_v(a.cv_gaps);

            // Highlight if lookahead
            string tag = (meth == "IMP_INTERP_LOOKAHEAD") ? " <<<" : "";

            cout << "  " << left << setw(24) << meth
                 << right << setw(6) << a.snrs.size()
                 << fixed << setprecision(2) << setw(10) << ms
                 << setprecision(2) << setw(10) << ss
                 << setprecision(4) << setw(12) << msc
                 << setprecision(4) << setw(10) << mcv
                 << tag << endl;

            sum_csv << dom << "," << meth << ","
                    << a.snrs.size() << ","
                    << fixed << setprecision(4)
                    << ms << "," << ss << ","
                    << msc << "," << ssc << "," << mcv << endl;
        }
        cout << endl;

        // Print gap-to-RDP for key methods
        auto rdp_key = make_pair(dom, string("RDP_OFFLINE"));
        if (agg.count(rdp_key)) {
            double rdp_snr = mean_v(agg[rdp_key].snrs);
            cout << "  Gap to RDP:" << endl;
            for (auto& meth : {"IMP_INTERP_LOOKAHEAD", "IMP_INTERP_ERROR", "IMP_INTERP_SPECTRAL"}) {
                auto key = make_pair(dom, string(meth));
                if (!agg.count(key)) continue;
                double ms = mean_v(agg[key].snrs);
                double gap = ms - rdp_snr;
                cout << "    " << left << setw(24) << meth
                     << right << fixed << setprecision(3)
                     << setw(10) << gap << " dB" << endl;
            }
            cout << endl;
        }
    }

    sum_csv.close();

    auto wall_end = chrono::high_resolution_clock::now();
    double wall_s = chrono::duration<double>(wall_end - wall_start).count();

    cout << "============================================================" << endl;
    cout << "  Wall-clock: " << fixed << setprecision(1) << wall_s << " s ("
         << setprecision(1) << wall_s / 60.0 << " min)" << endl;
    cout << "  Output files:" << endl;
    cout << "    " << out_dir << "/lookahead_eval_results.csv" << endl;
    cout << "    " << out_dir << "/lookahead_alpha_sweep.csv" << endl;
    cout << "    " << out_dir << "/lookahead_summary.csv" << endl;
    cout << "============================================================" << endl;

    return 0;
}
