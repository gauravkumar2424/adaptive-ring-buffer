// ============================================================
// duration_eval.cpp — Signal Duration Generalization Test
// DATE 2027: Adaptive Ring Buffer
//
// PURPOSE: Confirm that comparative results (proposed vs RDP,
// LTTB, external baselines) hold at longer signal durations.
// Current evaluation uses 2000-sample segments (~5.5s at 360Hz).
// This tests at 2000, 5000, and 10000 samples (~5.5s, ~13.9s,
// ~27.8s) to validate generalization.
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/duration_eval duration_eval.cpp
// Run:   cd ../build && ./duration_eval
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

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"

using namespace std;

// ============================================================
// Offline baselines
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
// DFT spectral correlation (vibration only)
// ============================================================
double compute_spectral_correlation(const vector<double>& original,
                                     const vector<double>& reconstructed,
                                     int fft_N) {
    int n = min({(int)original.size(), (int)reconstructed.size(), fft_N});
    int half = n / 2;
    if (half < 4) return -1.0;

    vector<double> mag_o(half), mag_r(half);
    for (int k = 0; k < half; ++k) {
        double re_o = 0, im_o = 0, re_r = 0, im_r = 0;
        for (int i = 0; i < n; ++i) {
            double angle = 2.0 * M_PI * k * i / n;
            double c = cos(angle), s = sin(angle);
            re_o += original[i] * c;   im_o -= original[i] * s;
            re_r += reconstructed[i] * c; im_r -= reconstructed[i] * s;
        }
        mag_o[k] = sqrt(re_o*re_o + im_o*im_o);
        mag_r[k] = sqrt(re_r*re_r + im_r*im_r);
    }

    double mo = accumulate(mag_o.begin(), mag_o.end(), 0.0) / half;
    double mr = accumulate(mag_r.begin(), mag_r.end(), 0.0) / half;
    double num = 0, d1 = 0, d2 = 0;
    for (int k = 0; k < half; ++k) {
        double a = mag_o[k] - mo, b = mag_r[k] - mr;
        num += a * b; d1 += a * a; d2 += b * b;
    }
    double den = sqrt(d1 * d2);
    return (den > 1e-15) ? num / den : 0.0;
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

struct Result {
    string signal_name, domain, method;
    int max_samples, buf_size, overload, trial;
    double snr_db, spec_corr;
    int n_dropped;
    bool snr_saturated;
};

Result run_online(const vector<double>& signal, const string& sig_name,
                  const string& domain, BufferMode mode, const string& method,
                  size_t buf_size, int overload, int trial, int max_samples) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());
    vector<double> sv; vector<int> si;
    sv.reserve(signal.size()); si.reserve(signal.size());

    thread prod(producer_thread, ref(buffer), cref(signal), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(sv), ref(si),
                cons_delay, (int)signal.size());
    prod.join(); buffer.finish(); cons.join();

    auto recon = reconstruct_signal(si, sv, (int)signal.size());
    int fft_n = min(512, (int)signal.size());

    Result r;
    r.signal_name = sig_name; r.domain = domain; r.method = method;
    r.max_samples = max_samples; r.buf_size = (int)buf_size;
    r.overload = overload; r.trial = trial;
    r.snr_db = compute_snr(signal, recon);
    r.spec_corr = (domain == "vibration") ?
        compute_spectral_correlation(signal, recon, fft_n) : -1.0;
    r.n_dropped = (int)signal.size() - (int)si.size();
    r.snr_saturated = !isfinite(r.snr_db);
    return r;
}

Result run_offline(const vector<double>& signal, const string& sig_name,
                   const string& domain, const string& method,
                   int target, int buf_size, int overload, int trial, int max_samples) {
    vector<int> sel = (method == "RDP_OFFLINE") ?
        offline_rdp(signal, target) : offline_lttb(signal, target);
    vector<double> sv;
    for (int idx : sel) sv.push_back(signal[idx]);
    auto recon = reconstruct_signal(sel, sv, (int)signal.size());
    int fft_n = min(512, (int)signal.size());

    Result r;
    r.signal_name = sig_name; r.domain = domain; r.method = method;
    r.max_samples = max_samples; r.buf_size = buf_size;
    r.overload = overload; r.trial = trial;
    r.snr_db = compute_snr(signal, recon);
    r.spec_corr = (domain == "vibration") ?
        compute_spectral_correlation(signal, recon, fft_n) : -1.0;
    r.n_dropped = (int)signal.size() - (int)sel.size();
    r.snr_saturated = !isfinite(r.snr_db);
    return r;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  SIGNAL DURATION GENERALIZATION TEST" << endl;
    cout << "  Validating results at 2000, 5000, 10000 samples" << endl;
    cout << "============================================================\n" << endl;

    auto wall_start = chrono::high_resolution_clock::now();

    string data_dir = "../data";
    string out_dir = "../results";

    vector<int> durations = {2000, 5000, 10000};
    vector<size_t> buf_sizes = {128, 256, 512};
    vector<int> overloads = {3, 5, 10};
    int num_trials = 5;

    ofstream csv(out_dir + "/duration_eval_results.csv");
    csv << "signal,domain,method,max_samples,buf_size,overload,trial,"
        << "snr_db,snr_saturated,spec_corr,n_dropped" << endl;

    // Accumulate per (duration, domain, method)
    struct Accum { vector<double> snrs, specs; };
    map<tuple<int,string,string>, Accum> agg;

    int done = 0;

    for (int max_samp : durations) {
        cout << "\n=== Duration: " << max_samp << " samples ===" << endl;
        auto signals = load_all_real_signals(data_dir, max_samp);

        for (auto& sig : signals) {
            for (size_t bs : buf_sizes) {
                for (int ol : overloads) {
                    for (int t = 0; t < num_trials; ++t) {
                        // INTERP_ERROR
                        auto r_ie = run_online(sig.data, sig.name, sig.domain,
                            BufferMode::IMPORTANCE_INTERP_ERROR, "IMP_INTERP_ERROR",
                            bs, ol, t, max_samp);

                        // INTERP_SPECTRAL
                        auto r_sp = run_online(sig.data, sig.name, sig.domain,
                            BufferMode::IMPORTANCE_INTERP_SPECTRAL, "IMP_INTERP_SPECTRAL",
                            bs, ol, t, max_samp);

                        // DROP
                        auto r_dr = run_online(sig.data, sig.name, sig.domain,
                            BufferMode::DROP, "DROP",
                            bs, ol, t, max_samp);

                        int target = max(2, (int)sig.data.size() - r_ie.n_dropped);

                        // RDP
                        auto r_rdp = run_offline(sig.data, sig.name, sig.domain,
                            "RDP_OFFLINE", target, (int)bs, ol, t, max_samp);

                        // LTTB
                        auto r_lttb = run_offline(sig.data, sig.name, sig.domain,
                            "LTTB_OFFLINE", target, (int)bs, ol, t, max_samp);

                        for (auto* r : {&r_ie, &r_sp, &r_dr, &r_rdp, &r_lttb}) {
                            csv << r->signal_name << "," << r->domain << ","
                                << r->method << "," << r->max_samples << ","
                                << r->buf_size << "," << r->overload << ","
                                << r->trial << ",";
                            if (r->snr_saturated) csv << "inf";
                            else csv << fixed << setprecision(4) << r->snr_db;
                            csv << "," << r->snr_saturated << ","
                                << fixed << setprecision(6) << r->spec_corr << ","
                                << r->n_dropped << endl;

                            if (!r->snr_saturated) {
                                auto& a = agg[{max_samp, r->domain, r->method}];
                                a.snrs.push_back(r->snr_db);
                                if (r->spec_corr >= 0) a.specs.push_back(r->spec_corr);
                            }
                            ++done;
                        }

                        if (done % 200 == 0) {
                            cout << "\r  Progress: " << done << " experiments" << flush;
                        }
                    }
                }
            }
        }
    }
    csv.close();
    cout << "\n\n  Total experiments: " << done << endl;

    // ============================================================
    // Summary
    // ============================================================
    auto mean_v = [](const vector<double>& v) {
        return v.empty() ? 0.0 : accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    cout << "\n============================================================" << endl;
    cout << "  RESULTS: SNR by duration and domain" << endl;
    cout << "============================================================\n" << endl;

    vector<string> methods = {"RDP_OFFLINE", "IMP_INTERP_SPECTRAL",
                               "IMP_INTERP_ERROR", "LTTB_OFFLINE", "DROP"};

    for (string dom : {"ecg", "vibration"}) {
        cout << "  --- " << dom << " ---" << endl;
        cout << "  " << left << setw(24) << "Method";
        for (int d : durations) cout << right << setw(10) << (to_string(d) + "s");
        cout << setw(12) << "Stable?" << endl;
        cout << "  " << string(58, '-') << endl;

        for (auto& meth : methods) {
            cout << "  " << left << setw(24) << meth;
            double first_gap = 0;
            bool stable = true;
            double rdp_first = 0;

            for (size_t di = 0; di < durations.size(); ++di) {
                int d = durations[di];
                auto key = make_tuple(d, string(dom), meth);
                double ms = mean_v(agg[key].snrs);
                cout << right << fixed << setprecision(2) << setw(10) << ms;

                // Track gap stability for proposed methods
                auto rdp_key = make_tuple(d, string(dom), string("RDP_OFFLINE"));
                double rdp_ms = mean_v(agg[rdp_key].snrs);
                double gap = ms - rdp_ms;
                if (di == 0) { first_gap = gap; rdp_first = rdp_ms; }
                else if (meth != "RDP_OFFLINE" && abs(gap - first_gap) > 2.0)
                    stable = false;
            }
            if (meth == "RDP_OFFLINE")
                cout << setw(12) << "ref";
            else
                cout << setw(12) << (stable ? "YES" : "NO");
            cout << endl;
        }
        cout << endl;

        // Gap to RDP table
        cout << "  Gap to RDP (dB):" << endl;
        cout << "  " << left << setw(24) << "Method";
        for (int d : durations) cout << right << setw(10) << (to_string(d) + "s");
        cout << endl;
        cout << "  " << string(54, '-') << endl;

        for (auto& meth : {"IMP_INTERP_SPECTRAL", "IMP_INTERP_ERROR", "LTTB_OFFLINE", "DROP"}) {
            cout << "  " << left << setw(24) << meth;
            for (int d : durations) {
                auto key = make_tuple(d, string(dom), string(meth));
                auto rdp_key = make_tuple(d, string(dom), string("RDP_OFFLINE"));
                double ms = mean_v(agg[key].snrs);
                double rdp = mean_v(agg[rdp_key].snrs);
                cout << right << fixed << setprecision(2) << setw(10) << (ms - rdp);
            }
            cout << endl;
        }
        cout << endl;
    }

    // Spectral correlation across durations (vibration)
    cout << "  --- Spectral Correlation (vibration) ---" << endl;
    cout << "  " << left << setw(24) << "Method";
    for (int d : durations) cout << right << setw(10) << (to_string(d) + "s");
    cout << endl;
    cout << "  " << string(54, '-') << endl;

    for (auto& meth : methods) {
        cout << "  " << left << setw(24) << meth;
        for (int d : durations) {
            auto key = make_tuple(d, string("vibration"), meth);
            double ms = mean_v(agg[key].specs);
            cout << right << fixed << setprecision(4) << setw(10) << ms;
        }
        cout << endl;
    }
    cout << endl;

    auto wall_end = chrono::high_resolution_clock::now();
    double wall_s = chrono::duration<double>(wall_end - wall_start).count();

    cout << "============================================================" << endl;
    cout << "  Wall-clock: " << fixed << setprecision(1) << wall_s << " s ("
         << setprecision(1) << wall_s / 60.0 << " min)" << endl;
    cout << "  Output: " << out_dir << "/duration_eval_results.csv" << endl;
    cout << "============================================================" << endl;

    return 0;
}
