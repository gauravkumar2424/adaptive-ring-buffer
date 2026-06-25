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

using namespace std;

// ============================================================
// Parameter Sensitivity Analysis (Professor feedback, item 4)
//
// PURPOSE: every result so far used ImportanceConfig's DEFAULT
// values (alpha=0.1, beta=0.3, gamma=0.6, window_half=3,
// boundary_protect=1) without ever varying them. This program
// performs a one-factor-at-a-time sensitivity sweep: vary ONE
// parameter across a coarse range while holding the others at
// default, and check whether IMP_INTERP_COMPOSITE's advantage
// over LTTB_OFFLINE (and its gap to RDP_OFFLINE) holds across
// that range, or whether it was an artifact of the specific
// default values used everywhere else in this project.
//
// This is deliberately NOT a full factorial grid. One-factor-at-
// a-time is the standard, defensible approach for a coarse
// robustness check.
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

struct SweepResult {
    string param_name;
    double param_value;
    string signal_name;
    string domain;
    double snr_composite;
    bool composite_saturated;
    size_t composite_drops;
    double snr_lttb;
    bool lttb_saturated;
    double snr_rdp;
    bool rdp_saturated;
};

SweepResult run_one_config(const RealSignal& sig, const string& param_name, double param_value,
                           ImportanceConfig cfg, size_t buf_size, int overload_ratio) {
    int prod_delay = 100;
    int cons_delay = prod_delay * overload_ratio;

    RingBuffer<double> buffer(buf_size, BufferMode::IMPORTANCE_INTERP_COMPOSITE,
                               chrono::milliseconds(2), cfg);

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

    SweepResult r;
    r.param_name = param_name;
    r.param_value = param_value;
    r.signal_name = sig.name;
    r.domain = sig.domain;
    r.snr_composite = compute_snr(sig.data, reconstructed);
    r.composite_saturated = !std::isfinite(r.snr_composite);
    r.composite_drops = buffer.getDropCount();

    int target = (int)sig.data.size() - (int)r.composite_drops;
    target = max(2, target);

    auto lttb_sel = offline_lttb(sig.data, target);
    vector<double> lttb_vals; vector<int> lttb_idx;
    for (int idx : lttb_sel) { lttb_idx.push_back(idx); lttb_vals.push_back(sig.data[idx]); }
    auto lttb_recon = reconstruct_signal(lttb_idx, lttb_vals, (int)sig.data.size());
    r.snr_lttb = compute_snr(sig.data, lttb_recon);
    r.lttb_saturated = !std::isfinite(r.snr_lttb);

    auto rdp_sel = offline_rdp(sig.data, target);
    vector<double> rdp_vals; vector<int> rdp_idx;
    for (int idx : rdp_sel) { rdp_idx.push_back(idx); rdp_vals.push_back(sig.data[idx]); }
    auto rdp_recon = reconstruct_signal(rdp_idx, rdp_vals, (int)sig.data.size());
    r.snr_rdp = compute_snr(sig.data, rdp_recon);
    r.rdp_saturated = !std::isfinite(r.snr_rdp);

    return r;
}

int main(int argc, char* argv[]) {
    cout << "=== Parameter Sensitivity Sweep ===" << endl;
    cout << "One-factor-at-a-time: window_half, boundary_protect, gamma" << endl << endl;

    string data_dir = "../data";
    int max_samples = 2000;
    bool quick = (argc > 1 && string(argv[1]) == "--quick");

    auto signals = load_all_real_signals(data_dir, max_samples);
    cout << "Loaded " << signals.size() << " signals\n" << endl;

    if (signals.empty()) {
        cerr << "ERROR: No signals loaded." << endl;
        return 1;
    }

    size_t buf_size = 256;
    int overload_ratio = 5;

    vector<size_t> window_half_values   = quick ? vector<size_t>{2, 4} : vector<size_t>{1, 2, 3, 4, 5};
    vector<size_t> boundary_protect_vals= quick ? vector<size_t>{0, 2} : vector<size_t>{0, 1, 2, 3};
    vector<double> gamma_values         = quick ? vector<double>{0.2, 0.8} : vector<double>{0.2, 0.4, 0.6, 0.8};

    ofstream csv("../results/sensitivity_results.csv");
    csv << "param_name,param_value,signal,domain,"
        << "snr_composite,composite_saturated,composite_drops,"
        << "snr_lttb,lttb_saturated,snr_rdp,rdp_saturated" << endl;

    auto write_row = [&](const SweepResult& r) {
        csv << r.param_name << "," << r.param_value << "," << r.signal_name << "," << r.domain << ","
            << fixed << setprecision(4) << r.snr_composite << "," << r.composite_saturated << "," << r.composite_drops << ","
            << r.snr_lttb << "," << r.lttb_saturated << ","
            << r.snr_rdp << "," << r.rdp_saturated << endl;
    };

    int done = 0;
    auto t0 = chrono::high_resolution_clock::now();

    cout << "--- Varying window_half (default=3) ---" << endl;
    for (size_t wh : window_half_values) {
        ImportanceConfig cfg;
        cfg.window_half = wh;
        for (auto& sig : signals) {
            auto r = run_one_config(sig, "window_half", (double)wh, cfg, buf_size, overload_ratio);
            write_row(r);
            ++done;
        }
        cout << "  window_half=" << wh << " done (" << done << " total)" << endl;
    }

    cout << "\n--- Varying boundary_protect (default=1) ---" << endl;
    for (size_t bp : boundary_protect_vals) {
        ImportanceConfig cfg;
        cfg.boundary_protect = bp;
        for (auto& sig : signals) {
            auto r = run_one_config(sig, "boundary_protect", (double)bp, cfg, buf_size, overload_ratio);
            write_row(r);
            ++done;
        }
        cout << "  boundary_protect=" << bp << " done (" << done << " total)" << endl;
    }

    cout << "\n--- Varying gamma (default=0.6) ---" << endl;
    for (double g : gamma_values) {
        ImportanceConfig cfg;
        cfg.gamma = g;
        for (auto& sig : signals) {
            auto r = run_one_config(sig, "gamma", g, cfg, buf_size, overload_ratio);
            write_row(r);
            ++done;
        }
        cout << "  gamma=" << g << " done (" << done << " total)" << endl;
    }

    csv.close();

    auto t1 = chrono::high_resolution_clock::now();
    double total_sec = chrono::duration<double>(t1 - t0).count();

    cout << "\n=== SWEEP COMPLETE ===" << endl;
    cout << "Total configurations run: " << done << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_sec << "s" << endl;
    cout << "Results: results/sensitivity_results.csv" << endl;
    cout << "\nNEXT STEP: check whether snr_composite > snr_lttb holds for EVERY" << endl;
    cout << "row above (excluding saturated rows) -- that is the actual robustness" << endl;
    cout << "answer your professor asked for, not just whether the program ran." << endl;

    return 0;
}
