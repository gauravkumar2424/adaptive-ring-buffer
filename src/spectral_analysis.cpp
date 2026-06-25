// ============================================================
// spectral_analysis.cpp — Spectral Mechanism Investigation
// DATE 2027: Adaptive Ring Buffer
//
// PURPOSE: Explain WHY the proposed method beats RDP on spectral
// correlation despite losing on pointwise SNR (vibration domain).
//
// Outputs:
//   1. Reconstructed waveform CSVs (proposed vs RDP vs LTTB vs original)
//   2. Magnitude spectrum CSVs for each
//   3. Surviving sample spacing distribution analysis
//   4. Per-frequency-bin error comparison
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/spectral_analysis spectral_analysis.cpp
// Run:   cd ../build && ./spectral_analysis
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
// Offline baselines (same as cross_domain.cpp)
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
// DFT magnitude spectrum
// ============================================================
struct Spectrum {
    vector<double> magnitude;
    vector<double> freq_bins;  // normalized frequency [0, 0.5)
};

Spectrum compute_spectrum(const vector<double>& signal, int N) {
    Spectrum spec;
    int n = min((int)signal.size(), N);
    int half = n / 2;
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

double spectral_correlation(const Spectrum& a, const Spectrum& b) {
    int n = min(a.magnitude.size(), b.magnitude.size());
    if (n < 4) return -1.0;
    double ma = accumulate(a.magnitude.begin(), a.magnitude.begin()+n, 0.0) / n;
    double mb = accumulate(b.magnitude.begin(), b.magnitude.begin()+n, 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < n; ++k) {
        double xa = a.magnitude[k] - ma;
        double xb = b.magnitude[k] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    double den = sqrt(da * db);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================
// Spacing statistics for surviving samples
// ============================================================
struct SpacingStats {
    double mean_gap, std_gap, median_gap, max_gap, min_gap;
    double cv;  // coefficient of variation (std/mean)
    int n_samples;
    vector<int> gaps; // raw gap sizes for histogram
};

SpacingStats analyze_spacing(const vector<int>& indices) {
    SpacingStats s{};
    s.n_samples = (int)indices.size();
    if (indices.size() < 2) return s;

    for (size_t i = 1; i < indices.size(); ++i)
        s.gaps.push_back(indices[i] - indices[i-1]);

    sort(s.gaps.begin(), s.gaps.end());
    s.min_gap = s.gaps.front();
    s.max_gap = s.gaps.back();
    s.median_gap = (s.gaps.size() % 2 == 0) ?
        (s.gaps[s.gaps.size()/2-1] + s.gaps[s.gaps.size()/2]) / 2.0 :
        s.gaps[s.gaps.size()/2];
    s.mean_gap = accumulate(s.gaps.begin(), s.gaps.end(), 0.0) / s.gaps.size();
    double var = 0;
    for (int g : s.gaps) var += (g - s.mean_gap) * (g - s.mean_gap);
    s.std_gap = sqrt(var / s.gaps.size());
    s.cv = (s.mean_gap > 1e-15) ? s.std_gap / s.mean_gap : 0.0;
    return s;
}

// ============================================================
// Per-frequency-bin error analysis
// ============================================================
void write_per_bin_error(ofstream& csv,
                         const Spectrum& orig, const Spectrum& recon,
                         const string& method, const string& signal_name) {
    int n = min(orig.magnitude.size(), recon.magnitude.size());
    for (int k = 0; k < n; ++k) {
        double err = fabs(orig.magnitude[k] - recon.magnitude[k]);
        double rel_err = (orig.magnitude[k] > 1e-10) ?
            err / orig.magnitude[k] : 0.0;
        csv << signal_name << "," << method << "," << k << ","
            << fixed << setprecision(6)
            << orig.freq_bins[k] << ","
            << orig.magnitude[k] << ","
            << recon.magnitude[k] << ","
            << err << ","
            << rel_err << endl;
    }
}

// ============================================================
// Producer / Consumer (same as cross_domain.cpp)
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

struct MethodResult {
    string method_name;
    vector<double> reconstructed;
    vector<int> surviving_indices;
    double snr;
    double spec_corr;
    SpacingStats spacing;
};

// ============================================================
// Run one online method, return full result
// ============================================================
MethodResult run_online(const vector<double>& signal, BufferMode mode,
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

    MethodResult r;
    r.method_name = name;
    r.surviving_indices = surv_idx;
    r.reconstructed = reconstruct_signal(surv_idx, surv_vals, (int)signal.size());
    r.snr = compute_snr(signal, r.reconstructed);
    r.spacing = analyze_spacing(surv_idx);
    return r;
}

MethodResult run_offline(const vector<double>& signal, const string& method,
                         int target_surviving) {
    vector<int> selected;
    if (method == "RDP_OFFLINE")
        selected = offline_rdp(signal, target_surviving);
    else
        selected = offline_lttb(signal, target_surviving);

    vector<double> surv_vals;
    for (int idx : selected) surv_vals.push_back(signal[idx]);

    MethodResult r;
    r.method_name = method;
    r.surviving_indices = selected;
    r.reconstructed = reconstruct_signal(selected, surv_vals, (int)signal.size());
    r.snr = compute_snr(signal, r.reconstructed);
    r.spacing = analyze_spacing(selected);
    return r;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  SPECTRAL MECHANISM ANALYSIS" << endl;
    cout << "  Explaining: proposed > RDP on spectral correlation" << endl;
    cout << "============================================================\n" << endl;

    string data_dir = "../data";
    string out_dir = "../results/spectral";
    system(("mkdir -p " + out_dir).c_str());

    // Load vibration signals only
    auto all_signals = load_all_real_signals(data_dir, 2000);
    vector<RealSignal*> vib_signals;
    for (auto& s : all_signals)
        if (s.domain == "vibration") vib_signals.push_back(&s);

    cout << "Vibration signals loaded: " << vib_signals.size() << "\n" << endl;

    if (vib_signals.empty()) {
        cerr << "ERROR: No vibration signals found." << endl;
        return 1;
    }

    // Parameters: use representative config that showed the effect
    size_t buf_size = 256;
    int overload = 5;
    int num_trials = 5;
    int fft_N = 512;

    // Output files
    ofstream waveform_csv(out_dir + "/waveforms.csv");
    waveform_csv << "signal,method,sample_idx,value" << endl;

    ofstream spectrum_csv(out_dir + "/spectra.csv");
    spectrum_csv << "signal,method,bin,norm_freq,magnitude" << endl;

    ofstream spacing_csv(out_dir + "/spacing_stats.csv");
    spacing_csv << "signal,method,trial,n_surviving,n_dropped,mean_gap,std_gap,"
                << "cv,median_gap,min_gap,max_gap,snr_db,spectral_corr" << endl;

    ofstream bin_error_csv(out_dir + "/per_bin_spectral_error.csv");
    bin_error_csv << "signal,method,bin,norm_freq,orig_mag,recon_mag,abs_error,rel_error" << endl;

    ofstream gap_hist_csv(out_dir + "/gap_histograms.csv");
    gap_hist_csv << "signal,method,trial,gap_size" << endl;

    ofstream summary_csv(out_dir + "/mechanism_summary.csv");
    summary_csv << "signal,method,avg_snr,avg_spec_corr,avg_cv,avg_mean_gap,avg_max_gap,n_trials" << endl;

    for (auto* sig_ptr : vib_signals) {
        auto& sig = *sig_ptr;
        cout << "=== Processing: " << sig.name << " (" << sig.data.size() << " samples) ===" << endl;

        auto orig_spec = compute_spectrum(sig.data, fft_N);

        // Write original waveform (once)
        for (int i = 0; i < (int)sig.data.size(); ++i)
            waveform_csv << sig.name << ",ORIGINAL," << i << "," << sig.data[i] << endl;

        // Write original spectrum (once)
        for (int k = 0; k < (int)orig_spec.magnitude.size(); ++k)
            spectrum_csv << sig.name << ",ORIGINAL," << k << ","
                         << orig_spec.freq_bins[k] << "," << orig_spec.magnitude[k] << endl;

        // Accumulate per-method stats across trials
        struct Accum { vector<double> snrs, spec_corrs, cvs, mean_gaps, max_gaps; };
        map<string, Accum> method_accum;

        for (int trial = 0; trial < num_trials; ++trial) {
            // Run proposed method
            auto proposed = run_online(sig.data, BufferMode::IMPORTANCE_INTERP_ERROR,
                                       "IMP_INTERP_ERROR", buf_size, overload);
            int drops = (int)sig.data.size() - (int)proposed.surviving_indices.size();
            int target = max(2, (int)sig.data.size() - drops);

            // Run offline baselines matched to same drop count
            auto rdp = run_offline(sig.data, "RDP_OFFLINE", target);
            auto lttb = run_offline(sig.data, "LTTB_OFFLINE", target);

            // Also run composite
            auto composite = run_online(sig.data, BufferMode::IMPORTANCE_INTERP_COMPOSITE,
                                         "IMP_INTERP_COMPOSITE", buf_size, overload);

            vector<MethodResult*> methods = {&proposed, &rdp, &lttb, &composite};

            for (auto* m : methods) {
                // Compute spectral correlation
                auto recon_spec = compute_spectrum(m->reconstructed, fft_N);
                m->spec_corr = spectral_correlation(orig_spec, recon_spec);

                // Write waveform (first trial only to keep file size sane)
                if (trial == 0) {
                    for (int i = 0; i < (int)m->reconstructed.size(); ++i)
                        waveform_csv << sig.name << "," << m->method_name << ","
                                     << i << "," << m->reconstructed[i] << endl;

                    // Write spectrum
                    for (int k = 0; k < (int)recon_spec.magnitude.size(); ++k)
                        spectrum_csv << sig.name << "," << m->method_name << ","
                                     << k << "," << recon_spec.freq_bins[k] << ","
                                     << recon_spec.magnitude[k] << endl;

                    // Write per-bin error
                    write_per_bin_error(bin_error_csv, orig_spec, recon_spec,
                                        m->method_name, sig.name);
                }

                // Spacing stats
                spacing_csv << sig.name << "," << m->method_name << "," << trial << ","
                            << m->spacing.n_samples << ","
                            << drops << ","
                            << fixed << setprecision(4)
                            << m->spacing.mean_gap << ","
                            << m->spacing.std_gap << ","
                            << m->spacing.cv << ","
                            << m->spacing.median_gap << ","
                            << m->spacing.min_gap << ","
                            << m->spacing.max_gap << ","
                            << m->snr << ","
                            << m->spec_corr << endl;

                // Gap histogram data
                for (int g : m->spacing.gaps)
                    gap_hist_csv << sig.name << "," << m->method_name << ","
                                << trial << "," << g << endl;

                // Accumulate
                auto& acc = method_accum[m->method_name];
                if (isfinite(m->snr)) acc.snrs.push_back(m->snr);
                acc.spec_corrs.push_back(m->spec_corr);
                acc.cvs.push_back(m->spacing.cv);
                acc.mean_gaps.push_back(m->spacing.mean_gap);
                acc.max_gaps.push_back(m->spacing.max_gap);
            }
        }

        // Print per-signal summary
        cout << "\n  " << left << setw(22) << "Method"
             << right << setw(10) << "SNR(dB)"
             << setw(12) << "Spec.Corr"
             << setw(10) << "CV(gap)"
             << setw(12) << "Mean Gap"
             << setw(10) << "Max Gap" << endl;
        cout << "  " << string(76, '-') << endl;

        vector<string> method_order = {"RDP_OFFLINE", "IMP_INTERP_ERROR",
                                        "IMP_INTERP_COMPOSITE", "LTTB_OFFLINE"};
        for (auto& mname : method_order) {
            auto& acc = method_accum[mname];
            auto avg = [](const vector<double>& v) {
                return v.empty() ? 0.0 : accumulate(v.begin(), v.end(), 0.0) / v.size();
            };
            double a_snr = avg(acc.snrs);
            double a_spec = avg(acc.spec_corrs);
            double a_cv = avg(acc.cvs);
            double a_mg = avg(acc.mean_gaps);
            double a_xg = avg(acc.max_gaps);

            string tag = "";
            if (mname.find("INTERP") != string::npos) tag = " <<<";

            cout << "  " << left << setw(22) << mname
                 << right << fixed
                 << setprecision(2) << setw(10) << a_snr
                 << setprecision(4) << setw(12) << a_spec
                 << setprecision(4) << setw(10) << a_cv
                 << setprecision(2) << setw(12) << a_mg
                 << setprecision(0) << setw(10) << a_xg
                 << tag << endl;

            summary_csv << sig.name << "," << mname << ","
                        << fixed << setprecision(4)
                        << a_snr << "," << a_spec << ","
                        << a_cv << "," << a_mg << "," << a_xg << ","
                        << acc.snrs.size() << endl;
        }
        cout << endl;

        // Key insight analysis
        auto& acc_prop = method_accum["IMP_INTERP_ERROR"];
        auto& acc_rdp = method_accum["RDP_OFFLINE"];
        auto avg = [](const vector<double>& v) {
            return v.empty() ? 0.0 : accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        double cv_proposed = avg(acc_prop.cvs);
        double cv_rdp = avg(acc_rdp.cvs);
        double maxgap_proposed = avg(acc_prop.max_gaps);
        double maxgap_rdp = avg(acc_rdp.max_gaps);

        cout << "  MECHANISM HYPOTHESIS TEST for " << sig.name << ":" << endl;
        if (cv_proposed < cv_rdp) {
            cout << "    CONFIRMED: Proposed has MORE UNIFORM spacing (CV="
                 << fixed << setprecision(4) << cv_proposed
                 << " vs RDP CV=" << cv_rdp << ")" << endl;
        } else {
            cout << "    REJECTED: Proposed spacing NOT more uniform (CV="
                 << fixed << setprecision(4) << cv_proposed
                 << " vs RDP CV=" << cv_rdp << ")" << endl;
        }
        if (maxgap_proposed < maxgap_rdp) {
            cout << "    CONFIRMED: Proposed has SMALLER max gap ("
                 << fixed << setprecision(0) << maxgap_proposed
                 << " vs RDP " << maxgap_rdp << ")" << endl;
        } else {
            cout << "    REJECTED: Proposed max gap NOT smaller ("
                 << fixed << setprecision(0) << maxgap_proposed
                 << " vs RDP " << maxgap_rdp << ")" << endl;
        }
        cout << "    INTERPRETATION: ";
        if (cv_proposed < cv_rdp && maxgap_proposed < maxgap_rdp) {
            cout << "Online local eviction avoids large gaps that RDP's global" << endl;
            cout << "      optimization creates; uniform spacing preserves Nyquist bandwidth." << endl;
        } else {
            cout << "Spacing alone does not explain the spectral advantage." << endl;
            cout << "      Further investigation needed (phase preservation, etc.)." << endl;
        }
        cout << endl;
    }

    waveform_csv.close();
    spectrum_csv.close();
    spacing_csv.close();
    bin_error_csv.close();
    gap_hist_csv.close();
    summary_csv.close();

    cout << "============================================================" << endl;
    cout << "  OUTPUT FILES (in " << out_dir << "):" << endl;
    cout << "    waveforms.csv           — time-domain traces for plotting" << endl;
    cout << "    spectra.csv             — magnitude spectra for plotting" << endl;
    cout << "    spacing_stats.csv       — gap distribution per method" << endl;
    cout << "    gap_histograms.csv      — raw gap data for histogram plots" << endl;
    cout << "    per_bin_spectral_error.csv — per-frequency error" << endl;
    cout << "    mechanism_summary.csv   — aggregated summary" << endl;
    cout << "============================================================" << endl;

    return 0;
}
