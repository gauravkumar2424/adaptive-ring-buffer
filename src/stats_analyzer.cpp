// ============================================================
// stats_analyzer.cpp — Post-hoc Statistical Analysis
// DATE 2027: Adaptive Ring Buffer
//
// Reads EXISTING CSVs (no re-run needed). Computes:
//   - Per-mode mean, std dev, 95% CI, n
//   - Wilcoxon signed-rank test (paired by config)
//   - Cohen's d effect size
//   - Per-domain breakdowns
//
// Build: g++ -std=c++17 -O2 -o ../build/stats_analyzer stats_analyzer.cpp
// Run:   cd ../build && ./stats_analyzer
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <tuple>
#include <cassert>

using namespace std;

// ============================================================
// Normal CDF (Abramowitz & Stegun 26.2.17)
// ============================================================
double normal_cdf(double x) {
    if (x < -8.0) return 0.0;
    if (x >  8.0) return 1.0;
    double a = fabs(x);
    double t = 1.0 / (1.0 + 0.2316419 * a);
    double d = 0.3989422804014327; // 1/sqrt(2*pi)
    double p = d * exp(-0.5 * x * x) *
        (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
         t * (-1.821255978 + t * 1.330274429)))));
    return (x > 0) ? 1.0 - p : p;
}

// ============================================================
// Wilcoxon Signed-Rank Test (two-sided)
// Returns {W_statistic, z_score, p_value, n_pairs}
// ============================================================
struct WilcoxonResult {
    double W;
    double z;
    double p_value;
    int n_pairs;
    int n_positive;  // pairs where proposed > baseline
    int n_negative;
};

WilcoxonResult wilcoxon_signed_rank(const vector<double>& a,
                                     const vector<double>& b) {
    WilcoxonResult res{0,0,1.0,0,0,0};
    int n = min(a.size(), b.size());

    // Compute differences, remove zeros
    struct DiffEntry { double abs_diff; int sign; }; // sign: +1 or -1
    vector<DiffEntry> diffs;
    for (int i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        if (fabs(d) < 1e-15) continue; // remove zeros
        diffs.push_back({fabs(d), (d > 0) ? 1 : -1});
    }

    int nr = (int)diffs.size();
    res.n_pairs = nr;
    if (nr < 5) { res.p_value = 1.0; return res; } // too few pairs

    // Rank by |d| (handle ties with average rank)
    vector<int> order(nr);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int x, int y) { return diffs[x].abs_diff < diffs[y].abs_diff; });

    vector<double> ranks(nr);
    int i = 0;
    double tie_correction = 0.0;
    while (i < nr) {
        int j = i;
        while (j < nr && fabs(diffs[order[j]].abs_diff - diffs[order[i]].abs_diff) < 1e-15)
            ++j;
        double avg_rank = (i + 1.0 + j) / 2.0; // 1-indexed average
        int tie_size = j - i;
        if (tie_size > 1)
            tie_correction += (double)tie_size * tie_size * tie_size - tie_size;
        for (int k = i; k < j; ++k)
            ranks[order[k]] = avg_rank;
        i = j;
    }

    // W+ and W-
    double w_plus = 0, w_minus = 0;
    for (int k = 0; k < nr; ++k) {
        if (diffs[k].sign > 0) {
            w_plus += ranks[k];
            res.n_positive++;
        } else {
            w_minus += ranks[k];
            res.n_negative++;
        }
    }

    double T = min(w_plus, w_minus);
    res.W = T;

    // Normal approximation with tie correction
    double mean_T = (double)nr * (nr + 1) / 4.0;
    double var_T = (double)nr * (nr + 1) * (2 * nr + 1) / 24.0 - tie_correction / 48.0;
    if (var_T <= 0) { res.p_value = 1.0; return res; }

    // Continuity correction
    res.z = (T - mean_T + 0.5) / sqrt(var_T);
    res.p_value = 2.0 * normal_cdf(res.z); // two-sided

    return res;
}

// ============================================================
// Cohen's d (pooled std)
// ============================================================
double cohens_d(const vector<double>& a, const vector<double>& b) {
    if (a.size() < 2 || b.size() < 2) return 0.0;
    double ma = accumulate(a.begin(), a.end(), 0.0) / a.size();
    double mb = accumulate(b.begin(), b.end(), 0.0) / b.size();
    double va = 0, vb = 0;
    for (double x : a) va += (x - ma) * (x - ma);
    for (double x : b) vb += (x - mb) * (x - mb);
    va /= (a.size() - 1);
    vb /= (b.size() - 1);
    double sp = sqrt(((a.size()-1)*va + (b.size()-1)*vb) / (a.size()+b.size()-2));
    return (sp > 1e-15) ? (ma - mb) / sp : 0.0;
}

// ============================================================
// Descriptive stats
// ============================================================
struct DescStats {
    double mean, std_dev, ci_lo, ci_hi, median;
    int n;
};

DescStats compute_desc(vector<double>& vals) {
    DescStats s{0,0,0,0,0,0};
    s.n = (int)vals.size();
    if (s.n == 0) return s;
    sort(vals.begin(), vals.end());
    s.mean = accumulate(vals.begin(), vals.end(), 0.0) / s.n;
    s.median = (s.n % 2 == 0) ?
        (vals[s.n/2-1] + vals[s.n/2]) / 2.0 : vals[s.n/2];
    if (s.n < 2) { s.std_dev = 0; s.ci_lo = s.ci_hi = s.mean; return s; }
    double var = 0;
    for (double x : vals) var += (x - s.mean) * (x - s.mean);
    s.std_dev = sqrt(var / (s.n - 1));
    double se = s.std_dev / sqrt((double)s.n);
    s.ci_lo = s.mean - 1.96 * se;
    s.ci_hi = s.mean + 1.96 * se;
    return s;
}

// ============================================================
// CSV row for cross_domain_results.csv
// ============================================================
struct CDRow {
    string signal, domain, mode;
    int buffer_size, overload, trial;
    double snr, mse, max_wait, deriv_ratio;
    int drops;
    bool degenerate, snr_saturated;
    double rpeak_f1, rpeak_precision, rpeak_recall, rpeak_timing;
    double spectral_corr;
};

// Config key for pairing
using ConfigKey = tuple<string, int, int, int>; // signal, buf_size, overload, trial

ConfigKey make_key(const CDRow& r) {
    return {r.signal, r.buffer_size, r.overload, r.trial};
}

vector<CDRow> read_cross_domain(const string& path) {
    vector<CDRow> rows;
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "Cannot open " << path << endl;
        return rows;
    }
    string line;
    getline(f, line); // header

    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        CDRow r;
        string tmp;
        int deg_i, sat_i;

        getline(ss, r.signal, ',');
        getline(ss, r.domain, ',');
        getline(ss, r.mode, ',');
        ss >> r.buffer_size; ss.ignore();
        ss >> r.overload; ss.ignore();
        ss >> r.trial; ss.ignore();

        // SNR might be "inf" or "-inf"
        getline(ss, tmp, ',');
        if (tmp.find("inf") != string::npos || tmp.find("nan") != string::npos)
            r.snr = numeric_limits<double>::infinity();
        else
            r.snr = stod(tmp);

        ss >> r.mse; ss.ignore();
        ss >> r.max_wait; ss.ignore();
        ss >> r.deriv_ratio; ss.ignore();
        ss >> r.drops; ss.ignore();
        ss >> deg_i; ss.ignore();
        ss >> sat_i; ss.ignore();
        r.degenerate = (deg_i != 0);
        r.snr_saturated = (sat_i != 0);

        ss >> r.rpeak_f1; ss.ignore();
        ss >> r.rpeak_precision; ss.ignore();
        ss >> r.rpeak_recall; ss.ignore();
        ss >> r.rpeak_timing; ss.ignore();
        ss >> r.spectral_corr;

        rows.push_back(r);
    }
    return rows;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    string results_dir = "../results";
    if (argc > 1) results_dir = argv[1];

    cout << "============================================================" << endl;
    cout << "  STATISTICAL ANALYSIS — Adaptive Ring Buffer" << endl;
    cout << "  Reading from: " << results_dir << endl;
    cout << "============================================================\n" << endl;

    // ---- Read cross-domain results ----
    string cd_path = results_dir + "/cross_domain_results.csv";
    auto rows = read_cross_domain(cd_path);
    cout << "Loaded " << rows.size() << " rows from cross_domain_results.csv\n" << endl;

    if (rows.empty()) {
        cerr << "No data. Exiting." << endl;
        return 1;
    }

    // ============================================================
    // 1. Per-mode descriptive statistics (overall + per-domain)
    // ============================================================
    cout << "============================================================" << endl;
    cout << "  SECTION 1: Descriptive Statistics (SNR dB)" << endl;
    cout << "============================================================\n" << endl;

    // Collect SNR values by (domain, mode) — excluding saturated
    map<string, map<string, vector<double>>> domain_mode_snr; // domain -> mode -> snrs
    map<string, map<string, vector<double>>> domain_mode_spec; // spectral corr
    map<string, map<string, vector<double>>> domain_mode_f1;   // rpeak f1

    for (auto& r : rows) {
        if (!r.snr_saturated) {
            domain_mode_snr[r.domain][r.mode].push_back(r.snr);
            domain_mode_snr["ALL"][r.mode].push_back(r.snr);
        }
        if (r.spectral_corr >= 0 && !r.snr_saturated)
            domain_mode_spec[r.domain][r.mode].push_back(r.spectral_corr);
        if (r.rpeak_f1 >= 0)
            domain_mode_f1[r.domain][r.mode].push_back(r.rpeak_f1);
    }

    // Output CSV with all stats
    ofstream stats_csv(results_dir + "/statistical_summary.csv");
    stats_csv << "domain,mode,metric,n,mean,std_dev,ci_lo,ci_hi,median" << endl;

    for (auto& domain_label : {"ALL", "ecg", "vibration"}) {
        string dl = domain_label;
        if (domain_mode_snr.find(dl) == domain_mode_snr.end()) continue;

        cout << "--- Domain: " << dl << " (SNR dB) ---" << endl;
        cout << left << setw(24) << "Mode"
             << right << setw(6) << "n"
             << setw(10) << "Mean"
             << setw(10) << "StdDev"
             << setw(12) << "95% CI Lo"
             << setw(12) << "95% CI Hi"
             << setw(10) << "Median" << endl;
        cout << string(84, '-') << endl;

        // Sort modes by mean SNR descending
        vector<pair<string, vector<double>>> sorted_modes(
            domain_mode_snr[dl].begin(), domain_mode_snr[dl].end());
        sort(sorted_modes.begin(), sorted_modes.end(),
             [](auto& a, auto& b) {
                 double ma = accumulate(a.second.begin(), a.second.end(), 0.0) / max((size_t)1, a.second.size());
                 double mb = accumulate(b.second.begin(), b.second.end(), 0.0) / max((size_t)1, b.second.size());
                 return ma > mb;
             });

        for (auto& [mode, vals] : sorted_modes) {
            auto s = compute_desc(vals);
            string tag = "";
            if (mode == "IMP_INTERP_ERROR" || mode == "IMP_INTERP_COMPOSITE") tag = " <<<";

            cout << left << setw(24) << mode
                 << right << setw(6) << s.n
                 << fixed << setprecision(2)
                 << setw(10) << s.mean
                 << setw(10) << s.std_dev
                 << setw(12) << s.ci_lo
                 << setw(12) << s.ci_hi
                 << setw(10) << s.median
                 << tag << endl;

            stats_csv << dl << "," << mode << ",snr_db,"
                      << s.n << "," << fixed << setprecision(4)
                      << s.mean << "," << s.std_dev << ","
                      << s.ci_lo << "," << s.ci_hi << "," << s.median << endl;
        }
        cout << endl;

        // Spectral correlation stats (vibration only)
        if (dl == "vibration" && domain_mode_spec.find(dl) != domain_mode_spec.end()) {
            cout << "--- Domain: " << dl << " (Spectral Correlation) ---" << endl;
            cout << left << setw(24) << "Mode"
                 << right << setw(6) << "n"
                 << setw(10) << "Mean"
                 << setw(10) << "StdDev"
                 << setw(12) << "95% CI Lo"
                 << setw(12) << "95% CI Hi" << endl;
            cout << string(72, '-') << endl;

            vector<pair<string, vector<double>>> spec_sorted(
                domain_mode_spec[dl].begin(), domain_mode_spec[dl].end());
            sort(spec_sorted.begin(), spec_sorted.end(),
                 [](auto& a, auto& b) {
                     double ma = accumulate(a.second.begin(), a.second.end(), 0.0) / max((size_t)1, a.second.size());
                     double mb = accumulate(b.second.begin(), b.second.end(), 0.0) / max((size_t)1, b.second.size());
                     return ma > mb;
                 });

            for (auto& [mode, vals] : spec_sorted) {
                auto s = compute_desc(vals);
                string tag = "";
                if (mode == "IMP_INTERP_ERROR" || mode == "IMP_INTERP_COMPOSITE") tag = " <<<";
                cout << left << setw(24) << mode
                     << right << setw(6) << s.n
                     << fixed << setprecision(4)
                     << setw(10) << s.mean
                     << setw(10) << s.std_dev
                     << setw(12) << s.ci_lo
                     << setw(12) << s.ci_hi
                     << tag << endl;

                stats_csv << dl << "," << mode << ",spectral_corr,"
                          << s.n << "," << fixed << setprecision(6)
                          << s.mean << "," << s.std_dev << ","
                          << s.ci_lo << "," << s.ci_hi << "," << s.median << endl;
            }
            cout << endl;
        }

        // R-peak F1 stats (ECG only)
        if (dl == "ecg" && domain_mode_f1.find(dl) != domain_mode_f1.end()) {
            cout << "--- Domain: " << dl << " (R-peak F1) ---" << endl;
            cout << left << setw(24) << "Mode"
                 << right << setw(6) << "n"
                 << setw(10) << "Mean"
                 << setw(10) << "StdDev"
                 << setw(12) << "95% CI Lo"
                 << setw(12) << "95% CI Hi" << endl;
            cout << string(72, '-') << endl;

            vector<pair<string, vector<double>>> f1_sorted(
                domain_mode_f1[dl].begin(), domain_mode_f1[dl].end());
            sort(f1_sorted.begin(), f1_sorted.end(),
                 [](auto& a, auto& b) {
                     double ma = accumulate(a.second.begin(), a.second.end(), 0.0) / max((size_t)1, a.second.size());
                     double mb = accumulate(b.second.begin(), b.second.end(), 0.0) / max((size_t)1, b.second.size());
                     return ma > mb;
                 });

            for (auto& [mode, vals] : f1_sorted) {
                auto s = compute_desc(vals);
                cout << left << setw(24) << mode
                     << right << setw(6) << s.n
                     << fixed << setprecision(4)
                     << setw(10) << s.mean
                     << setw(10) << s.std_dev
                     << setw(12) << s.ci_lo
                     << setw(12) << s.ci_hi << endl;

                stats_csv << dl << "," << mode << ",rpeak_f1,"
                          << s.n << "," << fixed << setprecision(6)
                          << s.mean << "," << s.std_dev << ","
                          << s.ci_lo << "," << s.ci_hi << "," << s.median << endl;
            }
            cout << endl;
        }
    }

    // ============================================================
    // 2. Paired Wilcoxon Signed-Rank Tests
    // ============================================================
    cout << "============================================================" << endl;
    cout << "  SECTION 2: Wilcoxon Signed-Rank Tests (Paired)" << endl;
    cout << "  Pairing key: (signal, buffer_size, overload, trial)" << endl;
    cout << "============================================================\n" << endl;

    // Build lookup: config_key -> mode -> snr
    map<ConfigKey, map<string, double>> config_mode_snr;
    map<ConfigKey, map<string, double>> config_mode_spec;
    map<ConfigKey, string> config_domain;

    for (auto& r : rows) {
        auto key = make_key(r);
        config_domain[key] = r.domain;
        if (!r.snr_saturated)
            config_mode_snr[key][r.mode] = r.snr;
        if (r.spectral_corr >= 0 && !r.snr_saturated)
            config_mode_spec[key][r.mode] = r.spectral_corr;
    }

    // Test pairs
    struct TestPair {
        string proposed;
        string baseline;
        string label;
    };
    vector<TestPair> tests = {
        {"IMP_INTERP_ERROR", "RDP_OFFLINE",  "Proposed vs RDP (offline upper bound)"},
        {"IMP_INTERP_ERROR", "LTTB_OFFLINE", "Proposed vs LTTB (offline industry std)"},
        {"IMP_INTERP_COMPOSITE", "RDP_OFFLINE",  "Composite vs RDP"},
        {"IMP_INTERP_COMPOSITE", "LTTB_OFFLINE", "Composite vs LTTB"},
        {"IMP_INTERP_ERROR", "IMP_INTERP_COMPOSITE", "InterpError vs InterpComposite"},
        {"IMP_INTERP_ERROR", "DROP",         "Proposed vs FIFO (naive baseline)"},
        {"IMP_INTERP_ERROR", "IMP_COMPOSITE","Proposed vs Proxy Composite"},
    };

    ofstream wilcox_csv(results_dir + "/wilcoxon_results.csv");
    wilcox_csv << "domain,proposed,baseline,metric,n_pairs,n_positive,n_negative,"
               << "W,z,p_value,cohens_d,mean_diff,interpretation" << endl;

    for (auto& domain_label : {"ALL", "ecg", "vibration"}) {
        string dl = domain_label;
        cout << "--- Domain: " << dl << " ---\n" << endl;

        for (auto& tp : tests) {
            // Collect paired SNR values
            vector<double> proposed_snr, baseline_snr;
            for (auto& [key, mode_snr] : config_mode_snr) {
                if (dl != "ALL" && config_domain[key] != dl) continue;
                auto it_p = mode_snr.find(tp.proposed);
                auto it_b = mode_snr.find(tp.baseline);
                if (it_p != mode_snr.end() && it_b != mode_snr.end()) {
                    proposed_snr.push_back(it_p->second);
                    baseline_snr.push_back(it_b->second);
                }
            }

            if (proposed_snr.empty()) continue;

            auto wres = wilcoxon_signed_rank(proposed_snr, baseline_snr);
            double cd = cohens_d(proposed_snr, baseline_snr);
            double mean_diff = 0;
            for (size_t i = 0; i < proposed_snr.size(); ++i)
                mean_diff += proposed_snr[i] - baseline_snr[i];
            mean_diff /= proposed_snr.size();

            string sig_label;
            if (wres.p_value < 0.001)      sig_label = "*** (p<0.001)";
            else if (wres.p_value < 0.01)  sig_label = "**  (p<0.01)";
            else if (wres.p_value < 0.05)  sig_label = "*   (p<0.05)";
            else                            sig_label = "n.s.";

            string effect_label;
            double abs_cd = fabs(cd);
            if (abs_cd >= 0.8) effect_label = "large";
            else if (abs_cd >= 0.5) effect_label = "medium";
            else if (abs_cd >= 0.2) effect_label = "small";
            else effect_label = "negligible";

            string interp;
            if (mean_diff > 0) interp = tp.proposed + " WINS by " + to_string(mean_diff).substr(0,5) + " dB";
            else interp = tp.baseline + " WINS by " + to_string(-mean_diff).substr(0,5) + " dB";

            cout << "  " << tp.label << endl;
            cout << "    n_pairs=" << wres.n_pairs
                 << "  W+_wins=" << wres.n_positive
                 << "  W-_wins=" << wres.n_negative << endl;
            cout << "    W=" << fixed << setprecision(1) << wres.W
                 << "  z=" << setprecision(4) << wres.z
                 << "  p=" << scientific << setprecision(4) << wres.p_value
                 << "  " << sig_label << endl;
            cout << "    mean_diff=" << fixed << setprecision(4) << mean_diff << " dB"
                 << "  Cohen's d=" << setprecision(4) << cd
                 << " (" << effect_label << ")" << endl;
            cout << "    => " << interp << "\n" << endl;

            wilcox_csv << dl << "," << tp.proposed << "," << tp.baseline << ",snr_db,"
                       << wres.n_pairs << "," << wres.n_positive << "," << wres.n_negative << ","
                       << fixed << setprecision(4) << wres.W << "," << wres.z << ","
                       << scientific << wres.p_value << ","
                       << fixed << setprecision(4) << cd << "," << mean_diff << ","
                       << "\"" << interp << "\"" << endl;
        }

        // Also test spectral correlation for vibration
        if (dl == "vibration" || dl == "ALL") {
            cout << "  --- Spectral Correlation Paired Tests ---\n" << endl;

            vector<TestPair> spec_tests = {
                {"IMP_INTERP_ERROR", "RDP_OFFLINE",  "Proposed vs RDP (spectral)"},
                {"IMP_INTERP_ERROR", "LTTB_OFFLINE", "Proposed vs LTTB (spectral)"},
            };

            for (auto& tp : spec_tests) {
                vector<double> proposed_spec, baseline_spec;
                for (auto& [key, mode_spec] : config_mode_spec) {
                    if (dl != "ALL" && config_domain[key] != dl) continue;
                    auto it_p = mode_spec.find(tp.proposed);
                    auto it_b = mode_spec.find(tp.baseline);
                    if (it_p != mode_spec.end() && it_b != mode_spec.end()) {
                        proposed_spec.push_back(it_p->second);
                        baseline_spec.push_back(it_b->second);
                    }
                }

                if (proposed_spec.empty()) continue;

                auto wres = wilcoxon_signed_rank(proposed_spec, baseline_spec);
                double cd = cohens_d(proposed_spec, baseline_spec);
                double mean_diff = 0;
                for (size_t i = 0; i < proposed_spec.size(); ++i)
                    mean_diff += proposed_spec[i] - baseline_spec[i];
                mean_diff /= proposed_spec.size();

                string sig_label;
                if (wres.p_value < 0.001)      sig_label = "*** (p<0.001)";
                else if (wres.p_value < 0.01)  sig_label = "**  (p<0.01)";
                else if (wres.p_value < 0.05)  sig_label = "*   (p<0.05)";
                else                            sig_label = "n.s.";

                cout << "  " << tp.label << endl;
                cout << "    n_pairs=" << wres.n_pairs
                     << "  p=" << scientific << setprecision(4) << wres.p_value
                     << "  " << sig_label
                     << "  mean_diff=" << fixed << setprecision(6) << mean_diff
                     << "  Cohen's d=" << setprecision(4) << cd << "\n" << endl;

                wilcox_csv << dl << "," << tp.proposed << "," << tp.baseline << ",spectral_corr,"
                           << wres.n_pairs << "," << wres.n_positive << "," << wres.n_negative << ","
                           << fixed << setprecision(4) << wres.W << "," << wres.z << ","
                           << scientific << wres.p_value << ","
                           << fixed << setprecision(4) << cd << "," << mean_diff << ","
                           << "\"spectral\"" << endl;
            }
        }
    }

    // ============================================================
    // 3. Per-buffer-size breakdown (scaling analysis)
    // ============================================================
    cout << "============================================================" << endl;
    cout << "  SECTION 3: Per-Buffer-Size Stats (IMP_INTERP_ERROR)" << endl;
    cout << "============================================================\n" << endl;

    map<int, vector<double>> bufsize_snr;
    for (auto& r : rows) {
        if (r.mode == "IMP_INTERP_ERROR" && !r.snr_saturated)
            bufsize_snr[r.buffer_size].push_back(r.snr);
    }

    cout << left << setw(12) << "BufSize"
         << right << setw(6) << "n"
         << setw(10) << "Mean"
         << setw(10) << "StdDev"
         << setw(12) << "95% CI Lo"
         << setw(12) << "95% CI Hi" << endl;
    cout << string(62, '-') << endl;

    for (auto& [bs, vals] : bufsize_snr) {
        auto s = compute_desc(vals);
        cout << left << setw(12) << bs
             << right << setw(6) << s.n
             << fixed << setprecision(2)
             << setw(10) << s.mean
             << setw(10) << s.std_dev
             << setw(12) << s.ci_lo
             << setw(12) << s.ci_hi << endl;
    }
    cout << endl;

    // ============================================================
    // 4. Per-overload-ratio breakdown
    // ============================================================
    cout << "============================================================" << endl;
    cout << "  SECTION 4: Per-Overload-Ratio Stats (IMP_INTERP_ERROR)" << endl;
    cout << "============================================================\n" << endl;

    map<int, vector<double>> overload_snr;
    for (auto& r : rows) {
        if (r.mode == "IMP_INTERP_ERROR" && !r.snr_saturated)
            overload_snr[r.overload].push_back(r.snr);
    }

    cout << left << setw(12) << "Overload"
         << right << setw(6) << "n"
         << setw(10) << "Mean"
         << setw(10) << "StdDev"
         << setw(12) << "95% CI Lo"
         << setw(12) << "95% CI Hi" << endl;
    cout << string(62, '-') << endl;

    for (auto& [ol, vals] : overload_snr) {
        auto s = compute_desc(vals);
        cout << left << setw(12) << ol
             << right << setw(6) << s.n
             << fixed << setprecision(2)
             << setw(10) << s.mean
             << setw(10) << s.std_dev
             << setw(12) << s.ci_lo
             << setw(12) << s.ci_hi << endl;
    }
    cout << endl;

    // ============================================================
    // 5. Win/Loss matrix (per-config pairwise)
    // ============================================================
    cout << "============================================================" << endl;
    cout << "  SECTION 5: Win/Loss/Tie Counts (per-config)" << endl;
    cout << "  IMP_INTERP_ERROR vs each other mode" << endl;
    cout << "============================================================\n" << endl;

    set<string> all_modes;
    for (auto& r : rows) all_modes.insert(r.mode);

    cout << left << setw(24) << "Opponent"
         << right << setw(8) << "Wins"
         << setw(8) << "Losses"
         << setw(8) << "Ties"
         << setw(10) << "WinRate" << endl;
    cout << string(58, '-') << endl;

    for (auto& opponent : all_modes) {
        if (opponent == "IMP_INTERP_ERROR") continue;
        int wins = 0, losses = 0, ties = 0;
        for (auto& [key, mode_snr] : config_mode_snr) {
            auto it_p = mode_snr.find("IMP_INTERP_ERROR");
            auto it_o = mode_snr.find(opponent);
            if (it_p != mode_snr.end() && it_o != mode_snr.end()) {
                double diff = it_p->second - it_o->second;
                if (diff > 0.01) ++wins;
                else if (diff < -0.01) ++losses;
                else ++ties;
            }
        }
        int total = wins + losses + ties;
        double wr = (total > 0) ? 100.0 * wins / total : 0;
        cout << left << setw(24) << opponent
             << right << setw(8) << wins
             << setw(8) << losses
             << setw(8) << ties
             << fixed << setprecision(1) << setw(9) << wr << "%" << endl;
    }
    cout << endl;

    stats_csv.close();
    wilcox_csv.close();

    cout << "============================================================" << endl;
    cout << "  OUTPUT FILES:" << endl;
    cout << "    " << results_dir << "/statistical_summary.csv" << endl;
    cout << "    " << results_dir << "/wilcoxon_results.csv" << endl;
    cout << "============================================================" << endl;

    return 0;
}
