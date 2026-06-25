// novelty_stats.cpp — Wilcoxon tests for novelty_eval_results.csv
// Build: g++ -std=c++17 -O2 -o ../build/novelty_stats novelty_stats.cpp
// Run:   cd ../build && ./novelty_stats

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

using namespace std;

double normal_cdf(double x) {
    if (x < -8.0) return 0.0;
    if (x >  8.0) return 1.0;
    double a = fabs(x);
    double t = 1.0 / (1.0 + 0.2316419 * a);
    double d = 0.3989422804014327;
    double p = d * exp(-0.5 * x * x) *
        (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
         t * (-1.821255978 + t * 1.330274429)))));
    return (x > 0) ? 1.0 - p : p;
}

struct WilcoxonResult { double W, z, p_value; int n_pairs, n_pos, n_neg; };

WilcoxonResult wilcoxon(const vector<double>& a, const vector<double>& b) {
    WilcoxonResult res{0,0,1.0,0,0,0};
    int n = min(a.size(), b.size());
    struct D { double ad; int sign; };
    vector<D> diffs;
    for (int i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        if (fabs(d) < 1e-15) continue;
        diffs.push_back({fabs(d), (d > 0) ? 1 : -1});
    }
    int nr = diffs.size();
    res.n_pairs = nr;
    if (nr < 5) return res;
    vector<int> order(nr);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int x, int y) { return diffs[x].ad < diffs[y].ad; });
    vector<double> ranks(nr);
    double tie_corr = 0;
    int i = 0;
    while (i < nr) {
        int j = i;
        while (j < nr && fabs(diffs[order[j]].ad - diffs[order[i]].ad) < 1e-15) ++j;
        double avg = (i + 1.0 + j) / 2.0;
        int ts = j - i;
        if (ts > 1) tie_corr += (double)ts * ts * ts - ts;
        for (int k = i; k < j; ++k) ranks[order[k]] = avg;
        i = j;
    }
    double wp = 0, wm = 0;
    for (int k = 0; k < nr; ++k) {
        if (diffs[k].sign > 0) { wp += ranks[k]; res.n_pos++; }
        else { wm += ranks[k]; res.n_neg++; }
    }
    double T = min(wp, wm);
    res.W = T;
    double mean_T = (double)nr * (nr + 1) / 4.0;
    double var_T = (double)nr * (nr + 1) * (2 * nr + 1) / 24.0 - tie_corr / 48.0;
    if (var_T <= 0) return res;
    res.z = (T - mean_T + 0.5) / sqrt(var_T);
    res.p_value = 2.0 * normal_cdf(res.z);
    return res;
}

double cohens_d(const vector<double>& a, const vector<double>& b) {
    if (a.size() < 2 || b.size() < 2) return 0.0;
    double ma = accumulate(a.begin(), a.end(), 0.0) / a.size();
    double mb = accumulate(b.begin(), b.end(), 0.0) / b.size();
    double va = 0, vb = 0;
    for (double x : a) va += (x - ma) * (x - ma);
    for (double x : b) vb += (x - mb) * (x - mb);
    va /= (a.size() - 1); vb /= (b.size() - 1);
    double sp = sqrt(((a.size()-1)*va + (b.size()-1)*vb) / (a.size()+b.size()-2));
    return (sp > 1e-15) ? (ma - mb) / sp : 0.0;
}

using Key = tuple<string, int, int, int>;

int main() {
    cout << "============================================================" << endl;
    cout << "  NOVELTY STATISTICAL ANALYSIS" << endl;
    cout << "  Wilcoxon tests: Proposed vs SDT/PLA/LTC + Spectral mode" << endl;
    cout << "============================================================\n" << endl;

    ifstream f("../results/novelty_eval_results.csv");
    if (!f.is_open()) { cerr << "Cannot open novelty_eval_results.csv" << endl; return 1; }

    string line;
    getline(f, line); // header

    // key -> mode -> {snr, spec}
    map<Key, map<string, double>> key_mode_snr;
    map<Key, map<string, double>> key_mode_spec;
    map<Key, string> key_domain;

    int total = 0;
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string sig, dom, mode;
        int bs, ol, trial;
        string snr_str;
        int sat;
        int drops;
        double f1, spec;

        getline(ss, sig, ','); getline(ss, dom, ','); getline(ss, mode, ',');
        ss >> bs; ss.ignore(); ss >> ol; ss.ignore(); ss >> trial; ss.ignore();
        getline(ss, snr_str, ',');
        ss >> sat; ss.ignore(); ss >> drops; ss.ignore();
        ss >> f1; ss.ignore(); ss >> spec;

        double snr = 0;
        if (snr_str.find("inf") != string::npos || snr_str.find("nan") != string::npos)
            sat = 1;
        else
            snr = stod(snr_str);

        Key key = {sig, bs, ol, trial};
        key_domain[key] = dom;

        if (!sat) key_mode_snr[key][mode] = snr;
        if (spec >= 0 && !sat) key_mode_spec[key][mode] = spec;
        ++total;
    }
    cout << "Loaded " << total << " rows\n" << endl;

    // Test pairs
    struct TP { string proposed; string baseline; string label; };
    vector<TP> snr_tests = {
        {"IMP_INTERP_SPECTRAL", "SDT_MATCHED",  "SPECTRAL vs SDT"},
        {"IMP_INTERP_SPECTRAL", "PLA_MATCHED",  "SPECTRAL vs PLA"},
        {"IMP_INTERP_SPECTRAL", "LTC_MATCHED",  "SPECTRAL vs LTC"},
        {"IMP_INTERP_SPECTRAL", "RDP_OFFLINE",  "SPECTRAL vs RDP"},
        {"IMP_INTERP_SPECTRAL", "LTTB_OFFLINE", "SPECTRAL vs LTTB"},
        {"IMP_INTERP_SPECTRAL", "IMP_INTERP_ERROR", "SPECTRAL vs INTERP_ERROR"},
        {"IMP_INTERP_ERROR",    "SDT_MATCHED",  "INTERP_ERROR vs SDT"},
        {"IMP_INTERP_ERROR",    "PLA_MATCHED",  "INTERP_ERROR vs PLA"},
        {"IMP_INTERP_ERROR",    "LTC_MATCHED",  "INTERP_ERROR vs LTC"},
        {"IMP_INTERP_ERROR",    "RDP_OFFLINE",  "INTERP_ERROR vs RDP"},
        {"IMP_INTERP_ERROR",    "DROP",         "INTERP_ERROR vs DROP"},
    };

    ofstream csv("../results/novelty_wilcoxon.csv");
    csv << "domain,proposed,baseline,metric,n_pairs,n_pos,n_neg,W,z,p_value,cohens_d,mean_diff" << endl;

    for (auto& domain_label : {"ALL", "ecg", "vibration"}) {
        string dl = domain_label;
        cout << "=== Domain: " << dl << " ===" << endl;
        cout << "\n--- SNR (dB) ---\n" << endl;

        for (auto& tp : snr_tests) {
            vector<double> pa, ba;
            for (auto& [key, ms] : key_mode_snr) {
                if (dl != "ALL" && key_domain[key] != dl) continue;
                auto ip = ms.find(tp.proposed), ib = ms.find(tp.baseline);
                if (ip != ms.end() && ib != ms.end()) {
                    pa.push_back(ip->second);
                    ba.push_back(ib->second);
                }
            }
            if (pa.empty()) continue;

            auto w = wilcoxon(pa, ba);
            double cd = cohens_d(pa, ba);
            double md = 0;
            for (size_t i = 0; i < pa.size(); ++i) md += pa[i] - ba[i];
            md /= pa.size();

            string sig_l = (w.p_value < 0.001) ? "***" : (w.p_value < 0.01) ? "**" : (w.p_value < 0.05) ? "*" : "n.s.";
            string eff = (fabs(cd) >= 0.8) ? "large" : (fabs(cd) >= 0.5) ? "medium" : (fabs(cd) >= 0.2) ? "small" : "negl.";
            string winner = (md > 0) ? tp.proposed : tp.baseline;

            cout << "  " << tp.label << endl;
            cout << "    n=" << w.n_pairs << "  W+=" << w.n_pos << "  W-=" << w.n_neg
                 << "  p=" << scientific << setprecision(4) << w.p_value << "  " << sig_l << endl;
            cout << "    mean_diff=" << fixed << setprecision(4) << md << " dB"
                 << "  d=" << cd << " (" << eff << ")"
                 << "  => " << winner << "\n" << endl;

            csv << dl << "," << tp.proposed << "," << tp.baseline << ",snr_db,"
                << w.n_pairs << "," << w.n_pos << "," << w.n_neg << ","
                << fixed << setprecision(4) << w.W << "," << w.z << ","
                << scientific << w.p_value << ","
                << fixed << setprecision(4) << cd << "," << md << endl;
        }

        // Spectral correlation tests
        if (dl == "vibration" || dl == "ALL") {
            cout << "--- Spectral Correlation ---\n" << endl;

            vector<TP> spec_tests = {
                {"IMP_INTERP_SPECTRAL", "SDT_MATCHED",  "SPECTRAL vs SDT (spec)"},
                {"IMP_INTERP_SPECTRAL", "PLA_MATCHED",  "SPECTRAL vs PLA (spec)"},
                {"IMP_INTERP_SPECTRAL", "LTC_MATCHED",  "SPECTRAL vs LTC (spec)"},
                {"IMP_INTERP_SPECTRAL", "RDP_OFFLINE",  "SPECTRAL vs RDP (spec)"},
                {"IMP_INTERP_SPECTRAL", "IMP_INTERP_ERROR", "SPECTRAL vs INTERP_ERROR (spec)"},
                {"IMP_INTERP_ERROR",    "SDT_MATCHED",  "INTERP_ERROR vs SDT (spec)"},
                {"IMP_INTERP_ERROR",    "PLA_MATCHED",  "INTERP_ERROR vs PLA (spec)"},
                {"IMP_INTERP_ERROR",    "RDP_OFFLINE",  "INTERP_ERROR vs RDP (spec)"},
            };

            for (auto& tp : spec_tests) {
                vector<double> pa, ba;
                for (auto& [key, ms] : key_mode_spec) {
                    if (dl != "ALL" && key_domain[key] != dl) continue;
                    auto ip = ms.find(tp.proposed), ib = ms.find(tp.baseline);
                    if (ip != ms.end() && ib != ms.end()) {
                        pa.push_back(ip->second);
                        ba.push_back(ib->second);
                    }
                }
                if (pa.empty()) continue;

                auto w = wilcoxon(pa, ba);
                double cd = cohens_d(pa, ba);
                double md = 0;
                for (size_t i = 0; i < pa.size(); ++i) md += pa[i] - ba[i];
                md /= pa.size();

                string sig_l = (w.p_value < 0.001) ? "***" : (w.p_value < 0.01) ? "**" : (w.p_value < 0.05) ? "*" : "n.s.";

                cout << "  " << tp.label << endl;
                cout << "    n=" << w.n_pairs
                     << "  p=" << scientific << setprecision(4) << w.p_value << "  " << sig_l
                     << "  mean_diff=" << fixed << setprecision(6) << md
                     << "  d=" << setprecision(4) << cd << "\n" << endl;

                csv << dl << "," << tp.proposed << "," << tp.baseline << ",spectral_corr,"
                    << w.n_pairs << "," << w.n_pos << "," << w.n_neg << ","
                    << fixed << setprecision(4) << w.W << "," << w.z << ","
                    << scientific << w.p_value << ","
                    << fixed << setprecision(6) << cd << "," << md << endl;
            }
        }
    }

    csv.close();
    cout << "Output: ../results/novelty_wilcoxon.csv" << endl;
    return 0;
}
