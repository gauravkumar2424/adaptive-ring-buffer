// ============================================================
// cross_domain_v4.cpp  --  S1 + S2 fixes
//
// CHANGES FROM v3 (cross_domain.cpp):
//
// S1  Deterministic drop-count-matched driver (no threads).
//     Old version: producer/consumer threads with sleeps; retention
//     set by OS scheduling. Drop counts differed between modes in
//     89.5% of pairs (r=0.355 with SNR advantage). Now every mode
//     retains an identical, closed-form count. Asserted per row.
//
// S1b trials removed. The algorithm is deterministic; 5 trials on
//     identical input were 5 copies of one number and inflated n 5x.
//
// S1c ADAPTIVE_TIMED_WAIT removed -- without a consumer thread it
//     degenerates to DROP plus a 2 ms sleep per eviction.
//
// S2  "RDP_OFFLINE" renamed IE_ORACLE_OFFLINE. The v3 offline_rdp()
//     was NOT Douglas-Peucker (top-down recursive split on
//     perpendicular distance). It was greedy bottom-up removal by
//     minimum linear interpolation error -- i.e. the OFFLINE ORACLE
//     FOR IE'S OWN OBJECTIVE. Keeping the "RDP" label is a
//     misattribution any line-simplification reviewer will catch.
//
// S2b VW_ORACLE_OFFLINE added: identical greedy loop scored by
//     Visvalingam-Whyatt triangle area. Without this control, "IE is
//     closer to the oracle than V-W" is circular -- the oracle
//     optimises IE's criterion. Each criterion now has its own
//     matched oracle; compare the GAPS.
//
// S3  R-peak evaluation removed from C++. Three different detectors
//     existed across the codebase (0.6*max in metrics.h, 0.5/100 in
//     rpeak_eval.h, NeuroKit2 in Python) producing three different
//     sets of numbers. NeuroKit2 is the survivor; F1 is computed
//     downstream in Python from the emitted survivor indices.
//
// S4  High-compression operating points added (ovl 10..50). At
//     ovl 2-6 the true compression ratio is ~1x, which is not a
//     compression regime.
//
// OUTPUT: results/cross_domain_v4.csv
//         results/survivors/<signal>_<mode>_<buf>_<ovl>.txt  (optional)
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <queue>
#include <map>
#include <chrono>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "deterministic_driver.h"

using namespace std;

// ============================================================
// Offline oracles -- greedy bottom-up removal, matched budget.
//
// Both use the SAME loop structure and the SAME stopping rule.
// They differ ONLY in the scoring function. This is what makes
// the criterion comparison fair.
// ============================================================

enum class OracleScore { INTERP_ERROR, VW_AREA };

static vector<int> offline_oracle(const vector<double>& sig,
                                  int target_points,
                                  OracleScore which)
{
    int n = (int)sig.size();
    if (target_points >= n) {
        vector<int> all(n);
        iota(all.begin(), all.end(), 0);
        return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<char> kept(n, 1);
    vector<int> pk(n), nk(n);
    for (int i = 0; i < n; ++i) { pk[i] = i - 1; nk[i] = i + 1; }
    nk[n - 1] = -1;

    auto score = [&](int i) -> double {
        int p = pk[i], s = nk[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        if (which == OracleScore::INTERP_ERROR) {
            double span = (double)(s - p);
            if (span <= 0) return 0.0;
            double t = (double)(i - p) / span;
            double xh = sig[p] + t * (sig[s] - sig[p]);
            return fabs(sig[i] - xh);
        } else {
            // Visvalingam-Whyatt effective area, (index, value) as 2-D.
            double tp = (double)p, ti = (double)i, ts = (double)s;
            return 0.5 * fabs(tp * (sig[i] - sig[s])
                            + ti * (sig[s] - sig[p])
                            + ts * (sig[p] - sig[i]));
        }
    };

    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({score(i), i});

    int count = n;
    while (count > target_points && !pq.empty()) {
        auto top = pq.top(); pq.pop();
        int idx = top.second;
        if (!kept[idx]) continue;
        double actual = score(idx);
        if (fabs(actual - top.first) > 1e-12) { pq.push({actual, idx}); continue; }
        kept[idx] = 0;
        int p = pk[idx], s = nk[idx];
        if (p >= 0) nk[p] = s;
        if (s >= 0 && s < n) pk[s] = p;
        --count;
        if (p > 0 && kept[p]) pq.push({score(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({score(s), s});
    }

    vector<int> out;
    for (int i = 0; i < n; ++i) if (kept[i]) out.push_back(i);
    return out;
}

static vector<int> offline_lttb(const vector<double>& sig, int target_points) {
    int n = (int)sig.size();
    if (target_points >= n) {
        vector<int> all(n); iota(all.begin(), all.end(), 0); return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<int> res; res.reserve(target_points); res.push_back(0);
    double bs = (double)(n - 2) / (target_points - 2);
    int prev = 0;
    for (int b = 0; b < target_points - 2; ++b) {
        int s0 = (int)(b * bs) + 1, s1 = min((int)((b + 1) * bs) + 1, n - 1);
        int t0 = (int)((b + 1) * bs) + 1, t1 = min((int)((b + 2) * bs) + 1, n);
        double ax = 0, ay = 0; int c = 0;
        for (int i = t0; i < t1; ++i) { ax += i; ay += sig[i]; ++c; }
        if (c > 0) { ax /= c; ay /= c; }
        double best = -1; int bi = s0;
        for (int i = s0; i < s1; ++i) {
            double a = fabs((prev - ax) * (sig[i] - sig[prev])
                          - (prev - i) * (ay - sig[prev])) * 0.5;
            if (a > best) { best = a; bi = i; }
        }
        res.push_back(bi); prev = bi;
    }
    res.push_back(n - 1);
    return res;
}

// ============================================================
// Spectral correlation.
//
// NOTE (S6, not yet fixed): naive O(N^2) DFT over the FIRST 512
// samples only, no window, correlation over all magnitude bins
// including DC. Under-specified for publication. Kept identical to
// v3 here so the S1/S2 deltas are isolated. Replace in S6 with a
// Hann-windowed FFT over the full record plus an envelope-spectrum
// fault-frequency error metric.
// ============================================================
static double spectral_corr(const vector<double>& o, const vector<double>& r) {
    if (o.size() < 64 || r.size() < 64) return -1.0;
    int N = min((int)min(o.size(), r.size()), 512);
    vector<double> mo(N / 2), mr(N / 2);
    for (int k = 0; k < N / 2; ++k) {
        double ro = 0, io = 0, rr = 0, ir = 0;
        for (int n = 0; n < N; ++n) {
            double a = 2.0 * M_PI * k * n / N;
            double ca = cos(a), sa = sin(a);
            ro += o[n] * ca; io -= o[n] * sa;
            rr += r[n] * ca; ir -= r[n] * sa;
        }
        mo[k] = sqrt(ro * ro + io * io);
        mr[k] = sqrt(rr * rr + ir * ir);
    }
    double meo = accumulate(mo.begin(), mo.end(), 0.0) / mo.size();
    double mer = accumulate(mr.begin(), mr.end(), 0.0) / mr.size();
    double num = 0, dO = 0, dR = 0;
    for (int k = 0; k < N / 2; ++k) {
        double a = mo[k] - meo, b = mr[k] - mer;
        num += a * b; dO += a * a; dR += b * b;
    }
    double den = sqrt(dO * dR);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================

struct Row {
    string signal, domain, mode;
    size_t buffer_size;
    int overload;
    size_t retained, expected, drops;
    double snr, mse, max_err, deriv_ratio, spec_corr;
    bool snr_saturated;
};

static void fill_metrics(Row& r, const vector<double>& orig,
                         const vector<int>& si, const vector<double>& sv)
{
    vector<double> rec = reconstruct_signal(si, sv, (int)orig.size());
    r.snr = compute_snr(orig, rec);
    r.mse = compute_mse_aligned(orig, rec);
    r.max_err = compute_max_error(orig, rec);
    r.deriv_ratio = compute_deriv_ratio(orig, rec);
    r.snr_saturated = !isfinite(r.snr);
    r.spec_corr = (r.domain == "vibration") ? spectral_corr(orig, rec) : -1.0;
}

int main(int argc, char** argv) {
    cout << "=== Cross-Domain v4 (deterministic, oracle-controlled) ===\n";

    bool quick = (argc > 1 && string(argv[1]) == "--quick");
    const int MAXS = 2000;

    auto signals = load_all_real_signals("../data", MAXS);
    if (signals.empty()) { cerr << "ERROR: no signals loaded\n"; return 1; }
    cout << "Loaded " << signals.size() << " signals\n";

    vector<size_t> bufs = quick ? vector<size_t>{128, 256}
                                : vector<size_t>{64, 128, 256, 512};
    // S4: low-compression points kept for continuity with v3;
    //     high-compression points added because CR ~= 1x at ovl 2-6.
    vector<int> ovls = quick ? vector<int>{3, 20}
                             : vector<int>{2, 3, 4, 5, 6, 10, 20, 30, 50};

    struct ME { string name; BufferMode mode; };
    vector<ME> online = {
        {"DROP",                 BufferMode::DROP},
        {"DROP_MIDDLE",          BufferMode::DROP_MIDDLE},
        {"DROP_LOW_VARIANCE",    BufferMode::DROP_LOW_VARIANCE},
        {"LEGACY_IMPORTANCE",    BufferMode::ADAPTIVE_IMPORTANCE},
        {"IMP_FIRST_ORDER",      BufferMode::IMPORTANCE_FIRST_ORDER},
        {"IMP_SECOND_ORDER",     BufferMode::IMPORTANCE_SECOND_ORDER},
        {"IMP_WINDOWED_ENERGY",  BufferMode::IMPORTANCE_WINDOWED_ENERGY},
        {"IMP_COMPOSITE",        BufferMode::IMPORTANCE_COMPOSITE},
        {"IMP_ADAPTIVE",         BufferMode::IMPORTANCE_ADAPTIVE},
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
        {"IMP_VW_AREA",          BufferMode::IMPORTANCE_VW_AREA},
    };
    // NOTE: RANDOM_DROP omitted -- ring_buffer.h seeds it from
    // std::random_device, which is nondeterministic. Re-add only
    // after patching that line to a fixed seed.

    ofstream csv("../results/cross_domain_v4.csv");
    csv << "signal,domain,mode,buffer_size,overload,retained,expected,drops,"
           "snr_db,mse,max_error,deriv_ratio,snr_saturated,spectral_correlation\n";

    auto emit = [&](const Row& r) {
        csv << r.signal << "," << r.domain << "," << r.mode << ","
            << r.buffer_size << "," << r.overload << ","
            << r.retained << "," << r.expected << "," << r.drops << ","
            << fixed << setprecision(6) << r.snr << ","
            << scientific << setprecision(6) << r.mse << ","
            << fixed << setprecision(6) << r.max_err << ","
            << r.deriv_ratio << "," << (r.snr_saturated ? 1 : 0) << ","
            << setprecision(6) << r.spec_corr << "\n";
    };

    size_t done = 0, mismatches = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& sig : signals) {
        for (size_t bs : bufs) {
            for (int ol : ovls) {
                size_t expect = expected_retained(sig.data.size(), bs, ol);

                for (auto& m : online) {
                    auto run = run_online_deterministic(sig, m.mode, bs, ol);

                    // HARD CHECK: this is the S1 invariant. If it ever
                    // fires, the comparison is not drop-count matched
                    // and no downstream statistic is valid.
                    if (run.surv_idx.size() != expect) {
                        cerr << "MISMATCH " << sig.name << " " << m.name
                             << " buf=" << bs << " ovl=" << ol
                             << " got=" << run.surv_idx.size()
                             << " expected=" << expect << "\n";
                        ++mismatches;
                    }

                    Row r;
                    r.signal = sig.name; r.domain = sig.domain; r.mode = m.name;
                    r.buffer_size = bs; r.overload = ol;
                    r.retained = run.surv_idx.size(); r.expected = expect;
                    r.drops = run.drops;
                    fill_metrics(r, sig.data, run.surv_idx, run.surv_vals);
                    emit(r); ++done;
                }

                // ---- Offline references, matched to the SAME budget ----
                int target = max(2, (int)expect);
                struct OF { string name; int kind; };
                vector<OF> offs = {
                    {"IE_ORACLE_OFFLINE", 0},
                    {"VW_ORACLE_OFFLINE", 1},
                    {"LTTB_OFFLINE",      2},
                };
                for (auto& o : offs) {
                    vector<int> sel;
                    if (o.kind == 0) sel = offline_oracle(sig.data, target, OracleScore::INTERP_ERROR);
                    else if (o.kind == 1) sel = offline_oracle(sig.data, target, OracleScore::VW_AREA);
                    else sel = offline_lttb(sig.data, target);

                    vector<double> sv; sv.reserve(sel.size());
                    for (int i : sel) sv.push_back(sig.data[i]);

                    Row r;
                    r.signal = sig.name; r.domain = sig.domain; r.mode = o.name;
                    r.buffer_size = bs; r.overload = ol;
                    r.retained = sel.size(); r.expected = expect;
                    r.drops = sig.data.size() - sel.size();
                    fill_metrics(r, sig.data, sel, sv);
                    emit(r); ++done;
                }

                if (done % 500 == 0) cout << "  " << done << " rows\n";
            }
        }
    }
    csv.close();

    double secs = chrono::duration<double>(
        chrono::high_resolution_clock::now() - t0).count();

    cout << "\nRows: " << done << "\n";
    cout << "Drop-count mismatches: " << mismatches
         << (mismatches ? "   <<< INVESTIGATE BEFORE USING RESULTS" : "   (matched)") << "\n";
    cout << "Time: " << fixed << setprecision(1) << secs << " s\n";
    cout << "Output: results/cross_domain_v4.csv\n";
    return 0;
}
