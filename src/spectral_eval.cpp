// ============================================================
// spectral_eval.cpp  --  S6: dump survivor indices for the
//                        envelope-spectrum fault-frequency study.
//
// WHY THIS EXISTS SEPARATELY FROM cross_domain_v4
//   Frequency resolution. At 12 kHz a 2,000-sample window gives 6 Hz
//   bins, but the bearing defect frequencies sit at BPFO ~107 Hz and
//   BPFI ~162 Hz. A "fault frequency error in Hz" metric quantised to
//   6 Hz steps cannot resolve anything meaningful. 12,000 samples
//   (1.0 s) gives 1 Hz bins, which is what the claim requires.
//
// WHY IT DUMPS INDICES INSTEAD OF COMPUTING METRICS
//   Every spectral quantity is computed once, in Python, with a real
//   FFT. The previous C++ spectral_corr() was a naive O(N^2) DFT over
//   the first 512 samples with no window -- leakage-dominated and
//   under-specified. Worse, the codebase had accumulated three
//   different R-peak detectors the same way. One implementation, in
//   one language, is the fix.
//
//   Values need not be stored: survivor VALUES are exactly sig[idx],
//   so indices alone are lossless. Indices are delta-coded (they are
//   monotonically increasing) which keeps the dump around 10 MB.
//
// GRID
//   40 CWRU recordings (10 conditions x 4 loads, all 12 kHz)
//   5 modes x 3 buffer sizes x 4 overloads = 60 runs per recording
//   CR spans roughly 4x to 33x -- the diagnostic-relevant regime.
//
// OUTPUT: results/spectral_survivors.csv
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <chrono>

#include "ring_buffer.h"
#include "signal_loader.h"
#include "deterministic_driver.h"

using namespace std;

static const int WINDOW = 12000;   // 1.0 s at 12 kHz -> 1 Hz FFT bins

// ---------- offline oracles (identical loop, different score) ----------
enum class OracleScore { INTERP_ERROR, VW_AREA };

static vector<int> offline_oracle(const vector<double>& sig, int target,
                                  OracleScore which)
{
    int n = (int)sig.size();
    if (target >= n) { vector<int> a(n); iota(a.begin(), a.end(), 0); return a; }
    if (target < 2) return {0, n - 1};

    vector<char> kept(n, 1);
    vector<int> pk(n), nk(n);
    for (int i = 0; i < n; ++i) { pk[i] = i - 1; nk[i] = i + 1; }
    nk[n - 1] = -1;

    auto score = [&](int i) -> double {
        int p = pk[i], s = nk[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p);
        if (span <= 0) return 0.0;
        double t = (double)(i - p) / span;
        double ie = fabs(sig[i] - (sig[p] + t * (sig[s] - sig[p])));
        // V-W area == 0.5 * span * ie exactly; use the well-conditioned
        // form, not the determinant (which cancels catastrophically).
        return (which == OracleScore::INTERP_ERROR) ? ie : 0.5 * span * ie;
    };

    priority_queue<pair<double,int>, vector<pair<double,int>>,
                   greater<pair<double,int>>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({score(i), i});

    int cnt = n;
    while (cnt > target && !pq.empty()) {
        auto top = pq.top(); pq.pop();
        int i = top.second;
        if (!kept[i]) continue;
        double a = score(i);
        if (fabs(a - top.first) > 1e-12) { pq.push({a, i}); continue; }
        kept[i] = 0;
        int p = pk[i], s = nk[i];
        if (p >= 0) nk[p] = s;
        if (s >= 0 && s < n) pk[s] = p;
        --cnt;
        if (p > 0 && kept[p]) pq.push({score(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({score(s), s});
    }
    vector<int> out;
    for (int i = 0; i < n; ++i) if (kept[i]) out.push_back(i);
    return out;
}

static vector<int> offline_lttb(const vector<double>& sig, int target) {
    int n = (int)sig.size();
    if (target >= n) { vector<int> a(n); iota(a.begin(), a.end(), 0); return a; }
    if (target < 2) return {0, n - 1};
    vector<int> res; res.reserve(target); res.push_back(0);
    double bs = (double)(n - 2) / (target - 2);
    int prev = 0;
    for (int b = 0; b < target - 2; ++b) {
        int s0 = (int)(b*bs)+1, s1 = min((int)((b+1)*bs)+1, n-1);
        int t0 = (int)((b+1)*bs)+1, t1 = min((int)((b+2)*bs)+1, n);
        double ax=0, ay=0; int c=0;
        for (int i=t0;i<t1;++i){ ax+=i; ay+=sig[i]; ++c; }
        if (c) { ax/=c; ay/=c; }
        double best=-1; int bi=s0;
        for (int i=s0;i<s1;++i) {
            double a = fabs((prev-ax)*(sig[i]-sig[prev])
                          - (prev-i)*(ay-sig[prev]))*0.5;
            if (a>best){best=a;bi=i;}
        }
        res.push_back(bi); prev=bi;
    }
    res.push_back(n-1);
    return res;
}

// ---------- manifest-driven loader ----------
struct VibRec { string name; vector<double> data; };

static vector<VibRec> load_from_manifest(const string& dir, int nmax) {
    vector<VibRec> out;
    ifstream man(dir + "/manifest.csv");
    if (!man.is_open()) {
        cerr << "ERROR: cannot open " << dir << "/manifest.csv\n"
             << "Run scripts/extract_cwru.py first.\n";
        return out;
    }
    string line;
    getline(man, line);                      // header
    while (getline(man, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string name;
        getline(ss, name, ',');              // first column is `name`
        ifstream f(dir + "/vib_" + name + ".txt");
        if (!f.is_open()) { cerr << "  WARN: missing vib_" << name << ".txt\n"; continue; }
        VibRec r; r.name = name; r.data.reserve(nmax);
        double v;
        while (f >> v && (int)r.data.size() < nmax) r.data.push_back(v);
        if ((int)r.data.size() < nmax) {
            cerr << "  WARN: " << name << " only " << r.data.size()
                 << " samples, need " << nmax << " -- skipped\n";
            continue;
        }
        out.push_back(std::move(r));
    }
    return out;
}

int main() {
    cout << "=== Spectral survivor dump (S6) ===\n";

    auto recs = load_from_manifest("../data/cwru-bearing", WINDOW);
    if (recs.empty()) { cerr << "No recordings loaded.\n"; return 1; }
    cout << "Loaded " << recs.size() << " vibration recordings ("
         << WINDOW << " samples each)\n";

    vector<size_t> bufs = {128, 256, 512};
    vector<int>    ovls = {5, 10, 20, 50};

    struct ME { string name; BufferMode mode; };
    vector<ME> online = {
        {"IE",   BufferMode::IMPORTANCE_INTERP_ERROR},
        {"VW",   BufferMode::IMPORTANCE_VW_AREA},
        {"DROP", BufferMode::DROP},
    };

    ofstream csv("../results/spectral_survivors.csv");
    csv << "signal,mode,buffer_size,overload,retained,idx_deltas\n";

    // Indices are monotonically increasing, so store first-difference.
    auto emit = [&](const string& sig, const string& mode, size_t bs, int ol,
                    const vector<int>& idx)
    {
        csv << sig << "," << mode << "," << bs << "," << ol << ","
            << idx.size() << ",";
        int prev = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (i) csv << ' ';
            csv << (idx[i] - prev);
            prev = idx[i];
        }
        csv << "\n";
    };

    size_t runs = 0, mism = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& r : recs) {
        RealSignal sig;
        sig.name = r.name; sig.domain = "vibration";
        sig.data = r.data; sig.has_rpeaks = false;

        for (size_t bs : bufs) {
            for (int ol : ovls) {
                size_t expect = expected_retained(r.data.size(), bs, ol);

                for (auto& m : online) {
                    auto run = run_online_deterministic(sig, m.mode, bs, ol);
                    if (run.surv_idx.size() != expect) ++mism;
                    emit(r.name, m.name, bs, ol, run.surv_idx);
                    ++runs;
                }

                int target = max(2, (int)expect);
                emit(r.name, "IE_ORACLE", bs, ol,
                     offline_oracle(r.data, target, OracleScore::INTERP_ERROR));
                emit(r.name, "LTTB", bs, ol, offline_lttb(r.data, target));
                runs += 2;
            }
        }
        cout << "  " << r.name << " done (" << runs << " runs)\n";
    }
    csv.close();

    double secs = chrono::duration<double>(
        chrono::high_resolution_clock::now() - t0).count();
    cout << "\nRuns: " << runs << "\n";
    cout << "Drop-count mismatches: " << mism
         << (mism ? "   <<< INVALID, STOP" : "   (matched)") << "\n";
    cout << "Time: " << secs << " s\n";
    cout << "Output: results/spectral_survivors.csv\n";
    return 0;
}
