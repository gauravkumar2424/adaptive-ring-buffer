// ============================================================
// spectral_eval_v2.cpp  --  S6 + Phase 2: adds SDT to the
//                           envelope-spectrum diagnostic study.
//
// THE QUESTION
//   SDT beats the proposed method on SNR, even when its output queue
//   is bounded and overflowing (Experiment A, section 5). But SNR
//   rewards small pointwise deviation, which is precisely what SDT's
//   swinging-door test minimises. It says nothing about whether the
//   PERIODIC IMPACT STRUCTURE -- the thing a diagnostician reads --
//   survives compression.
//
//   Every comparison where interpolation-error eviction looked strong
//   (dz ~ 2.0, 100% win over V-W / LTTB / FIFO) was on the envelope
//   spectrum. SDT has never been measured there. This closes that gap.
//
//   If SDT preserves fault signatures as well as it preserves SNR, the
//   quality argument against it is gone and the paper rests on the
//   epsilon-guarantee result plus the O(log N) architecture. If it does
//   not, then pointwise fidelity and diagnostic fidelity are different
//   properties, and only one of them matters to the end user. Either
//   answer is publishable. Only one of them is currently claimed.
//
// TWO SDT VARIANTS
//   SDT          unbounded output queue -- the charitable case,
//                what the SDT literature implicitly assumes
//   SDT_BOUNDED  output queue capped at the buffer budget N, FIFO
//                drop on overflow -- what actually runs on an MCU
//
// FAIRNESS
//   epsilon is binary-searched per (recording, N, overload) so SDT's
//   emission count matches the proposed method's retained count. Same
//   budget, same drain schedule, same reconstruction.
//
// OUTPUT: results/spectral_survivors.csv   (overwrites v1)
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
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

// ------------------------------------------------------------
// Offline oracles: identical greedy loop, different score.
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// Swinging-Door Trending with an explicit output queue.
// cap == 0 means unbounded. Consumer drains one queued point every
// `overload` input samples -- the same schedule the proposed buffer
// sees, so the comparison is like for like.
// ------------------------------------------------------------
struct SdtOut { vector<int> idx; long emitted = 0; long dropped = 0; };

static SdtOut sdt_stream(const vector<double>& sig, double eps,
                         size_t cap, int overload)
{
    SdtOut R;
    const int n = (int)sig.size();
    deque<pair<int,double>> q;
    int last_idx = -1;  double last_val = 0.0;
    int prev_idx = -1;  double prev_val = 0.0;
    double shi = 1e30, slo = -1e30;

    auto push = [&](int i, double v) {
        if (cap > 0 && q.size() >= cap) { q.pop_front(); ++R.dropped; }
        q.push_back({i, v});
        ++R.emitted;
    };

    for (int i = 0; i < n; ++i) {
        double v = sig[i];
        if (last_idx < 0) {
            push(i, v); last_idx = i; last_val = v;
            shi = 1e30; slo = -1e30;
        } else {
            double dt = (double)(i - last_idx);
            double dv = v - last_val;
            double su = (dv + eps) / dt, sl = (dv - eps) / dt;
            if (su < shi) shi = su;
            if (sl > slo) slo = sl;
            if (slo > shi && prev_idx >= 0) {
                push(prev_idx, prev_val);
                last_idx = prev_idx; last_val = prev_val;
                double d2 = (double)(i - last_idx);
                if (d2 > 0) {
                    double dv2 = v - last_val;
                    shi = (dv2 + eps) / d2; slo = (dv2 - eps) / d2;
                } else { shi = 1e30; slo = -1e30; }
            }
        }
        prev_idx = i; prev_val = v;

        if (overload > 0 && ((i + 1) % overload) == 0 && !q.empty()) {
            R.idx.push_back(q.front().first);
            q.pop_front();
        }
    }
    if (prev_idx >= 0 && (q.empty() || q.back().first != prev_idx))
        push(prev_idx, prev_val);
    while (!q.empty()) { R.idx.push_back(q.front().first); q.pop_front(); }

    sort(R.idx.begin(), R.idx.end());
    R.idx.erase(unique(R.idx.begin(), R.idx.end()), R.idx.end());
    return R;
}

// Binary-search epsilon so SDT emits `target` points (unbounded run).
static double tune_epsilon(const vector<double>& sig, long target, int overload) {
    double lo = 1e-9, hi = 1.0;
    for (int k = 0; k < 60 && hi < 1e9; ++k) {
        if (sdt_stream(sig, hi, 0, overload).emitted <= target) break;
        hi *= 2.0;
    }
    for (int it = 0; it < 50; ++it) {
        double mid = 0.5 * (lo + hi);
        if (sdt_stream(sig, mid, 0, overload).emitted > target) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ------------------------------------------------------------
struct VibRec { string name; vector<double> data; };

static vector<VibRec> load_from_manifest(const string& dir, int nmax) {
    vector<VibRec> out;
    ifstream man(dir + "/manifest.csv");
    if (!man.is_open()) {
        cerr << "ERROR: cannot open " << dir << "/manifest.csv\n";
        return out;
    }
    string line; getline(man, line);
    while (getline(man, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string name; getline(ss, name, ',');
        ifstream f(dir + "/vib_" + name + ".txt");
        if (!f.is_open()) continue;
        VibRec r; r.name = name; r.data.reserve(nmax);
        double v;
        while (f >> v && (int)r.data.size() < nmax) r.data.push_back(v);
        if ((int)r.data.size() == nmax) out.push_back(std::move(r));
    }
    return out;
}

int main() {
    cout << "=== Spectral survivor dump v2 (SDT included) ===\n";

    auto recs = load_from_manifest("../data/cwru-bearing", WINDOW);
    if (recs.empty()) { cerr << "No recordings loaded.\n"; return 1; }
    cout << "Loaded " << recs.size() << " recordings (" << WINDOW << " samples)\n";

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
    double worst_cr_gap = 0.0;
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

                // ---- SDT at matched budget ----
                double eps = tune_epsilon(r.data, (long)expect, ol);
                auto su = sdt_stream(r.data, eps, 0,  ol);   // unbounded queue
                auto sb = sdt_stream(r.data, eps, bs, ol);   // capped at budget
                emit(r.name, "SDT",         bs, ol, su.idx);
                emit(r.name, "SDT_BOUNDED", bs, ol, sb.idx);
                runs += 2;

                // Budget matching is integer-quantised, so track the worst
                // mismatch: if SDT ends up with materially more retained
                // samples than IE, it is not a fair comparison.
                double gap = fabs((double)su.idx.size() - (double)expect)
                           / (double)expect;
                worst_cr_gap = max(worst_cr_gap, gap);
            }
        }
        cout << "  " << r.name << " (" << runs << " runs)\n";
    }
    csv.close();

    cout << "\nRuns: " << runs << "\n";
    cout << "Drop-count mismatches (online modes): " << mism
         << (mism ? "   <<< INVALID, STOP" : "   (matched)") << "\n";
    cout << "Worst SDT budget mismatch: " << (100.0 * worst_cr_gap) << "%"
         << (worst_cr_gap > 0.05 ? "   <<< CHECK FAIRNESS" : "   (acceptable)") << "\n";
    cout << "Time: "
         << chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count()
         << " s\nOutput: results/spectral_survivors.csv\n";
    return 0;
}
