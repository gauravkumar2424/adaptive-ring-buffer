// ============================================================
// memory_contract.cpp  --  Phase 2, Experiment A
//
// THE CLAIM UNDER TEST
//   SDT / LTC bound ERROR and let MEMORY float.
//   The proposed buffer bounds MEMORY and lets ERROR float.
//   Under a fixed SRAM budget with a consumer that may be late, only
//   the second contract is implementable.
//
// WHY THE NAIVE COMPARISON IS UNFAIR TO US
//   Prior reports compared SDT and the proposed method at matched
//   output counts and concluded "SDT ties on ECG SNR and costs 151
//   cycles, so why use a heap?" That comparison silently grants SDT
//   unbounded output memory. It emits at a DATA-DEPENDENT rate: tune
//   epsilon so the AVERAGE rate matches the drain rate and it still
//   bursts, because a transient produces a run of closely spaced
//   segment endpoints. Those queue. With a bounded queue they
//   overflow, and the epsilon guarantee is void.
//
// WHAT IS MEASURED, per (signal, buffer N, overload r)
//   max_occupancy   queue SDT needs for zero loss (unbounded run).
//                   This is the headline: it is data-dependent and
//                   cannot be bounded a priori.
//   overflow        events when the queue is capped at N
//   FIFO policy     drop oldest queued point -> DATA LOST
//   STALL policy    emitter blocks, segment extends past epsilon
//                   -> GUARANTEE VOID (max_dev / epsilon reported)
//   Both policies are simulated because choosing one invites the
//   objection that we chose the unflattering one. SDT escapes
//   neither: it loses data or it loses its bound.
//
//   The proposed method: overflow is 0 BY CONSTRUCTION and the
//   retained count is closed-form. That is the entire point.
//
// FAIRNESS
//   epsilon is binary-searched per (signal, N, r) so SDT's total
//   emission equals the proposed method's retained count exactly.
//   Same budget, same drain schedule, same reconstruction, same SNR.
//
// OUTPUT: results/memory_contract.csv
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <limits>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "deterministic_driver.h"

using namespace std;

enum class Policy { UNBOUNDED, FIFO_DROP, STALL };

struct StreamResult {
    long emitted = 0;         // points the compressor produced
    long delivered = 0;       // points that reached the consumer + residual
    long max_occupancy = 0;   // peak queue depth
    long overflow_events = 0;
    long dropped = 0;         // FIFO policy: points lost
    long stalled = 0;         // STALL policy: samples the emitter was blocked
    double max_dev = 0.0;     // STALL policy: worst deviation from the
                              // anchor line, to compare against epsilon
    vector<int> idx;
    vector<double> val;
};

// ------------------------------------------------------------
// Swinging-Door Trending with an explicit bounded output queue.
//
// Consumer drains one queued point every `overload` input samples,
// which is exactly the drain schedule the proposed buffer sees.
// ------------------------------------------------------------
static StreamResult sdt_stream(const vector<double>& sig, double eps,
                               size_t cap, int overload, Policy pol)
{
    StreamResult R;
    const int n = (int)sig.size();
    deque<pair<int,double>> q;

    int    last_idx = -1;   double last_val = 0.0;
    int    prev_idx = -1;   double prev_val = 0.0;
    double shi = 1e30, slo = -1e30;
    bool   pending = false;         // an emission is owed but blocked

    auto queue_push = [&](int i, double v) -> bool {
        if (pol != Policy::UNBOUNDED && q.size() >= cap) {
            ++R.overflow_events;
            if (pol == Policy::FIFO_DROP) {
                q.pop_front();      // oldest emitted point is lost
                ++R.dropped;
            } else {
                return false;       // STALL: refuse, emitter blocks
            }
        }
        q.push_back({i, v});
        ++R.emitted;
        if ((long)q.size() > R.max_occupancy) R.max_occupancy = (long)q.size();
        return true;
    };

    auto reanchor = [&](int idx, double val) {
        double dt = (double)(idx - last_idx);
        if (dt <= 0) { shi = 1e30; slo = -1e30; return; }
        double dv = val - last_val;
        shi = (dv + eps) / dt;
        slo = (dv - eps) / dt;
    };

    for (int i = 0; i < n; ++i) {
        double v = sig[i];

        if (last_idx < 0) {
            if (queue_push(i, v)) { last_idx = i; last_val = v; }
            shi = 1e30; slo = -1e30;
        } else {
            double dt = (double)(i - last_idx);
            double dv = v - last_val;
            double su = (dv + eps) / dt;
            double sl = (dv - eps) / dt;
            if (su < shi) shi = su;
            if (sl > slo) slo = sl;

            if (slo > shi || pending) {
                // The doors have crossed: the current segment can no
                // longer represent the data within epsilon. Emit the
                // previous point as the new anchor.
                if (prev_idx >= 0) {
                    if (queue_push(prev_idx, prev_val)) {
                        last_idx = prev_idx; last_val = prev_val;
                        reanchor(i, v);
                        pending = false;
                    } else {
                        // STALL: cannot emit. The segment is forced to
                        // extend, so the reconstruction now deviates by
                        // MORE than epsilon. Record how much.
                        pending = true;
                        ++R.stalled;
                        double span = (double)(i - last_idx);
                        if (span > 0) {
                            double t = (double)(prev_idx - last_idx) / span;
                            double xh = last_val + t * (v - last_val);
                            R.max_dev = max(R.max_dev, fabs(prev_val - xh));
                        }
                    }
                }
            }
        }
        prev_idx = i; prev_val = v;

        // Consumer: one pop per `overload` input samples.
        if (overload > 0 && ((i + 1) % overload) == 0 && !q.empty()) {
            R.idx.push_back(q.front().first);
            R.val.push_back(q.front().second);
            q.pop_front();
            ++R.delivered;
        }
    }

    // Close the final segment, then drain whatever remains -- the same
    // accounting the proposed buffer gets for its residual contents.
    if (prev_idx >= 0 && (q.empty() || q.back().first != prev_idx))
        queue_push(prev_idx, prev_val);
    while (!q.empty()) {
        R.idx.push_back(q.front().first);
        R.val.push_back(q.front().second);
        q.pop_front();
        ++R.delivered;
    }
    return R;
}

// Binary-search epsilon so SDT's emission count matches `target`.
static double tune_epsilon(const vector<double>& sig, long target, int overload) {
    double lo = 1e-9, hi = 1.0;
    for (int k = 0; k < 60; ++k) {
        auto r = sdt_stream(sig, hi, 0, overload, Policy::UNBOUNDED);
        if (r.emitted <= target) break;
        hi *= 2.0;
        if (hi > 1e9) break;
    }
    for (int it = 0; it < 60; ++it) {
        double mid = 0.5 * (lo + hi);
        auto r = sdt_stream(sig, mid, 0, overload, Policy::UNBOUNDED);
        if (r.emitted > target) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ------------------------------------------------------------
static vector<RealSignal> load_vibration(const string& dir, int nmax) {
    vector<RealSignal> out;
    ifstream man(dir + "/manifest.csv");
    if (!man.is_open()) return out;
    string line; getline(man, line);
    while (getline(man, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string name; getline(ss, name, ',');
        ifstream f(dir + "/vib_" + name + ".txt");
        if (!f.is_open()) continue;
        RealSignal s; s.name = name; s.domain = "vibration"; s.has_rpeaks = false;
        double v;
        while (f >> v && (int)s.data.size() < nmax) s.data.push_back(v);
        if ((int)s.data.size() == nmax) out.push_back(std::move(s));
    }
    return out;
}

int main() {
    cout << "=== Experiment A: bounded memory vs bounded error ===\n";
    const int NSAMP = 4000;

    auto sigs = load_all_real_signals("../data", NSAMP);      // 16 ECG (+3 old vib)
    sigs.erase(remove_if(sigs.begin(), sigs.end(),
        [](const RealSignal& s){ return s.domain != "ecg"; }), sigs.end());
    auto vib = load_vibration("../data/cwru-bearing", NSAMP); // 40 CWRU
    for (auto& v : vib) sigs.push_back(v);

    cout << "Signals: " << sigs.size() << " ("
         << count_if(sigs.begin(), sigs.end(),
                     [](const RealSignal& s){ return s.domain=="ecg"; })
         << " ECG + " << vib.size() << " vibration), "
         << NSAMP << " samples each\n";
    if (sigs.empty()) { cerr << "no signals\n"; return 1; }

    vector<size_t> bufs = {64, 128, 256, 512};
    vector<int>    ovls = {2, 5, 10, 20};

    ofstream csv("../results/memory_contract.csv");
    csv << "signal,domain,buffer_size,overload,method,policy,epsilon,"
           "emitted,delivered,max_occupancy,occupancy_ratio,overflow_events,"
           "dropped,stalled,max_dev,eps_violation_ratio,snr_db\n";

    auto emit_row = [&](const RealSignal& s, size_t bs, int ol,
                        const string& meth, const string& pol, double eps,
                        const StreamResult& R, double snr)
    {
        csv << s.name << "," << s.domain << "," << bs << "," << ol << ","
            << meth << "," << pol << ","
            << scientific << setprecision(6) << eps << ","
            << R.emitted << "," << R.delivered << ","
            << R.max_occupancy << ","
            << fixed << setprecision(4) << (double)R.max_occupancy / (double)bs << ","
            << R.overflow_events << "," << R.dropped << "," << R.stalled << ","
            << scientific << setprecision(6) << R.max_dev << ","
            << fixed << setprecision(4)
            << (eps > 0 ? R.max_dev / eps : 0.0) << ","
            << setprecision(6) << snr << "\n";
    };

    size_t rows = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& s : sigs) {
        for (size_t bs : bufs) {
            for (int ol : ovls) {
                size_t expect = expected_retained(s.data.size(), bs, ol);

                // ---- proposed: bounded memory, zero overflow ----
                auto run = run_online_deterministic(
                    s, BufferMode::IMPORTANCE_INTERP_ERROR, bs, ol);
                auto rec = reconstruct_signal(run.surv_idx, run.surv_vals,
                                              (int)s.data.size());
                StreamResult P;
                P.emitted = P.delivered = (long)run.surv_idx.size();
                P.max_occupancy = (long)bs;   // exactly the budget, never more
                emit_row(s, bs, ol, "PROPOSED", "none", 0.0, P,
                         compute_snr(s.data, rec));
                ++rows;

                // ---- SDT at matched budget ----
                double eps = tune_epsilon(s.data, (long)expect, ol);

                for (auto pol : {Policy::UNBOUNDED, Policy::FIFO_DROP, Policy::STALL}) {
                    auto R = sdt_stream(s.data, eps, bs, ol, pol);
                    auto rc = reconstruct_signal(R.idx, R.val, (int)s.data.size());
                    const char* pn = (pol == Policy::UNBOUNDED) ? "unbounded"
                                   : (pol == Policy::FIFO_DROP) ? "fifo_drop" : "stall";
                    emit_row(s, bs, ol, "SDT", pn, eps, R,
                             compute_snr(s.data, rc));
                    ++rows;
                }
            }
        }
        cout << "  " << s.name << " (" << rows << " rows)\n";
    }
    csv.close();

    cout << "\nRows: " << rows << "\nTime: "
         << chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count()
         << " s\nOutput: results/memory_contract.csv\n";
    return 0;
}
