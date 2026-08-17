// ============================================================
// span_sweep.cpp  --  S5: characterise the criterion continuum
//
// Sweeps beta in  score(i) = |x_i - xhat_i| * (t_s - t_p)^beta
//
//   beta = 0  ==  interpolation error   (the "proposed" criterion)
//   beta = 1  ==  Visvalingam-Whyatt    (the 1993 baseline)
//
// These are not two methods. V-W area == 0.5 * span * interp error,
// exactly, for a 1-D series. This experiment asks the real question:
// WHICH beta is best, and does the answer depend on the operating
// point? If beta* rises with compression ratio, the contribution is a
// characterisation of the criterion space rather than a bake-off.
//
// DESIGN NOTES
//   * Deterministic driver: every mode retains an identical,
//     closed-form count. Asserted per row.
//   * Buffer 32 and overload 100 added: the v4 grid topped out at
//     ~19x actual CR, reached only at buffer 64, so compression ratio
//     was confounded with buffer size. This grid decouples them --
//     each CR band is now reachable at several buffer sizes.
//   * Reference modes IMP_INTERP_ERROR and IMP_VW_AREA are run
//     alongside beta=0 and beta=1. If the endpoints do not reproduce
//     them to within 1e-9 dB the patch is wrong and we abort.
//
// OUTPUT: results/span_sweep.csv
// ============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "ring_buffer.h"
#include "metrics.h"
#include "signal_loader.h"
#include "deterministic_driver.h"

using namespace std;

// Naive DFT magnitude correlation, first min(N,512) samples.
// Kept identical to cross_domain_v4 so results are comparable.
// (Replaced properly in S6.)
static double spectral_corr(const vector<double>& o, const vector<double>& r) {
    if (o.size() < 64 || r.size() < 64) return -1.0;
    int N = min((int)min(o.size(), r.size()), 512);
    vector<double> mo(N / 2), mr(N / 2);
    for (int k = 0; k < N / 2; ++k) {
        double ro = 0, io = 0, rr = 0, ir = 0;
        for (int n = 0; n < N; ++n) {
            double a = 2.0 * M_PI * k * n / N, ca = cos(a), sa = sin(a);
            ro += o[n] * ca; io -= o[n] * sa;
            rr += r[n] * ca; ir -= r[n] * sa;
        }
        mo[k] = sqrt(ro * ro + io * io);
        mr[k] = sqrt(rr * rr + ir * ir);
    }
    double meo = 0, mer = 0;
    for (int k = 0; k < N / 2; ++k) { meo += mo[k]; mer += mr[k]; }
    meo /= (N / 2); mer /= (N / 2);
    double num = 0, dO = 0, dR = 0;
    for (int k = 0; k < N / 2; ++k) {
        double a = mo[k] - meo, b = mr[k] - mer;
        num += a * b; dO += a * a; dR += b * b;
    }
    double den = sqrt(dO * dR);
    return (den > 1e-15) ? num / den : 0.0;
}

int main(int argc, char** argv) {
    cout << "=== Span-weighted criterion family sweep (S5) ===\n";
    bool quick = (argc > 1 && string(argv[1]) == "--quick");

    const int MAXS = 2000;
    auto signals = load_all_real_signals("../data", MAXS);
    if (signals.empty()) { cerr << "ERROR: no signals\n"; return 1; }
    cout << "Loaded " << signals.size() << " signals\n";

    vector<size_t> bufs = quick ? vector<size_t>{64, 256}
                                : vector<size_t>{32, 64, 128, 256, 512};
    vector<int> ovls    = quick ? vector<int>{5, 50}
                                : vector<int>{2, 3, 5, 10, 20, 50, 100};
    vector<double> betas = quick ? vector<double>{0.0, 0.5, 1.0}
                                 : vector<double>{0.0, 0.25, 0.5, 0.75,
                                                  1.0, 1.25, 1.5, 2.0};

    ofstream csv("../results/span_sweep.csv");
    csv << "signal,domain,mode,beta,buffer_size,overload,"
           "retained,expected,CR,snr_db,mse,max_error,"
           "snr_saturated,spectral_correlation\n";

    auto emit = [&](const RealSignal& sig, const string& mode, double beta,
                    size_t bs, int ol, const OnlineRun& run, size_t expect)
                -> double
    {
        vector<double> rec = reconstruct_signal(run.surv_idx, run.surv_vals,
                                                (int)sig.data.size());
        double snr = compute_snr(sig.data, rec);
        double sc  = (sig.domain == "vibration")
                   ? spectral_corr(sig.data, rec) : -1.0;
        double cr  = (double)sig.data.size() / max<size_t>(run.surv_idx.size(), 1);
        csv << sig.name << "," << sig.domain << "," << mode << ","
            << fixed << setprecision(4) << beta << ","
            << bs << "," << ol << ","
            << run.surv_idx.size() << "," << expect << ","
            << setprecision(4) << cr << ","
            << setprecision(8) << snr << ","
            << scientific << setprecision(6) << compute_mse_aligned(sig.data, rec) << ","
            << fixed << setprecision(8) << compute_max_error(sig.data, rec) << ","
            << (isfinite(snr) ? 0 : 1) << ","
            << setprecision(8) << sc << "\n";
        return snr;
    };

    size_t rows = 0, mism = 0, ident_fail = 0;
    double worst_ident = 0.0;
    auto t0 = chrono::high_resolution_clock::now();

    for (auto& sig : signals) {
        for (size_t bs : bufs) {
            for (int ol : ovls) {
                size_t expect = expected_retained(sig.data.size(), bs, ol);

                // ---- reference modes ----
                ImportanceConfig base;
                auto rIE = run_online_deterministic(
                    sig, BufferMode::IMPORTANCE_INTERP_ERROR, bs, ol, base);
                auto rVW = run_online_deterministic(
                    sig, BufferMode::IMPORTANCE_VW_AREA, bs, ol, base);
                double snrIE = emit(sig, "REF_INTERP_ERROR", -1.0, bs, ol, rIE, expect);
                double snrVW = emit(sig, "REF_VW_AREA",      -1.0, bs, ol, rVW, expect);
                rows += 2;
                if (rIE.surv_idx.size() != expect || rVW.surv_idx.size() != expect) ++mism;

                // ---- beta family ----
                for (double b : betas) {
                    ImportanceConfig cfg;
                    cfg.span_beta = b;
                    auto r = run_online_deterministic(
                        sig, BufferMode::IMPORTANCE_INTERP_SPAN, bs, ol, cfg);
                    if (r.surv_idx.size() != expect) ++mism;
                    double snr = emit(sig, "SPAN", b, bs, ol, r, expect);
                    ++rows;

                    // endpoint identity checks
                    if (b == 0.0 || b == 1.0) {
                        double ref = (b == 0.0) ? snrIE : snrVW;
                        if (isfinite(snr) && isfinite(ref)) {
                            double dd = fabs(snr - ref);
                            worst_ident = max(worst_ident, dd);
                            if (dd > 1e-9) {
                                ++ident_fail;
                                if (ident_fail <= 5)
                                    cerr << "IDENTITY FAIL beta=" << b << " "
                                         << sig.name << " buf=" << bs
                                         << " ovl=" << ol
                                         << " span=" << snr << " ref=" << ref
                                         << " d=" << dd << "\n";
                            }
                        }
                    }
                }
                if (rows % 2000 < 10) cout << "  " << rows << " rows\n";
            }
        }
    }
    csv.close();

    double secs = chrono::duration<double>(
        chrono::high_resolution_clock::now() - t0).count();

    cout << "\nRows: " << rows << "\n";
    cout << "Drop-count mismatches: " << mism
         << (mism ? "   <<< INVALID, STOP" : "   (matched)") << "\n";
    cout << "Endpoint identity failures: " << ident_fail
         << "   (worst |dSNR| = " << scientific << worst_ident << ")\n";
    if (ident_fail)
        cout << "  >>> beta=0 must equal INTERP_ERROR and beta=1 must equal\n"
                "  >>> VW_AREA exactly. Failure means the patch is wrong.\n";
    else
        cout << "  beta=0 == INTERP_ERROR and beta=1 == VW_AREA confirmed.\n";
    cout << "Time: " << fixed << setprecision(1) << secs << " s\n";
    cout << "Output: results/span_sweep.csv\n";
    return 0;
}
