// ============================================================
// task2_task3.cpp — HAR Classification + Long Duration Evaluation
// DAC 2027
//
// Part A: UCI HAR 6-class activity classification under compression
//   - 6 classes, 50 windows each, 128 samples/window, 50 Hz
//   - Train on uncompressed, test on compressed
//   - Compression: buf_size=32, overloads 2..20
//
// Part B: Long-duration SNR + spectral on CWRU/ECG at 10,000 samples
//   - 5x longer than the 2,000-sample experiments
//   - Same methods, same metrics
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/task2_task3 task2_task3.cpp
// Run:   cd ../build && ./task2_task3
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
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

using namespace std;

// ============================================================
// Offline baselines
// ============================================================
vector<int> offline_rdp(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) { vector<int> all(n); iota(all.begin(), all.end(), 0); return all; }
    if (target_points < 2) return {0, n - 1};
    vector<bool> kept(n, true);
    vector<int> prev_kept(n), next_kept(n);
    for (int i = 0; i < n; ++i) { prev_kept[i] = i - 1; next_kept[i] = i + 1; }
    next_kept[n - 1] = -1;
    auto ce = [&](int i) -> double {
        int p = prev_kept[i], s = next_kept[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p);
        if (span <= 0) return 0.0;
        double t = (double)(i - p) / span;
        return abs(signal[i] - (signal[p] + t * (signal[s] - signal[p])));
    };
    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({ce(i), i});
    int cc = n;
    while (cc > target_points && !pq.empty()) {
        auto [err, idx] = pq.top(); pq.pop();
        if (!kept[idx]) continue;
        double actual = ce(idx);
        if (abs(actual - err) > 1e-12) { pq.push({actual, idx}); continue; }
        kept[idx] = false;
        int p = prev_kept[idx], s = next_kept[idx];
        if (p >= 0) next_kept[p] = s;
        if (s >= 0 && s < n) prev_kept[s] = p;
        --cc;
        if (p > 0 && kept[p]) pq.push({ce(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({ce(s), s});
    }
    vector<int> result;
    for (int i = 0; i < n; ++i) if (kept[i]) result.push_back(i);
    return result;
}

// ============================================================
// DFT magnitude spectrum
// ============================================================
vector<double> compute_mag_spectrum(const vector<double>& signal) {
    int n = (int)signal.size(), half = n / 2;
    vector<double> mag(half);
    for (int k = 0; k < half; ++k) {
        double re = 0, im = 0;
        for (int i = 0; i < n; ++i) {
            double angle = 2.0 * M_PI * k * i / n;
            re += signal[i] * cos(angle);
            im -= signal[i] * sin(angle);
        }
        mag[k] = sqrt(re * re + im * im);
    }
    return mag;
}

double spec_corr(const vector<double>& a, const vector<double>& b) {
    int n = min(a.size(), b.size());
    if (n < 4) return -1.0;
    double ma = accumulate(a.begin(), a.begin()+n, 0.0) / n;
    double mb = accumulate(b.begin(), b.begin()+n, 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < n; ++k) {
        double xa = a[k] - ma, xb = b[k] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    double den = sqrt(da * db);
    return (den > 1e-15) ? num / den : 0.0;
}

// ============================================================
// Feature extraction for classification
// ============================================================
vector<double> extract_features(const vector<double>& signal, int n_bins) {
    auto mag = compute_mag_spectrum(signal);
    int half = (int)mag.size();
    vector<double> features(n_bins, 0.0);
    double bin_width = max(1.0, (double)half / n_bins);
    for (int b = 0; b < n_bins; ++b) {
        int start = (int)(b * bin_width);
        int end = min((int)((b + 1) * bin_width), half);
        for (int k = start; k < end; ++k) features[b] += mag[k];
    }
    // RMS
    double rms = 0;
    for (double v : signal) rms += v * v;
    rms = sqrt(rms / signal.size());
    features.push_back(rms);
    // Kurtosis
    double mean = accumulate(signal.begin(), signal.end(), 0.0) / signal.size();
    double m2 = 0, m4 = 0;
    for (double v : signal) { double d = v - mean; m2 += d*d; m4 += d*d*d*d; }
    m2 /= signal.size(); m4 /= signal.size();
    features.push_back((m2 > 1e-15) ? m4 / (m2*m2) : 0.0);
    // Crest factor
    double peak = 0;
    for (double v : signal) peak = max(peak, abs(v));
    features.push_back((rms > 1e-15) ? peak / rms : 0.0);
    return features;
}

// ============================================================
// Classifier
// ============================================================
struct Centroid { int label; vector<double> mean_f; };

double cosine_sim(const vector<double>& a, const vector<double>& b) {
    int n = min(a.size(), b.size());
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    double den = sqrt(na) * sqrt(nb);
    return (den > 1e-15) ? dot / den : 0.0;
}

vector<Centroid> train_centroids(const vector<vector<double>>& features, const vector<int>& labels) {
    map<int, vector<int>> ci;
    for (size_t i = 0; i < labels.size(); ++i) ci[labels[i]].push_back(i);
    vector<Centroid> centroids;
    for (auto& [label, indices] : ci) {
        Centroid c; c.label = label;
        int fd = (int)features[indices[0]].size();
        c.mean_f.assign(fd, 0.0);
        for (int idx : indices) for (int f = 0; f < fd; ++f) c.mean_f[f] += features[idx][f];
        for (int f = 0; f < fd; ++f) c.mean_f[f] /= indices.size();
        centroids.push_back(c);
    }
    return centroids;
}

int classify(const vector<double>& f, const vector<Centroid>& c) {
    double best = -1e30; int bl = -1;
    for (auto& ci : c) { double s = cosine_sim(f, ci.mean_f); if (s > best) { best = s; bl = ci.label; } }
    return bl;
}

// ============================================================
// Producer / Consumer
// ============================================================
void producer_thread(RingBuffer<double>& buffer, const vector<double>& signal, int delay_us) {
    for (size_t i = 0; i < signal.size(); ++i) {
        buffer.push(signal[i], (int)i);
        this_thread::sleep_for(chrono::microseconds(delay_us));
    }
}
void consumer_thread(RingBuffer<double>& buffer, vector<double>& v, vector<int>& ix, int delay_us, int exp) {
    try { for (int i = 0; i < exp; ++i) {
        IndexedSample s = buffer.pop_indexed(); v.push_back(s.value); ix.push_back(s.original_index);
        this_thread::sleep_for(chrono::microseconds(delay_us));
    } } catch (const runtime_error&) {}
}

struct CResult {
    vector<double> reconstructed;
    vector<int> surviving;
    double snr, sc;
};

CResult compress_online(const vector<double>& w, BufferMode mode, size_t buf, int ovl) {
    int pd = 50, cd = pd * ovl;
    RingBuffer<double> buffer(buf, mode, chrono::milliseconds(2), ImportanceConfig());
    vector<double> sv; vector<int> si;
    thread p(producer_thread, ref(buffer), cref(w), pd);
    thread c(consumer_thread, ref(buffer), ref(sv), ref(si), cd, (int)w.size());
    p.join(); buffer.finish(); c.join();
    CResult r;
    r.surviving = si;
    r.reconstructed = reconstruct_signal(si, sv, (int)w.size());
    r.snr = compute_snr(w, r.reconstructed);
    auto s1 = compute_mag_spectrum(w), s2 = compute_mag_spectrum(r.reconstructed);
    r.sc = spec_corr(s1, s2);
    return r;
}

CResult compress_rdp(const vector<double>& w, int target) {
    auto sel = offline_rdp(w, target);
    vector<double> sv; for (int i : sel) sv.push_back(w[i]);
    CResult r;
    r.surviving = sel;
    r.reconstructed = reconstruct_signal(sel, sv, (int)w.size());
    r.snr = compute_snr(w, r.reconstructed);
    auto s1 = compute_mag_spectrum(w), s2 = compute_mag_spectrum(r.reconstructed);
    r.sc = spec_corr(s1, s2);
    return r;
}

CResult compress_drop(const vector<double>& w, int target) {
    CResult r;
    int n = (int)w.size(), start = max(0, n - target);
    for (int i = start; i < n; ++i) r.surviving.push_back(i);
    vector<double> sv; for (int i : r.surviving) sv.push_back(w[i]);
    r.reconstructed = reconstruct_signal(r.surviving, sv, n);
    r.snr = compute_snr(w, r.reconstructed);
    r.sc = 0;
    return r;
}

// ============================================================
// Load HAR windowed data
// ============================================================
struct HARClass {
    string name;
    int label;
    vector<vector<double>> windows;
};

HARClass load_har_windows(const string& filepath, const string& name, int label) {
    HARClass hc; hc.name = name; hc.label = label;
    ifstream f(filepath);
    if (!f.is_open()) { cerr << "Cannot open " << filepath << endl; return hc; }
    int n_win, win_size;
    f >> n_win >> win_size;
    for (int w = 0; w < n_win; ++w) {
        vector<double> win(win_size);
        for (int i = 0; i < win_size; ++i) f >> win[i];
        hc.windows.push_back(win);
    }
    return hc;
}

vector<double> load_signal(const string& fp, int max_s = 100000) {
    vector<double> d; ifstream f(fp); double v;
    while (f >> v && (int)d.size() < max_s) d.push_back(v);
    return d;
}

// ============================================================
// PART A: HAR CLASSIFICATION
// ============================================================
void part_A_har(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "  PART A: UCI HAR Activity Classification" << endl;
    cout << "============================================================\n" << endl;

    string har_dir = "../data/uci-har/processed";
    vector<HARClass> classes = {
        load_har_windows(har_dir + "/har_walking_windows.txt", "walking", 0),
        load_har_windows(har_dir + "/har_walking_up_windows.txt", "walking_up", 1),
        load_har_windows(har_dir + "/har_walking_down_windows.txt", "walking_down", 2),
        load_har_windows(har_dir + "/har_sitting_windows.txt", "sitting", 3),
        load_har_windows(har_dir + "/har_standing_windows.txt", "standing", 4),
        load_har_windows(har_dir + "/har_laying_windows.txt", "laying", 5),
    };

    // Collect all windows
    struct Window { vector<double> data; int label; string cname; };
    vector<Window> all_w;
    for (auto& c : classes) {
        cout << "  " << c.name << ": " << c.windows.size() << " windows, "
             << (c.windows.empty() ? 0 : c.windows[0].size()) << " samples" << endl;
        for (auto& w : c.windows)
            all_w.push_back({w, c.label, c.name});
    }
    cout << "  Total: " << all_w.size() << " windows\n" << endl;

    int n_bins = 16;  // fewer bins for 128-sample windows (64 FFT bins)

    // Extract uncompressed features
    vector<vector<double>> orig_f;
    vector<int> orig_l;
    for (auto& w : all_w) {
        orig_f.push_back(extract_features(w.data, n_bins));
        orig_l.push_back(w.label);
    }
    auto centroids = train_centroids(orig_f, orig_l);

    // Baseline accuracy
    int correct = 0;
    for (size_t i = 0; i < all_w.size(); ++i)
        if (classify(orig_f[i], centroids) == orig_l[i]) correct++;
    cout << "  Uncompressed accuracy: " << fixed << setprecision(1)
         << 100.0 * correct / all_w.size() << "%" << endl;

    // Compression sweep
    size_t buf = 32;  // appropriate for 128-sample windows
    vector<int> overloads = {2, 3, 5, 8, 10, 15, 20};
    int num_trials = 3;

    ofstream csv(out_dir + "/har_classification.csv");
    csv << "method,overload,trial,accuracy,n_correct,n_total,avg_snr,avg_spec_corr" << endl;
    csv << "UNCOMPRESSED,1,0," << (double)correct/all_w.size() << ","
        << correct << "," << all_w.size() << ",inf,1.0" << endl;

    struct Method { string name; bool online; BufferMode mode; };
    vector<Method> methods = {
        {"PROPOSED", true, BufferMode::IMPORTANCE_INTERP_ERROR},
        {"RDP", false, BufferMode::DROP},
        {"DROP", true, BufferMode::DROP},
    };

    for (auto& m : methods) {
        for (int ovl : overloads) {
            int nt = m.online ? num_trials : 1;
            for (int trial = 0; trial < nt; ++trial) {
                int cor = 0; double tsnr = 0, tsc = 0; int nv = 0;
                for (size_t wi = 0; wi < all_w.size(); ++wi) {
                    auto& w = all_w[wi];
                    CResult cr;
                    if (m.name == "DROP") {
                        cr = compress_drop(w.data, max(2, (int)w.data.size() / ovl));
                    } else if (m.online) {
                        cr = compress_online(w.data, m.mode, buf, ovl);
                    } else {
                        auto ref = compress_online(w.data, BufferMode::IMPORTANCE_INTERP_ERROR, buf, ovl);
                        cr = compress_rdp(w.data, (int)ref.surviving.size());
                    }
                    auto cf = extract_features(cr.reconstructed, n_bins);
                    if (classify(cf, centroids) == w.label) cor++;
                    if (isfinite(cr.snr)) { tsnr += cr.snr; nv++; }
                    tsc += cr.sc;
                }
                double acc = (double)cor / all_w.size();
                csv << m.name << "," << ovl << "," << trial << ","
                    << fixed << setprecision(4) << acc << ","
                    << cor << "," << all_w.size() << ","
                    << setprecision(2) << (nv > 0 ? tsnr/nv : 0) << ","
                    << setprecision(4) << tsc/all_w.size() << endl;
                cout << "  " << left << setw(12) << m.name
                     << " ovl=" << setw(3) << ovl << " t=" << trial
                     << "  acc=" << fixed << setprecision(1) << acc*100 << "%"
                     << "  snr=" << setprecision(1) << (nv>0?tsnr/nv:0) << "dB"
                     << "  sc=" << setprecision(3) << tsc/all_w.size() << endl;
            }
        }
        cout << endl;
    }
    csv.close();
}

// ============================================================
// PART B: LONG DURATION EVALUATION (10,000 samples)
// ============================================================
void part_B_long_duration(const string& out_dir) {
    cout << "\n============================================================" << endl;
    cout << "  PART B: Long Duration (10,000 samples)" << endl;
    cout << "============================================================\n" << endl;

    struct Sig { string name; string path; string domain; };
    vector<Sig> signals = {
        {"ecg_100",    "../data/mit-bih/ecg_100.txt",              "ecg"},
        {"ecg_108",    "../data/mit-bih/ecg_108.txt",              "ecg"},
        {"ecg_228",    "../data/mit-bih/ecg_228.txt",              "ecg"},
        {"vib_normal", "../data/cwru-bearing/normal_0hp.txt",      "vibration"},
        {"vib_inner",  "../data/cwru-bearing/inner_race_007.txt",  "vibration"},
        {"vib_ball",   "../data/cwru-bearing/ball_fault_007.txt",  "vibration"},
    };

    ofstream csv(out_dir + "/long_duration.csv");
    csv << "signal,domain,n_samples,method,overload,trial,snr_db,spec_corr,n_surviving,n_dropped" << endl;

    size_t buf = 256;
    vector<int> overloads = {3, 5, 10, 20};
    int num_trials = 3;

    for (auto& s : signals) {
        auto data = load_signal(s.path, 10000);
        cout << "  " << s.name << ": " << data.size() << " samples" << endl;

        for (int ovl : overloads) {
            // Proposed online
            for (int trial = 0; trial < num_trials; ++trial) {
                auto cr = compress_online(data, BufferMode::IMPORTANCE_INTERP_ERROR, buf, ovl);
                csv << s.name << "," << s.domain << "," << data.size() << ","
                    << "PROPOSED," << ovl << "," << trial << ","
                    << fixed << setprecision(4) << cr.snr << ","
                    << cr.sc << ","
                    << cr.surviving.size() << ","
                    << data.size() - cr.surviving.size() << endl;
            }

            // RDP (match drop count from first proposed trial)
            {
                auto ref = compress_online(data, BufferMode::IMPORTANCE_INTERP_ERROR, buf, ovl);
                auto cr = compress_rdp(data, (int)ref.surviving.size());
                csv << s.name << "," << s.domain << "," << data.size() << ","
                    << "RDP," << ovl << ",0,"
                    << fixed << setprecision(4) << cr.snr << ","
                    << cr.sc << ","
                    << cr.surviving.size() << ","
                    << data.size() - cr.surviving.size() << endl;
            }

            // DROP
            {
                auto cr = compress_drop(data, max(2, (int)data.size() / ovl));
                csv << s.name << "," << s.domain << "," << data.size() << ","
                    << "DROP," << ovl << ",0,"
                    << fixed << setprecision(4) << cr.snr << ","
                    << cr.sc << ","
                    << cr.surviving.size() << ","
                    << data.size() - cr.surviving.size() << endl;
            }

            cout << "    ovl=" << ovl << ": done" << endl;
        }
    }
    csv.close();
    cout << "\n  Output: " << out_dir << "/long_duration.csv" << endl;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  TASKS 2 & 3: HAR Classification + Long Duration" << endl;
    cout << "============================================================" << endl;

    string out_dir = "../results/tasks23";
    system(("mkdir -p " + out_dir).c_str());

    auto t0 = chrono::high_resolution_clock::now();

    part_A_har(out_dir);
    part_B_long_duration(out_dir);

    auto t1 = chrono::high_resolution_clock::now();
    cout << "\n  Total wall-clock: " << fixed << setprecision(1)
         << chrono::duration<double>(t1 - t0).count() << " s" << endl;

    return 0;
}
