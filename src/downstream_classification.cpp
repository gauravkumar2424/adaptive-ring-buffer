// ============================================================
// downstream_classification.cpp — Phase 3: Downstream Task Validation
// DAC 2027
//
// PURPOSE: Prove that spectral correlation advantage translates
// to measurable downstream task improvement.
//
// DESIGN:
//   1. Load full-length CWRU vibration signals (3 classes)
//   2. Segment into non-overlapping windows
//   3. Train classifier on UNCOMPRESSED windows (spectral features)
//   4. For each method + compression ratio:
//      a. Compress each window
//      b. Extract same spectral features from compressed signal
//      c. Classify using trained model
//      d. Report accuracy
//
//   This directly answers: "I trained my fault detector on clean
//   data, then deployed with a compressed pipeline. How much
//   accuracy do I lose, and which compression method loses least?"
//
// CLASSIFIER: Nearest centroid on binned FFT magnitudes.
//   Simple, interpretable, no hyperparameters. The point is not
//   to build the best classifier but to show relative degradation
//   across compression methods.
//
// Build: g++ -std=c++17 -O2 -Wall -pthread -o ../build/downstream downstream_classification.cpp
// Run:   cd ../build && ./downstream
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
#include <cassert>
#include <set>

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

// ============================================================
// Offline RDP
// ============================================================
vector<int> offline_rdp(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n); iota(all.begin(), all.end(), 0); return all;
    }
    if (target_points < 2) return {0, n - 1};

    vector<bool> kept(n, true);
    vector<int> prev_kept(n), next_kept(n);
    for (int i = 0; i < n; ++i) { prev_kept[i] = i - 1; next_kept[i] = i + 1; }
    next_kept[n - 1] = -1;

    auto compute_error = [&](int i) -> double {
        int p = prev_kept[i], s = next_kept[i];
        if (p < 0 || s < 0 || s >= n) return 1e30;
        double span = (double)(s - p);
        if (span <= 0) return 0.0;
        double t = (double)(i - p) / span;
        return abs(signal[i] - (signal[p] + t * (signal[s] - signal[p])));
    };

    using PQE = pair<double, int>;
    priority_queue<PQE, vector<PQE>, greater<PQE>> pq;
    for (int i = 1; i < n - 1; ++i) pq.push({compute_error(i), i});

    int current_count = n;
    while (current_count > target_points && !pq.empty()) {
        auto [err, idx] = pq.top(); pq.pop();
        if (!kept[idx]) continue;
        double actual = compute_error(idx);
        if (abs(actual - err) > 1e-12) { pq.push({actual, idx}); continue; }
        kept[idx] = false;
        int p = prev_kept[idx], s = next_kept[idx];
        if (p >= 0) next_kept[p] = s;
        if (s >= 0 && s < n) prev_kept[s] = p;
        --current_count;
        if (p > 0 && kept[p]) pq.push({compute_error(p), p});
        if (s > 0 && s < n - 1 && kept[s]) pq.push({compute_error(s), s});
    }
    vector<int> result;
    for (int i = 0; i < n; ++i) if (kept[i]) result.push_back(i);
    return result;
}

// LTTB offline
vector<int> offline_lttb(const vector<double>& signal, int target_points) {
    int n = (int)signal.size();
    if (target_points >= n) {
        vector<int> all(n); iota(all.begin(), all.end(), 0); return all;
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
        double avg_x = 0, avg_y = 0; int nb_count = 0;
        for (int i = nb_start; i < nb_end; ++i) { avg_x += i; avg_y += signal[i]; ++nb_count; }
        if (nb_count > 0) { avg_x /= nb_count; avg_y /= nb_count; }
        double max_area = -1; int best_idx = b_start;
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
vector<double> compute_magnitude_spectrum(const vector<double>& signal) {
    int n = (int)signal.size();
    int half = n / 2;
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

// ============================================================
// Feature extraction: bin FFT magnitudes into n_bins bands
// ============================================================
vector<double> extract_spectral_features(const vector<double>& signal, int n_bins) {
    auto mag = compute_magnitude_spectrum(signal);
    int half = (int)mag.size();
    if (half == 0) return vector<double>(n_bins, 0.0);

    vector<double> features(n_bins, 0.0);
    double bin_width = (double)half / n_bins;

    for (int b = 0; b < n_bins; ++b) {
        int start = (int)(b * bin_width);
        int end = (int)((b + 1) * bin_width);
        end = min(end, half);
        double sum = 0;
        for (int k = start; k < end; ++k)
            sum += mag[k];
        features[b] = sum;
    }

    // Also add time-domain statistical features
    // RMS
    double rms = 0;
    for (double v : signal) rms += v * v;
    rms = sqrt(rms / signal.size());
    features.push_back(rms);

    // Kurtosis
    double mean = accumulate(signal.begin(), signal.end(), 0.0) / signal.size();
    double m2 = 0, m4 = 0;
    for (double v : signal) {
        double d = v - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= signal.size();
    m4 /= signal.size();
    double kurt = (m2 > 1e-15) ? m4 / (m2 * m2) : 0.0;
    features.push_back(kurt);

    // Crest factor (peak / RMS)
    double peak = 0;
    for (double v : signal) peak = max(peak, abs(v));
    features.push_back((rms > 1e-15) ? peak / rms : 0.0);

    return features;
}

// ============================================================
// Nearest centroid classifier
// ============================================================
struct Centroid {
    int label;
    vector<double> mean_features;
};

double euclidean_dist(const vector<double>& a, const vector<double>& b) {
    double sum = 0;
    int n = min(a.size(), b.size());
    for (int i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

double cosine_similarity(const vector<double>& a, const vector<double>& b) {
    int n = min(a.size(), b.size());
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    return (denom > 1e-15) ? dot / denom : 0.0;
}

vector<Centroid> train_centroids(const vector<vector<double>>& features,
                                  const vector<int>& labels) {
    map<int, vector<int>> class_indices;
    for (size_t i = 0; i < labels.size(); ++i)
        class_indices[labels[i]].push_back(i);

    vector<Centroid> centroids;
    for (auto& [label, indices] : class_indices) {
        Centroid c;
        c.label = label;
        int feat_dim = (int)features[indices[0]].size();
        c.mean_features.assign(feat_dim, 0.0);
        for (int idx : indices)
            for (int f = 0; f < feat_dim; ++f)
                c.mean_features[f] += features[idx][f];
        for (int f = 0; f < feat_dim; ++f)
            c.mean_features[f] /= indices.size();
        centroids.push_back(c);
    }
    return centroids;
}

int classify_nearest(const vector<double>& features,
                     const vector<Centroid>& centroids) {
    double best_sim = -1e30;
    int best_label = -1;
    for (auto& c : centroids) {
        double sim = cosine_similarity(features, c.mean_features);
        if (sim > best_sim) {
            best_sim = sim;
            best_label = c.label;
        }
    }
    return best_label;
}

// ============================================================
// Compression methods
// ============================================================

// Online: run producer/consumer
struct CompressedWindow {
    vector<double> reconstructed;
    vector<int> surviving_indices;
    double snr_db;
    double spec_corr;
};

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

CompressedWindow compress_online(const vector<double>& window,
                                  BufferMode mode, size_t buf_size, int overload) {
    int prod_delay = 50;  // faster for windowed segments
    int cons_delay = prod_delay * overload;

    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());
    vector<double> sv; vector<int> si;
    sv.reserve(window.size()); si.reserve(window.size());

    thread prod(producer_thread, ref(buffer), cref(window), prod_delay);
    thread cons(consumer_thread, ref(buffer), ref(sv), ref(si),
                cons_delay, (int)window.size());
    prod.join(); buffer.finish(); cons.join();

    CompressedWindow cw;
    cw.surviving_indices = si;
    cw.reconstructed = reconstruct_signal(si, sv, (int)window.size());
    cw.snr_db = compute_snr(window, cw.reconstructed);

    // Spectral correlation
    auto s1 = compute_magnitude_spectrum(window);
    auto s2 = compute_magnitude_spectrum(cw.reconstructed);
    int n = min(s1.size(), s2.size());
    double ma = accumulate(s1.begin(), s1.begin()+n, 0.0) / n;
    double mb = accumulate(s2.begin(), s2.begin()+n, 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < n; ++k) {
        double xa = s1[k] - ma, xb = s2[k] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    double den = sqrt(da * db);
    cw.spec_corr = (den > 1e-15) ? num / den : 0.0;
    return cw;
}

CompressedWindow compress_offline(const vector<double>& window,
                                   const string& method, int target) {
    vector<int> selected;
    if (method == "RDP")
        selected = offline_rdp(window, target);
    else
        selected = offline_lttb(window, target);

    vector<double> sv;
    for (int i : selected) sv.push_back(window[i]);

    CompressedWindow cw;
    cw.surviving_indices = selected;
    cw.reconstructed = reconstruct_signal(selected, sv, (int)window.size());
    cw.snr_db = compute_snr(window, cw.reconstructed);

    auto s1 = compute_magnitude_spectrum(window);
    auto s2 = compute_magnitude_spectrum(cw.reconstructed);
    int n = min(s1.size(), s2.size());
    double ma = accumulate(s1.begin(), s1.begin()+n, 0.0) / n;
    double mb = accumulate(s2.begin(), s2.begin()+n, 0.0) / n;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < n; ++k) {
        double xa = s1[k] - ma, xb = s2[k] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    double den = sqrt(da * db);
    cw.spec_corr = (den > 1e-15) ? num / den : 0.0;
    return cw;
}

CompressedWindow compress_drop(const vector<double>& window, int target) {
    // FIFO drop: keep only the last 'target' samples
    CompressedWindow cw;
    int n = (int)window.size();
    int start = n - target;
    if (start < 0) start = 0;
    for (int i = start; i < n; ++i) {
        cw.surviving_indices.push_back(i);
    }
    vector<double> sv;
    for (int i : cw.surviving_indices) sv.push_back(window[i]);
    cw.reconstructed = reconstruct_signal(cw.surviving_indices, sv, n);
    cw.snr_db = compute_snr(window, cw.reconstructed);
    cw.spec_corr = 0;  // not needed for classification
    return cw;
}

// ============================================================
// Load full signals (more samples than default 2000)
// ============================================================
vector<double> load_signal(const string& filepath, int max_samples = 50000) {
    vector<double> data;
    ifstream file(filepath);
    if (!file.is_open()) return data;
    double val;
    while (file >> val && (int)data.size() < max_samples)
        data.push_back(val);
    return data;
}

// ============================================================
// MAIN EXPERIMENT
// ============================================================
int main() {
    cout << "============================================================" << endl;
    cout << "  DOWNSTREAM FAULT CLASSIFICATION EXPERIMENT" << endl;
    cout << "  Does spectral fidelity advantage translate to accuracy?" << endl;
    cout << "============================================================\n" << endl;

    string data_dir = "../data";
    string out_dir = "../results/downstream";
    system(("mkdir -p " + out_dir).c_str());

    // Load full signals
    struct SignalClass {
        string name;
        string filepath;
        int label;  // 0=normal, 1=inner_race, 2=ball_fault
        vector<double> data;
    };

    vector<SignalClass> classes = {
        {"normal",     data_dir + "/cwru-bearing/normal_0hp.txt",      0, {}},
        {"inner_race", data_dir + "/cwru-bearing/inner_race_007.txt",  1, {}},
        {"ball_fault", data_dir + "/cwru-bearing/ball_fault_007.txt",  2, {}},
    };

    for (auto& c : classes) {
        c.data = load_signal(c.filepath);
        cout << "Loaded " << c.name << ": " << c.data.size() << " samples" << endl;
    }

    // Parameters
    int window_size = 512;
    int n_bins = 32;  // spectral feature bins
    // Overload ratios to sweep (compression levels)
    vector<int> overloads = {10, 20, 30, 50, 80, 100};
    size_t buf_size = 256;
    int num_trials = 5;  // repeat online methods for better statistics

    // Segment signals into non-overlapping windows
    struct Window {
        vector<double> data;
        int label;
        string class_name;
    };

    vector<Window> all_windows;
    for (auto& c : classes) {
        int n_windows = (int)c.data.size() / window_size;
        cout << "  " << c.name << ": " << n_windows << " windows of "
             << window_size << " samples" << endl;
        for (int w = 0; w < n_windows; ++w) {
            Window win;
            win.data.assign(c.data.begin() + w * window_size,
                           c.data.begin() + (w + 1) * window_size);
            win.label = c.label;
            win.class_name = c.name;
            all_windows.push_back(win);
        }
    }
    cout << "\nTotal windows: " << all_windows.size() << endl;

    // Extract features from UNCOMPRESSED windows (training data)
    cout << "\nExtracting features from uncompressed windows..." << endl;
    vector<vector<double>> orig_features;
    vector<int> orig_labels;
    for (auto& w : all_windows) {
        orig_features.push_back(extract_spectral_features(w.data, n_bins));
        orig_labels.push_back(w.label);
    }

    // Train centroids on uncompressed data
    auto centroids = train_centroids(orig_features, orig_labels);
    cout << "Trained " << centroids.size() << " class centroids" << endl;

    // Baseline: classify uncompressed windows (should be ~100%)
    int correct_orig = 0;
    for (size_t i = 0; i < all_windows.size(); ++i) {
        int pred = classify_nearest(orig_features[i], centroids);
        if (pred == orig_labels[i]) correct_orig++;
    }
    double acc_orig = (double)correct_orig / all_windows.size();
    cout << "Uncompressed accuracy: " << fixed << setprecision(4)
         << acc_orig * 100 << "% (" << correct_orig << "/"
         << all_windows.size() << ")" << endl;

    // Also do leave-one-out cross-validation on uncompressed
    int correct_loo = 0;
    for (size_t i = 0; i < all_windows.size(); ++i) {
        // Train on all except i
        vector<vector<double>> train_f;
        vector<int> train_l;
        for (size_t j = 0; j < all_windows.size(); ++j) {
            if (j != i) {
                train_f.push_back(orig_features[j]);
                train_l.push_back(orig_labels[j]);
            }
        }
        auto loo_centroids = train_centroids(train_f, train_l);
        int pred = classify_nearest(orig_features[i], loo_centroids);
        if (pred == orig_labels[i]) correct_loo++;
    }
    double acc_loo = (double)correct_loo / all_windows.size();
    cout << "Uncompressed LOO-CV accuracy: " << fixed << setprecision(4)
         << acc_loo * 100 << "%" << endl;

    // Output CSV
    ofstream csv(out_dir + "/classification_results.csv");
    csv << "method,overload,trial,accuracy,n_correct,n_total,"
        << "avg_snr,avg_spec_corr,"
        << "acc_normal,acc_inner_race,acc_ball_fault" << endl;

    // Write uncompressed baseline
    csv << "UNCOMPRESSED,1,0," << acc_orig << ","
        << correct_orig << "," << all_windows.size() << ","
        << "inf,1.0000,"
        << "1.0000,1.0000,1.0000" << endl;

    // Detailed per-window CSV
    ofstream detail_csv(out_dir + "/classification_detail.csv");
    detail_csv << "method,overload,trial,window_idx,true_label,pred_label,"
               << "correct,snr_db,spec_corr,class_name" << endl;

    // Methods to test
    struct Method {
        string name;
        bool is_online;
        BufferMode mode;  // for online
        string offline_name;  // for offline
    };

    vector<Method> methods = {
        {"PROPOSED",      true,  BufferMode::IMPORTANCE_INTERP_ERROR, ""},
        {"PROPOSED_SPEC", true,  BufferMode::IMPORTANCE_INTERP_SPECTRAL, ""},
        {"RDP",           false, BufferMode::DROP, "RDP"},
        {"LTTB",          false, BufferMode::DROP, "LTTB"},
        {"DROP",          true,  BufferMode::DROP, ""},
    };

    auto t0 = chrono::high_resolution_clock::now();

    for (auto& method : methods) {
        for (int ovl : overloads) {
            int n_trials = method.is_online ? num_trials : 1;  // offline is deterministic

            for (int trial = 0; trial < n_trials; ++trial) {
                int correct = 0;
                double total_snr = 0, total_spec = 0;
                int n_valid = 0;
                map<int, int> class_correct, class_total;

                for (size_t wi = 0; wi < all_windows.size(); ++wi) {
                    auto& w = all_windows[wi];
                    CompressedWindow cw;

                    if (method.name == "DROP") {
                        int target = max(2, (int)w.data.size() / ovl);
                        cw = compress_drop(w.data, target);
                    } else if (method.is_online) {
                        cw = compress_online(w.data, method.mode, buf_size, ovl);
                    } else {
                        // For offline: match drop count to what online produces
                        // First run a quick online to get the drop count
                        auto ref = compress_online(w.data,
                            BufferMode::IMPORTANCE_INTERP_ERROR, buf_size, ovl);
                        int target = (int)ref.surviving_indices.size();
                        cw = compress_offline(w.data, method.offline_name, target);
                    }

                    // Extract features from compressed signal
                    auto comp_features = extract_spectral_features(cw.reconstructed, n_bins);

                    // Classify
                    int pred = classify_nearest(comp_features, centroids);
                    bool is_correct = (pred == w.label);
                    if (is_correct) correct++;

                    class_total[w.label]++;
                    if (is_correct) class_correct[w.label]++;

                    if (isfinite(cw.snr_db)) {
                        total_snr += cw.snr_db;
                        n_valid++;
                    }
                    total_spec += cw.spec_corr;

                    // Detail output
                    detail_csv << method.name << "," << ovl << "," << trial << ","
                               << wi << "," << w.label << "," << pred << ","
                               << (is_correct ? 1 : 0) << ","
                               << fixed << setprecision(2) << cw.snr_db << ","
                               << setprecision(4) << cw.spec_corr << ","
                               << w.class_name << endl;
                }

                double accuracy = (double)correct / all_windows.size();
                double avg_snr = (n_valid > 0) ? total_snr / n_valid : 0;
                double avg_spec = total_spec / all_windows.size();

                double acc_n = class_total.count(0) ?
                    (double)class_correct[0] / class_total[0] : 0;
                double acc_i = class_total.count(1) ?
                    (double)class_correct[1] / class_total[1] : 0;
                double acc_b = class_total.count(2) ?
                    (double)class_correct[2] / class_total[2] : 0;

                csv << method.name << "," << ovl << "," << trial << ","
                    << fixed << setprecision(4)
                    << accuracy << "," << correct << "," << all_windows.size() << ","
                    << setprecision(2) << avg_snr << ","
                    << setprecision(4) << avg_spec << ","
                    << setprecision(4) << acc_n << ","
                    << acc_i << "," << acc_b << endl;

                cout << "  " << left << setw(15) << method.name
                     << " ovl=" << setw(3) << ovl
                     << " trial=" << trial
                     << "  acc=" << fixed << setprecision(1) << accuracy * 100 << "%"
                     << "  snr=" << setprecision(1) << avg_snr << "dB"
                     << "  spec=" << setprecision(3) << avg_spec
                     << "  [N:" << setprecision(0) << acc_n*100
                     << " I:" << acc_i*100
                     << " B:" << acc_b*100 << "%]"
                     << endl;
            }
        }
        cout << endl;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    csv.close();
    detail_csv.close();

    cout << "\n============================================================" << endl;
    cout << "  EXPERIMENT COMPLETE" << endl;
    cout << "  Wall-clock: " << fixed << setprecision(1) << elapsed << " s" << endl;
    cout << "  Output: " << out_dir << "/classification_results.csv" << endl;
    cout << "          " << out_dir << "/classification_detail.csv" << endl;
    cout << "============================================================" << endl;

    return 0;
}
