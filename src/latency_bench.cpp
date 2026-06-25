#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>

#include "ring_buffer.h"
#include "metrics.h"

using namespace std;

// ============================================================
// Latency Benchmark v3 — Single-buffer-size deep dive
//
// ADDED vs v2: optional --buffer=<size> --repeats=<n> arguments
// to run a deep, high-repeat investigation of ONE buffer size,
// without re-running the entire 32-4096 sweep each time. This
// exists specifically to investigate the buffer=2048 anomaly
// found in the v2 full sweep, where p99/max latency stayed
// disproportionately high even after CPU pinning eliminated most
// tail latency at every other buffer size.
// ============================================================

double bench_one_config(BufferMode mode, size_t buf_size, int stream_length,
                        EvictionProfile& out_profile) {
    RingBuffer<double> buffer(buf_size, mode, chrono::milliseconds(2), ImportanceConfig());

    mt19937 gen(42);
    normal_distribution<double> noise(0.0, 1.0);

    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < stream_length; ++i) {
        buffer.push(noise(gen), i);
    }
    auto t1 = chrono::high_resolution_clock::now();

    out_profile = buffer.getProfile();
    return chrono::duration<double, milli>(t1 - t0).count();
}

double percentile(vector<double>& sorted_data, double pct) {
    if (sorted_data.empty()) return 0.0;
    size_t idx = (size_t)(pct / 100.0 * (sorted_data.size() - 1));
    return sorted_data[idx];
}

int main(int argc, char* argv[]) {
    cout << "=== Eviction Latency Benchmark v3 (Single-Buffer Deep Dive) ===" << endl;

    size_t target_buffer = 2048;
    int repeats = 300;
    int stream_length = 50000;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg.rfind("--buffer=", 0) == 0)
            target_buffer = stoul(arg.substr(9));
        else if (arg.rfind("--repeats=", 0) == 0)
            repeats = stoi(arg.substr(10));
        else if (arg.rfind("--stream=", 0) == 0)
            stream_length = stoi(arg.substr(9));
    }

    cout << "Buffer size: " << target_buffer << ", repeats: " << repeats
         << ", stream length: " << stream_length << endl << endl;

    struct ModeEntry { string name; BufferMode mode; };
    vector<ModeEntry> modes = {
        {"IMP_INTERP_ERROR",     BufferMode::IMPORTANCE_INTERP_ERROR},
        {"IMP_INTERP_COMPOSITE", BufferMode::IMPORTANCE_INTERP_COMPOSITE},
    };

    ofstream csv("../results/latency_deepdive_buf" + to_string(target_buffer) + ".csv");
    csv << "mode,buffer_size,repeat,evictions,avg_us,trial_max_us,wall_clock_ms" << endl;

    for (auto& m : modes) {
        cout << "--- " << m.name << " (buffer=" << target_buffer << ") ---" << endl;
        vector<double> trial_maxes;

        for (int r = 0; r < repeats; ++r) {
            EvictionProfile prof;
            double wall_ms = bench_one_config(m.mode, target_buffer, stream_length, prof);

            csv << m.name << "," << target_buffer << "," << r << ","
                << prof.count << ","
                << fixed << setprecision(4) << prof.avg_us() << ","
                << prof.max_us() << ","
                << wall_ms << endl;

            trial_maxes.push_back(prof.max_us());

            // Flag any individual trial whose max exceeds 1ms (1000us),
            // so we can see directly HOW OFTEN the large spike recurs,
            // not just summary statistics about it.
            if (prof.max_us() > 1000.0) {
                cout << "  [SPIKE] repeat " << r << ": trial_max=" << prof.max_us() << "us" << endl;
            }
        }

        sort(trial_maxes.begin(), trial_maxes.end());
        int spike_count = 0;
        for (double v : trial_maxes) if (v > 1000.0) ++spike_count;

        cout << "  p50=" << percentile(trial_maxes, 50)
             << "us  p95=" << percentile(trial_maxes, 95)
             << "us  p99=" << percentile(trial_maxes, 99)
             << "us  max=" << trial_maxes.back() << "us" << endl;
        cout << "  Spikes >1ms: " << spike_count << "/" << repeats
             << " (" << fixed << setprecision(1) << (100.0 * spike_count / repeats) << "%)" << endl << endl;
    }

    csv.close();
    cout << "Raw data saved: results/latency_deepdive_buf" << target_buffer << ".csv" << endl;

    return 0;
}
