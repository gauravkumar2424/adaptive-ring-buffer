#ifndef SIGNAL_LOADER_H
#define SIGNAL_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// ============================================================
// Signal loader for real-world datasets
// Loads ECG (MIT-BIH) and vibration (CWRU) signals from text files
// Also loads R-peak ground truth annotations for ECG evaluation
// ============================================================

struct RealSignal {
    std::string name;        // e.g. "ecg_100", "cwru_normal"
    std::string domain;      // "ecg" or "vibration"
    std::vector<double> data;
    std::vector<int> rpeak_indices;  // Ground truth R-peaks (ECG only)
    bool has_rpeaks;
};

inline RealSignal load_signal_file(const std::string& filepath,
                                    const std::string& name,
                                    const std::string& domain,
                                    int max_samples = 2000) {
    RealSignal sig;
    sig.name = name;
    sig.domain = domain;
    sig.has_rpeaks = false;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "WARNING: Cannot open " << filepath << std::endl;
        return sig;
    }

    double val;
    while (file >> val && (int)sig.data.size() < max_samples) {
        sig.data.push_back(val);
    }
    return sig;
}

inline void load_rpeak_annotations(RealSignal& sig, const std::string& filepath,
                                    int max_sample_idx) {
    std::ifstream file(filepath);
    if (!file.is_open()) return;

    int idx;
    while (file >> idx) {
        if (idx < max_sample_idx) {
            sig.rpeak_indices.push_back(idx);
        }
    }
    sig.has_rpeaks = !sig.rpeak_indices.empty();
}

// Load all real-world signals from the data directory
inline std::vector<RealSignal> load_all_real_signals(const std::string& data_dir,
                                                      int max_samples = 2000) {
    std::vector<RealSignal> signals;

    // MIT-BIH ECG records
    std::vector<std::string> ecg_ids = {"100", "105", "108", "201", "228"};
    for (const auto& id : ecg_ids) {
        std::string sig_path = data_dir + "/mit-bih/ecg_" + id + ".txt";
        std::string rpeak_path = data_dir + "/mit-bih/rpeak_" + id + ".txt";

        RealSignal sig = load_signal_file(sig_path, "ecg_" + id, "ecg", max_samples);
        if (!sig.data.empty()) {
            load_rpeak_annotations(sig, rpeak_path, max_samples);
            std::cout << "  Loaded ECG " << id << ": " << sig.data.size()
                      << " samples, " << sig.rpeak_indices.size() << " R-peaks" << std::endl;
            signals.push_back(std::move(sig));
        }
    }

    // CWRU Bearing vibration signals
    std::vector<std::pair<std::string, std::string>> cwru_files = {
        {"normal_0hp", "Normal baseline"},
        {"inner_race_007", "Inner race fault"},
        {"ball_fault_007", "Ball fault"},
    };

    for (const auto& [fname, desc] : cwru_files) {
        std::string sig_path = data_dir + "/cwru-bearing/" + fname + ".txt";
        RealSignal sig = load_signal_file(sig_path, "vib_" + fname, "vibration", max_samples);
        if (!sig.data.empty()) {
            std::cout << "  Loaded CWRU " << desc << ": " << sig.data.size()
                      << " samples" << std::endl;
            signals.push_back(std::move(sig));
        }
    }

    return signals;
}

#endif // SIGNAL_LOADER_H
