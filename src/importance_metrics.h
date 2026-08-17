#ifndef IMPORTANCE_METRICS_H
#define IMPORTANCE_METRICS_H

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>

// ============================================================
// Importance Metrics for Adaptive Ring Buffer Eviction
// ============================================================

enum class ImportanceMetric {
    FIRST_ORDER,
    SECOND_ORDER,
    WINDOWED_ENERGY,
    COMPOSITE,
    ADAPTIVE_COMPOSITE,
    INTERP_ERROR,          // direct interpolation error
    INTERP_COMPOSITE       // interp error × (1 + γ·Ew)
};

enum class CombinationRule {
    WEIGHTED_LINEAR,
    WEIGHTED_GEOMETRIC,
    RANK_FUSION,
    HARMONIC_WEIGHTED,
    WORST_CASE_MIN
};

struct ImportanceConfig {
    double alpha = 0.1;
    double beta  = 0.3;
    double gamma = 0.6;
    size_t window_half = 3;
    double occupancy_shift = 0.15;
    size_t boundary_protect = 1;
    CombinationRule combination = CombinationRule::WEIGHTED_LINEAR;
    double span_beta = 0.0;        // === SPAN ADDITION === 0=interp error, 1=V-W area
    double lookahead_alpha = 0.5;  // === LOOKAHEAD ADDITION === cascade penalty weight
};

struct EvictionProfile {
    uint64_t total_ns   = 0;
    uint64_t count      = 0;
    uint64_t max_ns     = 0;
    uint64_t scan_ns    = 0;
    uint64_t removal_ns = 0;

    double avg_us()         const { return count > 0 ? (double)total_ns / count / 1000.0 : 0.0; }
    double max_us()         const { return (double)max_ns / 1000.0; }
    double avg_scan_us()    const { return count > 0 ? (double)scan_ns / count / 1000.0 : 0.0; }
    double avg_removal_us() const { return count > 0 ? (double)removal_ns / count / 1000.0 : 0.0; }
};

// ============================================================
// Standalone metric functions (backward compatibility)
// ============================================================

inline double raw_first_order(
    const double* buf, size_t cap, size_t head, size_t idx)
{
    if (idx == 0) return 0.0;
    return std::abs(buf[(head+idx)%cap] - buf[(head+idx-1)%cap]);
}

inline double raw_second_order(
    const double* buf, size_t cap, size_t head, size_t idx)
{
    if (idx < 2) return 0.0;
    double d1 = buf[(head+idx)%cap] - buf[(head+idx-1)%cap];
    double d0 = buf[(head+idx-1)%cap] - buf[(head+idx-2)%cap];
    return std::abs(d1 - d0);
}

inline double raw_windowed_energy(
    const double* buf, size_t cap, size_t head,
    size_t buf_size, size_t idx, size_t wh)
{
    double energy = 0.0;
    size_t start = (idx > wh) ? (idx - wh) : 0;
    size_t end = (idx + wh < buf_size) ? (idx + wh) : (buf_size - 1);
    for (size_t i = start + 1; i <= end; ++i) {
        double diff = buf[(head+i)%cap] - buf[(head+i-1)%cap];
        energy += diff * diff;
    }
    size_t wlen = end - start;
    return (wlen > 0) ? energy / wlen : 0.0;
}

inline size_t find_eviction_candidate(
    const double* buf, size_t cap,
    size_t head, size_t buf_size,
    ImportanceMetric metric,
    const ImportanceConfig& cfg,
    double occupancy_ratio = 0.0)
{
    if (buf_size <= 2) return 0;

    size_t protect = cfg.boundary_protect;
    size_t search_start = (protect > 0) ? protect : 1;
    size_t search_end = (buf_size > protect) ? (buf_size - protect) : buf_size;
    if (search_start >= search_end) return search_start;

    double min_score = std::numeric_limits<double>::max();
    size_t min_idx = search_start;

    double a = cfg.alpha, b = cfg.beta, g = cfg.gamma;
    if (metric == ImportanceMetric::ADAPTIVE_COMPOSITE) {
        double shift = cfg.occupancy_shift * occupancy_ratio;
        g = std::min(0.95, g + shift);
        a = std::max(0.01, a - shift * 0.5);
        double total = a + b + g;
        a /= total; b /= total; g /= total;
    }

    for (size_t i = search_start; i < search_end; ++i) {
        double score;
        switch (metric) {
            case ImportanceMetric::FIRST_ORDER:
                score = raw_first_order(buf, cap, head, i);
                break;
            case ImportanceMetric::SECOND_ORDER:
                score = raw_second_order(buf, cap, head, i);
                break;
            case ImportanceMetric::WINDOWED_ENERGY:
                score = raw_windowed_energy(buf, cap, head, buf_size, i, cfg.window_half);
                break;
            case ImportanceMetric::COMPOSITE:
            case ImportanceMetric::ADAPTIVE_COMPOSITE:
                score = a * raw_first_order(buf, cap, head, i)
                      + b * raw_second_order(buf, cap, head, i)
                      + g * raw_windowed_energy(buf, cap, head, buf_size, i, cfg.window_half);
                break;
            default:
                score = raw_first_order(buf, cap, head, i);
                break;
        }
        if (score < min_score) {
            min_score = score;
            min_idx = i;
        }
    }
    return min_idx;
}

inline double importance_first_order(const double* buf, size_t cap,
    size_t head, size_t, size_t idx) { return raw_first_order(buf, cap, head, idx); }

inline double importance_second_order(const double* buf, size_t cap,
    size_t head, size_t, size_t idx) { return raw_second_order(buf, cap, head, idx); }

inline double importance_windowed_energy(const double* buf, size_t cap,
    size_t head, size_t bs, size_t idx, size_t wh=3) {
    return raw_windowed_energy(buf, cap, head, bs, idx, wh); }

#endif // IMPORTANCE_METRICS_H
