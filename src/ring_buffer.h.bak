#ifndef RING_BUFFER_H
#define RING_BUFFER_H

// ============================================================
// Adaptive Ring Buffer — O(1) Eviction + Interpolation Error
// ============================================================
//
// Architecture: intrusive doubly-linked list over pre-allocated
// node array. O(N·W) importance scan + O(1) pointer unlink.
//
// Key innovation (INTERP_ERROR / INTERP_COMPOSITE modes):
//   Instead of proxy metrics (derivative, curvature, energy),
//   compute the EXACT reconstruction error that would result
//   from removing each sample via linear interpolation:
//
//     x̂_i = x_p + (t_i - t_p)/(t_s - t_p) · (x_s - x_p)
//     importance(i) = |x_i - x̂_i|
//
//   This is O(1) per sample using linked-list prev/next pointers
//   and the tracked original_index for position-aware interpolation.
//   Total scan: O(N) — faster than O(N·W) proxy metrics.
//
//   INTERP_COMPOSITE adds context: I(i) = ie(i) · (1 + γ·Ew(i))
//   boosting importance in high-activity regions.
//
//   INTERP_SPECTRAL: I(i) = ie(i) · freq_weight(i)
//   Weights interpolation error by local frequency content (ZCR),
//   boosting importance of samples in spectrally-active regions.
//
//   INTERP_LOOKAHEAD (NEW): Bounded one-step lookahead that
//   penalizes eviction candidates whose removal would create
//   vulnerable neighbors (importance collapse / cascade effect).
//   For each candidate i, simulates its removal and computes
//   what the interpolation errors of its neighbors p and s
//   WOULD BECOME with the new adjacency. The score is:
//
//     cascade_penalty = max(0, ie(i) - min(new_ie(p), new_ie(s)))
//     score(i) = ie(i) + α * cascade_penalty
//
//   The penalty ONLY fires when removing i would create a neighbor
//   more vulnerable than i itself (the cascade condition). When
//   neighbors stay safe, penalty = 0 and score = ie(i) exactly.
//   Reduces to INTERP_ERROR when α = 0.
//   Still O(N) — 3 interpolation computations per candidate
//   instead of 1, constant-factor overhead only.
// ============================================================

#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "importance_metrics.h"

enum class BufferMode {
    // Baselines
    DROP,
    WAIT,
    TIMED_WAIT,
    ADAPTIVE_TIMED_WAIT,
    ADAPTIVE_IMPORTANCE,
    RANDOM_DROP,
    DROP_MIDDLE,
    DROP_LOW_VARIANCE,

    // Ablation: single-metric proxy modes
    IMPORTANCE_FIRST_ORDER,
    IMPORTANCE_SECOND_ORDER,
    IMPORTANCE_WINDOWED_ENERGY,

    // Original composite (proxy-based)
    IMPORTANCE_COMPOSITE,
    IMPORTANCE_ADAPTIVE,

    // Direct interpolation error minimization
    IMPORTANCE_INTERP_ERROR,      // O(N) — exact removal error
    IMPORTANCE_INTERP_COMPOSITE,  // O(N·W) — error × (1 + γ·Ew)

    // Frequency-informed interpolation error
    IMPORTANCE_INTERP_SPECTRAL,   // O(N·W) — error × freq_weight

    // === LOOKAHEAD ADDITION ===
    // Bounded one-step lookahead: penalizes candidates whose removal
    // would create vulnerable neighbors (importance collapse).
    // cascade_penalty = max(0, ie(i) - min(new_ie(p), new_ie(s)))
    // score(i) = ie(i) + α * cascade_penalty
    // Still O(N) — constant factor overhead over INTERP_ERROR.
    IMPORTANCE_INTERP_LOOKAHEAD   // O(N) — error with cascade penalty
};

struct IndexedSample {
    double value;
    int original_index;
};

template <typename T>
class RingBuffer {
public:
    RingBuffer(size_t capacity, BufferMode mode = BufferMode::DROP,
               std::chrono::milliseconds timeout = std::chrono::milliseconds(20),
               ImportanceConfig imp_config = ImportanceConfig())
        : capacity_(capacity),
          mode_(mode),
          timeout_(timeout),
          imp_config_(imp_config),
          size_(0),
          head_(-1),
          tail_(-1),
          free_head_(0),
          drop_count_(0),
          max_producer_wait_ms_(0.0),
          profile_{}
    {
        nodes_.resize(capacity);
        for (size_t i = 0; i < capacity - 1; ++i) {
            nodes_[i].next = static_cast<int>(i + 1);
            nodes_[i].prev = -1;
            nodes_[i].occupied = false;
        }
        nodes_[capacity - 1].next = -1;
        nodes_[capacity - 1].prev = -1;
        nodes_[capacity - 1].occupied = false;
    }

    void push(const T& value, int original_index = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto start = std::chrono::high_resolution_clock::now();

        while (size_ == capacity_) {
            if (mode_ == BufferMode::WAIT) {
                cond_full_.wait(lock);
            } else if (mode_ == BufferMode::TIMED_WAIT ||
                       mode_ == BufferMode::ADAPTIVE_TIMED_WAIT) {
                auto effective_timeout = (mode_ == BufferMode::ADAPTIVE_TIMED_WAIT)
                    ? std::chrono::milliseconds(static_cast<long>(
                        timeout_.count() * (1.0 - static_cast<double>(size_) / capacity_)))
                    : timeout_;
                if (!cond_full_.wait_for(lock, effective_timeout,
                        [this] { return size_ < capacity_; })) {
                    performEviction();
                }
            } else {
                performEviction();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double wait_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (wait_ms > max_producer_wait_ms_) max_producer_wait_ms_ = wait_ms;

        int slot = free_head_;
        free_head_ = nodes_[slot].next;

        nodes_[slot].value = value;
        nodes_[slot].original_index = original_index;
        nodes_[slot].occupied = true;
        nodes_[slot].next = -1;
        nodes_[slot].prev = tail_;

        if (tail_ >= 0) nodes_[tail_].next = slot;
        else            head_ = slot;
        tail_ = slot;
        ++size_;

        cond_empty_.notify_one();
    }

    IndexedSample pop_indexed() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_empty_.wait(lock, [this] { return size_ > 0 || !running_; });

        if (size_ == 0)
            throw std::runtime_error("Buffer is empty and production finished");

        int slot = head_;
        IndexedSample result;
        result.value = static_cast<double>(nodes_[slot].value);
        result.original_index = nodes_[slot].original_index;

        head_ = nodes_[slot].next;
        if (head_ >= 0) nodes_[head_].prev = -1;
        else            tail_ = -1;

        recycleSlot(slot);
        --size_;
        cond_full_.notify_one();
        return result;
    }

    T pop() { return static_cast<T>(pop_indexed().value); }

    void finish() {
        std::unique_lock<std::mutex> lock(mutex_);
        running_ = false;
        cond_empty_.notify_all();
    }

    size_t getDropCount()         const { return drop_count_; }
    double getMaxProducerWaitMs() const { return max_producer_wait_ms_; }
    size_t getSize()              const { return size_; }
    size_t getCapacity()          const { return capacity_; }
    EvictionProfile getProfile()  const { return profile_; }

private:
    struct Node {
        T value;
        int original_index = -1;
        int prev = -1;
        int next = -1;
        bool occupied = false;
    };

    std::vector<Node> nodes_;
    size_t capacity_;
    BufferMode mode_;
    std::chrono::milliseconds timeout_;
    ImportanceConfig imp_config_;

    size_t size_;
    int head_;
    int tail_;
    int free_head_;

    bool running_ = true;
    mutable std::mutex mutex_;
    std::condition_variable cond_full_;
    std::condition_variable cond_empty_;

    size_t drop_count_ = 0;
    double max_producer_wait_ms_ = 0.0;
    EvictionProfile profile_;

    void recycleSlot(int slot) {
        nodes_[slot].occupied = false;
        nodes_[slot].prev = -1;
        nodes_[slot].next = free_head_;
        free_head_ = slot;
    }

    // ========================================================
    // Metric computations over linked-list
    // ========================================================

    double compute_d1(int slot) const {
        int p = nodes_[slot].prev;
        if (p < 0) return 0.0;
        return std::abs(static_cast<double>(nodes_[slot].value)
                      - static_cast<double>(nodes_[p].value));
    }

    double compute_d2(int slot) const {
        int p = nodes_[slot].prev;
        if (p < 0) return 0.0;
        int pp = nodes_[p].prev;
        if (pp < 0) return 0.0;
        double d1_curr = std::abs(static_cast<double>(nodes_[slot].value)
                                - static_cast<double>(nodes_[p].value));
        double d1_prev = std::abs(static_cast<double>(nodes_[p].value)
                                - static_cast<double>(nodes_[pp].value));
        return std::abs(d1_curr - d1_prev);
    }

    double compute_ew(int slot) const {
        double energy = 0.0;
        int count = 0;
        size_t wh = imp_config_.window_half;

        int cur = slot;
        for (size_t w = 0; w < wh && cur >= 0; ++w) {
            int p = nodes_[cur].prev;
            if (p >= 0) {
                double diff = static_cast<double>(nodes_[cur].value)
                            - static_cast<double>(nodes_[p].value);
                energy += diff * diff;
                ++count;
            }
            cur = p;
        }
        cur = slot;
        for (size_t w = 0; w < wh && cur >= 0; ++w) {
            int n = nodes_[cur].next;
            if (n >= 0) {
                double diff = static_cast<double>(nodes_[n].value)
                            - static_cast<double>(nodes_[cur].value);
                energy += diff * diff;
                ++count;
            }
            cur = n;
        }
        return (count > 0) ? energy / count : 0.0;
    }

    // ========================================================
    // Direct interpolation error — O(1) per sample
    // ========================================================
    double compute_interp_error(int slot) const {
        int p = nodes_[slot].prev;
        int s = nodes_[slot].next;

        if (p < 0 || s < 0)
            return std::numeric_limits<double>::max();

        double x_p = static_cast<double>(nodes_[p].value);
        double x_i = static_cast<double>(nodes_[slot].value);
        double x_s = static_cast<double>(nodes_[s].value);

        int t_p = nodes_[p].original_index;
        int t_i = nodes_[slot].original_index;
        int t_s = nodes_[s].original_index;

        double span = static_cast<double>(t_s - t_p);
        if (span <= 0.0) return 0.0;

        double t_frac = static_cast<double>(t_i - t_p) / span;
        double x_hat = x_p + t_frac * (x_s - x_p);

        return std::abs(x_i - x_hat);
    }

    // ========================================================
    // === LOOKAHEAD ADDITION ===
    // Hypothetical interpolation error — O(1)
    //
    // Computes what a point's interpolation error WOULD BE if
    // its neighbors were changed to (hyp_prev, hyp_next).
    // Used by the lookahead mode to simulate the effect of
    // removing a candidate on its neighbors' importance.
    // ========================================================
    double compute_hypothetical_ie(int slot, int hyp_prev, int hyp_next) const {
        if (hyp_prev < 0 || hyp_next < 0)
            return std::numeric_limits<double>::max();

        double x_p = static_cast<double>(nodes_[hyp_prev].value);
        double x_i = static_cast<double>(nodes_[slot].value);
        double x_s = static_cast<double>(nodes_[hyp_next].value);

        int t_p = nodes_[hyp_prev].original_index;
        int t_i = nodes_[slot].original_index;
        int t_s = nodes_[hyp_next].original_index;

        double span = static_cast<double>(t_s - t_p);
        if (span <= 0.0) return 0.0;

        double t_frac = static_cast<double>(t_i - t_p) / span;
        double x_hat = x_p + t_frac * (x_s - x_p);

        return std::abs(x_i - x_hat);
    }

    // ========================================================
    // Local frequency content estimator — O(W) per sample
    // ========================================================
    double compute_local_freq_weight(int slot) const {
        size_t wh = imp_config_.window_half;

        double sum = 0.0;
        int count = 0;
        int cur = slot;
        for (size_t w = 0; w <= wh && cur >= 0; ++w) {
            sum += static_cast<double>(nodes_[cur].value);
            ++count;
            if (w < wh) cur = nodes_[cur].prev;
        }
        cur = nodes_[slot].next;
        for (size_t w = 0; w < wh && cur >= 0; ++w) {
            sum += static_cast<double>(nodes_[cur].value);
            ++count;
            cur = nodes_[cur].next;
        }
        double local_mean = (count > 0) ? sum / count : 0.0;

        int zc = 0;
        int window_samples = 0;

        cur = slot;
        for (size_t w = 0; w < wh && cur >= 0; ++w)
            cur = nodes_[cur].prev;
        if (cur < 0) cur = head_;

        double prev_centered = static_cast<double>(nodes_[cur].value) - local_mean;
        cur = nodes_[cur].next;

        while (cur >= 0 && window_samples < (int)(2 * wh + 1)) {
            double cur_centered = static_cast<double>(nodes_[cur].value) - local_mean;
            if (prev_centered * cur_centered < 0.0)
                ++zc;
            prev_centered = cur_centered;
            cur = nodes_[cur].next;
            ++window_samples;
        }

        double zcr = (window_samples > 0) ? (double)zc / window_samples : 0.0;
        return 1.0 + imp_config_.gamma * zcr;
    }

    // ========================================================
    // Combination scoring (for original proxy-based modes)
    // ========================================================
    double computeProxyScore(int slot, double a, double b, double g,
                             CombinationRule rule) const {
        double d1 = compute_d1(slot);
        double d2 = compute_d2(slot);
        double ew = compute_ew(slot);
        constexpr double eps = 1e-12;

        switch (rule) {
            case CombinationRule::WEIGHTED_LINEAR:
                return a * d1 + b * d2 + g * ew;
            case CombinationRule::WEIGHTED_GEOMETRIC:
                return std::pow(d1+eps, a) * std::pow(d2+eps, b) * std::pow(ew+eps, g);
            case CombinationRule::HARMONIC_WEIGHTED: {
                double denom = a/(d1+eps) + b/(d2+eps) + g/(ew+eps);
                return (denom > eps) ? 1.0/denom : 0.0;
            }
            case CombinationRule::WORST_CASE_MIN:
                return std::min({a*d1, b*d2, g*ew});
            default:
                return a * d1 + b * d2 + g * ew;
        }
    }

    int findEvictionRankFusion(int search_start, int search_end,
                               double a, double b, double g) {
        struct Cand { int slot; double d1, d2, ew; };
        std::vector<Cand> cands;
        int cur = search_start;
        while (cur >= 0 && cur != search_end) {
            cands.push_back({cur, compute_d1(cur), compute_d2(cur), compute_ew(cur)});
            cur = nodes_[cur].next;
        }
        if (cands.empty()) return head_;
        int n = (int)cands.size();
        auto rank_by = [&](auto getter) {
            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                [&](int x, int y) { return getter(cands[x]) < getter(cands[y]); });
            std::vector<double> ranks(n);
            for (int r = 0; r < n; ++r)
                ranks[idx[r]] = (double)r / std::max(n-1, 1);
            return ranks;
        };
        auto r1 = rank_by([](const Cand& c) { return c.d1; });
        auto r2 = rank_by([](const Cand& c) { return c.d2; });
        auto r3 = rank_by([](const Cand& c) { return c.ew; });
        double min_s = std::numeric_limits<double>::max();
        int min_slot = cands[0].slot;
        for (int i = 0; i < n; ++i) {
            double s = a*r1[i] + b*r2[i] + g*r3[i];
            if (s < min_s) { min_s = s; min_slot = cands[i].slot; }
        }
        return min_slot;
    }

    // ========================================================
    // Mode classification
    // ========================================================
    bool isProxyImportanceMode() const {
        return mode_ == BufferMode::IMPORTANCE_FIRST_ORDER ||
               mode_ == BufferMode::IMPORTANCE_SECOND_ORDER ||
               mode_ == BufferMode::IMPORTANCE_WINDOWED_ENERGY ||
               mode_ == BufferMode::IMPORTANCE_COMPOSITE ||
               mode_ == BufferMode::IMPORTANCE_ADAPTIVE;
    }

    bool isInterpMode() const {
        return mode_ == BufferMode::IMPORTANCE_INTERP_ERROR ||
               mode_ == BufferMode::IMPORTANCE_INTERP_COMPOSITE ||
               mode_ == BufferMode::IMPORTANCE_INTERP_SPECTRAL ||
               mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD;  // === LOOKAHEAD ADDITION ===
    }

    // ========================================================
    // Core eviction: O(N) or O(N·W) scan + O(1) unlink
    // ========================================================
    void performEviction() {
        if (size_ == 0) return;

        auto ev_start = std::chrono::high_resolution_clock::now();

        // Search boundaries (protect edges)
        int search_start = head_;
        int search_end = -1;
        size_t bp = imp_config_.boundary_protect;

        if (bp > 0 && size_ > 2 * bp + 1) {
            for (size_t i = 0; i < bp && search_start >= 0; ++i)
                search_start = nodes_[search_start].next;
            int stop = tail_;
            for (size_t i = 0; i < bp && stop >= 0; ++i)
                stop = nodes_[stop].prev;
            search_end = (stop >= 0) ? nodes_[stop].next : -1;
        }
        if (search_start < 0) search_start = head_;

        auto scan_start = std::chrono::high_resolution_clock::now();

        int evict_slot = -1;

        // ==== Interpolation-error-based eviction modes ====
        if (isInterpMode()) {
            double min_score = std::numeric_limits<double>::max();
            int cur = search_start;

            // === LOOKAHEAD ADDITION ===
            if (mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD) {
                // Bounded one-step lookahead — O(N), 3 ie computations per candidate
                //
                // For each candidate i with neighbors p (prev) and s (next):
                //   1. ie(i) = interpolation error if i is removed
                //   2. After removing i, p's new next = s, s's new prev = p
                //   3. new_ie_p = hypothetical ie of p with neighbors (p.prev, s)
                //   4. new_ie_s = hypothetical ie of s with neighbors (p, s.next)
                //   5. downstream_min = min(new_ie_p, new_ie_s)
                //   6. cascade_penalty = max(0, ie(i) - downstream_min)
                //   7. score = ie(i) + α * cascade_penalty
                //
                // The penalty ONLY activates when removing i would create a
                // neighbor MORE vulnerable than i itself (cascade condition).
                // When neighbors stay safe (downstream_min >= ie), penalty = 0
                // and this reduces exactly to INTERP_ERROR.
                // At α = 0 this also reduces exactly to INTERP_ERROR.
                double alpha = imp_config_.lookahead_alpha;
                while (cur >= 0 && cur != search_end) {
                    double ie = compute_interp_error(cur);

                    int p = nodes_[cur].prev;
                    int s = nodes_[cur].next;

                    // Compute hypothetical neighbor ies after removing cur
                    double new_ie_p = std::numeric_limits<double>::max();
                    double new_ie_s = std::numeric_limits<double>::max();

                    if (p >= 0 && s >= 0) {
                        // After removing cur: p's neighbors become (p.prev, s)
                        int pp = nodes_[p].prev;
                        new_ie_p = compute_hypothetical_ie(p, pp, s);

                        // After removing cur: s's neighbors become (p, s.next)
                        int ss = nodes_[s].next;
                        new_ie_s = compute_hypothetical_ie(s, p, ss);
                    }

                    double downstream_min = std::min(new_ie_p, new_ie_s);

                    // Clamp to avoid inf contamination
                    if (!std::isfinite(downstream_min))
                        downstream_min = ie;  // no penalty when boundary

                    // Penalty fires ONLY when removing cur would create a
                    // neighbor more vulnerable than cur itself
                    double cascade_penalty = std::max(0.0, ie - downstream_min);
                    double score = ie + alpha * cascade_penalty;

                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
            // === END LOOKAHEAD ADDITION ===
            else if (mode_ == BufferMode::IMPORTANCE_INTERP_SPECTRAL) {
                // Spectral-aware: ie × local_freq_weight — O(N·W)
                while (cur >= 0 && cur != search_end) {
                    double ie = compute_interp_error(cur);
                    double fw = compute_local_freq_weight(cur);
                    double score = ie * fw;
                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            } else if (mode_ == BufferMode::IMPORTANCE_INTERP_ERROR) {
                // Pure interpolation error — O(N)
                while (cur >= 0 && cur != search_end) {
                    double ie = compute_interp_error(cur);
                    if (ie < min_score) {
                        min_score = ie;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            } else {
                // Interp composite: ie × (1 + γ·Ew) — O(N·W)
                double g = imp_config_.gamma;
                while (cur >= 0 && cur != search_end) {
                    double ie = compute_interp_error(cur);
                    double ew = compute_ew(cur);
                    double score = ie * (1.0 + g * ew);
                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
        }
        // ==== Proxy-based importance modes (original) ====
        else if (isProxyImportanceMode()) {
            double a = imp_config_.alpha;
            double b = imp_config_.beta;
            double g = imp_config_.gamma;

            if (mode_ == BufferMode::IMPORTANCE_ADAPTIVE) {
                double occ = (double)size_ / capacity_;
                double shift = imp_config_.occupancy_shift * occ;
                g = std::min(0.95, g + shift);
                a = std::max(0.01, a - shift * 0.5);
                double total = a + b + g;
                a /= total; b /= total; g /= total;
            }

            if (mode_ == BufferMode::IMPORTANCE_FIRST_ORDER)
                { a = 1.0; b = 0.0; g = 0.0; }
            if (mode_ == BufferMode::IMPORTANCE_SECOND_ORDER)
                { a = 0.0; b = 1.0; g = 0.0; }
            if (mode_ == BufferMode::IMPORTANCE_WINDOWED_ENERGY)
                { a = 0.0; b = 0.0; g = 1.0; }

            CombinationRule rule = imp_config_.combination;

            if (rule == CombinationRule::RANK_FUSION) {
                evict_slot = findEvictionRankFusion(search_start, search_end, a, b, g);
            } else {
                double min_score = std::numeric_limits<double>::max();
                int cur = search_start;
                while (cur >= 0 && cur != search_end) {
                    double score = computeProxyScore(cur, a, b, g, rule);
                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
        }
        // ==== Legacy first-order importance ====
        else if (mode_ == BufferMode::ADAPTIVE_IMPORTANCE) {
            double min_d = std::numeric_limits<double>::max();
            int cur = (head_ >= 0) ? nodes_[head_].next : -1;
            while (cur >= 0) {
                double d = compute_d1(cur);
                if (d < min_d) { min_d = d; evict_slot = cur; }
                cur = nodes_[cur].next;
            }
        }
        // ==== Baseline eviction modes ====
        else {
            switch (mode_) {
                case BufferMode::DROP:
                case BufferMode::TIMED_WAIT:
                case BufferMode::ADAPTIVE_TIMED_WAIT:
                    evict_slot = head_;
                    break;

                case BufferMode::RANDOM_DROP: {
                    static thread_local std::mt19937 rng(std::random_device{}());
                    std::uniform_int_distribution<size_t> dist(0, size_ - 1);
                    size_t target = dist(rng);
                    evict_slot = head_;
                    for (size_t i = 0; i < target && evict_slot >= 0; ++i)
                        evict_slot = nodes_[evict_slot].next;
                    break;
                }

                case BufferMode::DROP_MIDDLE: {
                    evict_slot = head_;
                    for (size_t i = 0; i < size_/2 && evict_slot >= 0; ++i)
                        evict_slot = nodes_[evict_slot].next;
                    break;
                }

                case BufferMode::DROP_LOW_VARIANCE: {
                    double min_var = std::numeric_limits<double>::max();
                    evict_slot = head_;
                    int cur = (head_ >= 0) ? nodes_[head_].next : -1;
                    while (cur >= 0 && nodes_[cur].next >= 0) {
                        int p = nodes_[cur].prev;
                        int n = nodes_[cur].next;
                        double v1 = static_cast<double>(nodes_[p].value);
                        double v2 = static_cast<double>(nodes_[cur].value);
                        double v3 = static_cast<double>(nodes_[n].value);
                        double mean = (v1+v2+v3) / 3.0;
                        double var = ((v1-mean)*(v1-mean) + (v2-mean)*(v2-mean)
                                    + (v3-mean)*(v3-mean)) / 3.0;
                        if (var < min_var) {
                            min_var = var;
                            evict_slot = cur;
                        }
                        cur = nodes_[cur].next;
                    }
                    break;
                }

                default:
                    evict_slot = head_;
                    break;
            }
        }

        auto scan_end = std::chrono::high_resolution_clock::now();

        if (evict_slot < 0) evict_slot = head_;

        // O(1) UNLINK — 4 pointer writes
        int p = nodes_[evict_slot].prev;
        int n = nodes_[evict_slot].next;
        if (p >= 0) nodes_[p].next = n;
        else        head_ = n;
        if (n >= 0) nodes_[n].prev = p;
        else        tail_ = p;

        recycleSlot(evict_slot);
        --size_;
        ++drop_count_;

        auto ev_end = std::chrono::high_resolution_clock::now();
        uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ev_end - ev_start).count();
        uint64_t s_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(scan_end - scan_start).count();
        uint64_t r_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ev_end - scan_end).count();
        profile_.total_ns += total_ns;
        profile_.scan_ns += s_ns;
        profile_.removal_ns += r_ns;
        ++profile_.count;
        if (total_ns > profile_.max_ns) profile_.max_ns = total_ns;
    }
};

#endif // RING_BUFFER_H
