#ifndef RING_BUFFER_V2_H
#define RING_BUFFER_V2_H

// ============================================================
// RingBufferV2 — O(1) Eviction via Intrusive Doubly-Linked List
// ============================================================
//
// Key improvement over V1: eviction removal is O(1) (pointer
// unlink) instead of O(N) (element shift). The importance scan
// remains O(N·W), but the removal step — the part that matters
// for worst-case latency on hardware — drops from O(N) to O(1).
//
// Architecture:
//   - Fixed-size array of Node structs (pre-allocated)
//   - Doubly-linked "active list" maintains logical sample order
//   - Singly-linked "free list" recycles evicted slots
//   - Push: O(1) — take from free list, append to active tail
//   - Pop:  O(1) — unlink from active head, return to free list
//   - Evict: O(N·W) scan + O(1) unlink (no memory shifting)
//
// The linked-list traversal also gives correct logical adjacency
// for importance computation — evicted gaps are automatically
// skipped, unlike V1 which required a contiguous cache copy.
//
// Cache locality note: for typical embedded buffer sizes
// (64–1024 samples), the entire node array fits in L1/L2 cache.
// On STM32F4 with 128 KB SRAM, a 512-sample buffer of doubles
// uses ~20 KB — well within cache.
// ============================================================

#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

// ============================================================
// Combination strategies for multi-metric importance scoring
// ============================================================
enum class CombinationRule {
    WEIGHTED_LINEAR,     // S = α·d1 + β·d2 + γ·Ew
    WEIGHTED_GEOMETRIC,  // S = (d1+ε)^α · (d2+ε)^β · (Ew+ε)^γ
    RANK_FUSION,         // S = α·rank(d1) + β·rank(d2) + γ·rank(Ew)
    HARMONIC_WEIGHTED,   // S = 1 / (α/(d1+ε) + β/(d2+ε) + γ/(Ew+ε))
    WORST_CASE_MIN       // S = min(α·d1, β·d2, γ·Ew)
};

// ============================================================
// Buffer modes (V2 — compatible with V1 mode names)
// ============================================================
enum class BufferModeV2 {
    // Baselines
    DROP,                      // Drop oldest (FIFO eviction)
    RANDOM_DROP,               // Drop random sample
    DROP_MIDDLE,               // Drop middle sample
    DROP_LOW_VARIANCE,         // Drop lowest local variance
    ADAPTIVE_TIMED_WAIT,       // Adaptive timeout before eviction
    ADAPTIVE_IMPORTANCE,       // Legacy first-order only

    // Proposed: single-metric ablation
    IMPORTANCE_FIRST_ORDER,    // |x_i - x_{i-1}|
    IMPORTANCE_SECOND_ORDER,   // |d1(i) - d1(i-1)|
    IMPORTANCE_WINDOWED_ENERGY,// windowed energy Ew(i)

    // Proposed: multi-metric composite
    IMPORTANCE_COMPOSITE,      // Fixed weights α,β,γ
    IMPORTANCE_ADAPTIVE        // Occupancy-adaptive weights
};

// ============================================================
// Configuration
// ============================================================
struct ImportanceConfigV2 {
    double alpha = 0.1;          // First-order weight
    double beta  = 0.3;          // Second-order weight
    double gamma = 0.6;          // Windowed energy weight
    size_t window_half = 3;      // Energy window half-size (diameter = 2W+1)
    double occupancy_shift = 0.15; // Adaptive shift parameter δ
    size_t boundary_protect = 1; // Protect this many samples at head/tail
    CombinationRule combination = CombinationRule::WEIGHTED_LINEAR;
};

struct IndexedSampleV2 {
    double value;
    int original_index;
};

// ============================================================
// Eviction profiling statistics
// ============================================================
struct EvictionProfile {
    uint64_t total_ns;       // Cumulative eviction time
    uint64_t count;          // Number of evictions
    uint64_t max_ns;         // Worst-case single eviction
    uint64_t scan_ns;        // Time spent in importance scan
    uint64_t removal_ns;     // Time spent in actual removal (unlink)

    double avg_us() const { return count > 0 ? (double)total_ns / count / 1000.0 : 0.0; }
    double max_us() const { return (double)max_ns / 1000.0; }
    double avg_scan_us() const { return count > 0 ? (double)scan_ns / count / 1000.0 : 0.0; }
    double avg_removal_us() const { return count > 0 ? (double)removal_ns / count / 1000.0 : 0.0; }
};

// ============================================================
// RingBufferV2
// ============================================================
template <typename T>
class RingBufferV2 {
public:
    RingBufferV2(size_t capacity,
                 BufferModeV2 mode = BufferModeV2::DROP,
                 std::chrono::milliseconds timeout = std::chrono::milliseconds(20),
                 ImportanceConfigV2 config = ImportanceConfigV2())
        : capacity_(capacity), mode_(mode), timeout_(timeout), config_(config),
          size_(0), head_(-1), tail_(-1), free_head_(0),
          drop_count_(0), max_producer_wait_ms_(0.0),
          profile_{0, 0, 0, 0, 0}
    {
        nodes_.resize(capacity);
        // Build free list (singly linked through 'next')
        for (size_t i = 0; i < capacity - 1; ++i) {
            nodes_[i].next = static_cast<int>(i + 1);
            nodes_[i].prev = -1;
            nodes_[i].occupied = false;
        }
        nodes_[capacity - 1].next = -1;
        nodes_[capacity - 1].prev = -1;
        nodes_[capacity - 1].occupied = false;
    }

    // --------------------------------------------------------
    // Push — O(1) amortized, triggers eviction when full
    // --------------------------------------------------------
    void push(const T& value, int original_index = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto start = std::chrono::high_resolution_clock::now();

        while (size_ == capacity_) {
            if (mode_ == BufferModeV2::ADAPTIVE_TIMED_WAIT) {
                auto eff = std::chrono::milliseconds(static_cast<long>(
                    timeout_.count() * (1.0 - (double)size_ / capacity_)));
                if (!cond_full_.wait_for(lock, eff,
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

        // Allocate slot from free list
        int slot = free_head_;
        free_head_ = nodes_[slot].next;

        // Initialize node
        nodes_[slot].value = value;
        nodes_[slot].original_index = original_index;
        nodes_[slot].occupied = true;
        nodes_[slot].next = -1;
        nodes_[slot].prev = tail_;

        // Append to active list tail
        if (tail_ >= 0) {
            nodes_[tail_].next = slot;
        } else {
            head_ = slot;  // First node in buffer
        }
        tail_ = slot;
        ++size_;

        cond_empty_.notify_one();
    }

    // --------------------------------------------------------
    // Pop — O(1), blocks until data available
    // --------------------------------------------------------
    IndexedSampleV2 pop_indexed() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_empty_.wait(lock, [this] { return size_ > 0 || !running_; });

        if (size_ == 0) {
            throw std::runtime_error("Buffer empty and production finished");
        }

        int slot = head_;
        IndexedSampleV2 result;
        result.value = static_cast<double>(nodes_[slot].value);
        result.original_index = nodes_[slot].original_index;

        // Unlink from active list head
        head_ = nodes_[slot].next;
        if (head_ >= 0) {
            nodes_[head_].prev = -1;
        } else {
            tail_ = -1;  // Buffer is now empty
        }

        // Return slot to free list
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

    // --------------------------------------------------------
    // Accessors
    // --------------------------------------------------------
    size_t getDropCount()         const { return drop_count_; }
    double getMaxProducerWaitMs() const { return max_producer_wait_ms_; }
    size_t getSize()              const { return size_; }
    size_t getCapacity()          const { return capacity_; }
    EvictionProfile getProfile()  const { return profile_; }

private:
    // ========================================================
    // Node structure — intrusive doubly-linked list element
    // ========================================================
    struct Node {
        T value;
        int original_index = -1;
        int prev = -1;          // Active list predecessor (-1 = head)
        int next = -1;          // Active list successor / free list next
        bool occupied = false;
    };

    std::vector<Node> nodes_;
    size_t capacity_;
    BufferModeV2 mode_;
    std::chrono::milliseconds timeout_;
    ImportanceConfigV2 config_;

    size_t size_;
    int head_;        // First node in active list
    int tail_;        // Last node in active list
    int free_head_;   // First node in free list

    bool running_ = true;
    mutable std::mutex mutex_;
    std::condition_variable cond_full_;
    std::condition_variable cond_empty_;

    size_t drop_count_ = 0;
    double max_producer_wait_ms_ = 0.0;
    EvictionProfile profile_;

    // ========================================================
    // Free list management
    // ========================================================
    void recycleSlot(int slot) {
        nodes_[slot].occupied = false;
        nodes_[slot].prev = -1;
        nodes_[slot].next = free_head_;
        free_head_ = slot;
    }

    // ========================================================
    // Importance metric computation (over linked list)
    // ========================================================

    // First-order derivative: |x_i - x_{i-1}| — O(1)
    double compute_d1(int slot) const {
        int p = nodes_[slot].prev;
        if (p < 0) return 0.0;
        return std::abs(static_cast<double>(nodes_[slot].value)
                      - static_cast<double>(nodes_[p].value));
    }

    // Second-order derivative: |d1(i) - d1(i-1)| — O(1)
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

    // Windowed energy over 2W+1 neighbors — O(W)
    double compute_ew(int slot) const {
        double energy = 0.0;
        int count = 0;
        size_t wh = config_.window_half;

        // Walk backward
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

        // Walk forward
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
    // Combination strategies
    // ========================================================

    // Score a single node using the specified combination rule
    double computeScore(int slot, double a, double b, double g,
                        CombinationRule rule) const {
        double d1 = compute_d1(slot);
        double d2 = compute_d2(slot);
        double ew = compute_ew(slot);
        constexpr double eps = 1e-12;

        switch (rule) {
            case CombinationRule::WEIGHTED_LINEAR:
                return a * d1 + b * d2 + g * ew;

            case CombinationRule::WEIGHTED_GEOMETRIC:
                return std::pow(d1 + eps, a)
                     * std::pow(d2 + eps, b)
                     * std::pow(ew + eps, g);

            case CombinationRule::HARMONIC_WEIGHTED: {
                double denom = a / (d1 + eps) + b / (d2 + eps) + g / (ew + eps);
                return (denom > eps) ? 1.0 / denom : 0.0;
            }

            case CombinationRule::WORST_CASE_MIN:
                return std::min({a * d1, b * d2, g * ew});

            default: // RANK_FUSION handled separately
                return a * d1 + b * d2 + g * ew;
        }
    }

    // Rank-fusion eviction: collect all scores, rank independently,
    // combine weighted ranks. O(N·W + N·log N) per eviction.
    int findEvictionRankFusion(int search_start, int search_end,
                               double a, double b, double g) {
        struct Candidate { int slot; double d1, d2, ew; };
        std::vector<Candidate> cands;

        int cur = search_start;
        while (cur >= 0 && cur != search_end) {
            Candidate c;
            c.slot = cur;
            c.d1 = compute_d1(cur);
            c.d2 = compute_d2(cur);
            c.ew = compute_ew(cur);
            cands.push_back(c);
            cur = nodes_[cur].next;
        }

        if (cands.empty()) return head_;
        int n = (int)cands.size();

        // Rank by each metric (ascending: lowest value = rank 0 = most evictable)
        auto rank_by = [&](auto getter) {
            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                [&](int x, int y) { return getter(cands[x]) < getter(cands[y]); });
            std::vector<double> ranks(n);
            for (int r = 0; r < n; ++r) ranks[idx[r]] = (double)r / std::max(n - 1, 1);
            return ranks;
        };

        auto r1 = rank_by([](const Candidate& c) { return c.d1; });
        auto r2 = rank_by([](const Candidate& c) { return c.d2; });
        auto r3 = rank_by([](const Candidate& c) { return c.ew; });

        double min_score = std::numeric_limits<double>::max();
        int min_slot = cands[0].slot;
        for (int i = 0; i < n; ++i) {
            double score = a * r1[i] + b * r2[i] + g * r3[i];
            if (score < min_score) {
                min_score = score;
                min_slot = cands[i].slot;
            }
        }
        return min_slot;
    }

    // ========================================================
    // Core eviction — O(N·W) scan + O(1) removal
    // ========================================================
    void performEviction() {
        if (size_ == 0) return;

        auto ev_start = std::chrono::high_resolution_clock::now();
        int evict_slot = -1;

        // --- Determine search boundaries (protect edges) ---
        int search_start = head_;
        int search_end = -1;  // -1 means "go to end"
        size_t bp = config_.boundary_protect;

        if (bp > 0 && size_ > 2 * bp + 1) {
            // Skip first bp nodes
            for (size_t i = 0; i < bp && search_start >= 0; ++i)
                search_start = nodes_[search_start].next;
            // Find node bp from the tail
            int stop = tail_;
            for (size_t i = 0; i < bp && stop >= 0; ++i)
                stop = nodes_[stop].prev;
            search_end = (stop >= 0) ? nodes_[stop].next : -1;
        }
        if (search_start < 0) search_start = head_;

        // --- Importance-based modes ---
        bool is_importance = (mode_ == BufferModeV2::IMPORTANCE_FIRST_ORDER ||
                              mode_ == BufferModeV2::IMPORTANCE_SECOND_ORDER ||
                              mode_ == BufferModeV2::IMPORTANCE_WINDOWED_ENERGY ||
                              mode_ == BufferModeV2::IMPORTANCE_COMPOSITE ||
                              mode_ == BufferModeV2::IMPORTANCE_ADAPTIVE);

        auto scan_start = std::chrono::high_resolution_clock::now();

        if (is_importance) {
            // Effective weights
            double a = config_.alpha, b = config_.beta, g = config_.gamma;

            if (mode_ == BufferModeV2::IMPORTANCE_ADAPTIVE) {
                double occ = (double)size_ / capacity_;
                double shift = config_.occupancy_shift * occ;
                g = std::min(0.95, g + shift);
                a = std::max(0.01, a - shift * 0.5);
                double total = a + b + g;
                a /= total; b /= total; g /= total;
            }

            // Single-metric modes override weights
            if (mode_ == BufferModeV2::IMPORTANCE_FIRST_ORDER)
                { a = 1.0; b = 0.0; g = 0.0; }
            if (mode_ == BufferModeV2::IMPORTANCE_SECOND_ORDER)
                { a = 0.0; b = 1.0; g = 0.0; }
            if (mode_ == BufferModeV2::IMPORTANCE_WINDOWED_ENERGY)
                { a = 0.0; b = 0.0; g = 1.0; }

            CombinationRule rule = config_.combination;

            if (rule == CombinationRule::RANK_FUSION) {
                evict_slot = findEvictionRankFusion(search_start, search_end, a, b, g);
            } else {
                // Single-pass scan over active linked list
                double min_score = std::numeric_limits<double>::max();
                int cur = search_start;
                while (cur >= 0 && cur != search_end) {
                    double score = computeScore(cur, a, b, g, rule);
                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
        } else {
            // --- Baseline eviction modes ---
            switch (mode_) {
                case BufferModeV2::DROP:
                case BufferModeV2::ADAPTIVE_TIMED_WAIT:
                    evict_slot = head_;
                    break;

                case BufferModeV2::RANDOM_DROP: {
                    static thread_local std::mt19937 rng(std::random_device{}());
                    std::uniform_int_distribution<size_t> dist(0, size_ - 1);
                    size_t target = dist(rng);
                    evict_slot = head_;
                    for (size_t i = 0; i < target && evict_slot >= 0; ++i)
                        evict_slot = nodes_[evict_slot].next;
                    break;
                }

                case BufferModeV2::DROP_MIDDLE: {
                    evict_slot = head_;
                    for (size_t i = 0; i < size_ / 2 && evict_slot >= 0; ++i)
                        evict_slot = nodes_[evict_slot].next;
                    break;
                }

                case BufferModeV2::DROP_LOW_VARIANCE: {
                    double min_var = std::numeric_limits<double>::max();
                    evict_slot = head_;
                    int cur = (head_ >= 0) ? nodes_[head_].next : -1;
                    while (cur >= 0 && nodes_[cur].next >= 0) {
                        int p = nodes_[cur].prev;
                        int n = nodes_[cur].next;
                        double v1 = static_cast<double>(nodes_[p].value);
                        double v2 = static_cast<double>(nodes_[cur].value);
                        double v3 = static_cast<double>(nodes_[n].value);
                        double mean = (v1 + v2 + v3) / 3.0;
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

                case BufferModeV2::ADAPTIVE_IMPORTANCE: {
                    double min_d = std::numeric_limits<double>::max();
                    evict_slot = head_;
                    int cur = (head_ >= 0) ? nodes_[head_].next : -1;
                    while (cur >= 0) {
                        double d = compute_d1(cur);
                        if (d < min_d) { min_d = d; evict_slot = cur; }
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

        // ====================================================
        // O(1) REMOVAL — the core architectural improvement
        // V1: O(N) element shift — two array copies per position
        // V2: O(1) pointer unlink — exactly 4 pointer writes
        // ====================================================
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

        // Profile timing
        uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            ev_end - ev_start).count();
        uint64_t s_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            scan_end - scan_start).count();
        uint64_t r_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            ev_end - scan_end).count();

        profile_.total_ns += total_ns;
        profile_.scan_ns += s_ns;
        profile_.removal_ns += r_ns;
        ++profile_.count;
        if (total_ns > profile_.max_ns) profile_.max_ns = total_ns;
    }
};

#endif // RING_BUFFER_V2_H
