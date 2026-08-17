#ifndef DETERMINISTIC_DRIVER_H
#define DETERMINISTIC_DRIVER_H

// ============================================================
// S1 FIX: Deterministic, drop-count-matched experiment driver.
//
// Replaces the producer/consumer thread pair in cross_domain.cpp.
//
// WHY: with threads, retention was set by OS scheduling. Each mode
// ran its own race, so IE and V-W were compared at DIFFERENT
// retained-sample counts. More retained samples => higher SNR for
// free. The reported IE-vs-V-W margin was therefore confounded.
//
// HOW: explicit drain schedule -- consumer pops one sample every
// `overload` pushes. The NUMBER of evictions then depends only on
// (signal length, capacity, schedule), never on WHICH node is
// evicted. Every mode retains exactly the same count.
//
// Fully deterministic: same inputs => bit-identical outputs.
// ============================================================

#include <vector>
#include <chrono>
#include "ring_buffer.h"
#include "signal_loader.h"

struct OnlineRun {
    std::vector<double> surv_vals;
    std::vector<int>    surv_idx;
    size_t drops = 0;
};

inline OnlineRun run_online_deterministic(
        const RealSignal& sig,
        BufferMode mode,
        size_t buf_size,
        int overload,
        const ImportanceConfig& cfg = ImportanceConfig())
{
    OnlineRun out;
    const size_t N = sig.data.size();
    out.surv_vals.reserve(N);
    out.surv_idx.reserve(N);

    RingBuffer<double> buffer(buf_size, mode,
                              std::chrono::milliseconds(2), cfg);

    for (size_t i = 0; i < N; ++i) {
        buffer.push(sig.data[i], static_cast<int>(i));

        // Drain schedule: one pop per `overload` pushes.
        if (overload > 0 && ((i + 1) % static_cast<size_t>(overload)) == 0) {
            if (buffer.getSize() > 0) {
                IndexedSample s = buffer.pop_indexed();
                out.surv_vals.push_back(s.value);
                out.surv_idx.push_back(s.original_index);
            }
        }
    }

    // Residual buffer contents are retained samples too.
    buffer.finish();
    while (buffer.getSize() > 0) {
        IndexedSample s = buffer.pop_indexed();
        out.surv_vals.push_back(s.value);
        out.surv_idx.push_back(s.original_index);
    }

    out.drops = buffer.getDropCount();
    return out;
}

// Closed form of the retained count -- depends ONLY on the schedule.
// Use this to assert matching across modes.
inline size_t expected_retained(size_t N, size_t cap, int overload) {
    size_t held = 0, popped = 0, evicted = 0;
    for (size_t i = 0; i < N; ++i) {
        if (held == cap) { ++evicted; --held; }   // evict to make room
        ++held;                                    // push
        if (overload > 0 && ((i + 1) % (size_t)overload) == 0 && held > 0) {
            --held; ++popped;
        }
    }
    (void)evicted;
    return popped + held;
}

#endif // DETERMINISTIC_DRIVER_H
