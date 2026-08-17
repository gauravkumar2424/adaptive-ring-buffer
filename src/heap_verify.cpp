/*
 * heap_verify.cpp — Verify O(log N) heap gives identical results to O(N) scan
 *
 * Build: g++ -std=c++17 -O2 -Wall -o ../build/heap_verify heap_verify.cpp
 * Run:   cd ../build && ./heap_verify
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>
#include "signal_loader.h"
#include "metrics.h"
#include "ring_buffer_heap.h"

using namespace std;

/* ============================================================
 * O(N) reference implementation (same logic as ring_buffer.h)
 * Struct-based to run alongside heap version
 * ============================================================ */

#define REF_MAX_CAP 2048

typedef struct {
    float value;
    int32_t original_index;
    int16_t prev, next;
} RefNode;

typedef struct {
    RefNode nodes[REF_MAX_CAP];
    int16_t head, tail, free_head;
    uint16_t size, capacity;
    uint32_t drops;
} RefRingBuffer;

static void ref_init(RefRingBuffer* rb, uint16_t cap) {
    rb->capacity = cap;
    rb->size = 0;
    rb->head = -1;
    rb->tail = -1;
    rb->free_head = 0;
    rb->drops = 0;
    for (uint16_t i = 0; i < cap - 1; i++) {
        rb->nodes[i].next = i + 1;
        rb->nodes[i].prev = -1;
    }
    rb->nodes[cap-1].next = -1;
    rb->nodes[cap-1].prev = -1;
}

static float ref_interp_error(const RefRingBuffer* rb, int16_t slot) {
    int16_t p = rb->nodes[slot].prev;
    int16_t s = rb->nodes[slot].next;
    if (p < 0 || s < 0) return FLT_MAX;
    float x_p = rb->nodes[p].value;
    float x_i = rb->nodes[slot].value;
    float x_s = rb->nodes[s].value;
    float span = (float)(rb->nodes[s].original_index - rb->nodes[p].original_index);
    if (span <= 0.0f) return 0.0f;
    float frac = (float)(rb->nodes[slot].original_index - rb->nodes[p].original_index) / span;
    float x_hat = x_p + frac * (x_s - x_p);
    return fabsf(x_i - x_hat);
}

static void ref_evict(RefRingBuffer* rb) {
    /* Scan all interior nodes for minimum IE — O(N) */
    int16_t start = rb->nodes[rb->head].next;
    int16_t stop = rb->tail;
    if (start < 0 || start == stop) { start = rb->head; stop = -1; }

    float min_ie = FLT_MAX;
    int16_t victim = start;
    int16_t cur = start;
    while (cur >= 0 && cur != stop) {
        float ie = ref_interp_error(rb, cur);
        if (ie < min_ie || (ie == min_ie &&
            rb->nodes[cur].original_index < rb->nodes[victim].original_index)) {
            min_ie = ie;
            victim = cur;
        }
        cur = rb->nodes[cur].next;
    }

    int16_t p = rb->nodes[victim].prev;
    int16_t n = rb->nodes[victim].next;
    if (p >= 0) rb->nodes[p].next = n; else rb->head = n;
    if (n >= 0) rb->nodes[n].prev = p; else rb->tail = p;
    rb->nodes[victim].prev = -1;
    rb->nodes[victim].next = rb->free_head;
    rb->free_head = victim;
    rb->size--;
    rb->drops++;
}

static void ref_push(RefRingBuffer* rb, float val, int32_t idx) {
    if (rb->size == rb->capacity) ref_evict(rb);
    int16_t slot = rb->free_head;
    rb->free_head = rb->nodes[slot].next;
    rb->nodes[slot].value = val;
    rb->nodes[slot].original_index = idx;
    rb->nodes[slot].next = -1;
    rb->nodes[slot].prev = rb->tail;
    if (rb->tail >= 0) rb->nodes[rb->tail].next = slot;
    else rb->head = slot;
    rb->tail = slot;
    rb->size++;
}

static int ref_read(const RefRingBuffer* rb, float* vals, int32_t* idxs, int max_n) {
    int count = 0;
    int16_t cur = rb->head;
    while (cur >= 0 && count < max_n) {
        vals[count] = rb->nodes[cur].value;
        idxs[count] = rb->nodes[cur].original_index;
        count++;
        cur = rb->nodes[cur].next;
    }
    return count;
}

/* ============================================================
 * MAIN — verify identical decisions
 * ============================================================ */

static RefRingBuffer ref_rb;
static HeapRingBuffer heap_rb;

static float ref_vals[2000], heap_vals[2000];
static int32_t ref_idxs[2000], heap_idxs[2000];

int main() {
    cout << "============================================================" << endl;
    cout << "  HEAP O(log N) vs SCAN O(N) VERIFICATION" << endl;
    cout << "  Confirming identical eviction decisions" << endl;
    cout << "============================================================\n" << endl;

    auto all_signals = load_all_real_signals("../data", 2000);
    if (all_signals.empty()) {
        cerr << "ERROR: no signals loaded." << endl;
        return 1;
    }

    uint16_t buf_sizes[] = {32, 64, 128, 256, 512};
    int n_sizes = 5;
    int total_tests = 0;
    int total_pass = 0;
    int total_fail = 0;

    cout << left << setw(18) << "Signal"
         << setw(6)  << "Buf"
         << setw(8)  << "Match?"
         << setw(10) << "Ref SNR"
         << setw(10) << "Heap SNR"
         << setw(12) << "Ref(us)"
         << setw(12) << "Heap(us)"
         << setw(8)  << "Speedup"
         << endl;
    cout << string(84, '-') << endl;

    for (auto& sig : all_signals) {
        vector<float> fsig(sig.data.begin(), sig.data.end());
        int sig_len = (int)fsig.size();

        for (int b = 0; b < n_sizes; b++) {
            uint16_t cap = buf_sizes[b];

            /* === Run O(N) reference === */
            auto t0 = chrono::high_resolution_clock::now();
            ref_init(&ref_rb, cap);
            for (int i = 0; i < sig_len; i++)
                ref_push(&ref_rb, fsig[i], i);
            auto t1 = chrono::high_resolution_clock::now();

            int ref_n = ref_read(&ref_rb, ref_vals, ref_idxs, 2000);

            /* === Run O(log N) heap === */
            auto t2 = chrono::high_resolution_clock::now();
            hrb_init(&heap_rb, cap);
            for (int i = 0; i < sig_len; i++)
                hrb_push(&heap_rb, fsig[i], i);
            auto t3 = chrono::high_resolution_clock::now();

            int heap_n = hrb_read_surviving(&heap_rb, heap_vals, heap_idxs, 2000);

            /* === Compare surviving indices === */
            bool match = (ref_n == heap_n);
            if (match) {
                for (int i = 0; i < ref_n; i++) {
                    if (ref_idxs[i] != heap_idxs[i]) {
                        match = false;
                        break;
                    }
                }
            }

            /* === Compute SNR for both === */
            auto ref_recon = reconstruct_signal(
                vector<int>(ref_idxs, ref_idxs + ref_n),
                vector<double>(ref_vals, ref_vals + ref_n),
                sig_len);
            auto heap_recon = reconstruct_signal(
                vector<int>(heap_idxs, heap_idxs + heap_n),
                vector<double>(heap_vals, heap_vals + heap_n),
                sig_len);
            double ref_snr = compute_snr(sig.data, ref_recon);
            double heap_snr = compute_snr(sig.data, heap_recon);

            double ref_us = chrono::duration<double, micro>(t1 - t0).count();
            double heap_us = chrono::duration<double, micro>(t3 - t2).count();
            double speedup = ref_us / heap_us;

            total_tests++;
            if (match) total_pass++; else total_fail++;

            cout << left << setw(18) << sig.name
                 << setw(6)  << cap
                 << setw(8)  << (match ? "YES" : "FAIL")
                 << fixed << setprecision(2)
                 << setw(10) << ref_snr
                 << setw(10) << heap_snr
                 << setw(12) << ref_us
                 << setw(12) << heap_us
                 << setw(8)  << setprecision(1) << speedup << "x"
                 << endl;

            if (!match && total_fail <= 3) {
                /* Print first mismatch for debugging */
                cout << "    MISMATCH: ref_n=" << ref_n << " heap_n=" << heap_n << endl;
                for (int i = 0; i < min(ref_n, heap_n) && i < 10; i++) {
                    if (ref_idxs[i] != heap_idxs[i]) {
                        cout << "    First diff at pos " << i
                             << ": ref=" << ref_idxs[i]
                             << " heap=" << heap_idxs[i] << endl;
                        break;
                    }
                }
            }
        }
    }

    cout << "\n============================================================" << endl;
    cout << "  RESULT: " << total_pass << "/" << total_tests << " PASS, "
         << total_fail << " FAIL" << endl;
    if (total_fail == 0)
        cout << "  All eviction decisions IDENTICAL — heap is verified." << endl;
    else
        cout << "  MISMATCHES FOUND — heap has a bug, do NOT use." << endl;
    cout << "============================================================" << endl;

    return 0;
}
