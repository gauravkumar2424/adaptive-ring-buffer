/**
 * Heap-Augmented Ring Buffer — O(log N) Eviction
 *
 * Core idea: after evicting sample i, only TWO samples change their
 * interpolation error (i's linked-list predecessor and successor).
 * An indexed min-heap gives O(log N) extract-min + O(log N) for the
 * two neighbor updates = O(log N) total per eviction.
 *
 * Optimizations over naive heap approach:
 *   (a) Heap entries store {error, node_idx, original_index} — all
 *       comparisons touch only the heap array, zero indirection into
 *       the node array during sift operations.
 *   (b) Deterministic tie-breaking by original stream index ensures
 *       bit-identical eviction decisions vs O(N) scan.
 *   (c) Boundary detection inlined — head/tail nodes never enter heap.
 *
 * Complexity per eviction:
 *   O(N) original:    N * (load + FP divide + FP subtract + compare)
 *   O(log N) this:    extract-min(log N sifts) + 2 error recomputes
 *                     + 2 heap-updates(log N sifts each)
 *
 * Memory: original node + 10 bytes/slot overhead (heap entry + heap_pos)
 */

#ifndef HEAP_RING_BUFFER_H
#define HEAP_RING_BUFFER_H

#include <cstdint>
#include <cmath>
#include <cassert>

struct HeapRingBuffer {

    static constexpr int MAX_CAP = 8192;

    /* ---- Node: linked-list element ---- */
    struct Node {
        float    value;
        int32_t  original_index;
        int16_t  prev;            // -1 = head
        int16_t  next;            // -1 = tail
        int16_t  heap_pos;        // -1 = not in heap
    };

    /* ---- Heap entry: self-contained, no indirection needed ---- */
    struct HeapEntry {
        float    err;             // interpolation error (sort key)
        int32_t  orig_idx;        // tie-break key (full 32-bit)
        int16_t  node_idx;        // back-reference to node
    };

    /* ---- Storage (all static) ---- */
    Node      nodes[MAX_CAP];
    HeapEntry heap[MAX_CAP];
    int16_t   heap_size;
    int16_t   head, tail, free_list;
    uint16_t  size, capacity;
    uint32_t  drops;

    /* ---- Init ---- */
    void init(uint16_t cap) {
        capacity = cap; size = 0;
        head = tail = -1;
        free_list = 0;
        drops = 0;
        heap_size = 0;
        for (uint16_t i = 0; i + 1 < cap; i++) {
            nodes[i].next = i + 1;
            nodes[i].prev = -1;
            nodes[i].heap_pos = -1;
        }
        nodes[cap - 1].next = -1;
        nodes[cap - 1].prev = -1;
        nodes[cap - 1].heap_pos = -1;
    }

    /* ---- Interpolation error ---- */
    float interp_error(int16_t slot) const {
        int16_t p = nodes[slot].prev, s = nodes[slot].next;
        if (p < 0 || s < 0) return 1e30f;

        float xp = nodes[p].value, xi = nodes[slot].value, xs = nodes[s].value;
        float span = (float)(nodes[s].original_index - nodes[p].original_index);
        if (span <= 0.0f) return 0.0f;
        float frac = (float)(nodes[slot].original_index - nodes[p].original_index) / span;
        return std::fabs(xi - (xp + frac * (xs - xp)));
    }

    /* ---- Heap: compare (touches only heap array) ---- */
    bool hless(int16_t a, int16_t b) const {
        if (heap[a].err != heap[b].err) return heap[a].err < heap[b].err;
        return heap[a].orig_idx < heap[b].orig_idx;
    }

    /* ---- Heap: swap + update reverse pointers ---- */
    void hswap(int16_t a, int16_t b) {
        HeapEntry t = heap[a]; heap[a] = heap[b]; heap[b] = t;
        nodes[heap[a].node_idx].heap_pos = a;
        nodes[heap[b].node_idx].heap_pos = b;
    }

    void sift_up(int16_t p) {
        while (p > 0) {
            int16_t par = (p - 1) >> 1;
            if (hless(p, par)) { hswap(p, par); p = par; }
            else break;
        }
    }

    void sift_down(int16_t p) {
        for (;;) {
            int16_t best = p;
            int16_t l = (p << 1) + 1, r = l + 1;
            if (l < heap_size && hless(l, best)) best = l;
            if (r < heap_size && hless(r, best)) best = r;
            if (best != p) { hswap(p, best); p = best; }
            else break;
        }
    }

    void hfix(int16_t p) { sift_up(p); sift_down(p); }

    /* ---- Heap: insert node ---- */
    void hinsert(int16_t ni) {
        int16_t p = heap_size++;
        heap[p].err      = interp_error(ni);
        heap[p].orig_idx = nodes[ni].original_index;
        heap[p].node_idx = ni;
        nodes[ni].heap_pos = p;
        sift_up(p);
    }

    /* ---- Heap: remove specific node ---- */
    void hremove(int16_t ni) {
        int16_t p = nodes[ni].heap_pos;
        if (p < 0) return;
        --heap_size;
        nodes[ni].heap_pos = -1;
        if (p == heap_size) return;         // was last slot
        heap[p] = heap[heap_size];
        nodes[heap[p].node_idx].heap_pos = p;
        hfix(p);
    }

    /* ---- Heap: refresh node's error ---- */
    void hrefresh(int16_t ni) {
        int16_t p = nodes[ni].heap_pos;
        if (p < 0) return;
        heap[p].err = interp_error(ni);
        hfix(p);
    }

    /* ---- Heap: extract min ---- */
    int16_t hextract_min() {
        int16_t mi = heap[0].node_idx;
        --heap_size;
        nodes[mi].heap_pos = -1;
        if (heap_size > 0) {
            heap[0] = heap[heap_size];
            nodes[heap[0].node_idx].heap_pos = 0;
            sift_down(0);
        }
        return mi;
    }

    /* ---- Evict: O(log N) ---- */
    void evict() {
        int16_t v = hextract_min();
        int16_t p = nodes[v].prev, n = nodes[v].next;

        // Unlink
        if (p >= 0) nodes[p].next = n; else head = n;
        if (n >= 0) nodes[n].prev = p; else tail = p;

        // Free
        nodes[v].prev = -1;
        nodes[v].next = free_list;
        nodes[v].heap_pos = -1;
        free_list = v;
        --size; ++drops;

        // Refresh predecessor
        if (p >= 0) {
            if (nodes[p].prev >= 0 && nodes[p].next >= 0) hrefresh(p);
            else if (nodes[p].heap_pos >= 0) hremove(p);
        }
        // Refresh successor
        if (n >= 0) {
            if (nodes[n].prev >= 0 && nodes[n].next >= 0) hrefresh(n);
            else if (nodes[n].heap_pos >= 0) hremove(n);
        }
    }

    /* ---- Push: O(log N) ---- */
    void push(float val, int32_t oi) {
        int16_t s = free_list;
        free_list = nodes[s].next;

        nodes[s].value = val;
        nodes[s].original_index = oi;
        nodes[s].next = -1;
        nodes[s].heap_pos = -1;

        int16_t ot = tail;
        nodes[s].prev = tail;
        if (tail >= 0) nodes[tail].next = s; else head = s;
        tail = s;
        ++size;

        // Old tail became interior — add to heap
        if (ot >= 0 && nodes[ot].prev >= 0) {
            if (nodes[ot].heap_pos < 0) hinsert(ot);
            else hrefresh(ot);
        }
    }

    /* ---- Stream push: evict-if-full + push ---- */
    void stream_push(float val, int32_t oi) {
        if (size == capacity) evict();
        push(val, oi);
    }

    /* ---- Read survivors ---- */
    int read_surviving(float* v, int32_t* ix, int mx) const {
        int c = 0; int16_t cur = head;
        while (cur >= 0 && c < mx) {
            v[c] = nodes[cur].value;
            ix[c] = nodes[cur].original_index;
            ++c; cur = nodes[cur].next;
        }
        return c;
    }

    /* ---- Debug ---- */
    bool verify_heap() const {
        for (int16_t i = 0; i < heap_size; i++) {
            if (nodes[heap[i].node_idx].heap_pos != i) return false;
            int16_t l = (i<<1)+1, r = l+1;
            if (l < heap_size && heap[l].err < heap[i].err) return false;
            if (r < heap_size && heap[r].err < heap[i].err) return false;
        }
        return true;
    }
    bool verify_errors() const {
        int16_t c = head;
        while (c >= 0) {
            if (nodes[c].prev >= 0 && nodes[c].next >= 0) {
                float act = interp_error(c);
                int16_t p = nodes[c].heap_pos;
                if (p < 0) return false;
                if (std::fabs(act - heap[p].err) > 1e-6f) return false;
            }
            c = nodes[c].next;
        }
        return true;
    }
};

#endif
