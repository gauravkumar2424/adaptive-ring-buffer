#ifndef RING_BUFFER_HEAP_H
#define RING_BUFFER_HEAP_H

#include <stdint.h>
#include <math.h>
#include <float.h>

#ifndef HEAP_RB_MAX_CAP
#define HEAP_RB_MAX_CAP 2048
#endif

typedef struct {
    float value;
    int32_t original_index;
    int16_t ll_prev;
    int16_t ll_next;
    int16_t heap_idx;
    float cached_ie;
} HeapRBNode;

typedef struct {
    HeapRBNode nodes[HEAP_RB_MAX_CAP];
    int16_t heap[HEAP_RB_MAX_CAP];
    int16_t heap_size;
    int16_t ll_head;
    int16_t ll_tail;
    int16_t free_head;
    uint16_t size;
    uint16_t capacity;
    uint32_t drops;
    uint32_t evict_count;
    uint32_t total_evict_cycles;
    uint32_t max_evict_cycles;
} HeapRingBuffer;

static inline float hrb_compute_ie(const HeapRingBuffer* rb, int16_t slot) {
    int16_t p = rb->nodes[slot].ll_prev;
    int16_t s = rb->nodes[slot].ll_next;
    if (p < 0 || s < 0) return FLT_MAX;

    float x_p = rb->nodes[p].value;
    float x_i = rb->nodes[slot].value;
    float x_s = rb->nodes[s].value;
    int32_t t_p = rb->nodes[p].original_index;
    int32_t t_i = rb->nodes[slot].original_index;
    int32_t t_s = rb->nodes[s].original_index;

    float span = (float)(t_s - t_p);
    if (span <= 0.0f) return 0.0f;
    float frac = (float)(t_i - t_p) / span;
    float x_hat = x_p + frac * (x_s - x_p);
    return fabsf(x_i - x_hat);
}

static inline int hrb_heap_less(const HeapRingBuffer* rb, int a, int b) {
    float ie_a = rb->nodes[rb->heap[a]].cached_ie;
    float ie_b = rb->nodes[rb->heap[b]].cached_ie;
    if (ie_a != ie_b) return ie_a < ie_b;
    return rb->nodes[rb->heap[a]].original_index < rb->nodes[rb->heap[b]].original_index;
}

static inline void hrb_heap_swap(HeapRingBuffer* rb, int i, int j) {
    int16_t si = rb->heap[i];
    int16_t sj = rb->heap[j];
    rb->heap[i] = sj;
    rb->heap[j] = si;
    rb->nodes[si].heap_idx = (int16_t)j;
    rb->nodes[sj].heap_idx = (int16_t)i;
}

static void hrb_heap_sift_up(HeapRingBuffer* rb, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (hrb_heap_less(rb, i, parent)) {
            hrb_heap_swap(rb, i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

static void hrb_heap_sift_down(HeapRingBuffer* rb, int i) {
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        if (left < rb->heap_size && hrb_heap_less(rb, left, smallest))
            smallest = left;
        if (right < rb->heap_size && hrb_heap_less(rb, right, smallest))
            smallest = right;
        if (smallest != i) {
            hrb_heap_swap(rb, i, smallest);
            i = smallest;
        } else {
            break;
        }
    }
}

static void hrb_heap_insert(HeapRingBuffer* rb, int16_t slot) {
    int i = rb->heap_size;
    rb->heap[i] = slot;
    rb->nodes[slot].heap_idx = (int16_t)i;
    rb->heap_size++;
    hrb_heap_sift_up(rb, i);
}

static int16_t hrb_heap_extract_min(HeapRingBuffer* rb) {
    int16_t min_slot = rb->heap[0];
    rb->heap_size--;
    rb->nodes[min_slot].heap_idx = -1;
    if (rb->heap_size > 0) {
        rb->heap[0] = rb->heap[rb->heap_size];
        rb->nodes[rb->heap[0]].heap_idx = 0;
        hrb_heap_sift_down(rb, 0);
    }
    return min_slot;
}

static void hrb_heap_update(HeapRingBuffer* rb, int16_t slot) {
    int i = rb->nodes[slot].heap_idx;
    if (i < 0) return;
    hrb_heap_sift_up(rb, i);
    i = rb->nodes[slot].heap_idx;
    hrb_heap_sift_down(rb, i);
}

static void hrb_init(HeapRingBuffer* rb, uint16_t cap) {
    if (cap > HEAP_RB_MAX_CAP) cap = HEAP_RB_MAX_CAP;
    rb->capacity = cap;
    rb->size = 0;
    rb->ll_head = -1;
    rb->ll_tail = -1;
    rb->free_head = 0;
    rb->drops = 0;
    rb->heap_size = 0;
    rb->evict_count = 0;
    rb->total_evict_cycles = 0;
    rb->max_evict_cycles = 0;
    for (uint16_t i = 0; i < cap - 1; i++) {
        rb->nodes[i].ll_next = (int16_t)(i + 1);
        rb->nodes[i].ll_prev = -1;
        rb->nodes[i].heap_idx = -1;
        rb->nodes[i].cached_ie = FLT_MAX;
    }
    rb->nodes[cap - 1].ll_next = -1;
    rb->nodes[cap - 1].ll_prev = -1;
    rb->nodes[cap - 1].heap_idx = -1;
    rb->nodes[cap - 1].cached_ie = FLT_MAX;
}

static inline void hrb_recycle(HeapRingBuffer* rb, int16_t slot) {
    rb->nodes[slot].ll_prev = -1;
    rb->nodes[slot].ll_next = rb->free_head;
    rb->nodes[slot].heap_idx = -1;
    rb->free_head = slot;
}

static void hrb_evict(HeapRingBuffer* rb) {
    if (rb->size == 0) return;
    int16_t victim = hrb_heap_extract_min(rb);
    int16_t p = rb->nodes[victim].ll_prev;
    int16_t s = rb->nodes[victim].ll_next;
    if (p >= 0) rb->nodes[p].ll_next = s;
    else        rb->ll_head = s;
    if (s >= 0) rb->nodes[s].ll_prev = p;
    else        rb->ll_tail = p;
    hrb_recycle(rb, victim);
    rb->size--;
    rb->drops++;
    if (p >= 0) {
        rb->nodes[p].cached_ie = hrb_compute_ie(rb, p);
        hrb_heap_update(rb, p);
    }
    if (s >= 0) {
        rb->nodes[s].cached_ie = hrb_compute_ie(rb, s);
        hrb_heap_update(rb, s);
    }
}

static void hrb_push(HeapRingBuffer* rb, float value, int32_t original_index) {
    if (rb->size == rb->capacity) {
        hrb_evict(rb);
    }
    int16_t slot = rb->free_head;
    rb->free_head = rb->nodes[slot].ll_next;
    rb->nodes[slot].value = value;
    rb->nodes[slot].original_index = original_index;
    rb->nodes[slot].ll_next = -1;
    rb->nodes[slot].ll_prev = rb->ll_tail;
    int16_t old_tail = rb->ll_tail;
    if (old_tail >= 0) {
        rb->nodes[old_tail].ll_next = slot;
    } else {
        rb->ll_head = slot;
    }
    rb->ll_tail = slot;
    rb->size++;
    rb->nodes[slot].cached_ie = FLT_MAX;
    hrb_heap_insert(rb, slot);
    if (old_tail >= 0) {
        rb->nodes[old_tail].cached_ie = hrb_compute_ie(rb, old_tail);
        hrb_heap_update(rb, old_tail);
    }
}

static int hrb_read_surviving(const HeapRingBuffer* rb,
                               float* out_values,
                               int32_t* out_indices,
                               int max_out) {
    int count = 0;
    int16_t cur = rb->ll_head;
    while (cur >= 0 && count < max_out) {
        out_values[count] = rb->nodes[cur].value;
        out_indices[count] = rb->nodes[cur].original_index;
        count++;
        cur = rb->nodes[cur].ll_next;
    }
    return count;
}

#endif /* RING_BUFFER_HEAP_H */
