#ifndef NATIVE_HOST_MIRROR_SLOTS_H
#define NATIVE_HOST_MIRROR_SLOTS_H
#include <stdint.h>
#include <string.h>

/* At most N committed pages and N candidate pages can own slots. Before any
 * candidate allocation there are at most 2N-1 live slots, even for a full reset.
 * References never mutate bytes: only a fresh slot may be written. The caller
 * appends candidates in page order and commits only after publication succeeds. */
#define NH_MIRROR_SLOT_COUNT (2u * NH_MAX_PAGES)
struct nh_mirror_slots {
    uint32_t next[NH_MIRROR_SLOT_COUNT], free_head, free_count;
    uint8_t references[NH_MIRROR_SLOT_COUNT];
    uint32_t mappings[2][NH_MAX_PAGES], *shadow, *capture;
    uint32_t shadow_count, capture_count;
};
static void nh_mirror_slots_init(struct nh_mirror_slots *pool) {
    uint32_t i;
    for (i = 0; i < NH_MIRROR_SLOT_COUNT; ++i) pool->next[i] = i + 1;
    pool->next[NH_MIRROR_SLOT_COUNT - 1] = UINT32_MAX;
    memset(pool->references, 0, sizeof(pool->references));
    pool->free_head = 0; pool->free_count = NH_MIRROR_SLOT_COUNT;
    pool->shadow = pool->mappings[0]; pool->capture = pool->mappings[1];
    pool->shadow_count = pool->capture_count = 0;
}
static uint32_t nh_mirror_slots_allocate(struct nh_mirror_slots *pool) {
    uint32_t slot = pool->free_head;
    if (pool->capture_count == NH_MAX_PAGES || !pool->free_count ||
        slot >= NH_MIRROR_SLOT_COUNT || pool->references[slot]) return UINT32_MAX;
    pool->free_head = pool->next[slot]; --pool->free_count;
    pool->references[slot] = 1;
    pool->capture[pool->capture_count++] = slot;
    return slot;
}
static uint32_t nh_mirror_slots_reference(struct nh_mirror_slots *pool, uint32_t old) {
    uint32_t slot;
    if (pool->capture_count == NH_MAX_PAGES || old >= pool->shadow_count) return UINT32_MAX;
    slot = pool->shadow[old];
    /* Each unique shadow page can be referenced at most once by the candidate. */
    if (slot >= NH_MIRROR_SLOT_COUNT || pool->references[slot] != 1) return UINT32_MAX;
    ++pool->references[slot];
    pool->capture[pool->capture_count++] = slot;
    return slot;
}
static void nh_mirror_slots_release(struct nh_mirror_slots *pool, uint32_t slot) {
    if (--pool->references[slot]) return;
    pool->next[slot] = pool->free_head; pool->free_head = slot; ++pool->free_count;
}
static void nh_mirror_slots_rollback(struct nh_mirror_slots *pool) {
    uint32_t i;
    for (i = 0; i < pool->capture_count; ++i) nh_mirror_slots_release(pool, pool->capture[i]);
    pool->capture_count = 0;
}
static void nh_mirror_slots_commit(struct nh_mirror_slots *pool) {
    uint32_t i, *mapping;
    for (i = 0; i < pool->shadow_count; ++i) nh_mirror_slots_release(pool, pool->shadow[i]);
    mapping = pool->shadow; pool->shadow = pool->capture; pool->capture = mapping;
    pool->shadow_count = pool->capture_count; pool->capture_count = 0;
}
#endif
