import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

test('slot captures share exact bytes, commit removals and roll back without leaks or shadow writes', () => {
  const directory = mkdtempSync(join(tmpdir(), 'native-mirror-slots-'));
  const source = join(directory, 'test.c');
  const binary = join(directory, 'test');
  const include = resolve(fileURLToPath(new URL('../../../packages/native-host/agent/', import.meta.url)));
  writeFileSync(
    source,
    `
#define NH_MAX_PAGES 64u
#define NH_PAGE_SIZE 4096u
#include "mirror-slots.h"
#include <assert.h>
#include <stdlib.h>
static struct nh_mirror_slots pool;
static uint8_t bytes[NH_MIRROR_SLOT_COUNT][NH_PAGE_SIZE];
static uint8_t committed[NH_MAX_PAGES][NH_PAGE_SIZE], image[NH_MAX_PAGES][NH_PAGE_SIZE];
static uint32_t addresses[NH_MAX_PAGES], candidate[NH_MAX_PAGES], count;
static uint32_t seed = 0x842a735bu;
static uint32_t random_u32(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed;
}
/* Reconstruct ownership independently from both maps and the free chain after
 * every operation. A duplicate free, leak or dangling reference fails here. */
static void check(void) {
    uint32_t expected[NH_MIRROR_SLOT_COUNT] = {0}, seen[NH_MIRROR_SLOT_COUNT] = {0};
    uint32_t free_count = 0, slot, i;
    assert(pool.shadow_count == count);
    for (i = 0; i < count; ++i) {
        slot = pool.shadow[i]; assert(slot < NH_MIRROR_SLOT_COUNT);
        assert(++expected[slot] == 1);
        assert(!memcmp(bytes[slot], committed[i], NH_PAGE_SIZE));
    }
    for (i = 0; i < pool.capture_count; ++i) {
        slot = pool.capture[i]; assert(slot < NH_MIRROR_SLOT_COUNT);
        assert(!seen[slot]++); ++expected[slot];
        assert(!memcmp(bytes[slot], image[i], NH_PAGE_SIZE));
    }
    memset(seen, 0, sizeof(seen));
    for (slot = pool.free_head; slot != UINT32_MAX; slot = pool.next[slot]) {
        assert(slot < NH_MIRROR_SLOT_COUNT && !seen[slot]++);
        assert(!expected[slot]); ++free_count;
    }
    assert(free_count == pool.free_count);
    for (i = 0; i < NH_MIRROR_SLOT_COUNT; ++i) {
        assert(expected[i] == pool.references[i]);
        assert((expected[i] == 0) == (seen[i] == 1));
    }
}
static void finish(int commit) {
    uint32_t next_count = pool.capture_count;
    if (commit) {
        memcpy(addresses, candidate, next_count * sizeof(*addresses));
        memcpy(committed, image, next_count * NH_PAGE_SIZE);
        count = next_count;
        nh_mirror_slots_commit(&pool);
    } else nh_mirror_slots_rollback(&pool);
    check();
    assert(!pool.capture_count && pool.free_count == NH_MIRROR_SLOT_COUNT - count);
    nh_mirror_slots_rollback(&pool); check(); /* Common done after commit is harmless. */
}
static void append(uint32_t address, uint32_t old, int reset, int change) {
    uint32_t at = pool.capture_count, slot;
    candidate[at] = address;
    if (old != UINT32_MAX) memcpy(image[at], committed[old], NH_PAGE_SIZE);
    else memset(image[at], (uint8_t)random_u32(), NH_PAGE_SIZE);
    if (change) image[at][random_u32() % NH_PAGE_SIZE] ^= 1;
    if (!reset && old != UINT32_MAX && !memcmp(image[at], committed[old], NH_PAGE_SIZE)) {
        slot = nh_mirror_slots_reference(&pool, old);
        assert(slot == pool.shadow[old]);
    } else {
        slot = nh_mirror_slots_allocate(&pool);
        assert(slot != UINT32_MAX);
        /* A write into any committed slot is forbidden, even during a reset. */
        for (uint32_t i = 0; i < count; ++i) assert(pool.shadow[i] != slot);
        memcpy(bytes[slot], image[at], NH_PAGE_SIZE);
    }
    assert(slot == pool.capture[at]); check();
}
int main(void) {
    nh_mirror_slots_init(&pool); check();
    assert(nh_mirror_slots_reference(&pool, 0) == UINT32_MAX); check();
    for (uint32_t i = 0; i < NH_MAX_PAGES; ++i) append(i, UINT32_MAX, 0, 0);
    finish(1);
    /* Full unchanged capture shares every slot; duplicate references fail closed. */
    for (uint32_t i = 0; i < count; ++i) append(i, i, 0, 0);
    assert(pool.free_count == NH_MAX_PAGES);
    assert(nh_mirror_slots_reference(&pool, 0) == UINT32_MAX); check();
    finish(0);
    append(0, 0, 0, 0);
    assert(nh_mirror_slots_reference(&pool, 0) == UINT32_MAX); check(); finish(0);
    /* Worst case: old and new full images coexist, with exactly 2N slots. */
    for (uint32_t i = 0; i < count; ++i) append(i, i, 0, 1);
    assert(pool.free_count == 0);
    assert(nh_mirror_slots_allocate(&pool) == UINT32_MAX); check(); finish(0);
    for (uint32_t i = 0; i < count; ++i) append(i, i, 1, 0);
    assert(pool.free_count == 0); finish(1); /* Equal bytes still copied for reset. */
    for (uint32_t i = 0; i < count; ++i) append(i + NH_MAX_PAGES, UINT32_MAX, 0, 0);
    assert(pool.free_count == 0); finish(1); /* Entire old membership removed. */
    finish(1); /* An empty committed image releases every slot. */
    for (uint32_t attempt = 0; attempt < 3000; ++attempt) {
        uint32_t target = random_u32() % (NH_MAX_PAGES + 1), old = 0, address = 0;
        uint32_t stop = random_u32() % (target + 1), mode = random_u32() % 4;
        int reset = random_u32() % 7 == 0;
        for (uint32_t i = 0; i < target; ++i) {
            uint32_t previous;
            if (mode == 0 && i == stop) break; /* Failure partway through stage 2. */
            address += 1 + random_u32() % 3;
            while (old < count && addresses[old] < address) ++old;
            previous = old < count && addresses[old] == address ? old : UINT32_MAX;
            append(address, previous, reset, random_u32() % 3 == 0);
        }
        /* Complete but rejected/late-stage failed captures also roll back. */
        finish(mode >= 2);
        if (attempt % 113 == 0) {
            nh_mirror_slots_init(&pool); count = 0; check(); /* Disconnect/reconnect. */
        }
    }
    finish(1); assert(pool.free_count == NH_MIRROR_SLOT_COUNT);
    return 0;
}
`,
  );
  try {
    const build = spawnSync('cc', ['-std=c99', '-Wall', '-Wextra', '-Werror', '-I', include, source, '-o', binary], {
      encoding: 'utf8',
      timeout: 10000,
    });
    assert.equal(build.status, 0, build.error?.message ?? build.stderr);
    const run = spawnSync(binary, [], { encoding: 'utf8', timeout: 30000 });
    assert.equal(run.status, 0, run.error?.message ?? run.stderr);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
