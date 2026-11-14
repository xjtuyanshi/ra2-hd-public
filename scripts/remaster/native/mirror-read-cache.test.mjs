import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

test('follow page cache is boundary-scoped and retries actual failed reads', () => {
  const directory = mkdtempSync(join(tmpdir(), 'native-read-cache-'));
  const source = join(directory, 'test.c');
  const binary = join(directory, 'test');
  const include = resolve(fileURLToPath(new URL('../../../packages/native-host/agent/', import.meta.url)));
  writeFileSync(
    source,
    `
#define NH_MAX_MESSAGE (32u * 1024u * 1024u)
#define NH_MAX_PAGES 4096u
#define NH_PAGE_SIZE 4096u
#define NH_MIRROR_HASH_SIZE 16384u
#include "mirror-read-cache.h"
#include "mirror-region-read.h"
#include <assert.h>
#include <stdlib.h>
static struct nh_read_cache cache;
static unsigned reads, epoch, failing;
static int page_read(uint32_t address, void *out, uint32_t bytes) {
    assert(!(address % NH_PAGE_SIZE) && bytes == NH_PAGE_SIZE);
    ++reads;
    if (failing && address == 0x2000) return 0;
    for (unsigned i = 0; i < bytes; ++i)
        ((uint8_t *)out)[i] = (uint8_t)((address >> 12) + i + epoch);
    return 1;
}
static struct nh_read_region_cursor cursor;
static unsigned queries, probes, deadline;
static int query(uint32_t address, struct nh_read_region *region) {
    ++queries;
    region->base = address & ~0xffffu;
    region->end = region->base + 0x10000u; region->readable = 1;
    return 1;
}
static int scoped_page_read(uint32_t address, void *out, uint32_t length) {
    ++probes;
    if (deadline) { nh_read_region_clear(&cursor); return 0; }
    return nh_read_region_copy(&cursor, address, out, length, query, page_read);
}
int main(void) {
    uint8_t bytes[8], *scratch = malloc(NH_MAX_PAGES * NH_PAGE_SIZE);
    assert(scratch);
    nh_read_cache_clear(&cache);
    assert(nh_read_cache_copy(&cache, scratch, 0x1ffe, bytes, 4, page_read));
    assert(reads == 2 && cache.count == 2);
    assert(bytes[0] == 255 && bytes[1] == 0 && bytes[2] == 2 && bytes[3] == 3);
    epoch = 7;
    assert(nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, page_read));
    assert(reads == 2 && bytes[0] == 2); /* Stable within a boundary. */
    nh_read_cache_clear(&cache);
    assert(nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, page_read));
    assert(reads == 3 && bytes[0] == 9); /* Fresh validation at the next boundary. */
    nh_read_cache_clear(&cache); failing = 1;
    assert(!nh_read_cache_copy(&cache, scratch, 0x1ffe, bytes, 4, page_read));
    assert(reads == 5 && cache.count == 1);
    failing = 0;
    assert(nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, page_read));
    assert(reads == 6 && cache.count == 2 && bytes[0] == 9);
    assert(!nh_read_cache_copy(&cache, scratch, 0xffffffff, bytes, 2, page_read));
    assert(reads == 6);
    assert(nh_read_cache_copy(&cache, scratch, 0xffffffff, bytes, 1, page_read));
    assert(reads == 7);
    nh_read_cache_clear(&cache); reads = 0;
    for (unsigned i = 0; i < NH_MAX_PAGES; ++i) {
        unsigned address = ((i * 4051u) & 4095u) * NH_PAGE_SIZE;
        assert(nh_read_cache_copy(&cache, scratch, address, bytes, 4, page_read));
    }
    for (unsigned i = 0; i < NH_MAX_PAGES; ++i)
        assert(nh_read_cache_copy(&cache, scratch, i * NH_PAGE_SIZE, bytes, 4, page_read));
    assert(reads == NH_MAX_PAGES && cache.count == NH_MAX_PAGES);
    assert(!nh_read_cache_copy(&cache, scratch, NH_MAX_PAGES * NH_PAGE_SIZE, bytes, 4, page_read));
    assert(reads == NH_MAX_PAGES);
    /* Stage-1 region metadata can be shared within this boundary, but each
     * cache fill still probes the budget and performs a fresh exact page read. */
    nh_read_cache_clear(&cache); nh_read_region_clear(&cursor); reads = 0;
    assert(nh_read_cache_copy(&cache, scratch, 0x1004, bytes, 4, scoped_page_read));
    assert(nh_read_cache_copy(&cache, scratch, 0x2004, bytes, 4, scoped_page_read));
    assert(nh_read_cache_copy(&cache, scratch, 0x1008, bytes, 4, scoped_page_read));
    assert(reads == 2 && probes == 2 && queries == 1);
    epoch = 12;
    nh_read_cache_clear(&cache); nh_read_region_clear(&cursor);
    assert(nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, scoped_page_read));
    assert(reads == 3 && probes == 3 && queries == 2 && bytes[0] == 14);
    nh_read_cache_clear(&cache); failing = 1;
    assert(!nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, scoped_page_read));
    assert(reads == 4 && probes == 4 && queries == 2 && !cursor.region.readable);
    assert(cache.count == 0);
    failing = 0;
    assert(nh_read_cache_copy(&cache, scratch, 0x2000, bytes, 4, scoped_page_read));
    assert(reads == 5 && probes == 5 && queries == 3);
    deadline = 1;
    assert(!nh_read_cache_copy(&cache, scratch, 0x3000, bytes, 4, scoped_page_read));
    assert(reads == 5 && probes == 6 && queries == 3 && !cursor.region.readable);
    deadline = 0;
    assert(nh_read_cache_copy(&cache, scratch, 0x3000, bytes, 4, scoped_page_read));
    assert(reads == 6 && probes == 7 && queries == 4);
    free(scratch);
    return 0;
}
`,
  );
  try {
    const build = spawnSync('cc', ['-std=c11', '-Wall', '-Wextra', '-Werror', '-I', include, source, '-o', binary], {
      encoding: 'utf8',
      timeout: 10000,
    });
    assert.equal(build.status, 0, build.error?.message ?? build.stderr);
    const run = spawnSync(binary, [], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 0, run.error?.message ?? run.stderr);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
