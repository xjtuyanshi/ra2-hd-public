import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

test('a tracked Bink handle is not copied before it decoded a frame; untracked handles copy as before', () => {
  const directory = mkdtempSync(join(tmpdir(), 'native-movie-copy-'));
  const source = join(directory, 'test.c');
  const binary = join(directory, 'test');
  const include = resolve(fileURLToPath(new URL('../../../packages/native-host/agent/', import.meta.url)));
  writeFileSync(
    source,
    `
#include "movie-copy.h"
#include <assert.h>
int main(void) {
    assert(nh_movie_copy_skipped(1, 0));  /* YR menu clip: copy right after BinkOpen, before BinkDoFrame. */
    assert(!nh_movie_copy_skipped(1, 1)); /* Every copy after the first decode goes through. */
    assert(!nh_movie_copy_skipped(0, 0)); /* Untracked handles keep the game's own behaviour. */
    assert(!nh_movie_copy_skipped(0, 1));
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
