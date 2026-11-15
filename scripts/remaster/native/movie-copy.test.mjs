import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const agent = resolve(fileURLToPath(new URL('../../../packages/native-host/agent/', import.meta.url)));
const hook = resolve(fileURLToPath(new URL('../../../packages/native-host/third_party/cnc-ddraw/src/hook.c', import.meta.url)));

function runC(body) {
  const directory = mkdtempSync(join(tmpdir(), 'native-movie-copy-'));
  const source = join(directory, 'test.c');
  const binary = join(directory, 'test');
  writeFileSync(source, `#include "movie-copy.h"\n#include <assert.h>\nint main(void) {\n${body}\n    return 0;\n}\n`);
  try {
    const build = spawnSync('cc', ['-std=c11', '-Wall', '-Wextra', '-Werror', '-I', agent, source, '-o', binary], {
      encoding: 'utf8',
      timeout: 10000,
    });
    assert.equal(build.status, 0, build.error?.message ?? build.stderr);
    const run = spawnSync(binary, [], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 0, run.error?.message ?? run.stderr);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test('a tracked Bink handle is not copied before its first decode, within a bounded window', () => {
  runC(`
    /* YR menu clip: copies right after BinkOpen, before BinkDoFrame (live: within 90 ms). */
    assert(nh_movie_copy_decide(1, 0, 0) == NH_MOVIE_COPY_SKIP);
    assert(nh_movie_copy_decide(1, 0, 90000) == NH_MOVIE_COPY_SKIP);
    assert(nh_movie_copy_decide(1, 0, NH_MOVIE_UNDECODED_COPY_LIMIT_US - 1) == NH_MOVIE_COPY_SKIP);
    /* A handle never decoded through the hooks is copied again after the limit, never left blank. */
    assert(nh_movie_copy_decide(1, 0, NH_MOVIE_UNDECODED_COPY_LIMIT_US) == NH_MOVIE_COPY_LATE);
    assert(nh_movie_copy_decide(1, 0, 60000000) == NH_MOVIE_COPY_LATE);
    /* Decoded handles and untracked handles keep the game's own copies at any age. */
    assert(nh_movie_copy_decide(1, 1, 0) == NH_MOVIE_COPY);
    assert(nh_movie_copy_decide(1, 1, 60000000) == NH_MOVIE_COPY);
    assert(nh_movie_copy_decide(0, 0, 0) == NH_MOVIE_COPY);
    assert(nh_movie_copy_decide(0, 1, 0) == NH_MOVIE_COPY);
    assert(NH_MOVIE_UNDECODED_COPY_LIMIT_US > 90000u);`);
});

test('handle bookkeeping: open clears, a decode (DoFrame or Goto) sets, reopen clears again', () => {
  runC(`
    struct nh_movie_copy_track track = { 1 };
    nh_movie_copy_opened(&track);
    assert(nh_movie_copy_decide(1, track.decoded, 1000) == NH_MOVIE_COPY_SKIP);
    nh_movie_copy_decoded(&track);
    assert(nh_movie_copy_decide(1, track.decoded, 1000) == NH_MOVIE_COPY);
    nh_movie_copy_decoded(&track);
    assert(track.decoded == 1);
    nh_movie_copy_opened(&track); /* Bink may hand out the same address for the next clip. */
    assert(nh_movie_copy_decide(1, track.decoded, 1000) == NH_MOVIE_COPY_SKIP);`);
});

test('cnc-ddraw routes the three Bink imports YR 1.001 uses through the agent with matching stdcall sizes', () => {
  const source = readFileSync(hook, 'utf8');
  for (const [name, wrapper, real] of [
    ['_BinkDoFrame@4', 'nh_movie_do_frame', 'nh_real_bink_do_frame'],
    ['_BinkGoto@12', 'nh_movie_goto', 'nh_real_bink_goto'],
    ['_BinkCopyToBuffer@28', 'nh_movie_copy', 'nh_real_bink_copy'],
  ]) {
    assert.match(source, new RegExp(`\\{ "${name}", \\(PROC\\)${wrapper}, \\(PROC\\*\\)&${real}, HOOK_SKIP_2 \\}`));
    assert.match(source, new RegExp(`${real} = \\([a-z_]+\\)real_GetProcAddress\\(bink, "${name}"\\)`));
  }
  const header = readFileSync(join(agent, 'agent.h'), 'utf8');
  // 4, 12 and 28 argument bytes: one, three and seven 32-bit stdcall parameters.
  assert.match(header, /int32_t WINAPI nh_movie_do_frame\(void \*movie\);/);
  assert.match(header, /void WINAPI nh_movie_goto\(void \*movie, uint32_t frame, int32_t flags\);/);
  assert.match(
    header,
    /int32_t WINAPI nh_movie_copy\(void \*movie, void \*dest, int32_t pitch, uint32_t height, uint32_t x, uint32_t y,\s+uint32_t flags\);/,
  );
});
