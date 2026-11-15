import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { describe, expect, it } from 'vitest';

const script = resolve('packages/hd-renderer/scripts/hardSurfaceSpec.ts');
function invoke(...args: string[]): string {
  return execFileSync(process.execPath, ['--import', 'tsx', script, ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}
describe('offline hard surface spec adapter (synthetic cache only)', () => {
  it('decodes only the first section, writes a deterministic spec and emits per-part JSON metrics', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hard-surface-'));
    try {
      const model = join(dir, 'model.json'),
        seed = join(dir, 'seed.json'),
        out = join(dir, 'spec.json');
      writeFileSync(
        model,
        JSON.stringify({
          parts: [
            {
              role: 'body',
              sections: [
                { size: [1, 1, 1], voxels: Buffer.from([0, 0, 0, 20, 0]).toString('base64') },
                { size: [1, 1, 1], voxels: '' },
              ],
            },
          ],
        }),
      );
      writeFileSync(
        seed,
        JSON.stringify({
          parts: {
            body: {
              components: [
                {
                  region: [
                    [0, 0, 0],
                    [1, 1, 1],
                  ],
                  dirs: 'axis',
                  chamfer: 0,
                  paint: 'house',
                },
              ],
            },
          },
        }),
      );
      const first = invoke(model, seed, out, '--metrics')
        .trim()
        .split('\n')
        .map((s) => JSON.parse(s));
      expect(first[0].parts).toEqual({ body: 1 });
      expect(first[1]).toMatchObject({ part: 'body', iou: 1, voxels: 1, missing: 0, extra: 0 });
      const before = readFileSync(out, 'utf8');
      invoke(model, seed, out);
      expect(readFileSync(out, 'utf8')).toBe(before);
      expect(JSON.parse(before).parts.body.primitives[0].paint).toBe('house');
      expect(() => invoke(model, seed, model)).toThrow('overwrite');
      // Both directory and final-file symlinks must resolve before the repository boundary check.
      const dest = resolve('packages/hd-renderer/tests'),
        inside = join(dest, 'hard-surface-forbidden-output.json');
      expect(existsSync(inside)).toBe(false);
      symlinkSync(dest, join(dir, 'repo'));
      expect(() => invoke(model, seed, join(dir, 'repo', 'hard-surface-forbidden-output.json'))).toThrow(
        'outside the repository',
      );
      symlinkSync(script, join(dir, 'linked-file.json'));
      expect(() => invoke(model, seed, join(dir, 'linked-file.json'))).toThrow('outside the repository');
      symlinkSync(inside, join(dir, 'dangling.json'));
      expect(() => invoke(model, seed, join(dir, 'dangling.json'))).toThrow('ENOENT');
      expect(existsSync(inside)).toBe(false);
      writeFileSync(
        model,
        JSON.stringify({
          parts: [{ role: 'body', sections: [{ size: [1, 1, 1], voxels: Buffer.from([0, 0]).toString('base64') }] }],
        }),
      );
      expect(() => invoke(model, seed, out)).toThrow('quintuples');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
