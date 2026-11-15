import { mkdtemp, mkdir, writeFile, readFile, rm, symlink, realpath } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  decodeSection,
  discoverModel,
  externalOutputDirectory,
  sectionAffine,
  transformPoint,
} from '../../../scripts/remaster/cad/cache';
import type { VoxelSection } from '../../ra2-assets/src/manifest';
import { encodePng } from '../../ra2-assets/src/node/png';

const section: VoxelSection = {
  name: 'synthetic',
  size: [2, 3, 4],
  scale: 1,
  minBounds: [1, 2, 3],
  maxBounds: [5, 8, 11],
  transform: [0, -1, 0, 10, 1, 0, 0, 20, 0, 0, 2, 30],
  voxels: Buffer.from([1, 2, 3, 20, 0]).toString('base64'),
};
describe('offline local cache export boundaries', () => {
  it('runs model-ID export, both sections, formal metrics and delivery hashing in one asset-free command', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'cad-command-'));
    try {
      const cache = join(dir, 'cache'),
        out = join(dir, 'output'),
        palette = join(dir, 'unit.png');
      await mkdir(join(cache, 'voxels'), { recursive: true });
      const voxels: number[] = [];
      for (let x = 0; x < 6; x++) for (let y = 0; y < 6; y++) for (let z = 0; z < 6; z++) voxels.push(x, y, z, 20, 0);
      const cube: VoxelSection = {
        ...section,
        size: [6, 6, 6],
        minBounds: [0, 0, 0],
        maxBounds: [6, 6, 6],
        transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        voxels: Buffer.from(voxels).toString('base64'),
      };
      await writeFile(
        join(cache, 'voxels', 'SYNTH.json'),
        JSON.stringify({
          id: 'SYNTH',
          palette: 'unit',
          normalsMode: 4,
          parts: [
            {
              role: 'body',
              sections: [cube, { ...cube, name: 'second', transform: [1, 0, 0, 8, 0, 1, 0, 0, 0, 0, 1, 0] }],
            },
          ],
        }),
      );
      await writeFile(
        palette,
        encodePng({ width: 256, height: 1, channels: 3, data: new Uint8Array(256 * 3).fill(128) }),
      );
      execFileSync(
        process.execPath,
        [
          '--import',
          'tsx',
          resolve('scripts/remaster/cad/run.ts'),
          '--id',
          'SYNTH',
          '--cache-root',
          cache,
          '--palette',
          palette,
          '--house',
          'neutral',
          '--out',
          out,
          '--evaluations',
          '1',
          '--search',
          'multi',
          '--no-render',
        ],
        { stdio: 'pipe' },
      );
      const exported = JSON.parse(await readFile(join(out, 'export.json'), 'utf8'));
      expect(Object.keys(exported.parts)).toEqual(['body:0', 'body:1']);
      expect(exported.parts['body:1'].maxTransformError).toBeLessThan(1e-5);
      const spec = JSON.parse(await readFile(join(out, 'spec.json'), 'utf8'));
      expect(Object.keys(spec.parts)).toHaveLength(2);
      const manifest = JSON.parse(await readFile(join(out, 'manifest.json'), 'utf8'));
      expect(manifest.files['spec.json'].sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(manifest.files['HOURS.md']).toBeDefined();
      expect(manifest.colour.rgb).toEqual([0.6, 0.6, 0.6]);
      expect(manifest.runtime).toContain('UNVERIFIED');
      expect(manifest.sourceFilesChangedDuringRun).toEqual([]);
      const searches = JSON.parse(await readFile(join(out, 'searches.json'), 'utf8'));
      expect(Object.keys(searches['body:0'])).toEqual(['strict', 'broad']);
      expect(Object.keys(searches['body:1'])).toEqual(['strict', 'broad']);
      const selection = JSON.parse(await readFile(join(out, 'search_selection.json'), 'utf8'));
      expect(selection).toEqual({ 'body:0': 'strict', 'body:1': 'strict' });
      const selected = join(dir, 'selected');
      execFileSync(
        process.execPath,
        [
          '--import',
          'tsx',
          resolve('scripts/remaster/cad/select.ts'),
          '--candidate',
          out,
          '--out',
          selected,
          '--no-render',
        ],
        { stdio: 'pipe' },
      );
      const selectedManifest = JSON.parse(await readFile(join(selected, 'manifest.json'), 'utf8'));
      expect(selectedManifest.kind).toBe('automatic-candidate-selection');
      expect(selectedManifest.metricPass).toBe(true);
      const chosen = JSON.parse(await readFile(join(selected, 'selection.json'), 'utf8'));
      expect(chosen['body:0'].directory).toBe(await realpath(out));
      expect(chosen['body:1'].directory).toBe(await realpath(out));
      // A stored spec/export mutation cannot enter selection under its old manifest hash.
      const stored = await readFile(join(out, 'export.json'), 'utf8');
      await writeFile(join(out, 'export.json'), stored + ' ');
      expect(() =>
        execFileSync(
          process.execPath,
          [
            '--import',
            'tsx',
            resolve('scripts/remaster/cad/select.ts'),
            '--candidate',
            out,
            '--out',
            join(dir, 'rejected'),
            '--no-render',
          ],
          { stdio: 'pipe' },
        ),
      ).toThrow('manifest hash mismatch');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
  it('preserves bounds, axis swap, HVA translation and pivot without sampling a live VM', () => {
    const p = transformPoint([1, 1, 1], sectionAffine(section, 0.25));
    const unit = 1 / (30 * Math.SQRT2);
    expect(p[0]).toBeCloseTo(6 * unit + 0.25);
    expect(p[1]).toBeCloseTo(40 * unit);
    expect(p[2]).toBeCloseTo(-23 * unit);
    expect(decodeSection(section).data.reduce((s, v) => s + v, 0)).toBe(1);
    expect(() => decodeSection({ ...section, voxels: Buffer.from([3, 0, 0, 20, 0]).toString('base64') })).toThrow(
      'outside',
    );
    expect(() => decodeSection({ ...section, voxels: 'AA==' })).toThrow('quintuples');
  });
  it('discovers source sections, rejects conflicting copies and guards symlinked repository outputs', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'cad-cache-'));
    try {
      const repo = join(dir, 'repo'),
        cache = join(dir, 'cache');
      await mkdir(repo);
      await mkdir(join(cache, 'map', 'voxels'), { recursive: true });
      const source = join(cache, 'map', 'voxels', 'SYNTH.json');
      await writeFile(source, JSON.stringify({ id: 'SYNTH', parts: [{ role: 'body', sections: [section] }] }));
      expect(await discoverModel(cache, 'synth')).toBe(await realpath(source));
      await mkdir(join(cache, 'other'));
      await writeFile(join(cache, 'other', 'SYNTH.json'), JSON.stringify({ id: 'SYNTH', parts: [] }));
      await expect(discoverModel(cache, 'SYNTH')).rejects.toThrow('Ambiguous');
      await symlink(repo, join(dir, 'linked-repo'));
      await expect(externalOutputDirectory(join(dir, 'linked-repo', 'outputs'), repo)).rejects.toThrow('outside');
      await expect(externalOutputDirectory(join(repo, 'new', 'outputs'), repo)).rejects.toThrow('outside');
      expect(await externalOutputDirectory(join(dir, 'derived'), repo)).toBe(await realpath(resolve(dir, 'derived')));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
