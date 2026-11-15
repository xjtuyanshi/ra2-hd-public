import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { summarizeRoundBatch } from '../../../scripts/remaster/cad/roundSummary';
import { grid } from './hardSurfaceFixtures';

describe('revised round reconstruction evidence', () => {
  it('keeps guide/target shortfalls separate from hard assembly, budgets and required author review', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'round-summary-'));
    try {
      const ids = ['GUIDE', 'TARGET', 'MISSING_VIEW', 'NO_REVIEW', 'BUDGET', 'PROCESS'];
      await writeFile(
        join(dir, 'round_batch_status.json'),
        JSON.stringify({
          startedAt: 'synthetic',
          ids,
          units: ids.map((id) => ({ id, exitCode: id === 'PROCESS' ? 1 : 0, automaticMinutes: 1 })),
        }),
      );
      for (const id of ids) {
        const unit = join(dir, 'units', id);
        await mkdir(unit, { recursive: true });
        const role = id === 'TARGET' ? 'barrel' : 'body',
          count = id === 'BUDGET' ? 131 : 1;
        await writeFile(
          join(unit, 'spec.json'),
          JSON.stringify({
            units: 'synthetic',
            parts: {
              shape: {
                primitives: Array.from({ length: count }, () => ({
                  ...(role === 'barrel'
                    ? { type: 'cyl', axis: 'z', c: [0.5, 0.5], r: 0.5, range: [0, 8] }
                    : { type: 'box', min: [0, 0, 0], max: [1, 1, 1] }),
                  chamfer: 0,
                })),
              },
            },
          }),
        );
        await writeFile(
          join(unit, 'export.json'),
          JSON.stringify({
            parts: { shape: { role, solid: { size: [1, 1, 8], data: Array.from(grid([1, 1, 8]).data) } } },
          }),
        );
        await writeFile(
          join(unit, 'metrics.json'),
          JSON.stringify({
            shape: {
              metrics: { primitives: count, dev_vox_to_cad: { p99: 2 }, dev_cad_to_vox: { p99: 1.5 } },
              silhouettes: Array.from({ length: 8 }, () => ({ iou: id === 'TARGET' ? 0.85 : 0.99 })),
            },
          }),
        );
        await writeFile(
          join(unit, 'silhouette.json'),
          JSON.stringify({ assembled: Array.from({ length: id === 'MISSING_VIEW' ? 7 : 8 }, () => ({ iou: 0.98 })) }),
        );
        if (id !== 'NO_REVIEW')
          await writeFile(
            join(unit, 'visual_review.json'),
            JSON.stringify({
              pass: true,
              reason: 'Synthetic evidence only',
              agentDirectedMinutes: 0.1,
              sourceNothingAdded: true,
              proportionsPreserved: true,
              scalesOpened: ['1x', '2x', 'turntable_8'],
            }),
          );
      }
      await summarizeRoundBatch(dir);
      const result = JSON.parse(await readFile(join(dir, 'round_acceptance.json'), 'utf8'));
      expect(result.units.GUIDE.result).toBe('PASS');
      expect(result.units.GUIDE.p99guide).toBe(false);
      expect(result.units.TARGET.result).toBe('PASS');
      expect(result.units.TARGET.barrelTarget).toBe(false);
      expect(result.units.NO_REVIEW.result).toBe('UNVERIFIED');
      for (const id of ['MISSING_VIEW', 'BUDGET', 'PROCESS']) expect(result.units[id].result).toBe('FAIL');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
  it('records all actual failed children rather than accepting successful process creation', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'round-batch-fail-'));
    try {
      const out = join(dir, 'out');
      expect(() =>
        execFileSync(
          process.execPath,
          [
            '--import',
            'tsx',
            resolve('scripts/remaster/cad/roundBatch.ts'),
            '--ids',
            'A,B',
            '--input-root',
            join(dir, 'missing'),
            '--out',
            out,
            '--concurrency',
            '2',
          ],
          { stdio: 'pipe' },
        ),
      ).toThrow();
      const status = JSON.parse(await readFile(join(out, 'round_batch_status.json'), 'utf8'));
      expect(status.finishedAt).toBeTruthy();
      expect(status.units.map((u: { id: string }) => u.id).sort()).toEqual(['A', 'B']);
      expect(status.units.every((u: { exitCode: number }) => u.exitCode !== 0)).toBe(true);
      const result = JSON.parse(await readFile(join(out, 'round_acceptance.json'), 'utf8'));
      expect(result.fullPasses).toBe(0);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
