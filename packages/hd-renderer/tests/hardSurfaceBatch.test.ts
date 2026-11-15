import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { summarizeBatch, type BatchStatus } from '../../../scripts/remaster/cad/summary';

describe('offline batch evidence aggregation', () => {
  it('requires independent numerical and visual evidence and retains every requested unit', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'cad-summary-'));
    try {
      const status: BatchStatus = {
        startedAt: '2026-01-01T00:00:00Z',
        preGate: true,
        configuration: {},
        units: ['A', 'B', 'C'].map((id) => ({ id, startedAt: '', finishedAt: '', exitCode: 0, automaticMinutes: 1 })),
      };
      await writeFile(join(dir, 'batch_status.json'), JSON.stringify(status));
      for (const id of ['A', 'B', 'C']) {
        const unit = join(dir, 'units', id);
        await mkdir(unit, { recursive: true });
        await writeFile(join(unit, 'manifest.json'), JSON.stringify({ metricPass: id !== 'B' }));
        await writeFile(
          join(unit, 'metrics.json'),
          JSON.stringify({
            body: {
              metrics: { primitives: 1, dev_vox_to_cad: { p99: 0.1 }, dev_cad_to_vox: { p99: 0.2 } },
              silhouettes: [{ iou: 0.99 }],
            },
          }),
        );
        if (id !== 'A')
          await writeFile(
            join(unit, 'visual_review.json'),
            JSON.stringify({ pass: true, agentDirectedMinutes: 0.5, reason: 'Synthetic fixture' }),
          );
      }
      await summarizeBatch(dir);
      const result = JSON.parse(await readFile(join(dir, 'batch_acceptance.json'), 'utf8'));
      expect(result.fullPasses).toBe(1);
      expect(result.units.A.result).toBe('UNVERIFIED');
      expect(result.units.B.result).toBe('FAIL');
      expect(result.units.C.result).toBe('PASS');
      expect(await readFile(join(dir, 'SUMMARY.md'), 'utf8')).toContain('human UNVERIFIED');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
  it('continues after missing source failures and reports each real child exit', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'cad-batch-failure-'));
    try {
      const out = join(dir, 'out');
      expect(() =>
        execFileSync(
          process.execPath,
          [
            '--import',
            'tsx',
            resolve('scripts/remaster/cad/batch.ts'),
            '--ids',
            'A,B,C',
            '--out',
            out,
            '--cache-root',
            join(dir, 'missing'),
            '--palette',
            join(dir, 'missing.png'),
            '--house',
            'neutral',
            '--concurrency',
            '2',
            '--pre-gate',
          ],
          { stdio: 'pipe' },
        ),
      ).toThrow();
      const status = JSON.parse(await readFile(join(out, 'batch_status.json'), 'utf8')) as BatchStatus;
      expect(status.units.map((u) => u.id)).toEqual(['A', 'B', 'C']);
      expect(status.units.every((u) => u.exitCode !== 0)).toBe(true);
      expect(status.finishedAt).toBeTruthy();
      const result = JSON.parse(await readFile(join(out, 'batch_acceptance.json'), 'utf8'));
      expect(result.fullPasses).toBe(0);
      expect(Object.values(result.units).every((u: unknown) => (u as { result: string }).result === 'FAIL')).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
