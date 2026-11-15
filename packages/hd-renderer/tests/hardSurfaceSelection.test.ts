import { describe, expect, it } from 'vitest';
import { measureCandidate, candidateRank, compareCandidates } from '../src/hardSurfaceSelection';
import { grid } from './hardSurfaceFixtures';
import { proposeHardSurface } from '../src/hardSurfacePropose';
import { refineHardSurface } from '../src/hardSurfaceRefine';
import type { SilhouetteView } from '../src/hardSurfaceSilhouette';

describe('automatic candidate comparison under unchanged gates', () => {
  it('remeasures the same initial part exactly and prefers a real pass over a failed low-loss record', () => {
    const solid = grid([4, 4, 4]),
      model = proposeHardSurface(solid, { category: 'chassis', budget: 2, chamfer: 0 });
    const views: SilhouetteView[] = [
      { name: 'front', width: 80, height: 80, project: ([x, y, z]) => [20 + 4 * x + 2 * y, 50 + y - 4 * z] },
    ];
    const original = refineHardSurface(solid, model, views, { maxEvaluations: 1, maxMilliseconds: 10000 });
    const fresh = measureCandidate(solid, model.part, views, 2);
    expect(fresh.metrics).toEqual(original.result.metrics);
    expect(fresh.silhouettes).toEqual(original.result.silhouettes);
    expect(fresh.pass).toBe(true);
    const failed = structuredClone(fresh);
    failed.pass = false;
    failed.loss = 0;
    expect(compareCandidates(fresh, failed)).toBeLessThan(0);
    const poor = structuredClone(fresh);
    poor.pass = false;
    poor.metrics.dev_vox_to_cad.p99 = 1;
    expect(candidateRank(poor)[1]).toBe(1);
    poor.silhouettes[0].iou = 0.96;
    expect(candidateRank(poor)[1]).toBe(2);
    poor.metrics.dev_vox_to_cad.p99 = Infinity;
    expect(compareCandidates(poor, failed)).toBeGreaterThan(0);
  });
});
