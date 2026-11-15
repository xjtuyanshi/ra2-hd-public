import { describe, expect, it } from 'vitest';
import { grid } from './hardSurfaceFixtures';
import { proposeHardSurface } from '../src/hardSurfacePropose';
import { refineHardSurface } from '../src/hardSurfaceRefine';
import {
  compareSilhouettes,
  rasterSilhouette,
  solidSurfaceMesh,
  type SilhouetteView,
} from '../src/hardSurfaceSilhouette';

const views: SilhouetteView[] = Array.from({ length: 8 }, (_, i) => {
  const angle = (i * Math.PI) / 4;
  return {
    name: String(i * 45),
    width: 100,
    height: 100,
    project: ([x, y, z]) => [
      50 + (x * Math.cos(angle) - y * Math.sin(angle)) * 2,
      65 + (x * Math.sin(angle) + y * Math.cos(angle)) - 3 * z,
    ],
  };
});
describe('mesh silhouette and acceptance-driven refinement', () => {
  it('retains holes and compares all eight fixed camera facings', () => {
    const full = grid([8, 8, 1]);
    const hollow = grid([8, 8, 1], ([x, y]) => x < 1 || x >= 7 || y < 1 || y >= 7);
    const comparison = views.map((v) =>
      compareSilhouettes(
        rasterSilhouette([solidSurfaceMesh(full)], v),
        rasterSilhouette([solidSurfaceMesh(hollow)], v),
        v.name,
      ),
    );
    expect(comparison).toHaveLength(8);
    expect(comparison.some((c) => c.iou < 1)).toBe(true);
    const a = rasterSilhouette([solidSurfaceMesh(hollow)], views[0]);
    expect(compareSilhouettes(a, a, 'same').iou).toBe(1);
  });
  it('rejects clipped geometry and empty reference masks instead of reporting a false pass', () => {
    expect(() => rasterSilhouette([solidSurfaceMesh(grid([8, 8, 8]))], { ...views[0], width: 10 })).toThrow('clips');
    expect(() => compareSilhouettes(new Uint8Array(4), new Uint8Array(4), 'empty')).toThrow('empty');
  });
  it('closes the formal surface and every-facing gates on a simple source without spending the budget', () => {
    const source = grid([6, 6, 6]);
    const model = proposeHardSurface(source, { category: 'chassis', budget: 2, chamfer: 0 });
    const result = refineHardSurface(source, model, views, { maxEvaluations: 5, maxMilliseconds: 10000 });
    expect(result.stop).toBe('pass');
    expect(result.history).toHaveLength(1);
    expect(result.result.silhouettes.every((v) => v.iou >= 0.97)).toBe(true);
    expect(result.result.metrics.dev_cad_to_vox.p99).toBeLessThanOrEqual(0.85);
  });
  it('keeps non-finite initial failures as failed checkpoints and returns the best in-budget state', () => {
    const source = grid([16, 16, 6], ([x, y]) => x < 3 || y < 3);
    const model = proposeHardSurface(source, { category: 'chassis', budget: 1 });
    const result = refineHardSurface(source, model, views, { maxEvaluations: 1, maxMilliseconds: 10000 });
    expect(result.result.pass).toBe(false);
    expect(result.stop).toBe('evaluation-budget');
    expect(result.history[0].result).toBeDefined();
    expect(result.part.primitives.length).toBeLessThanOrEqual(1);
  });
  it('moves every axial segment together when correcting a biased tube axis', () => {
    const source = grid([12, 8, 8], ([, y, z]) => Math.hypot(y - 3, z - 3) < 2.3);
    const base = proposeHardSurface(source, { category: 'tube', budget: 2 });
    const region = base.regions[0],
      primitive = region.primitive;
    if (primitive.type !== 'cyl') throw new Error('Expected tube proposal');
    const parts = [
      {
        ...structuredClone(primitive),
        name: 'barrel_a',
        range: [0, 6] as [number, number],
        c: [primitive.c[0] + 0.45, primitive.c[1]] as [number, number],
      },
      {
        ...structuredClone(primitive),
        name: 'barrel_b',
        range: [6, 12] as [number, number],
        c: [primitive.c[0] + 0.45, primitive.c[1]] as [number, number],
      },
    ];
    const model = {
      ...base,
      part: { primitives: parts },
      regions: parts.map((p, i) => ({
        ...structuredClone(region),
        id: String(i),
        primitive: p,
        cells: region.cells.filter((c) => (i === 0 ? c[0] < 6 : c[0] >= 6)),
      })),
    };
    const refined = refineHardSurface(source, model, views, { maxEvaluations: 8, maxMilliseconds: 10000 });
    expect(refined.history.some((h) => h.action.startsWith('axis:') && h.accepted)).toBe(true);
    const centres = refined.part.primitives.map((p) => (p.type === 'cyl' ? p.c : []));
    expect(centres[0]).toEqual(centres[1]);
    expect(centres[0][0]).toBeLessThan(primitive.c[0] + 0.45);
  });
  it('revisits radius deltas after an accepted change instead of stopping after a single sweep', () => {
    const source = grid([3, 5, 5], ([, y, z]) => Math.hypot(y - 2, z - 2) < 1.4);
    const model = proposeHardSurface(source, { category: 'tube', budget: 1 });
    const q = model.part.primitives[0];
    if (q.type !== 'cyl') throw new Error('Expected tube');
    q.r = 2.9;
    const result = refineHardSurface(source, model, views, { maxEvaluations: 50, maxMilliseconds: 15000 });
    const after = result.part.primitives[0];
    if (after.type !== 'cyl') throw new Error('Expected tube');
    expect(after.r).toBeLessThan(2.65);
    expect(result.history.filter((h) => h.accepted && h.action.startsWith('offset:')).length).toBeGreaterThanOrEqual(3);
  });
});
