import { describe, expect, it } from 'vitest';
import { fitHardSurface } from '../src/hardSurfaceFit';
import { buildPart, insidePrimitive, type PartSpec } from '../src/hardSurfaceMesh';
import {
  cadSurfaceSamples,
  deviation,
  evaluateHardSurface,
  nearestDistances,
  rayVisible,
  revoxelise,
  voxelSurfaceSamples,
} from '../src/hardSurfaceMetrics';
import { grid } from './hardSurfaceFixtures';
describe('hard surface metrics', () => {
  it('reconstructs a box with IoU 1 and exactly coincident analytic surfaces (sample NN retains discretisation error)', () => {
    const s = grid([3, 3, 3]),
      part: PartSpec = { primitives: [{ type: 'box', min: [0, 0, 0], max: [3, 3, 3], chamfer: 0 }] };
    const { primitives } = buildPart(part),
      vs = voxelSurfaceSamples(s);
    expect(revoxelise(primitives, s.size).every((v) => v === 1)).toBe(true);
    expect(vs.every((v) => insidePrimitive(v.point, primitives[0]) && v.point.some((c) => c === 0 || c === 3))).toBe(
      true,
    );
    expect(
      deviation(
        nearestDistances(
          vs.map((v) => v.point),
          vs.map((v) => v.point),
        ),
      ),
    ).toEqual({ max: 0, p99: 0, mean: 0 });
    const result = evaluateHardSurface(s, part);
    expect(result.iou).toBe(1);
    expect(result.missing).toBe(0);
    expect(result.extra).toBe(0);
    expect(result.dev_vox_to_cad.max).toBeLessThan(0.35);
    expect(result.dev_cad_to_vox.max).toBeLessThan(0.24);
    expect(result.area_share).toEqual({ planar: 1, cylinder: 0, chamfer: 0 });
  });
  it('fits a 1:2 wedge within one voxel in both surface directions', () => {
    const s = grid([12, 4, 6], (p) => p[0] + 0.5 + 2 * (p[2] + 0.5) <= 12.5);
    const spec = fitHardSurface(
      {
        parts: {
          wedge: {
            components: [
              {
                region: [
                  [0, 0, 0],
                  [12, 4, 6],
                ],
                chamfer: 0,
              },
            ],
          },
        },
      },
      { wedge: s },
    );
    const p = spec.parts.wedge.primitives[0];
    expect(p.type).toBe('poly');
    if (p.type === 'poly') expect(p.planes.some((n) => n[0] === 1 && n[1] === 0 && n[2] === 2)).toBe(true);
    const result = evaluateHardSurface(s, spec.parts.wedge);
    // Synthetic Python baseline: 168 source cells, 20 extra cells after majority re-voxelisation.
    expect(result.iou).toBe(0.8936);
    expect(result.missing).toBe(0);
    expect(result.extra).toBe(20);
    expect(result.dev_vox_to_cad.max).toBeLessThanOrEqual(1);
    expect(result.dev_cad_to_vox.max).toBeLessThanOrEqual(1);
  });
  it('re-voxelises a cylinder drum with IoU above 0.9 and classifies radial area', () => {
    const s = grid([14, 14, 5], (p) => (p[0] + 0.5 - 7) ** 2 + (p[1] + 0.5 - 7) ** 2 <= 36);
    const part: PartSpec = { primitives: [{ type: 'cyl', axis: 'z', c: [7, 7], r: 6, range: [0, 5] }] };
    const result = evaluateHardSurface(s, part);
    expect(result.iou).toBeGreaterThan(0.9);
    expect(result.area_share.cylinder).toBeGreaterThan(0.3);
    expect(result.area_share.chamfer).toBeGreaterThan(0);
    expect(result.dev_cad_to_vox.max).toBeLessThan(1);
    expect(evaluateHardSurface(s, part)).toEqual(result);
  });
  it('filters blocked voxel-face rays and CAD samples buried in another primitive', () => {
    const s = grid([5, 3, 3], ([x]) => x === 0 || x === 4);
    expect(rayVisible(s, { point: [1, 1.5, 1.5], normal: [1, 0, 0] })).toBe(false);
    expect(rayVisible(s, { point: [0, 1.5, 1.5], normal: [-1, 0, 0] })).toBe(true);
    const part: PartSpec = {
      primitives: [
        { type: 'box', min: [0, 0, 0], max: [3, 3, 3], chamfer: 0 },
        { type: 'box', min: [1, 1, 1], max: [2, 2, 2], chamfer: 0 },
      ],
    };
    const { primitives, meshes } = buildPart(part),
      samples = cadSurfaceSamples(primitives, meshes);
    expect(samples.length).toBeGreaterThan(0);
    expect(samples.every((s) => s.primitive === 0)).toBe(true);
    expect(cadSurfaceSamples(primitives, meshes)).toEqual(samples);
    expect(cadSurfaceSamples(primitives, meshes, 0.2, 5)).not.toEqual(samples);
  });
  it('retains the reference bucket radius, percentile interpolation and empty-union semantics', () => {
    expect([
      ...nearestDistances(
        [
          [0, 0, 0],
          [20, 0, 0],
        ],
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
      ),
    ]).toEqual([0, Infinity]);
    expect(deviation([0, 1, 2, 3, 4])).toEqual({ max: 4, p99: 3.96, mean: 2 });
    const empty = evaluateHardSurface(
      grid([1, 1, 1], () => false),
      { primitives: [] },
    );
    expect(empty.iou).toBe(1);
    expect(empty.n_cad).toBe(0);
  });
});
