import { describe, expect, it } from 'vitest';
import { grid } from './hardSurfaceFixtures';
import { detectMirror, proposeHardSurface, cylinderProposal, occupiedCells } from '../src/hardSurfacePropose';
import { buildPart } from '../src/hardSurfaceMesh';

describe('automatic source-supported broad regions', () => {
  it('finds a part-local mirror plane and reports rather than discards asymmetric source cells', () => {
    const symmetric = grid([8, 14, 5], ([x, y, z]) => x < 6 && y >= 2 && y < 10 && z < 4);
    expect(detectMirror(symmetric)).toMatchObject({ axis: 'y', at: 6, iou: 1, mismatched: 0 });
    symmetric.data[0] = 1;
    const evidence = detectMirror(symmetric);
    expect(evidence.at).toBe(6);
    expect(evidence.iou).toBeLessThan(1);
    expect(evidence.mismatched).toBeGreaterThan(0);
  });
  it('snaps an extended staircase to a continuous plate within the primitive budget', () => {
    const wedge = grid([14, 9, 9], ([x, y, z]) => x < 12 && y < 8 && z < 8 && x + z < 12);
    const model = proposeHardSurface(wedge, { category: 'chassis', budget: 5 });
    expect(buildPart(model.part).primitives.length).toBeLessThanOrEqual(5);
    expect(model.part.primitives.some((q) => q.type === 'poly' && q.planes.some((p) => p[0] === 1 && p[2] === 1))).toBe(
      true,
    );
    expect(model.part.primitives.every((q) => q.type !== 'box')).toBe(true);
    expect(() =>
      proposeHardSurface(
        grid([2, 2, 2], () => false),
        { category: 'chassis', budget: 2 },
      ),
    ).toThrow('empty');
  });
  it('keeps disconnected parallel tubes round and their refinements coaxial', () => {
    const source = grid(
      [18, 13, 7],
      ([x, y, z]) =>
        x < 17 &&
        ((y < 5 && Math.hypot(y - 2, z - 3) <= (x < 6 ? 2.3 : 1.8)) || (y >= 8 && Math.hypot(y - 10, z - 3) <= 2)),
    );
    const model = proposeHardSurface(source, { category: 'tube', budget: 3 });
    expect(model.regions.length).toBeGreaterThanOrEqual(2);
    expect(model.regions.length).toBeLessThanOrEqual(3);
    expect(model.part.primitives.every((q) => q.type === 'cyl' && q.axis === 'x')).toBe(true);
    for (const group of ['0', '1']) {
      const centres = model.regions
        .filter((r) => r.id.startsWith(group))
        .map((r) => (r.primitive.type === 'cyl' ? r.primitive.c : []));
      expect(new Set(centres.map((c) => JSON.stringify(c))).size).toBe(1);
    }
    expect(() => proposeHardSurface(source, { category: 'tube', budget: 1 })).toThrow('lanes');
  });
  it('proposes circular source-supported side discs without using a box-cover fallback', () => {
    const source = grid([11, 3, 11], ([x, , z]) => Math.hypot(x - 5, z - 5) < 4.7);
    const model = proposeHardSurface(source, { category: 'chassis', budget: 1 });
    expect(model.part.primitives[0].type).toBe('cyl');
    expect(model.regions[0].support).toBeGreaterThanOrEqual(0.85);
  });
  it('estimates a shared tube axis from axial slices rather than a short asymmetric base', () => {
    const source = grid(
      [20, 9, 9],
      ([x, y, z]) => x < 19 && (x < 4 ? Math.hypot(y - 4, z - 4) < 2.8 : Math.hypot(y - 3, z - 4) < 1.8),
    );
    const tube = cylinderProposal(occupiedCells(source), 'x');
    expect(tube.type).toBe('cyl');
    if (tube.type === 'cyl') expect(tube.c).toEqual([3.5, 4.5]);
  });
});
