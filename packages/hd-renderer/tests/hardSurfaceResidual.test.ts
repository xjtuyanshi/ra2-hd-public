import { describe, expect, it } from 'vitest';
import { triangleDistanceSquared, surfaceDistanceQuery } from '../src/hardSurfaceResidual';
import { buildPart, type Vec3 } from '../src/hardSurfaceMesh';

describe('independent triangle surface residual diagnostic', () => {
  it('handles face, edge, vertex and degenerate triangles analytically', () => {
    const a: Vec3 = [0, 0, 0],
      b: Vec3 = [2, 0, 0],
      c: Vec3 = [0, 2, 0];
    expect(triangleDistanceSquared([0.5, 0.5, 3], a, b, c)).toBe(9);
    expect(triangleDistanceSquared([2, 2, 0], a, b, c)).toBe(2);
    expect(triangleDistanceSquared([-1, -1, 0], a, b, c)).toBe(2);
    expect(triangleDistanceSquared([1, 1, 0], a, b, b)).toBe(1);
    expect(triangleDistanceSquared([1, 1, 0], a, a, a)).toBe(2);
  });
  it('matches a box analytically and agrees with brute force across a multilevel BVH', () => {
    const { meshes } = buildPart({ primitives: [{ type: 'box', min: [0, 0, 0], max: [2, 3, 4], chamfer: 0 }] });
    const query = surfaceDistanceQuery(meshes);
    expect(query([1, 1, 1])).toBe(1);
    expect(query([3, 4, 5])).toBeCloseTo(Math.sqrt(3));
    expect(query([1, 1, 4])).toBe(0);
    const multi = buildPart({
      primitives: Array.from({ length: 6 }, (_, i) => ({
        type: 'box' as const,
        min: [i * 4, 0, 0] as Vec3,
        max: [i * 4 + 2, 3, 4] as Vec3,
        chamfer: 0,
      })),
    }).meshes;
    const nearest = surfaceDistanceQuery(multi);
    for (let i = 0; i < 30; i++) {
      const p: Vec3 = [i * 0.83 - 1, Math.sin(i) * 4, Math.cos(i) * 5];
      let brute = Infinity;
      for (const m of multi)
        for (let j = 0; j < m.indices.length; j += 3) {
          const t = [0, 1, 2].map(
            (k) => Array.from(m.positions.slice(m.indices[j + k] * 3, m.indices[j + k] * 3 + 3)) as Vec3,
          );
          brute = Math.min(brute, triangleDistanceSquared(p, t[0], t[1], t[2]));
        }
      expect(nearest(p)).toBeCloseTo(Math.sqrt(brute), 10);
    }
    expect(() => surfaceDistanceQuery([])).toThrow('empty');
  });
});
