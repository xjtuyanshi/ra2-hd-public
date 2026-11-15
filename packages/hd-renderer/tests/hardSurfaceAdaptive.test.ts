import { describe, expect, it } from 'vitest';
import { adaptiveHardSurfaceMesh, type SurfaceSampler } from '../src/hardSurfaceAdaptive';
import { buildPart, cross, sub, type HardSurfaceMesh, type Vec3 } from '../src/hardSurfaceMesh';
import { subdivideHardSurfaceMesh } from '../src/hardSurfaceLiveMesh';

function plate(width = 8, height = 5): HardSurfaceMesh {
  return {
    positions: Float32Array.from([0, 0, 0, width, 0, 0, width, height, 0, 0, height, 0]),
    normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    chamfer: new Uint8Array(4),
    primitive: new Uint32Array(4),
    face: new Uint32Array(4),
    indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
  };
}
function triangles(mesh: HardSurfaceMesh): Vec3[][] {
  return Array.from({ length: mesh.indices.length / 3 }, (_, t) =>
    Array.from({ length: 3 }, (_, k) => {
      const i = mesh.indices[t * 3 + k] * 3;
      return Array.from(mesh.positions.subarray(i, i + 3)) as Vec3;
    }),
  );
}
function barycentric([a, b, c]: Vec3[], p: Vec3): number[] {
  const det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  const u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det;
  const v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det;
  return [u, v, 1 - u - v];
}
function area(tri: Vec3[]): number {
  return Math.hypot(...cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))) / 2;
}
function expectNoTJunctions(mesh: HardSurfaceMesh) {
  const tris = triangles(mesh),
    points = tris.flat();
  let junctions = 0;
  for (const tri of tris)
    for (let k = 0; k < 3; k++) {
      const a = tri[k],
        b = tri[(k + 1) % 3],
        d = sub(b, a),
        length = Math.hypot(...d);
      expect(length).toBeGreaterThan(1e-7);
      for (const p of points) {
        const ap = sub(p, a),
          t = ap.reduce((s, x, i) => s + x * d[i], 0) / length ** 2;
        if (t > 1e-5 / length && t < 1 - 1e-5 / length && Math.hypot(...cross(ap, d)) / length < 1e-6) junctions++;
      }
    }
  expect(junctions).toBe(0);
}
const plain: SurfaceSampler = () => ({ index: 80, shade: 80 });

describe('adaptive CAD plates', () => {
  it('reduces a large uniform plate to two triangles, independently of fine resolution', () => {
    for (const maxEdge of [0.5, 0.25]) {
      const mesh = adaptiveHardSurfaceMesh(plate(), maxEdge, plain, () => 0);
      expect(mesh.indices.length / 3).toBe(2);
      expect(triangles(mesh).reduce((sum, t) => sum + area(t), 0)).toBeCloseTo(40, 5);
      expectNoTJunctions(mesh);
    }
  });

  it.each(['stripe', 'hole', 'touching'] as const)('preserves every fine colour triangle for %s boundaries', (kind) => {
    const source = plate(4, 3);
    const paint: SurfaceSampler = (p) => ({
      index: (
        kind === 'stripe'
          ? p[0] + p[1] * 0.6 < 2.17
          : kind === 'hole'
            ? Math.hypot(p[0] - 2, p[1] - 1.5) < 0.8
            : p[0] < 2 === p[1] < 1.5
      )
        ? 16
        : 80,
      shade: 16,
    });
    const fine = triangles(subdivideHardSurfaceMesh(source));
    const mesh = adaptiveHardSurfaceMesh(source, 0.5, paint, () => 0),
      coarse = triangles(mesh);
    expect(coarse.length).toBeLessThan(fine.length);
    for (const tri of fine) {
      const centre = [0, 1, 2].map((a) => tri.reduce((s, p) => s + p[a] / 3, 0)) as Vec3;
      const i = coarse.findIndex((t) => Math.min(...barycentric(t, centre)) >= -1e-6);
      expect(i).toBeGreaterThanOrEqual(0);
      expect(mesh.pindex[mesh.indices[i * 3]]).toBe(paint(centre, [0, 0, 1]).index);
    }
    expect(coarse.reduce((sum, t) => sum + area(t), 0)).toBeCloseTo(12, 5);
    expectNoTJunctions(mesh);
  });

  it('refines shade and AO changes, including an interior cavity, and stitches the resulting coarse/fine edges', () => {
    const ao = (p: Vec3) => Math.exp(-((p[0] - 1.5) ** 2 + (p[1] - 1.5) ** 2) * 5) * 0.7;
    const paint: SurfaceSampler = (p) => ({ index: 80, shade: 80 + p[0] * 2 });
    const mesh = adaptiveHardSurfaceMesh(plate(3, 3), 0.5, paint, ao),
      coarse = triangles(mesh);
    expect(coarse.length).toBeGreaterThan(2);
    for (const tri of triangles(subdivideHardSurfaceMesh(plate(3, 3))))
      for (const p of tri) {
        const i = coarse.findIndex((t) => Math.min(...barycentric(t, p)) >= -1e-6),
          w = barycentric(coarse[i], p);
        const sample = (array: Float32Array) => w.reduce((s, v, k) => s + v * array[mesh.indices[i * 3 + k]], 0);
        expect(Math.abs(sample(mesh.pshade) - paint(p, [0, 0, 1]).shade)).toBeLessThanOrEqual(0.50001);
        expect(Math.abs(sample(mesh.occlusion) - ao(p))).toBeLessThanOrEqual(0.12501);
      }
    expectNoTJunctions(mesh);
  });

  it('stitches creases as well as coplanar regions and retains the cylinder angular facets', () => {
    const source = buildPart({ primitives: [{ type: 'box', min: [0, 0, 0], max: [2, 2, 2], chamfer: 0 }] }).mesh;
    const mesh = adaptiveHardSurfaceMesh(
      source,
      0.5,
      (p) => ({ index: p[0] + p[1] < 2.1 ? 16 : 80, shade: 16 }),
      () => 0,
    );
    expectNoTJunctions(mesh);
    const cylinder = buildPart({
      primitives: [{ type: 'cyl', axis: 'x', c: [0, 0], r: 1, range: [0, 6], chamfer: 0 }],
    }).mesh;
    const uniform = adaptiveHardSurfaceMesh(cylinder, 0.5, plain, () => 0);
    expect(uniform.indices.length).toBe(cylinder.indices.length);
  });

  it('stitches an inclined plate with fine colour boundaries and coarse AO regions', () => {
    const source = plate(3, 3);
    for (let i = 0; i < source.positions.length; i += 3) {
      const x = source.positions[i],
        y = source.positions[i + 1];
      source.positions.set([x * 0.8 - y * 0.36 + 0.13, x * 0.6 + y * 0.48 - 0.27, y * 0.8 + 1.19], i);
      source.normals.set([0.48, -0.64, 0.6], i);
    }
    const mesh = adaptiveHardSurfaceMesh(
      source,
      0.5,
      (p) => ({ index: p[0] < 1.3 ? 16 : 80, shade: 16 }),
      () => 0,
    );
    expectNoTJunctions(mesh);
    expect(triangles(mesh).reduce((sum, t) => sum + area(t), 0)).toBeCloseTo(9, 5);
  });
});
