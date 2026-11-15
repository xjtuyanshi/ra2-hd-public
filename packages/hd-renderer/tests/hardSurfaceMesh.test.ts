import { describe, expect, it } from 'vitest';
import {
  buildPart,
  chamferPlanes,
  dot,
  expand,
  insidePrimitive,
  mirror,
  partPrimitives,
  polytopeFaces,
  sub,
  cross,
  withCylinderEndCaps,
  type CylinderPrimitive,
  type PartSpec,
  type Primitive,
  type Vec3,
} from '../src/hardSurfaceMesh';
const box: Primitive = { type: 'box', min: [0, 0, 0], max: [4, 4, 4], chamfer: 0.2 };
describe('hard surface mesh', () => {
  it('enumerates box vertices, outward faces and exactly twelve chamfer edges', () => {
    const p = expand(box),
      poly = polytopeFaces(p.planes),
      ch = chamferPlanes(p.planes, 0.2);
    expect(poly.vertices).toHaveLength(8);
    expect(poly.faces).toHaveLength(6);
    expect(ch).toHaveLength(12);
    expect(poly.faces.every((f) => f.area === 16)).toBe(true);
    const { mesh } = buildPart({ primitives: [box] });
    expect(new Set([...mesh.face].filter((_, i) => mesh.chamfer[i])).size).toBe(12);
    expect(mesh.positions).toBeInstanceOf(Float32Array);
    expect(mesh.indices).toBeInstanceOf(Uint32Array);
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const [a, b, c] = [0, 1, 2].map(
        (k) => Array.from(mesh.positions.slice(mesh.indices[t + k] * 3, mesh.indices[t + k] * 3 + 3)) as Vec3,
      );
      const n = Array.from(mesh.normals.slice(mesh.indices[t] * 3, mesh.indices[t] * 3 + 3));
      expect(dot(cross(sub(b, a), sub(c, a)), n)).toBeGreaterThan(0);
    }
  });
  it('suppresses the four flush shared-face edges using the outside probe rule', () => {
    const left = expand(box),
      right = expand({ ...box, min: [4, -1, -1], max: [8, 5, 5] });
    const planes = chamferPlanes(left.planes, 0.2, [left, right], 0);
    expect(planes).toHaveLength(8);
    expect(planes.every((n) => n[0] <= 0)).toBe(true);
    // The covered-face probes also suppress a matching-width continuation whose bisector probes miss both solids.
    const continuation = expand({ ...box, min: [4, 0, 0], max: [8, 4, 4] });
    expect(chamferPlanes(left.planes, 0.2, [left, continuation], 0)).toHaveLength(8);
  });
  it('requires all three covered-face probes while retaining the any-probe flush rule', () => {
    const left = expand(box),
      partial = expand({ ...box, min: [4, 0, 0], max: [8, 1.5, 4] }),
      ch = chamferPlanes(left.planes, 0.2, [left, partial], 0);
    // Only one covered-face probe on each horizontal seam meets the partial continuation: neither is hidden.
    expect(ch.some((p) => p[0] > 0 && p[2] > 0)).toBe(true);
    expect(ch.some((p) => p[0] > 0 && p[2] < 0)).toBe(true);
    const flushPatch = expand({ ...box, min: [4, 1.5, 3.9], max: [5, 2.5, 5] });
    expect(chamferPlanes(left.planes, 0.2, [left, flushPatch], 0).some((p) => p[0] > 0 && p[2] > 0)).toBe(false);
  });
  it('splits cylinder end caps without mutating the input or losing metadata', () => {
    const prim: CylinderPrimitive = {
      type: 'cyl',
      name: 'vessel',
      axis: 'y',
      c: [7, 9],
      r: 3,
      range: [-2, 8],
      chamfer: 0.25,
      paint: 'coated',
      mirror: false,
      centred: true,
      groove: { depth: 0.1 },
      cuts: [[1, 0, 0, 10]],
    };
    const before = structuredClone(prim),
      [body, a, b] = withCylinderEndCaps(prim, { depth: 0.3, inset: 0.35, chamfer: 0.14 });
    expect(body).toEqual({ ...prim, range: [-1.7, 7.7] });
    expect(a).toEqual({ ...prim, name: 'vessel_capa', r: 2.65, range: [-2, -1.7], chamfer: 0.14 });
    expect(b).toEqual({ ...prim, name: 'vessel_capb', r: 2.65, range: [7.7, 8], chamfer: 0.14 });
    expect(prim).toEqual(before);
    expect(body).not.toBe(prim);
    expect(new Set([body.range, a.range, b.range, prim.range]).size).toBe(4);
  });
  it('mirrors halfspace offsets, cylinder centres, and part opt-outs without mutating inputs', () => {
    const p = expand({ type: 'box', min: [1, 2, 3], max: [4, 6, 7] }),
      m = mirror(p, 'y', 10);
    expect(insidePrimitive([2, 16, 4], m)).toBe(true);
    expect(insidePrimitive([2, 4, 4], m)).toBe(false);
    expect(mirror(m, 'y', 10).planes).toEqual(p.planes);
    const cyl = mirror(expand({ type: 'cyl', axis: 'z', c: [2, 4], r: 2, range: [0, 5] }), 'y', 10);
    expect(cyl.cylinder?.centre).toEqual([2, 16, 0]);
    const part: PartSpec = {
      mirror: { axis: 'y', at: 10 },
      primitives: [box, { ...box, centred: true }, { ...box, mirror: false }, { ...box, off: true }],
    };
    const before = structuredClone(part),
      ps = partPrimitives(part);
    expect(ps).toHaveLength(4);
    expect(part).toEqual(before);
    const mesh = buildPart(part).mesh;
    expect(new Set(mesh.primitive)).toEqual(new Set([0, 1, 2, 3]));
    expect([...mesh.face].every((f, i) => Math.floor(f / 1000) === mesh.primitive[i])).toBe(true);
  });
  it('uses an area-corrected 48-gon and radial cylinder side normals', () => {
    const prim: Primitive = { type: 'cyl', axis: 'x', c: [3, 4], r: 2, range: [1, 6], chamfer: 0 };
    const p = expand(prim),
      poly = polytopeFaces(p.planes);
    expect(p.planes).toHaveLength(50);
    expect(poly.vertices).toHaveLength(96);
    const cap = poly.faces.find((f) => f.plane === 48)!;
    expect(cap.area).toBeCloseTo(Math.PI * 4, 8);
    const { mesh } = buildPart({ primitives: [prim] });
    for (let i = 0; i < mesh.chamfer.length; i++) {
      const n = Array.from(mesh.normals.slice(i * 3, i * 3 + 3));
      if (Math.abs(n[0]) > 0.9) continue;
      const radial: [number, number] = [mesh.positions[i * 3 + 1] - 3, mesh.positions[i * 3 + 2] - 4],
        len = Math.hypot(...radial);
      expect(n[1]).toBeCloseTo(radial[0] / len, 5);
      expect(n[2]).toBeCloseTo(radial[1] / len, 5);
    }
  });
  it('matches fused halfspace accumulation at exact lattice subsample boundaries', () => {
    const plane = expand({ type: 'poly', planes: [[-2, 0, -1, -9.5]], chamfer: 0 });
    expect(insidePrimitive([1 / 6, 1 / 6, 9 + 1 / 6], plane)).toBe(true);
    expect(insidePrimitive([1 / 6, 1 / 6, 9 + 1 / 6 - 1e-5], plane)).toBe(false);
    expect(insidePrimitive([1 / 6, 1 / 6, 9 + 1 / 6 + 1e-5], plane)).toBe(true);
  });
  it('supports box/cylinder cuts, rejects invalid normals and is byte deterministic', () => {
    const part: PartSpec = {
      primitives: [
        { ...box, cuts: [[1, 1, 0, 5]] },
        { type: 'cyl', axis: 'z', c: [8, 8], r: 2, range: [0, 3], cuts: [[1, 0, 0, 8]] },
      ],
    };
    const before = structuredClone(part),
      a = buildPart(part),
      b = buildPart(part);
    expect(a.mesh).toEqual(b.mesh);
    expect(part).toEqual(before);
    expect(a.primitives[0].planes).toHaveLength(7);
    expect(a.primitives[1].planes).toHaveLength(51);
    expect(() => expand({ type: 'poly', planes: [[0, 0, 0, 1]] })).toThrow('halfspace');
    expect(buildPart({ primitives: [] }).mesh.indices).toHaveLength(0);
  });
});
