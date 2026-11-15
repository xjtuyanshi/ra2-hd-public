import { describe, it, expect } from 'vitest';
import { ellipticCylinder } from '../src/hardSurfaceRound';
import { buildPart, expand, mirror, insidePrimitive, type Vec3 } from '../src/hardSurfaceMesh';
import { parseHardSurfaceSpec } from '../src/hardSurfaceLiveMesh';
import { detectRoundShoulders } from '../src/hardSurfaceShoulder';
import { grid } from './hardSurfaceFixtures';
describe('exposed elliptic shoulders', () => {
  it('extrudes an oriented ellipse without stretching either transverse radius', () => {
    const q = expand(ellipticCylinder([2, 3, 4], [8, 9, 4], [0, 0, 1], [2, 4]));
    expect(insidePrimitive([5, 6, 5.9], q)).toBe(true);
    expect(insidePrimitive([5, 6, 6.1], q)).toBe(false);
    expect(insidePrimitive([5 + 3.9 / Math.sqrt(2), 6 - 3.9 / Math.sqrt(2), 4], q)).toBe(true);
    expect(insidePrimitive([5 + 4.1 / Math.sqrt(2), 6 - 4.1 / Math.sqrt(2), 4], q)).toBe(false);
    expect(insidePrimitive([1.9, 2.9, 4], q)).toBe(false);
  });

  it('keeps axial caps and caller cutters flat while shading the ellipse analytically', () => {
    const q = ellipticCylinder([0, 0, 0], [8, 0, 0], [0, 1, 0], [4, 2]);
    if (q.type !== 'poly') throw new Error('poly');
    q.planes.push([0, -1, 0, -2]);
    const built = buildPart({ primitives: [q] });
    expect(built.primitives).toHaveLength(1);
    const mesh = built.mesh;
    let cap = 0,
      cut = 0,
      curve = 0;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const p = Array.from(mesh.positions.slice(i, i + 3)),
        n = Array.from(mesh.normals.slice(i, i + 3));
      if (Math.abs(Math.abs(n[0]) - 1) < 1e-6) cap++;
      if (n[1] < -0.999999) cut++;
      if (Math.abs(n[0]) < 1e-6 && n[2] > 0.15 && n[1] > 0.15) curve++;
      expect(n.every(Number.isFinite)).toBe(true);
      expect(p[1]).toBeGreaterThanOrEqual(2 - 1e-5);
    }
    expect(cap).toBeGreaterThan(0);
    expect(cut).toBeGreaterThan(0);
    expect(curve).toBeGreaterThan(0);
    const reflected = mirror(expand(q), 'y', 0);
    expect(insidePrimitive([4, -3, 0], reflected)).toBe(true);
    expect(insidePrimitive([4, 3, 0], reflected)).toBe(false);
    expect(() => parseHardSurfaceSpec({ units: 'voxel', parts: { body: { primitives: [q] } } })).not.toThrow();
    const bad = structuredClone(q);
    if (bad.round?.type === 'elliptic') bad.round.radial = [1, 0, 0];
    expect(() => parseHardSurfaceSpec({ units: 'voxel', parts: { body: { primitives: [bad] } } })).toThrow();
  });
  it('requires measured curved profiles across a stable source run, and rejects a plane or straight chamfer', () => {
    const profile = (y: number) =>
      y < 7
        ? 7
        : y < 12
          ? Math.floor(7 + 6 * Math.sqrt(Math.max(0, 1 - ((y + 0.5 - 12) / 5) ** 2)) + 0.5)
          : y < 19
            ? 13
            : y < 24
              ? Math.floor(7 + 6 * Math.sqrt(Math.max(0, 1 - ((y + 0.5 - 19) / 5) ** 2)) + 0.5)
              : 7;
    const source = grid([16, 30, 18], ([x, y, z]) => x >= 2 && x < 14 && z < profile(y));
    const found = detectRoundShoulders(source);
    expect(found.length).toBeGreaterThan(0);
    expect(
      found.every((f) => f.evidence.slices >= 6 && f.profileError <= 0.8 && f.evidence.sourceSupport >= 0.95),
    ).toBe(true);
    expect(detectRoundShoulders(grid([16, 30, 18], ([x, y, z]) => z < 13))).toHaveLength(0);
    expect(detectRoundShoulders(grid([16, 30, 18], ([x, y, z]) => z < Math.min(13, 7 + y)))).toHaveLength(0);
    expect(() => ellipticCylinder([0, 0, 0], [0, 0, 0], [0, 1, 0], [2, 3])).toThrow();
  });
});
