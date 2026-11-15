import { describe, it, expect } from 'vitest';
import { coalescePart } from '../src/hardSurfaceCoalesce';
import { buildPart, insideAny, expand, type PartSpec, type Vec3 } from '../src/hardSurfaceMesh';
import { detectRoundRidges } from '../src/hardSurfaceRoundRidge';
import { patchUpperPipe } from '../src/hardSurfaceRoundPatch';
import { grid } from './hardSurfaceFixtures';
const slab = (min: Vec3, max: Vec3, paint = 'steel') => ({
  type: 'poly' as const,
  planes: expand({ type: 'box', min, max }).planes,
  chamfer: 0.1,
  paint,
  groove: 0,
});
const sameUnion = (a: PartSpec, b: PartSpec) => {
  const p = buildPart(a).primitives,
    q = buildPart(b).primitives;
  for (let x = -0.15; x < 8.3; x += 0.53)
    for (let y = -0.15; y < 4.3; y += 0.53)
      for (let z = -0.15; z < 4.3; z += 0.53) expect(insideAny([x, y, z], p)).toBe(insideAny([x, y, z], q));
};
describe('convex union coalescing and source ridges', () => {
  it('removes a shared boundary without altering the solid or opaque paint/groove metadata', () => {
    const part = { primitives: [slab([0, 0, 0], [4, 4, 4]), slab([4, 0, 0], [8, 4, 4])] };
    const result = coalescePart(part);
    expect(result.merges).toHaveLength(1);
    expect(buildPart(result.part).primitives).toHaveLength(1);
    sameUnion(part, result.part);
    expect(result.part.primitives[0].paint).toBe('steel');
    expect(result.part.primitives[0].groove).toBe(0);
  });
  it('does not bridge a concavity or merge incompatible caller-owned paint', () => {
    expect(coalescePart({ primitives: [slab([0, 0, 0], [4, 4, 4]), slab([4, 0, 0], [8, 2, 4])] }).merges).toHaveLength(
      0,
    );
    expect(
      coalescePart({ primitives: [slab([0, 0, 0], [4, 4, 4]), slab([4, 0, 0], [8, 4, 4], 'rubber')] }).merges,
    ).toHaveLength(0);
  });
  it('preserves existing subtraction when expanded flat pieces are coalesced', () => {
    const part: PartSpec = {
      primitives: [slab([0, 0, 0], [4, 4, 4]), slab([4, 0, 0], [8, 4, 4])],
      carve: [{ min: [2, 1, 1], max: [6, 3, 3] }],
    };
    const result = coalescePart(part);
    expect(result.merges.length).toBeGreaterThan(0);
    sameUnion(part, result.part);
  });
  it('fits only source-seeded narrow upper ridges and leaves a featureless plate alone', () => {
    const source = grid(
      [28, 18, 14],
      ([x, y, z]) => z < 5 || (x >= 3 && x < 25 && Math.hypot(y + 0.5 - 9, z + 0.5 - 6) < 2),
    );
    const ridges = detectRoundRidges(source);
    expect(ridges.some((f) => f.axis === 'x' && f.evidence.slices >= 15)).toBe(true);
    expect(ridges.every((f) => f.maximumHeightError <= 0.56)).toBe(true);
    expect(detectRoundRidges(grid([28, 18, 14], ([, , z]) => z < 5))).toHaveLength(0);
    const ridge = ridges.find((f) => f.axis === 'x')!,
      part = { primitives: [slab([0, 0, 0], [28, 18, 8])] };
    const p = patchUpperPipe(part, ridge, 'round', 'z', ridge.floorOffset, ridge.floorSlope);
    expect(insideAny([10, 9, 4], buildPart(p).primitives)).toBe(true);
  });
  it('retains the original curved owner and applies existing cuts exactly once after flat merges', () => {
    const tube = {
      type: 'cyl' as const,
      axis: 'x' as const,
      c: [2, 2] as [number, number],
      r: 1.3,
      range: [0, 8] as [number, number],
      paint: 'tube',
      groove: 0,
    };
    const part: PartSpec = {
      primitives: [slab([0, 0, 0], [4, 4, 4]), slab([4, 0, 0], [8, 4, 4]), tube],
      carve: [{ min: [2, 1, 1], max: [6, 3, 3] }],
    };
    const result = coalescePart(part);
    expect(result.merges.length).toBeGreaterThan(0);
    sameUnion(part, result.part);
    expect(result.part.primitives.find((q) => q.type === 'cyl')).toEqual(tube);
  });
});
