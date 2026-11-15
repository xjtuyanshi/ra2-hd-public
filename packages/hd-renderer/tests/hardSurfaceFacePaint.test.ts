import { describe, expect, it } from 'vitest';
import { foldHardSurfaceFacePaint, type FacePaintPolicy } from '../src/hardSurfaceFacePaint';
import type { HardSurfaceMesh } from '../src/hardSurfaceMesh';
import { buildHardSurfaceLiveMesh } from '../src/hardSurfaceLiveMesh';

const plate = (): HardSurfaceMesh => ({
  positions: Float32Array.from([0, 0, 0, 10, 0, 0, 10, 4, 0, 0, 4, 0]),
  normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
  indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
  chamfer: new Uint8Array(4),
  primitive: new Uint32Array(4),
  face: new Uint32Array(4).fill(17),
});
const policy = (fraction = 0.05): FacePaintPolicy => ({
  region: (p) => (p[0] < fraction * 10 ? 1 : 0),
  paint: (id) => ({ index: id + 80, shade: id + 80 }),
  luma: (id) => (id ? 0.1 : 0.5),
  house: () => false,
  glint: () => false,
  flat: () => false,
});
const fragment = [0.1, 2, 0],
  normal = [0, 0, 1];

describe('whole-face paint folding', () => {
  it('folds a 5% dark fragment and measures changed look area deterministically', () => {
    const mesh = plate(),
      p = policy(),
      fold = foldHardSurfaceFacePaint(mesh, p);
    expect(fold.sample(fragment, normal, 17).index).toBe(80);
    expect(fold.sample([8, 2, 0], normal, 17).index).toBe(80);
    const stats = fold.measure(mesh);
    expect(stats.surfaceArea).toBeCloseTo(40, 8);
    expect(stats.changedFraction).toBeCloseTo(0.05, 2);
    expect(foldHardSurfaceFacePaint(mesh, p).measure(mesh)).toEqual(stats);
    expect(() => fold.sample(fragment, normal, 18)).toThrow('Unknown paint face');
  });
  it('retains a 20% dark fragment', () => {
    const fold = foldHardSurfaceFacePaint(plate(), policy(0.2));
    expect(fold.sample(fragment, normal, 17).index).toBe(81);
    expect(fold.measure(plate()).changedArea).toBe(0);
  });
  it('retains a 5% house fragment but folds a sub-4% one', () => {
    for (const [fraction, expected] of [
      [0.05, 81],
      [0.02, 80],
    ]) {
      const fold = foldHardSurfaceFacePaint(plate(), { ...policy(fraction), house: (id) => id === 1 });
      expect(fold.sample(fragment, normal, 17).index).toBe(expected);
    }
  });
  it('retains a lighter 2 square-unit feature but folds a smaller or insufficiently lighter one', () => {
    for (const [fraction, luma, expected] of [
      [0.05, 0.7, 81],
      [0.02, 0.7, 80],
      [0.05, 0.57, 80],
    ]) {
      const fold = foldHardSurfaceFacePaint(plate(), { ...policy(fraction), luma: (id) => (id ? luma : 0.5) });
      expect(fold.sample(fragment, normal, 17).index).toBe(expected);
    }
  });
  it.each([3, 6])('folds an entire flat/hatch coded face (%s), including otherwise retained house paint', (code) => {
    const fold = foldHardSurfaceFacePaint(plate(), {
      ...policy(0.2),
      house: (id) => id === 1,
      flat: (_face, primitive) => [3, 6].includes([code][primitive]),
    });
    expect(fold.sample(fragment, normal, 17).index).toBe(80);
    expect(fold.measure(plate()).changedFraction).toBeCloseTo(0.2, 2);
  });
  it('folds glints even when their area would retain a lighter feature', () => {
    const fold = foldHardSurfaceFacePaint(plate(), {
      ...policy(0.2),
      glint: (id) => id === 1,
      luma: (id) => (id ? 0.95 : 0.5),
    });
    expect(fold.sample(fragment, normal, 17).index).toBe(80);
  });
  it('keeps adjacent face decisions isolated even when positions and normals coincide', () => {
    const mesh = plate();
    mesh.face[3] = 18;
    mesh.indices = Uint32Array.from([0, 1, 2, 3, 2, 1]);
    const fold = foldHardSurfaceFacePaint(mesh, { ...policy(), region: (p) => (p[1] > 2 ? 1 : 0), flat: () => true });
    expect(fold.sample([5, 2, 0], normal, 17).index).toBe(80);
    expect(fold.sample([5, 2, 0], normal, 18).index).toBe(81);
  });
  it('merges folded fragments before adaptive meshing and preserves the unfolded comparison mesh', () => {
    const part = {
      primitives: [
        {
          type: 'box' as const,
          min: [0, 0, 0] as [number, number, number],
          max: [10, 4, 1] as [number, number, number],
          chamfer: 0,
        },
      ],
    };
    const p = policy(),
      raw = (position: readonly number[], n: readonly number[]) => p.paint(p.region(position, n), position, n, false);
    const legacy = buildHardSurfaceLiveMesh(part, 0.5, raw);
    const folded = buildHardSurfaceLiveMesh(
      part,
      0.5,
      undefined,
      {},
      (mesh) => foldHardSurfaceFacePaint(mesh, p).sample,
    );
    const topIndices = (mesh: typeof folded) =>
      Array.from(mesh.pindex).filter((_, i) => mesh.normals[i * 3 + 2] > 0.99);
    expect(new Set(topIndices(legacy))).toEqual(new Set([80, 81]));
    expect(new Set(topIndices(folded))).toEqual(new Set([80]));
    expect(topIndices(folded).length).toBeLessThan(topIndices(legacy).length);
    expect(buildHardSurfaceLiveMesh(part, 0.5, raw)).toEqual(legacy);
  });
});
