import { describe, expect, it } from 'vitest';
import { buildPart, type PartSpec } from '../src/hardSurfaceMesh';
import { buildHardSurfaceLiveMesh } from '../src/hardSurfaceLiveMesh';
import {
  bakeLookVolume,
  vectoriseSeams,
  type LookVolume,
  type LookVolumeMesh,
  type VectorSeams,
} from '../src/hardSurfaceLookVolume';
const settings = { lines: { width: 0.42, aoWidth: 1 }, comb: { minLength: 2 } };
const meshFor = (part: PartSpec): LookVolumeMesh => {
  const m = buildPart(part).mesh;
  return { ...m, lattice: m.positions };
};
function at(v: LookVolume, p: number[], ch: number) {
  const q = p.map((x, a) => Math.floor((x - v.origin[a]) * v.res));
  return v.data[(q[0] + v.size[0] * (q[1] + v.size[1] * q[2])) * 4 + ch] / 255;
}
describe('hard surface look volume', () => {
  it('greedily vectorises exposed line runs in axial and diagonal directions and leaves isolated dots', () => {
    const row = Array.from({ length: 5 }, (_, i) => ({
      position: [i, 0, 0] as const,
      strength: 0.7,
      surface: true,
      kind: 'line',
    }));
    const diagonal = Array.from({ length: 3 }, (_, i) => ({ ...row[0], position: [i, 8 + i, 0] as const }));
    const v = vectoriseSeams([
      ...row,
      ...diagonal,
      { ...row[0], position: [20, 20, 0] },
      { ...row[0], position: [30, 30, 0], surface: false },
      { ...row[0], position: [40, 40, 0], kind: 'dot' },
      { ...row[0], position: [50, 50, 0], strength: 0.1 },
    ]);
    expect(v.segments).toHaveLength(2);
    expect(v.dots).toEqual([[20.5, 20.5, 0.5]]);
    expect(v.segments[0]).toEqual({ start: [0.5, 0.5, 0.5], end: [4.5, 0.5, 0.5], strength: 0.7 });
    expect(vectoriseSeams(row)).toEqual(vectoriseSeams(row));
  });
  it('bakes a seam, halo, dot and extended endpoints, leaving distant surface texels clear', () => {
    const mesh = meshFor({ primitives: [{ type: 'box', min: [0, 0, 0], max: [8, 8, 2], chamfer: 0 }] });
    const seams: VectorSeams = {
      segments: [{ start: [2.5, 1.5, 1.5], end: [2.5, 6.5, 1.5], strength: 1 }],
      dots: [[6.5, 1.5, 1.5]],
    };
    const v = bakeLookVolume(mesh, { ...settings, seams });
    expect(v.res).toBe(4);
    expect(v.origin).toEqual([-1, -1, -1]);
    expect(v.size).toEqual([40, 40, 16]);
    expect(at(v, [2.5, 4.5, 2.1], 0)).toBeGreaterThan(0.8);
    expect(at(v, [3, 4.5, 2.1], 1)).toBeGreaterThan(0);
    expect(at(v, [2.5, 1.2, 2.1], 0)).toBeGreaterThan(0.3);
    expect(at(v, [6.5, 1.5, 2.1], 0)).toBeGreaterThan(0.3);
    expect(at(v, [6, 6, 2.1], 0)).toBe(0);
    expect(v.surfaceTexels).toBeGreaterThan(0);
    expect(v.grooveTexels).toBeGreaterThan(0);
    expect(v.data.filter((_, i) => i % 4 === 3).every((x) => x === 0)).toBe(true);
  });
  it('rejects staircase seams on a sloped face', () => {
    const mesh: LookVolumeMesh = {
      lattice: new Float32Array([0, 0, 0, 8, 0, 0, 0, 8, 8]),
      indices: new Uint32Array([0, 1, 2]),
      primitive: new Uint32Array(3),
    };
    const v = bakeLookVolume(mesh, {
      ...settings,
      seams: { segments: [{ start: [0.5, 2.5, 2.5], end: [5.5, 2.5, 2.5], strength: 1 }], dots: [] },
    });
    expect(v.grooveTexels).toBe(0);
  });
  it('uses cylinder metadata for distinct tyre/disc/hub values and a wheel ring', () => {
    const mesh = meshFor({ primitives: [{ type: 'cyl', axis: 'z', c: [0, 0], range: [0, 1], r: 2, chamfer: 0 }] });
    const v = bakeLookVolume(mesh, {
      ...settings,
      seams: { segments: [], dots: [] },
      cylinders: new Map([[0, { centre: [0, 0, 0], axis: [0, 0, 1], range: [0, 1], profile: 'wheel' }]]),
    });
    expect(at(v, [0.1, 0.1, 1.1], 2)).toBeCloseTo(0.75, 2);
    expect(at(v, [1.3, 0.1, 1.1], 2)).toBeCloseTo(0.5, 2);
    expect(at(v, [1.8, 0.1, 1.1], 2)).toBeCloseTo(0.25, 2);
    expect(at(v, [1.1, 0.1, 1.1], 0)).toBeGreaterThan(0.4);
  });
  it('emits caller-owned codes for expanded primitives, including mirrored copies', () => {
    const mesh = buildHardSurfaceLiveMesh(
      {
        mirror: { axis: 'y', at: 0 },
        primitives: [{ type: 'box', min: [0, 1, 0], max: [2, 3, 2], paint: 'custom', chamfer: 0 }],
      },
      1,
      undefined,
      { custom: 17 },
    );
    expect(mesh.lookCode.length).toBe(mesh.pindex.length);
    expect([...new Set(mesh.lookCode)]).toEqual([17]);
    expect(new Set(mesh.primitive).size).toBe(2);
  });
});
