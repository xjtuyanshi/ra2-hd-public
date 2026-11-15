import { describe, expect, it } from 'vitest';
import { attachVoxelPaint, mergeCleanVertexPaint } from '../src/voxelCleanPaint';
import type { VoxelPaintResult } from '../src/voxelPaint';
import { buildVoxelShapeMesh } from '../src/voxelShapeMesh';

// Two cells side by side along x: cell 0 grey paint, cell 1 house colour carrying a seam line along y.
const grid = { size: [2, 1, 1] as const, cells: new Uint8Array([0, 0, 0, 10, 0, 1, 0, 0, 20, 0]) };
const paint: VoxelPaintResult = {
  surface: new Uint8Array([1, 1]),
  region: new Int32Array([0, 1]),
  seamFeature: new Int32Array([-1, 0]),
  seam: new Float32Array([0, 0.75]),
  mask: new Uint8Array([0, 1]),
  albedo: new Uint8Array([120, 120, 120, 200, 0, 0]),
  regions: [
    {
      id: 0,
      family: 1,
      familyName: 'grey',
      material: 'painted',
      masked: false,
      voxels: 1,
      surfaceVoxels: 1,
      albedo: [120, 120, 120],
      lightness: 50,
      indices: { 10: 1 },
    },
    {
      id: 1,
      family: 0,
      familyName: 'house',
      material: 'rubber',
      masked: true,
      voxels: 1,
      surfaceVoxels: 1,
      albedo: [200, 0, 0],
      lightness: 40,
      indices: { 20: 1 },
    },
  ],
  seams: [
    {
      id: 0,
      kind: 'line',
      region: 1,
      voxels: 1,
      length: 3,
      axis: 'y',
      darkness: 30,
      bboxMin: [1, 0, 0],
      bboxMax: [1, 0, 0],
      indices: { 20: 1 },
    },
  ],
};

describe('attachVoxelPaint', () => {
  const mesh = buildVoxelShapeMesh(grid);

  it('records the owning cell of every quad vertex', () => {
    expect(mesh.cell.length).toBe(mesh.material.length);
    for (let v = 0; v < mesh.cell.length; v++) expect(mesh.material[v]).toBe(mesh.cell[v] === 0 ? 10 : 20);
  });

  it('gives each vertex the flat paint, material, mask and seam of its cell', () => {
    const out = attachVoxelPaint(mesh, paint);
    for (let v = 0; v < mesh.cell.length; v++) {
      const house = mesh.cell[v] === 1;
      expect([...out.albedo.subarray(v * 3, v * 3 + 3)]).toEqual(house ? [200, 0, 0] : [120, 120, 120]);
      expect(out.mask[v]).toBe(house ? 1 : 0);
      expect(out.material[v]).toBe(house ? 2 : 0);
      expect(out.seam[v]).toBeCloseTo(house ? 0.75 : 0);
      expect(out.seamKind[v]).toBe(house ? 1 : 0);
      expect(out.seamAxis[v]).toBe(house ? 1 : 255);
    }
  });

  it('rejects paint that does not cover the mesh cells', () => {
    expect(() => attachVoxelPaint(mesh, { ...paint, region: new Int32Array([0]) })).toThrow(RangeError);
  });

  it('merges sections in vertex order', () => {
    const a = attachVoxelPaint(mesh, paint);
    const merged = mergeCleanVertexPaint([a, a]);
    expect(merged.material.length).toBe(a.material.length * 2);
    expect([...merged.mask.subarray(a.mask.length)]).toEqual([...a.mask]);
  });
});
