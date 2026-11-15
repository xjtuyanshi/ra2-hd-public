import { describe, expect, it } from 'vitest';
import { buildPart, type PartSpec } from '../src/hardSurfaceMesh';
import {
  buildHardSurfaceLiveMesh,
  hardSurfacePaintSampler,
  parseHardSurfaceSpec,
  subdivideHardSurfaceMesh,
  type VoxelPaint,
} from '../src/hardSurfaceLiveMesh';
import { forCells, gridCell, gridIndex, hardSurfaceSolid } from '../src/hardSurfaceSolid';

const box: PartSpec = { primitives: [{ type: 'box', min: [0, 0, 0], max: [3, 2, 2], chamfer: 0.15 }] };
describe('live CAD lattice geometry', () => {
  it('bounds every edge, preserves face/chamfer normals and radial cylinder normals', () => {
    const part: PartSpec = {
      mirror: { axis: 'y', at: 3 },
      primitives: [...box.primitives, { type: 'cyl', axis: 'x', c: [1, 1], r: 0.7, range: [3, 5], chamfer: 0 }],
    };
    const original = buildPart(part).mesh,
      mesh = subdivideHardSurfaceMesh(original);
    expect(mesh.indices.length).toBeGreaterThan(original.indices.length);
    for (let t = 0; t < mesh.indices.length; t += 3)
      for (let j = 0; j < 3; j++) {
        const a = mesh.indices[t + j] * 3,
          b = mesh.indices[t + ((j + 1) % 3)] * 3;
        expect(Math.hypot(...[0, 1, 2].map((k) => mesh.positions[a + k] - mesh.positions[b + k]))).toBeLessThanOrEqual(
          0.500001,
        );
        expect(Math.hypot(...mesh.normals.subarray(a, a + 3))).toBeCloseTo(1, 5);
        expect(mesh.face[a / 3]).toBe(mesh.face[b / 3]);
      }
    expect(new Set(mesh.chamfer)).toEqual(new Set([0, 1]));
    const radial = Array.from(mesh.primitive).findIndex((p, i) => p === 2 && Math.abs(mesh.normals[i * 3]) < 0.1);
    expect(radial).toBeGreaterThanOrEqual(0);
    const y = mesh.positions[radial * 3 + 1] - 1,
      z = mesh.positions[radial * 3 + 2] - 1;
    expect(mesh.normals[radial * 3 + 1]).toBeCloseTo(y / Math.hypot(y, z), 4);
  });
  it('samples CAD cavities for finite, bounded AO and retains lattice positions', () => {
    const part: PartSpec = {
      primitives: [
        { type: 'box', min: [0, 0, 0], max: [4, 4, 1], chamfer: 0 },
        { type: 'box', min: [0, 0, 1], max: [4, 1, 4], chamfer: 0 },
      ],
    };
    const mesh = buildHardSurfaceLiveMesh(part, 0.75);
    expect(mesh.lattice).toEqual(mesh.positions);
    expect(mesh.occlusion.every((v) => v >= 0 && v <= 1 && Number.isFinite(v))).toBe(true);
    expect(mesh.occlusion.some((v) => v > 0.1)).toBe(true);
    expect(mesh.occlusion.some((v) => v === 0)).toBe(true);
    expect(() => subdivideHardSurfaceMesh(mesh, 0)).toThrow();
  });
  it('rejects malformed specs before geometric expansion', () => {
    expect(parseHardSurfaceSpec({ units: 'lattice', parts: { body: box } }).parts.body).toEqual(box);
    for (const value of [
      null,
      {},
      { units: 'x', parts: {} },
      { units: 'x', parts: { body: { primitives: [{ type: 'cyl', axis: 'q', r: 1, c: [0, 0], range: [0, 1] }] } } },
      { units: 'x', parts: { body: { primitives: [{ type: 'poly', planes: [[0, 0, 0, 1]] }] } } },
    ])
      expect(() => parseHardSurfaceSpec(value)).toThrow();
  });
});

it('directional distance transforms match exhaustive exposed-face distances, including zero paint and hidden voxels', () => {
  const size = [5, 4, 3] as const;
  const paint: VoxelPaint = {
    size,
    data: new Uint8Array(60),
    index: new Int16Array(60).fill(-1),
    shade: new Float32Array(60),
  };
  forCells(size, (p, i) => {
    if ((p[0] * 17 + p[1] * 13 + p[2] * 7) % 5) {
      paint.data[i] = 1;
      paint.index[i] = i;
      paint.shade[i] = i + 0.25;
    }
  });
  paint.data[0] = 1;
  paint.index[0] = 0;
  const solid = hardSurfaceSolid(paint),
    sample = hardSurfacePaintSampler(solid, paint);
  for (let axis = 0; axis < 3; axis++)
    for (const sign of [-1, 1])
      forCells(size, (p) => {
        const n = [0, 0, 0];
        n[axis] = sign;
        const position = p.map((v) => v + 0.5),
          hit = sample(position, n);
        let best = Infinity;
        forCells(size, (q, i) => {
          const neighbour = [...q];
          neighbour[axis] += sign;
          if (!solid.data[i] || paint.index[i] < 0 || gridCell(solid, neighbour)) return;
          const d = q.reduce((s, v, a) => s + (v + 0.5 + (a === axis ? sign * 0.5 : 0) - position[a]) ** 2, 0);
          best = Math.min(best, d);
        });
        const cell = hit.cell,
          x = Math.floor(cell / (size[1] * size[2])),
          y = Math.floor(cell / size[2]) % size[1],
          z = cell % size[2];
        const distance = [x, y, z].reduce(
          (s, v, a) => s + (v + 0.5 + (a === axis ? sign * 0.5 : 0) - position[a]) ** 2,
          0,
        );
        expect(distance).toBe(best);
        expect(hit.index).toBe(paint.index[gridIndex(size, x, y, z)]);
        expect(hit.shade).toBe(paint.shade[cell]);
      });
});
