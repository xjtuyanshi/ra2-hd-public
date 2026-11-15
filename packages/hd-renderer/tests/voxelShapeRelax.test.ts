import { describe, expect, it } from 'vitest';
import { buildVoxelShapeMesh, voxelShapeOptions } from '../src/voxelShapeMesh';
import { relaxVoxelShapeMesh, smoothVoxelShapeNormals } from '../src/voxelShapeRelax';

function box(nx: number, ny: number, nz: number, keep = (_x: number, _y: number, _z: number) => true) {
  const cells: number[] = [];
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) if (keep(x, y, z)) cells.push(x, y, z, 7, 0);
  return buildVoxelShapeMesh(
    { size: [nx, ny, nz], cells: Uint8Array.from(cells) },
    voxelShapeOptions({ maxOffset: 0 }),
  );
}
function displacement(a: Float32Array, b: Float32Array, v: number) {
  return Math.hypot(a[v * 3]! - b[v * 3]!, a[v * 3 + 1]! - b[v * 3 + 1]!, a[v * 3 + 2]! - b[v * 3 + 2]!);
}
function face(p: Float32Array, idx: Uint32Array, t: number) {
  const a = idx[t]! * 3,
    b = idx[t + 1]! * 3,
    c = idx[t + 2]! * 3;
  const u = [0, 1, 2].map((k) => p[b + k]! - p[a + k]!);
  const v = [0, 1, 2].map((k) => p[c + k]! - p[a + k]!);
  const n = [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
  const len = Math.hypot(...n);
  return n.map((x) => x / len);
}
function checkFaces(mesh: ReturnType<typeof box>, p: Float32Array) {
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const a = face(mesh.positions, mesh.indices, t),
      b = face(p, mesh.indices, t);
    expect(a.reduce((s, x, k) => s + x * b[k]!, 0)).toBeGreaterThanOrEqual(0.25 - 1e-6);
  }
}

describe('shape-locked relaxation', () => {
  it('keeps an open flat plane flat, including its boundary', () => {
    const mesh = {
      positions: new Float32Array([0, 0, 3, 1, 0, 3, 1, 1, 3, 0, 1, 3]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
    const out = relaxVoxelShapeMesh(mesh, { voxelSize: 1 });
    for (let v = 0; v < 4; v++) {
      expect(out.positions[v * 3 + 2]).toBe(3);
      expect([...out.normals.subarray(v * 3, v * 3 + 3)]).toEqual([0, 0, 1]);
    }
    expect(out.zone).toEqual(new Uint8Array(4));
  });
  it('limits every big-box vertex to the bevel and keeps the broad face nearly planar', () => {
    const mesh = box(10, 10, 10),
      before = mesh.positions.slice();
    const out = relaxVoxelShapeMesh(mesh, { voxelSize: 1 });
    expect(mesh.positions).toEqual(before);
    expect(out.stats.maxDisplacementVoxels).toBeLessThanOrEqual(0.2 + 1e-9);
    expect(out.stats.pinned).toBe(0);
    for (let v = 0; v < mesh.material.length; v++) {
      expect(displacement(mesh.positions, out.positions, v)).toBeLessThanOrEqual(0.2 + 1e-6);
      expect(out.zone[v]).toBe(0);
      const i = v * 3;
      if (mesh.positions[i] === 0 && mesh.positions[i + 1] === 5 && mesh.positions[i + 2] === 5)
        expect(Math.abs(out.positions[i]!)).toBeLessThan(0.02);
    }
    checkFaces(mesh, out.positions);
  });
  it('rounds the corners of a small 3x3x8 box more than its face centre, towards a cylindrical section', () => {
    const mesh = box(3, 3, 8),
      smallVertices = new Uint8Array(mesh.material.length).fill(1);
    const out = relaxVoxelShapeMesh(mesh, { smallVertices });
    let corner = -1,
      centre = -1;
    for (let v = 0; v < mesh.material.length; v++)
      if (mesh.positions[v * 3] === 0 && mesh.positions[v * 3 + 2] === 4) {
        if (mesh.positions[v * 3 + 1] === 0) corner = v;
        if (mesh.positions[v * 3 + 1] === 1) centre = v;
      }
    expect(corner).toBeGreaterThanOrEqual(0);
    expect(centre).toBeGreaterThanOrEqual(0);
    expect(displacement(mesh.positions, out.positions, corner)).toBeGreaterThan(
      displacement(mesh.positions, out.positions, centre),
    );
    const radius = (p: Float32Array) => Math.hypot(p[corner * 3]! - 1.5, p[corner * 3 + 1]! - 1.5);
    expect(Math.abs(radius(out.positions) - 1.5)).toBeLessThan(Math.abs(radius(mesh.positions) - 1.5));
    expect(out.stats.maxDisplacementVoxels).toBeLessThanOrEqual(0.62 + 1e-9);
    for (const z of out.zone) expect(z).toBe(2);
    checkFaces(mesh, out.positions);
  });
  it('marks concave stair neighbourhoods and lets a caller small mask take priority across welded copies', () => {
    const mesh = box(5, 4, 5, (x, _y, z) => z <= x);
    const out = relaxVoxelShapeMesh(mesh);
    expect([...out.zone]).toContain(1);
    const smallVertices = new Uint8Array(mesh.material.length);
    smallVertices[0] = 1;
    const small = relaxVoxelShapeMesh(mesh, { smallVertices, voxelSize: 1 });
    for (let v = 0; v < mesh.material.length; v++) {
      const same = [0, 1, 2].every((k) => mesh.positions[v * 3 + k] === mesh.positions[k]);
      if (same) expect(small.zone[v]).toBe(2);
      const clamp = small.zone[v] === 2 ? 0.62 : small.zone[v] === 1 ? 0.35 : 0.2;
      expect(displacement(mesh.positions, small.positions, v)).toBeLessThanOrEqual(clamp + 1e-6);
    }
    checkFaces(mesh, small.positions);
  });
  it('pins vertices on edges used by more than two triangles', () => {
    const mesh = box(2, 2, 1, (x, y) => x === y),
      out = relaxVoxelShapeMesh(mesh, { smallVertices: new Uint8Array(mesh.material.length).fill(1) });
    expect(out.stats.pinned).toBe(2);
    let found = 0;
    for (let v = 0; v < mesh.material.length; v++)
      if (mesh.positions[v * 3] === 1 && mesh.positions[v * 3 + 1] === 1) {
        expect(displacement(mesh.positions, out.positions, v)).toBe(0);
        found++;
      }
    expect(found).toBeGreaterThan(2);
    checkFaces(mesh, out.positions);
  });
  it('never flips thin surfaces, including aggressive clamps and repeated pin/rerun attempts', () => {
    for (const mesh of [box(1, 1, 8), box(6, 6, 1), box(6, 6, 4, (x, y, z) => z === 0 || x === y)]) {
      const out = relaxVoxelShapeMesh(mesh, { smallVertices: new Uint8Array(mesh.material.length).fill(1), small: 3 });
      expect(out.stats.flips).toBe(0);
      checkFaces(mesh, out.positions);
    }
  });
  it('is deterministic, supports physical scale and orients normals like the input', () => {
    const mesh = box(3, 3, 8);
    mesh.positions = Float32Array.from(mesh.positions, (x) => x * 0.125);
    mesh.normals = Float32Array.from(mesh.normals, (x) => -x);
    const options = { smallVertices: new Uint8Array(mesh.material.length).fill(1) };
    const a = relaxVoxelShapeMesh(mesh, options),
      b = relaxVoxelShapeMesh(mesh, options);
    expect(a).toEqual(b);
    expect(a.stats.maxDisplacementVoxels).toBeLessThanOrEqual(0.62 + 1e-9);
    for (let i = 0; i < mesh.normals.length; i += 3) {
      expect(
        a.normals[i]! * mesh.normals[i]! +
          a.normals[i + 1]! * mesh.normals[i + 1]! +
          a.normals[i + 2]! * mesh.normals[i + 2]!,
      ).toBeGreaterThanOrEqual(0);
      expect(Math.hypot(a.normals[i]!, a.normals[i + 1]!, a.normals[i + 2]!)).toBeCloseTo(1, 6);
    }
  });
  it('smooths shading across neighbouring faces without bleeding across opposing thin faces', () => {
    const mesh = box(8, 8, 1),
      normals = smoothVoxelShapeNormals(mesh);
    expect(normals).toEqual(smoothVoxelShapeNormals(mesh));
    for (let v = 0; v < mesh.material.length; v++) {
      const i = v * 3;
      if (mesh.positions[i] === 4 && mesh.positions[i + 1] === 4) {
        expect(Math.abs(normals[i + 2]!)).toBeCloseTo(1, 6);
        expect(normals[i + 2]! * mesh.normals[i + 2]!).toBeGreaterThan(0);
      }
    }
  });
});
