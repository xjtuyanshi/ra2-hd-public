import { describe, expect, it } from 'vitest';
import {
  buildVoxelShapeMesh,
  DEFAULT_VOXEL_SHAPE,
  voxelShapeOptions,
  type VoxelShapeMesh,
} from '../src/voxelShapeMesh';
import { buildVoxelShapeModel, buildVoxelShapeSection } from '../src/ra2/voxelShapeModel';
import { VOXEL_TO_WORLD } from '../src/ra2/voxelScale';
import type { VoxelModel, VoxelSection } from '../../ra2-assets/src/manifest';

/** Synthetic grids only: no proprietary voxel art. */
function cells(list: [number, number, number][], material = 7): Uint8Array {
  const out = new Uint8Array(list.length * 5);
  list.forEach(([x, y, z], i) => out.set([x, y, z, material, 0], i * 5));
  return out;
}
function box(nx: number, ny: number, nz: number, keep: (x: number, y: number, z: number) => boolean = () => true) {
  const list: [number, number, number][] = [];
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (keep(x, y, z)) list.push([x, y, z]);
  return list;
}
function windingAgrees(mesh: VoxelShapeMesh): number {
  // Fraction of triangles whose geometric normal agrees with the interpolated shading normals.
  let ok = 0;
  const p = mesh.positions;
  const n = mesh.normals;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [mesh.indices[t]!, mesh.indices[t + 1]!, mesh.indices[t + 2]!];
    const u = [0, 1, 2].map((k) => p[b * 3 + k]! - p[a * 3 + k]!);
    const v = [0, 1, 2].map((k) => p[c * 3 + k]! - p[a * 3 + k]!);
    const g = [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
    const s = [0, 1, 2].map((k) => n[a * 3 + k]! + n[b * 3 + k]! + n[c * 3 + k]!);
    if (g[0]! * s[0]! + g[1]! * s[1]! + g[2]! * s[2]! > 0) ok++;
  }
  return ok / (mesh.indices.length / 3);
}

describe('voxel shape-locked mesh', () => {
  it('keeps a box exactly: flat faces on their planes, sharp corners, one quad per exposed face', () => {
    const mesh = buildVoxelShapeMesh({ size: [6, 4, 3], cells: cells(box(6, 4, 3)) });
    expect(mesh.stats.faces).toBe(2 * (6 * 4 + 6 * 3 + 4 * 3));
    expect(mesh.stats.triangles).toBe(mesh.stats.faces * 2);
    expect(mesh.stats.maxOffset).toBe(0);
    expect(mesh.stats.stairFaces).toBe(0);
    for (const v of mesh.lattice) expect(Number.isInteger(v)).toBe(true);
    // Interior vertices of a face carry the exact face normal; corner vertices are bevel-tilted towards neighbours.
    const top = [...mesh.direction.keys()].filter(
      (i) =>
        mesh.direction[i] === 4 &&
        mesh.lattice[i * 3]! > 0 &&
        mesh.lattice[i * 3]! < 6 &&
        mesh.lattice[i * 3 + 1]! > 0 &&
        mesh.lattice[i * 3 + 1]! < 4,
    );
    expect(top.length).toBeGreaterThan(0);
    for (const i of top) expect(mesh.normals[i * 3 + 2]).toBeCloseTo(1, 6);
    expect(windingAgrees(mesh)).toBe(1);
  });

  it('turns a 45-degree staircase into one inclined plane within the offset bound', () => {
    // Ramp: column x has height x + 1 over a 10 x 4 footprint.
    const mesh = buildVoxelShapeMesh({ size: [10, 4, 10], cells: cells(box(10, 4, 10, (x, _y, z) => z <= x)) });
    expect(mesh.stats.stairFaces).toBeGreaterThan(20);
    expect(mesh.stats.maxOffset).toBeLessThanOrEqual(DEFAULT_VOXEL_SHAPE.maxOffset + 1e-9);
    // Stair vertices in the middle of the ramp lie on the plane z = x + 0.5 (between the treads and risers).
    let checked = 0;
    for (let i = 0; i < mesh.stair.length; i++) {
      const x = mesh.lattice[i * 3]!,
        y = mesh.lattice[i * 3 + 1]!,
        z = mesh.lattice[i * 3 + 2]!;
      if (!mesh.stair[i] || x < 3 || x > 7 || y < 1 || y > 3) continue;
      expect(Math.abs(z - x - 0.5)).toBeLessThan(0.15);
      expect(mesh.normals[i * 3]! / mesh.normals[i * 3 + 2]!).toBeCloseTo(-1, 1);
      checked++;
    }
    expect(checked).toBeGreaterThan(10);
    expect(windingAgrees(mesh)).toBeGreaterThan(0.99);
  });

  it('keeps isolated one-cell steps (hatches, ledges) hard', () => {
    const plate = box(12, 12, 1).concat(box(12, 12, 2, (x, y, z) => z === 1 && x >= 4 && x < 8 && y >= 4 && y < 8));
    const mesh = buildVoxelShapeMesh({ size: [12, 12, 2], cells: cells(plate) });
    expect(mesh.stats.stairFaces).toBe(0);
    expect(mesh.stats.maxOffset).toBe(0);
  });

  it('keeps one-cell parts at full width and edge-touching cells apart', () => {
    const rod = buildVoxelShapeMesh({ size: [1, 1, 8], cells: cells(box(1, 1, 8)) });
    expect(rod.stats.maxOffset).toBe(0);
    const pair = buildVoxelShapeMesh({
      size: [2, 2, 1],
      cells: cells([
        [0, 0, 0],
        [1, 1, 0],
      ]),
    });
    // Two cubes sharing an edge: 8 corners each, none welded.
    expect(pair.stats.cornerVertices).toBe(16);
  });

  it('is deterministic and honours the offset clamp', () => {
    const grid = {
      size: [9, 9, 9] as [number, number, number],
      cells: cells(box(9, 9, 9, (x, y, z) => (x - 4) ** 2 + (y - 4) ** 2 + (z - 4) ** 2 <= 17)),
    };
    const a = buildVoxelShapeMesh(grid);
    const b = buildVoxelShapeMesh(grid);
    expect(Buffer.from(a.positions.buffer).equals(Buffer.from(b.positions.buffer))).toBe(true);
    expect(Buffer.from(a.normals.buffer).equals(Buffer.from(b.normals.buffer))).toBe(true);
    const tight = buildVoxelShapeMesh(grid, voxelShapeOptions({ maxOffset: 0.2 }));
    expect(tight.stats.maxOffset).toBeLessThanOrEqual(0.2 + 1e-9);
    const cubes = buildVoxelShapeMesh(grid, voxelShapeOptions({ maxOffset: 0 }));
    for (const v of cubes.lattice) expect(Number.isInteger(v)).toBe(true);
    expect(() => voxelShapeOptions({ maxOffset: 0.6 })).toThrow(RangeError);
  });
});

describe('RA2 voxel shape model', () => {
  const section = (transform: number[], list: [number, number, number][]): VoxelSection => ({
    name: 'S',
    size: [2, 2, 2],
    scale: 1,
    minBounds: [-1, -1, 0],
    maxBounds: [1, 1, 2],
    transform,
    voxels: Buffer.from(cells(list)).toString('base64'),
  });

  it('places sections like voxels.ts: bounds, HVA matrix, axis swap, VOXEL_TO_WORLD', () => {
    const mesh = buildVoxelShapeSection(section([1, 0, 0, 3, 0, 1, 0, 2, 0, 0, 1, 1], [[1, 1, 0]]));
    // Cell (1, 1, 0) spans voxel X 0..1, Y 0..1, Z 0..1; translated by (3, 2, 1) -> model x 3..4, y 1..2, z -3..-2.
    const xs = [],
      ys = [],
      zs = [];
    for (let i = 0; i < mesh.positions.length; i += 3) {
      xs.push(mesh.positions[i]! / VOXEL_TO_WORLD);
      ys.push(mesh.positions[i + 1]! / VOXEL_TO_WORLD);
      zs.push(mesh.positions[i + 2]! / VOXEL_TO_WORLD);
    }
    expect(Math.min(...xs)).toBeCloseTo(3, 5);
    expect(Math.max(...xs)).toBeCloseTo(4, 5);
    expect(Math.min(...ys)).toBeCloseTo(1, 5);
    expect(Math.max(...ys)).toBeCloseTo(2, 5);
    expect(Math.min(...zs)).toBeCloseTo(-3, 5);
    expect(Math.max(...zs)).toBeCloseTo(-2, 5);
    expect(windingAgrees(mesh)).toBe(1);
  });

  it('keeps outward winding under a mirroring matrix and passes the turret pivot in world units', () => {
    const mirrored = buildVoxelShapeSection(
      section(
        [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
        [
          [0, 0, 0],
          [1, 0, 0],
        ],
      ),
    );
    expect(windingAgrees(mirrored)).toBe(1);
    const model: VoxelModel = {
      id: 'T',
      palette: 'unit',
      normalsMode: 4,
      turretOffset: 64,
      parts: [
        { role: 'body', sections: [section([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], [[0, 0, 0]])] },
        { role: 'turret', sections: [section([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], [[0, 0, 1]])] },
      ],
    };
    const parts = buildVoxelShapeModel(model);
    expect(parts.map((p) => p.role)).toEqual(['body', 'turret']);
    expect(parts[1]!.turretOffset).toBe(0.25);
  });
});
