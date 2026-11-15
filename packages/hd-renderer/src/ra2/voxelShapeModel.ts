import type { VoxelModel, VoxelSection } from '../../../ra2-assets/src/manifest';
import {
  buildVoxelShapeMesh,
  DEFAULT_VOXEL_SHAPE,
  type VoxelShapeMesh,
  type VoxelShapeOptions,
  type VoxelShapeStats,
} from '../voxelShapeMesh';
import { voxelSectionFrame } from './voxelSectionFrame';

/** Engine leptons per cell edge (art TurretOffset is in leptons; one world unit is one cell edge). */
const LEPTONS_PER_CELL = 256;

/**
 * One RA2/YR voxel part as a shape-locked mesh in the renderer's model frame (x = nose, y = up, z = right of the
 * VXL's -Y; world units of one cell edge), the same frame and section placement as voxels.ts buildVoxelGeometry:
 * lattice -> section bounds -> HVA frame-0 matrix (translation pre-scaled) -> axis swap -> VOXEL_TO_WORLD.
 * `material` is the VXL palette index; house colour stays whatever indices the original paints there.
 */
export interface VoxelShapePart {
  role: 'body' | 'turret' | 'barrel';
  /** Pivot of turret and barrel along +x (world units), as instantiateVoxel places its turret group. */
  turretOffset: number;
  mesh: VoxelShapeMesh;
}

function decode(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** A section's mesh, moved from the section's physical frame into the model frame. */
export function buildVoxelShapeSection(
  section: VoxelSection,
  options: Readonly<VoxelShapeOptions> = DEFAULT_VOXEL_SHAPE,
): VoxelShapeMesh {
  const [sx, sy, sz] = section.size;
  const step: [number, number, number] = [
    (section.maxBounds[0] - section.minBounds[0]) / Math.max(1, sx),
    (section.maxBounds[1] - section.minBounds[1]) / Math.max(1, sy),
    (section.maxBounds[2] - section.minBounds[2]) / Math.max(1, sz),
  ];
  const mesh = buildVoxelShapeMesh(
    { size: section.size, cells: decode(section.voxels), step, origin: section.minBounds },
    options,
  );
  const frame = voxelSectionFrame(section);
  for (let i = 0; i < mesh.positions.length; i += 3) {
    mesh.positions.set(
      frame.position(...(Array.from(mesh.positions.subarray(i, i + 3)) as [number, number, number]), false),
      i,
    );
    mesh.normals.set(
      frame.normal(...(Array.from(mesh.normals.subarray(i, i + 3)) as [number, number, number]), 'physical'),
      i,
    );
  }
  if (frame.mirrored)
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const swap = mesh.indices[t + 1]!;
      mesh.indices[t + 1] = mesh.indices[t + 2]!;
      mesh.indices[t + 2] = swap;
    }
  return mesh;
}

/** Every part of a model (sections concatenated per part), with the turret pivot instantiateVoxel uses. */
export function buildVoxelShapeModel(
  model: VoxelModel,
  options: Readonly<VoxelShapeOptions> = DEFAULT_VOXEL_SHAPE,
): VoxelShapePart[] {
  const turretOffset = (model.turretOffset ?? 0) / LEPTONS_PER_CELL;
  return model.parts.map((part) => ({
    role: part.role,
    turretOffset,
    mesh: mergeShapeMeshes(part.sections.map((section) => buildVoxelShapeSection(section, options))),
  }));
}

export function mergeShapeMeshes(meshes: readonly VoxelShapeMesh[]): VoxelShapeMesh {
  if (meshes.length === 1) return meshes[0]!;
  const sum = (f: (m: VoxelShapeMesh) => number) => meshes.reduce((s, m) => s + f(m), 0);
  const vertices = sum((m) => m.material.length);
  const out: VoxelShapeMesh = {
    positions: new Float32Array(vertices * 3),
    normals: new Float32Array(vertices * 3),
    lattice: new Float32Array(vertices * 3),
    material: new Uint8Array(vertices),
    direction: new Uint8Array(vertices),
    stair: new Uint8Array(vertices),
    // Section-local record indices: callers joining per-cell data do so per section before merging.
    cell: new Int32Array(vertices),
    indices: new Uint32Array(sum((m) => m.indices.length)),
    stats: mergeStats(meshes.map((m) => m.stats)),
  };
  let v = 0;
  let t = 0;
  for (const m of meshes) {
    out.positions.set(m.positions, v * 3);
    out.normals.set(m.normals, v * 3);
    out.lattice.set(m.lattice, v * 3);
    out.material.set(m.material, v);
    out.direction.set(m.direction, v);
    out.stair.set(m.stair, v);
    out.cell.set(m.cell, v);
    for (let i = 0; i < m.indices.length; i++) out.indices[t + i] = m.indices[i]! + v;
    v += m.material.length;
    t += m.indices.length;
  }
  return out;
}

function mergeStats(stats: readonly VoxelShapeStats[]): VoxelShapeStats {
  const total = (k: keyof VoxelShapeStats) => stats.reduce((s, x) => s + x[k], 0);
  const corners = total('cornerVertices');
  return {
    cells: total('cells'),
    faces: total('faces'),
    triangles: total('triangles'),
    vertices: total('vertices'),
    weldedVertices: total('weldedVertices'),
    cornerVertices: corners,
    stairFaces: total('stairFaces'),
    maxOffset: Math.max(0, ...stats.map((s) => s.maxOffset)),
    meanOffset: corners ? stats.reduce((s, x) => s + x.meanOffset * x.cornerVertices, 0) / corners : 0,
  };
}
