import type { VoxelModel, VoxelSection } from '../../../ra2-assets/src/manifest';
import { buildHardSurfaceLiveMesh, hardSurfacePaintSampler, type VoxelPaint } from '../hardSurfaceLiveMesh';
import type { HardSurfaceSpec, PartSpec } from '../hardSurfaceMesh';
import { gridIndex, hardSurfaceSolid } from '../hardSurfaceSolid';
import type { SmoothMeshData } from './voxelSmooth';
import { voxelSectionFrame } from './voxelSectionFrame';

/** Pure CPU adapter; both the Worker and offline measurements run this exact implementation. */
export function buildCadSection(section: VoxelSection, spec: PartSpec, maxEdge = 0.5): SmoothMeshData {
  const bytes = Uint8Array.from(atob(section.voxels), (c) => c.charCodeAt(0));
  const count = section.size.reduce((a, b) => a * b, 1);
  if (bytes.length % 5 || count > 4_000_000) throw new Error('Invalid CAD source section');
  const paint: VoxelPaint = {
    size: section.size,
    data: new Uint8Array(count),
    index: new Int16Array(count).fill(-1),
    shade: new Float32Array(count),
  };
  for (let i = 0; i < bytes.length; i += 5) {
    const [x, y, z, index] = bytes.subarray(i, i + 4);
    if (x >= section.size[0] || y >= section.size[1] || z >= section.size[2])
      throw new Error('Voxel outside CAD source');
    const k = gridIndex(section.size, x, y, z);
    paint.data[k] = 1;
    paint.index[k] = index;
    paint.shade[k] = index;
  }
  const sample = hardSurfacePaintSampler(hardSurfaceSolid(paint), paint);
  const mesh = buildHardSurfaceLiveMesh(spec, maxEdge, sample),
    vertices = mesh.chamfer.length;
  const out: SmoothMeshData = {
    position: new Float32Array(vertices * 3),
    normal: new Float32Array(vertices * 3),
    vpos: mesh.lattice,
    pindex: mesh.pindex,
    pshade: mesh.pshade,
    occlusion: mesh.occlusion,
    curvature: Float32Array.from(mesh.chamfer, (ch) => ch * 2),
    indices: mesh.indices,
  };
  const frame = voxelSectionFrame(section);
  for (let t = 0; t < vertices; t += 3) {
    for (let v = t; v < t + 3; v++) {
      const p = Array.from(mesh.positions.subarray(v * 3, v * 3 + 3)) as [number, number, number];
      const n = Array.from(mesh.normals.subarray(v * 3, v * 3 + 3)) as [number, number, number];
      out.position.set(frame.position(...p), v * 3);
      out.normal.set(frame.normal(...n), v * 3);
    }
    if (frame.mirrored) {
      const tmp = out.indices[t + 1];
      out.indices[t + 1] = out.indices[t + 2];
      out.indices[t + 2] = tmp;
    }
  }
  return out;
}

export function buildCadModel(model: VoxelModel, spec: HardSurfaceSpec): SmoothMeshData[] {
  if (Object.keys(spec.parts).some((role) => !model.parts.some((part) => part.role === role)))
    throw new Error('CAD part has no source part');
  return model.parts.map((part) => {
    const section = part.sections[0],
      ps = spec.parts[part.role];
    if (!section || !ps) throw new Error(`Missing CAD part: ${part.role}`);
    return buildCadSection(section, ps);
  });
}
