import type { VoxelModel, VoxelSection } from '../../../ra2-assets/src/manifest';
import { buildHardSurfaceLiveMesh, hardSurfacePaintSampler, type VoxelPaint } from '../hardSurfaceLiveMesh';
import { foldHardSurfaceFacePaint, type FacePaintFoldStats } from '../hardSurfaceFacePaint';
import { buildPart, type HardSurfaceSpec, type PartSpec } from '../hardSurfaceMesh';
import { bakeLookVolume, vectoriseSeams, type LookCylinder, type LookVolume } from '../hardSurfaceLookVolume';
import { assignDetailCharts, expandVertexAttribute, type DetailChartTable } from '../hardSurfaceDetailAtlas';
import { analyseVoxelPaint } from '../voxelPaint';
import { voxelPaintInputFor } from './voxelCleanModel';
import { ra2VoxelPaintPolicy } from './voxelPaintPolicy';
import { RA2_HARD_SURFACE_LOOK } from './hardSurfaceLook';
import { gridIndex, hardSurfaceSolid } from '../hardSurfaceSolid';
import type { SmoothMeshData } from './voxelSmooth';
import { voxelSectionFrame } from './voxelSectionFrame';

export interface CadMeshData extends SmoothMeshData {
  /** Unfolded legacy mesh, built through the original sampler for exact old-PBR comparison. */
  pbr: SmoothMeshData;
  lookCode: Float32Array;
  /** Folded host paint for the look material; pbr retains the exact legacy attributes and topology. */
  lookIndex: Float32Array;
  lookVolume: LookVolume;
  /** Baked detail atlas coordinates in texels, (-1, -1) where no chart applies (every vertex when no atlas). */
  detailUv: Float32Array;
  /** Atlas size in texels; null when the part has no detail table. */
  detailSize: [number, number] | null;
  /** Decoded atlas, attached on the presenting thread after the build (never crosses the Worker boundary). */
  detailImage?: ImageBitmap | null;
}

export interface CadDetailStats {
  triangles: number;
  matched: number;
}

/** Pure CPU adapter; both the Worker and offline measurements run this exact implementation. */
export function buildCadSection(
  section: VoxelSection,
  spec: PartSpec,
  palette: Uint8Array,
  maxEdge = 0.5,
  res = 4,
  reportFold?: (stats: FacePaintFoldStats) => void,
  detail?: DetailChartTable,
  reportDetail?: (stats: CadDetailStats) => void,
): CadMeshData {
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
  const legacy = buildHardSurfaceLiveMesh(spec, maxEdge, sample, RA2_HARD_SURFACE_LOOK.paintCodes);
  const input = voxelPaintInputFor(section),
    analysis = analyseVoxelPaint(input, ra2VoxelPaintPolicy(palette));
  const source = new Int32Array(count).fill(-1);
  for (let i = 0; i < input.x.length; i++) source[gridIndex(section.size, input.x[i], input.y[i], input.z[i])] = i;
  const regionIndex = analysis.regions.map((region) => {
    let best = 0,
      distance = Infinity;
    for (const index of Object.keys(region.indices).map(Number)) {
      const d = region.albedo.reduce((sum, v, a) => sum + (palette[index * 3 + a] - v) ** 2, 0);
      if (d < distance) {
        best = index;
        distance = d;
      }
    }
    return best;
  });
  const regionAt = (p: readonly number[], n: readonly number[]) => analysis.region[source[sample(p, n).cell]];
  let fold!: ReturnType<typeof foldHardSurfaceFacePaint>;
  const mesh = buildHardSurfaceLiveMesh(
    spec,
    maxEdge,
    undefined,
    RA2_HARD_SURFACE_LOOK.paintCodes,
    (surface, primitives) => {
      fold = foldHardSurfaceFacePaint(surface, {
        region: regionAt,
        paint: (id, p, n, dominant) => {
          const region = analysis.regions[id];
          // Retained house regions keep the exact palette entry; folded samples use their new host's representative.
          const index = region.masked && !dominant ? sample(p, n).index : regionIndex[id];
          return { index, shade: index };
        },
        luma: (id) => analysis.regions[id].albedo.reduce((sum, v, a) => sum + (v / 255) * [0.3, 0.59, 0.11][a], 0),
        house: (id) => analysis.regions[id].masked,
        glint: (id) => {
          const region = analysis.regions[id];
          return (
            !region.masked &&
            region.material === 'painted' &&
            region.albedo.reduce((sum, v) => sum + v / 255, 0) / 3 > 0.8
          );
        },
        flat: (_face, primitive) => {
          const code = RA2_HARD_SURFACE_LOOK.paintCodes[primitives[primitive].paint ?? ''];
          return code === 3 || code === 6;
        },
      });
      return fold.sample;
    },
  );
  if (reportFold) reportFold(fold.measure(legacy));
  const seams = vectoriseSeams(
    Array.from(input.x, (x, i) => ({
      position: [x, input.y[i], input.z[i]] as const,
      surface: analysis.surface[i] !== 0,
      strength: analysis.seam[i],
      kind: analysis.seams[analysis.seamFeature[i]]?.kind ?? '',
    })),
  );
  const cylinders = new Map<number, LookCylinder>();
  mesh.primitives.forEach((p, i) => {
    if (!p.cylinder) return;
    const code = RA2_HARD_SURFACE_LOOK.paintCodes[p.paint ?? ''];
    const profile = code === 4 ? 'wheel' : code === 5 ? 'hoop' : code === 6 ? 'ring' : null;
    if (!profile) return;
    const { centre, axis } = p.cylinder;
    // Expanded halfspaces include mirrored ranges and carved fragments, unlike the original spec metadata.
    const range: [number, number] = [-Infinity, Infinity];
    for (const plane of p.planes) {
      const d = plane[0] * axis[0] + plane[1] * axis[1] + plane[2] * axis[2];
      if (Math.abs(d) < 0.9999) continue;
      const at = (plane[3] - plane[0] * centre[0] - plane[1] * centre[1] - plane[2] * centre[2]) / d;
      if (d > 0) range[1] = Math.min(range[1], at);
      else range[0] = Math.max(range[0], at);
    }
    cylinders.set(i, { centre, axis, range, profile });
  });
  const lookVolume = bakeLookVolume(mesh, {
    res,
    seams,
    cylinders,
    lines: RA2_HARD_SURFACE_LOOK.lines,
    comb: RA2_HARD_SURFACE_LOOK.seams,
  });
  // Baked detail charts are matched on the folded look mesh only; the legacy PBR twin keeps its exact topology.
  let detailUv: Float32Array;
  if (detail) {
    const assigned = assignDetailCharts(mesh.lattice, mesh.indices, detail);
    reportDetail?.({ triangles: assigned.triangles, matched: assigned.matched });
    if (assigned.source.length !== mesh.chamfer.length) {
      const src = assigned.source;
      mesh.normals = expandVertexAttribute(mesh.normals, 3, src);
      mesh.lattice = expandVertexAttribute(mesh.lattice, 3, src);
      mesh.pindex = expandVertexAttribute(mesh.pindex, 1, src);
      mesh.pshade = expandVertexAttribute(mesh.pshade, 1, src);
      mesh.occlusion = expandVertexAttribute(mesh.occlusion, 1, src);
      mesh.chamfer = expandVertexAttribute(mesh.chamfer, 1, src);
      mesh.lookCode = expandVertexAttribute(mesh.lookCode, 1, src);
    }
    mesh.indices = assigned.indices;
    detailUv = assigned.uv;
  } else detailUv = new Float32Array(mesh.chamfer.length * 2).fill(-1);
  const out: CadMeshData = {
    detailUv,
    detailSize: detail ? [detail.size[0], detail.size[1]] : null,
    lookCode: mesh.lookCode,
    lookIndex: mesh.pindex,
    pbr: {
      position: legacy.positions.slice(),
      normal: legacy.normals.slice(),
      vpos: legacy.lattice,
      pindex: legacy.pindex,
      pshade: legacy.pshade,
      occlusion: legacy.occlusion,
      curvature: Float32Array.from(legacy.chamfer, (ch) => ch * 2),
      indices: legacy.indices,
    },
    lookVolume,
    position: new Float32Array(mesh.chamfer.length * 3),
    normal: mesh.normals.slice(),
    vpos: mesh.lattice,
    pindex: mesh.pindex,
    pshade: mesh.pshade,
    occlusion: mesh.occlusion,
    curvature: Float32Array.from(mesh.chamfer, (ch) => ch * 2),
    indices: mesh.indices,
  };
  const frame = voxelSectionFrame(section);
  for (const target of [out, out.pbr]) {
    for (let t = 0; t < target.pindex.length; t += 3) {
      for (let v = t; v < t + 3; v++) {
        const p = Array.from(target.vpos.subarray(v * 3, v * 3 + 3)) as [number, number, number];
        const n = Array.from(target.normal.subarray(v * 3, v * 3 + 3)) as [number, number, number];
        target.position.set(frame.position(...p), v * 3);
        target.normal.set(frame.normal(...n), v * 3);
      }
      if (frame.mirrored) {
        const tmp = target.indices[t + 1];
        target.indices[t + 1] = target.indices[t + 2];
        target.indices[t + 2] = tmp;
      }
    }
  }
  return out;
}

export function buildCadModel(
  model: VoxelModel,
  spec: HardSurfaceSpec,
  palette: Uint8Array,
  reportFold?: (part: string, stats: FacePaintFoldStats) => void,
  detail?: Readonly<Record<string, DetailChartTable>>,
  reportDetail?: (part: string, stats: CadDetailStats) => void,
): CadMeshData[] {
  if (Object.keys(spec.parts).some((role) => !model.parts.some((part) => part.role === role)))
    throw new Error('CAD part has no source part');
  // Full RGBA8 AABBs at 4x exceed 8 MB for the Rhino. Choose one density for the whole model, retaining
  // the generic baker's 4x default while bounding model textures to 4 MB (the Rhino uses 3x, ~3.6 MB).
  let cells = 0;
  for (const ps of Object.values(spec.parts)) {
    const positions = buildPart(ps).mesh.positions,
      lo = [Infinity, Infinity, Infinity],
      hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i++) {
      const a = i % 3;
      lo[a] = Math.min(lo[a], Math.floor(positions[i]) - 1);
      hi[a] = Math.max(hi[a], Math.ceil(positions[i]) + 1);
    }
    cells += (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  }
  const res = Math.min(4, Math.floor(Math.cbrt(4_000_000 / Math.max(1, cells * 4))));
  if (res < 1) throw new Error('CAD look volume exceeds model budget');
  return model.parts.map((part) => {
    const section = part.sections[0],
      ps = spec.parts[part.role];
    if (!section || !ps) throw new Error(`Missing CAD part: ${part.role}`);
    return buildCadSection(
      section,
      ps,
      palette,
      0.5,
      res,
      reportFold && ((stats) => reportFold(part.role, stats)),
      detail?.[part.role],
      reportDetail && ((stats) => reportDetail(part.role, stats)),
    );
  });
}
