import { PAINT_MATERIALS, type VoxelPaintResult } from './voxelPaint';
import type { VoxelShapeMesh } from './voxelShapeMesh';

/**
 * Joins a de-dithered paint analysis (voxelPaint) to a shape-locked mesh (voxelShapeMesh) of the same voxel section:
 * every vertex takes the flat paint of the cell that owns its quad. Generic mechanism, no game knowledge; the paint
 * policy (which palette entries are house colour, which are metal) was already applied by the analysis.
 *
 * The result is what a "clean voxel" material needs per vertex: flat albedo (no per-voxel shade or dither), the
 * material class for roughness/metalness, the masked (house-colour) flag, and the seam map the original painted as
 * dark lines, with the line's axis so a shader can draw it as a groove narrower than one voxel.
 */
export interface CleanVertexPaint {
  /** Flat region albedo, 8-bit sRGB, 3 per vertex. */
  albedo: Uint8Array;
  /** Index into PAINT_MATERIALS, 1 per vertex. */
  material: Uint8Array;
  /** 1 where the cell's palette entry is in a masked family (e.g. house colour), 1 per vertex. */
  mask: Uint8Array;
  /** Seam strength 0..1 (0 = no seam), 1 per vertex. */
  seam: Float32Array;
  /** 0 = none, 1 = line, 2 = dot. */
  seamKind: Uint8Array;
  /** Lattice axis a line runs along (0, 1, 2), 255 for none or a dot. */
  seamAxis: Uint8Array;
  /** Paint region id (section-local), 1 per vertex. */
  region: Int32Array;
}

const AXIS = { x: 0, y: 1, z: 2 } as const;

function emptyPaint(n: number): CleanVertexPaint {
  return {
    albedo: new Uint8Array(n * 3),
    material: new Uint8Array(n),
    mask: new Uint8Array(n),
    seam: new Float32Array(n),
    seamKind: new Uint8Array(n),
    seamAxis: new Uint8Array(n).fill(255),
    region: new Int32Array(n),
  };
}

/** Paint per vertex, looked up through `mesh.cell` (the mesh and the analysis must come from the same cell records). */
export function attachVoxelPaint(mesh: VoxelShapeMesh, paint: VoxelPaintResult): CleanVertexPaint {
  const n = mesh.cell.length;
  const out = emptyPaint(n);
  const cells = paint.region.length;
  for (let v = 0; v < n; v++) {
    const c = mesh.cell[v]!;
    if (c < 0 || c >= cells) throw new RangeError(`vertex ${v} owns cell ${c}, paint covers ${cells}`);
    out.albedo[v * 3] = paint.albedo[c * 3]!;
    out.albedo[v * 3 + 1] = paint.albedo[c * 3 + 1]!;
    out.albedo[v * 3 + 2] = paint.albedo[c * 3 + 2]!;
    const r = paint.region[c]!;
    out.region[v] = r;
    out.material[v] = Math.max(0, PAINT_MATERIALS.indexOf(paint.regions[r]?.material ?? 'painted'));
    out.mask[v] = paint.mask[c]!;
    const s = paint.seamFeature[c]!;
    if (s >= 0) {
      const feature = paint.seams[s]!;
      out.seam[v] = paint.seam[c]!;
      out.seamKind[v] = feature.kind === 'line' ? 1 : 2;
      if (feature.kind === 'line') out.seamAxis[v] = AXIS[feature.axis];
    }
  }
  return out;
}

/** Concatenates per-section results in the same vertex order as mergeShapeMeshes. */
export function mergeCleanVertexPaint(parts: readonly CleanVertexPaint[]): CleanVertexPaint {
  if (parts.length === 1) return parts[0]!;
  const out = emptyPaint(parts.reduce((s, p) => s + p.material.length, 0));
  let v = 0;
  for (const p of parts) {
    out.albedo.set(p.albedo, v * 3);
    out.material.set(p.material, v);
    out.mask.set(p.mask, v);
    out.seam.set(p.seam, v);
    out.seamKind.set(p.seamKind, v);
    out.seamAxis.set(p.seamAxis, v);
    out.region.set(p.region, v);
    v += p.material.length;
  }
  return out;
}
