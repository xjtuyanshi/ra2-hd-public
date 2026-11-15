import { buildPart, type PartSpec } from './hardSurfaceMesh';
import { forCells, gridCell, validateGrid, type SolidGrid, type Vec3 } from './hardSurfaceSolid';

export interface TriangleMesh {
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
}
export interface SilhouetteView {
  name: string;
  width: number;
  height: number;
  /** Lattice corner coordinates to fixed screen pixels; the caller owns camera and part placement. */
  project: (p: Vec3) => [number, number];
}
export interface SilhouetteComparison {
  name: string;
  iou: number;
  missing: number;
  extra: number;
  referencePixels: number;
}
export function solidSurfaceMesh(solid: SolidGrid): TriangleMesh {
  validateGrid(solid);
  const positions: number[] = [],
    indices: number[] = [];
  forCells(solid.size, (p, i) => {
    if (!solid.data[i]) return;
    for (let a = 0; a < 3; a++)
      for (const sign of [-1, 1]) {
        const q = [...p] as Vec3;
        q[a] += sign;
        if (gridCell(solid, q)) continue;
        const [u, v] = [0, 1, 2].filter((k) => k !== a),
          base = positions.length / 3;
        for (const [du, dv] of [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ]) {
          const c = [...p] as Vec3;
          c[a] += sign === 1 ? 1 : 0;
          c[u] += du;
          c[v] += dv;
          positions.push(...c);
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      }
  });
  return { positions, indices };
}
export function partMeshes(part: PartSpec): TriangleMesh[] {
  return buildPart(part).meshes.map((m) => ({ positions: m.positions, indices: m.indices }));
}
/** Deterministic two-sided silhouette. Pixel centres and a fixed camera are used for both geometries. */
export function rasterSilhouette(meshes: readonly TriangleMesh[], view: SilhouetteView): Uint8Array {
  if (!Number.isInteger(view.width) || !Number.isInteger(view.height) || view.width < 1 || view.height < 1)
    throw new Error('Invalid silhouette viewport');
  const out = new Uint8Array(view.width * view.height);
  for (const mesh of meshes) {
    if (mesh.positions.length % 3 || mesh.indices.length % 3) throw new Error('Invalid triangle mesh');
    const points: [number, number][] = [];
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const p = view.project([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
      if (!p.every(Number.isFinite)) throw new Error('Non-finite projection');
      // A clipped object can otherwise report a misleadingly high overlap.
      if (p[0] < 0 || p[0] > view.width || p[1] < 0 || p[1] > view.height)
        throw new Error('Silhouette viewport clips geometry');
      points.push(p);
    }
    for (let i = 0; i < mesh.indices.length; i += 3) {
      const tri = [points[mesh.indices[i]], points[mesh.indices[i + 1]], points[mesh.indices[i + 2]]];
      if (tri.some((p) => !p)) throw new Error('Triangle index outside mesh');
      const [a, b, c] = tri;
      const edge = (p: number[], q: number[], x: number, y: number) =>
        (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
      const area = edge(a, b, c[0], c[1]);
      if (Math.abs(area) < 1e-10) continue;
      const sign = area > 0 ? 1 : -1;
      const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0]))),
        x1 = Math.min(view.width - 1, Math.ceil(Math.max(a[0], b[0], c[0])));
      const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1]))),
        y1 = Math.min(view.height - 1, Math.ceil(Math.max(a[1], b[1], c[1])));
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++)
          if (
            edge(a, b, x + 0.5, y + 0.5) * sign >= -1e-8 &&
            edge(b, c, x + 0.5, y + 0.5) * sign >= -1e-8 &&
            edge(c, a, x + 0.5, y + 0.5) * sign >= -1e-8
          )
            out[y * view.width + x] = 1;
    }
  }
  return out;
}
export function compareSilhouettes(reference: Uint8Array, candidate: Uint8Array, name: string): SilhouetteComparison {
  if (reference.length !== candidate.length) throw new Error('Silhouette mask dimensions differ');
  let intersection = 0,
    union = 0,
    missing = 0,
    extra = 0,
    referencePixels = 0;
  for (let i = 0; i < reference.length; i++) {
    if (reference[i]) referencePixels++;
    if (reference[i] && candidate[i]) intersection++;
    if (reference[i] || candidate[i]) union++;
    if (reference[i] && !candidate[i]) missing++;
    if (!reference[i] && candidate[i]) extra++;
  }
  if (!referencePixels) throw new Error('Cannot accept an empty reference silhouette');
  return { name, iou: intersection / union, missing, extra, referencePixels };
}
