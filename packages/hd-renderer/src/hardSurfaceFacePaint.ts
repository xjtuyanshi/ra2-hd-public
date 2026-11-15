import type { SurfacePaint, SurfaceSampler } from './hardSurfaceAdaptive';
import { cross, sub, type HardSurfaceMesh, type Vec3 } from './hardSurfaceMesh';

export interface FacePaintPolicy {
  region(p: readonly number[], n: readonly number[]): number;
  paint(region: number, p: readonly number[], n: readonly number[], dominant: boolean): SurfacePaint;
  luma(region: number): number;
  house(region: number): boolean;
  glint(region: number): boolean;
  /** Caller-owned primitive policy; the generic fold does not interpret paint codes. */
  flat(face: number, primitive: number): boolean;
}
export interface FacePaintFoldStats {
  surfaceArea: number;
  changedArea: number;
  changedFraction: number;
}

/** Deterministic uniform triangle samples, at least four per triangle and approximately 24 per square unit.
 * Weight by area/count so tiny fan triangles do not outweigh larger ones because of the minimum sample count.
 */
function surfaceSamples(
  mesh: HardSurfaceMesh,
  visit: (p: Vec3, n: Vec3, face: number, primitive: number, area: number) => void,
): void {
  const ordinals = new Map<number, number>(),
    normals = new Map<number, Vec3>();
  const position = (i: number) => Array.from(mesh.positions.subarray(i * 3, i * 3 + 3)) as Vec3;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [ia, ib, ic] = mesh.indices.subarray(t, t + 3),
      a = position(ia),
      u = sub(position(ib), a),
      v = sub(position(ic), a),
      area = Math.hypot(...cross(u, v)) * 0.5;
    if (area < 1e-10) continue;
    const face = mesh.face[ia],
      ordinal = (ordinals.get(face) ?? 0) + 1;
    ordinals.set(face, ordinal);
    let n = normals.get(face);
    if (!n) {
      n = Array.from(mesh.normals.subarray(ia * 3, ia * 3 + 3)) as Vec3;
      normals.set(face, n);
    }
    const count = Math.max(4, Math.floor(area * 24));
    const seed = (Math.imul(face, 7) + ordinal) >>> 0;
    for (let i = 0; i < count; i++) {
      // Scrambled Hammersley points avoid clustering in the tiny features this histogram decides to retain.
      let bits = (i ^ seed) >>> 0;
      bits = ((bits >>> 1) & 0x55555555) | ((bits & 0x55555555) << 1);
      bits = ((bits >>> 2) & 0x33333333) | ((bits & 0x33333333) << 2);
      bits = ((bits >>> 4) & 0x0f0f0f0f) | ((bits & 0x0f0f0f0f) << 4);
      bits = ((bits >>> 8) & 0x00ff00ff) | ((bits & 0x00ff00ff) << 8);
      bits = ((bits >>> 16) | (bits << 16)) >>> 0;
      let r = (i + 0.5) / count,
        s = (bits + 0.5) / 4294967296;
      if (r + s > 1) {
        r = 1 - r;
        s = 1 - s;
      }
      visit(a.map((x, k) => x + r * u[k] + s * v[k]) as Vec3, n, face, mesh.primitive[ia], area / count);
    }
  }
}

/** Analyse whole faces before adaptive contour merging; only retained regions can introduce paint boundaries. */
export function foldHardSurfaceFacePaint(mesh: HardSurfaceMesh, policy: FacePaintPolicy) {
  const histograms = new Map<number, { area: number; primitive: number; regions: Map<number, number> }>();
  surfaceSamples(mesh, (p, n, face, primitive, area) => {
    let histogram = histograms.get(face);
    if (!histogram) {
      histogram = { area: 0, primitive, regions: new Map() };
      histograms.set(face, histogram);
    }
    const region = policy.region(p, n);
    histogram.area += area;
    histogram.regions.set(region, (histogram.regions.get(region) ?? 0) + area);
  });
  const faces = new Map<number, { dominant: number; keep: Set<number>; flat: boolean }>();
  for (const [face, histogram] of histograms) {
    // Stable region-ID tie break, independent of triangle traversal order.
    const regions = [...histogram.regions].sort((a, b) => b[1] - a[1] || a[0] - b[0]),
      dominant = regions[0][0],
      keep = new Set([dominant]);
    for (const [region, area] of regions) {
      const fraction = area / histogram.area;
      if (
        fraction >= 0.12 ||
        (policy.house(region) && fraction >= 0.04) ||
        (area >= 1.5 && policy.luma(region) > policy.luma(dominant) + 0.08)
      )
        keep.add(region);
    }
    faces.set(face, { dominant, keep, flat: policy.flat(face, histogram.primitive) });
  }
  const foldedRegion = (region: number, face: number) => {
    const rule = faces.get(face);
    if (!rule) throw new Error('Unknown paint face');
    const dominant = rule.flat || policy.glint(region) || !rule.keep.has(region);
    return { region: dominant ? rule.dominant : region, dominant };
  };
  const sample: SurfaceSampler = (p, n, face) => {
    const result = foldedRegion(policy.region(p, n), face);
    return policy.paint(result.region, p, n, result.dominant);
  };
  return {
    sample,
    /** Measure visible (culled) triangles separately from the full-face histogram, using the same area sampling. */
    measure(surface: HardSurfaceMesh): FacePaintFoldStats {
      let surfaceArea = 0,
        changedArea = 0;
      surfaceSamples(surface, (p, n, face, _primitive, area) => {
        const region = policy.region(p, n);
        surfaceArea += area;
        const result = foldedRegion(region, face);
        if (policy.paint(region, p, n, false).index !== policy.paint(result.region, p, n, result.dominant).index)
          changedArea += area;
      });
      return { surfaceArea, changedArea, changedFraction: surfaceArea ? changedArea / surfaceArea : 0 };
    },
  };
}
