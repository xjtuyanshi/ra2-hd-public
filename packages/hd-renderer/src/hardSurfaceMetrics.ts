import {
  add,
  buildPart,
  cross,
  dot,
  insideAny,
  scale,
  sub,
  type ExpandedPrimitive,
  type HardSurfaceMesh,
  type PartSpec,
  type Vec3,
} from './hardSurfaceMesh';
import { forCells, gridCell, validateGrid, type SolidGrid } from './hardSurfaceSolid';

export interface SurfaceSample {
  point: Vec3;
  normal: Vec3;
}
export interface CadSample extends SurfaceSample {
  kind: 0 | 1 | 2;
  slope: boolean;
  primitive: number;
}
export interface Deviation {
  max: number;
  p99: number;
  mean: number;
}
export interface HardSurfaceMetrics {
  iou: number;
  missing: number;
  extra: number;
  voxels: number;
  dev_vox_to_cad: Deviation;
  dev_cad_to_vox: Deviation & { max_flat: number; max_slope: number };
  n_over_half_flat: number;
  n_cad: number;
  /** Sample-weighted reference shares: planar includes chamfers, so chamfer overlaps planar. */
  area_share: { planar: number; cylinder: number; chamfer: number };
  primitives: number;
}
export function revoxelise(primitives: readonly ExpandedPrimitive[], size: SolidGrid['size']): Float64Array {
  const out = new Float64Array(size[0] * size[1] * size[2]),
    g = [1 / 6, 0.5, 5 / 6];
  forCells(size, (p, i) => {
    let count = 0;
    for (const a of g)
      for (const b of g) for (const c of g) if (insideAny([p[0] + a, p[1] + b, p[2] + c], primitives)) count++;
    out[i] = count / 27;
  });
  return out;
}
export function voxelSurfaceSamples(solid: SolidGrid, n = 3): SurfaceSample[] {
  const out: SurfaceSample[] = [];
  for (let a = 0; a < 3; a++)
    for (const sign of [-1, 1]) {
      const [u, v] = [0, 1, 2].filter((k) => k !== a),
        exposed: Vec3[] = [];
      forCells(solid.size, (p, i) => {
        const q = [...p] as Vec3;
        q[a] += sign;
        if (solid.data[i] && !gridCell(solid, q)) exposed.push(p);
      });
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          for (const cell of exposed) {
            const p = [...cell] as Vec3,
              normal: Vec3 = [0, 0, 0];
            p[a] += sign === 1 ? 1 : 0;
            p[u] += (i + 0.5) / n;
            p[v] += (j + 0.5) / n;
            normal[a] = sign;
            out.push({ point: p, normal });
          }
    }
  return out;
}
export function rayVisible(solid: SolidGrid, sample: SurfaceSample): boolean {
  const cell = add(sample.point, scale(sample.normal, 0.5)).map(Math.floor) as Vec3;
  for (let step = 0; step < Math.max(...solid.size); step++)
    if (gridCell(solid, add(cell, scale(sample.normal, step)))) return false;
  return true;
}
/** Mulberry32, local to each triangle; reproducible without a platform RNG or shared state. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function cadSurfaceSamples(
  primitives: readonly ExpandedPrimitive[],
  meshes: readonly HardSurfaceMesh[],
  step = 0.2,
  seed = 0,
): CadSample[] {
  if (!(step > 0)) throw new Error('Surface sample step must be positive');
  const out: CadSample[] = [];
  meshes.forEach((m, i) => {
    for (let t = 0; t < m.indices.length / 3; t++) {
      const ids = [m.indices[t * 3], m.indices[t * 3 + 1], m.indices[t * 3 + 2]];
      const [a, b, c] = ids.map((k) => Array.from(m.positions.slice(k * 3, k * 3 + 3)) as Vec3);
      const ab = sub(b, a),
        ac = sub(c, a),
        nf = cross(ab, ac),
        area = 0.5 * Math.hypot(...nf),
        normal = scale(nf, 1 / (Math.hypot(...nf) + 1e-12));
      const count = Math.max(3, Math.ceil(area / (step * step)) * 2),
        random = seededRandom(seed + t + 1000 * i);
      const fi = ids[0],
        shading = Array.from(m.normals.slice(fi * 3, fi * 3 + 3)),
        cylinder = primitives[i].cylinder;
      const kind = cylinder && Math.abs(dot(shading, cylinder.axis)) < 0.2 ? 2 : m.chamfer[fi] ? 1 : 0;
      const slope = Math.max(...normal.map(Math.abs)) <= 0.999;
      for (let k = 0; k < count; k++) {
        let r = random(),
          s = random();
        if (r + s > 1) {
          r = 1 - r;
          s = 1 - s;
        }
        const point = add(a, add(scale(ab, r), scale(ac, s)));
        // Test the point just outside the face against the original (unchamfered) halfspaces.
        if (!insideAny(add(point, scale(normal, 0.05)), primitives, i, 1e-6))
          out.push({ point, normal, kind, slope, primitive: i });
      }
    }
  });
  return out;
}

/** Reference NN: only the 5^3 neighbouring buckets are searched. No candidates yields Infinity. */
export function nearestDistances(queries: readonly Vec3[], reference: readonly Vec3[], cell = 1): Float64Array {
  if (!(cell > 0)) throw new Error('Bucket size must be positive');
  const buckets = new Map<string, Vec3[]>(),
    qgroups = new Map<string, { key: Vec3; ids: number[] }>();
  const key = (p: Vec3): Vec3 => p.map((v) => Math.floor(v / cell)) as Vec3;
  for (const p of reference) {
    const k = key(p).join(',');
    const b = buckets.get(k);
    if (b) b.push(p);
    else buckets.set(k, [p]);
  }
  queries.forEach((p, i) => {
    const k = key(p),
      s = k.join(','),
      b = qgroups.get(s);
    if (b) b.ids.push(i);
    else qgroups.set(s, { key: k, ids: [i] });
  });
  const out = new Float64Array(queries.length).fill(Infinity);
  for (const { key: k, ids } of qgroups.values()) {
    const candidates: Vec3[] = [];
    for (let a = -2; a <= 2; a++)
      for (let b = -2; b <= 2; b++)
        for (let c = -2; c <= 2; c++) {
          const bucket = buckets.get([k[0] + a, k[1] + b, k[2] + c].join(','));
          if (bucket) for (const p of bucket) candidates.push(p);
        }
    for (const i of ids) {
      const p = queries[i];
      let best = Infinity;
      for (const q of candidates) {
        const x = p[0] - q[0],
          y = p[1] - q[1],
          z = p[2] - q[2],
          d = x * x + y * y + z * z;
        if (d < best) best = d;
      }
      out[i] = Math.sqrt(best);
    }
  }
  return out;
}
const rounded = (n: number, digits: number) => roundFinite(n, 10 ** digits);
function roundFinite(n: number, factor: number): number {
  return Math.round(n * factor) / factor;
}
export function deviation(values: ArrayLike<number>): Deviation {
  if (!values.length) return { max: 0, p99: 0, mean: 0 };
  const sorted = Array.from(values).sort((a, b) => a - b),
    pos = (sorted.length - 1) * 0.99,
    lo = Math.floor(pos),
    hi = Math.ceil(pos);
  const p99 = lo === hi || sorted[lo] === sorted[hi] ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  return {
    max: rounded(sorted[sorted.length - 1], 3),
    p99: rounded(p99, 3),
    mean: rounded(sorted.reduce((s, v) => s + v, 0) / sorted.length, 3),
  };
}
export function evaluateHardSurface(
  solid: SolidGrid,
  part: PartSpec,
  options: { seed?: number; sampleStep?: number } = {},
): HardSurfaceMetrics {
  validateGrid(solid);
  const { primitives, meshes } = buildPart(part),
    fraction = revoxelise(primitives, solid.size);
  let inter = 0,
    union = 0,
    missing = 0,
    extra = 0,
    voxels = 0;
  for (let i = 0; i < fraction.length; i++) {
    const v = fraction[i] >= 0.5,
      s = !!solid.data[i];
    if (v && s) inter++;
    if (v || s) union++;
    if (s && !v) missing++;
    if (v && !s) extra++;
    if (s) voxels++;
  }
  const vs = voxelSurfaceSamples(solid),
    cs = cadSurfaceSamples(primitives, meshes, options.sampleStep ?? 0.2, options.seed ?? 0);
  const d2 = nearestDistances(
    cs.map((s) => s.point),
    vs.map((s) => s.point),
  );
  const d1 = nearestDistances(
    vs.filter((s) => rayVisible(solid, s)).map((s) => s.point),
    cs.map((s) => s.point),
  );
  const flat: number[] = [],
    slope: number[] = [];
  cs.forEach((s, i) => (s.slope || s.kind === 2 ? slope : flat).push(d2[i]));
  const share = (kind: 1 | 2) => (cs.length ? cs.filter((s) => s.kind === kind).length / cs.length : 0);
  return {
    iou: rounded(union ? inter / union : 1, 4),
    missing,
    extra,
    voxels,
    dev_vox_to_cad: deviation(d1),
    dev_cad_to_vox: { ...deviation(d2), max_flat: deviation(flat).max, max_slope: deviation(slope).max },
    n_over_half_flat: flat.filter((d) => d > 0.5).length,
    n_cad: cs.length,
    area_share: {
      planar: cs.length ? rounded(1 - share(2), 4) : 0,
      cylinder: rounded(share(2), 4),
      chamfer: rounded(share(1), 4),
    },
    primitives: primitives.length,
  };
}
