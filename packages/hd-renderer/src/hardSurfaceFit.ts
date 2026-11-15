import {
  dot,
  roundEven,
  type Plane,
  type Primitive,
  type PrimitiveOptions,
  type HardSurfaceSpec,
  type PartSpec,
} from './hardSurfaceMesh';
import {
  axisIndex,
  forCells,
  hardSurfaceSolid,
  validateGrid,
  type Axis,
  type SolidGrid,
  type Vec3,
} from './hardSurfaceSolid';

export type Region = [Vec3, Vec3];
export type DirectionKind = 'axis' | 'all' | '45' | [number, number][];
export interface SeedComponent extends PrimitiveOptions {
  type?: 'raw' | 'cyl' | 'poly';
  prim?: Primitive;
  region?: Region;
  exclude?: Region[];
  colours?: [number, number];
  dirs?: DirectionKind;
  extra_ratios?: [number, number][];
  minsteps?: number;
  cuts?: Plane[];
  set?: Partial<Record<`${'+' | '-'}${Axis}`, number>>;
  axis?: Axis;
  c?: [number, number];
  r?: number;
  range?: [number, number];
}
export interface HardSurfaceSeed {
  parts: Record<string, { mirror?: PartSpec['mirror']; components: SeedComponent[] }>;
}
export interface FitGrid extends SolidGrid {
  colours?: Int16Array;
}
export interface FitOptions {
  down?: Axis;
  units?: string;
  onEmpty?: (role: string, component: SeedComponent) => void;
}

export function directions(kind: DirectionKind = 'all'): Vec3[] {
  const ds: Vec3[] = [];
  for (let a = 0; a < 3; a++)
    for (const s of [-1, 1]) {
      const n: Vec3 = [0, 0, 0];
      n[a] = s;
      ds.push(n);
    }
  if (kind === 'axis') return ds;
  const ratios =
    kind === 'all'
      ? [
          [1, 1],
          [1, 2],
          [2, 1],
        ]
      : kind === '45'
        ? [[1, 1]]
        : kind;
  for (let a = 0; a < 3; a++)
    for (let b = a + 1; b < 3; b++)
      for (const [p, q] of ratios)
        for (const sa of [-1, 1])
          for (const sb of [-1, 1]) {
            const n: Vec3 = [0, 0, 0];
            n[a] = sa * p;
            n[b] = sb * q;
            ds.push(n);
          }
  return ds;
}
export function robustTop(n: Vec3, centres: readonly Vec3[]): number {
  const values = centres.map((c) => dot(c, n)),
    free = n.findIndex((v) => v === 0);
  let lo = Infinity,
    hi = -Infinity;
  if (free >= 0)
    for (const c of centres) {
      lo = Math.min(lo, c[free]);
      hi = Math.max(hi, c[free]);
    }
  const ext = free < 0 ? 1 : Math.max(1, hi - lo + 1),
    threshold = Math.max(3, Math.trunc(0.25 * ext));
  const levels = [...new Set(values.map((v) => roundEven(v * 1000) / 1000))].sort((a, b) => b - a);
  for (const level of levels) if (values.filter((v) => Math.abs(v - level) < 1e-3).length >= threshold) return level;
  return values.reduce((max, v) => Math.max(max, v), -Infinity);
}
export function stairOK(n: Vec3, centres: readonly Vec3[], minSteps = 3): boolean {
  const top = robustTop(n, centres),
    C = centres.filter((c) => dot(c, n) <= top + 1e-3);
  const len = Math.hypot(...n),
    s = C.map((c) => dot(c, n) / len),
    max = s.reduce((max, v) => Math.max(max, v), -Infinity);
  const sup = C.filter((_, i) => s[i] >= max - (0.5 * Math.max(...n.map(Math.abs))) / len - 1e-6);
  const ax = [0, 1, 2].filter((i) => n[i] !== 0),
    steps = ax.map((a) => new Set(sup.map((c) => c[a])).size),
    major = ax.map((a) => Math.abs(n[a]));
  const k = major[0] === major[1] ? 0 : major.indexOf(Math.max(...major));
  return Math.min(...steps) >= (minSteps || 3) - (major[0] !== major[1] ? 1 : 0) && steps[k] >= 2;
}
export function fitPoly(centres: readonly Vec3[], dirs: readonly Vec3[], minSteps = 3): Plane[] {
  return dirs.flatMap((n) => {
    if (n.filter((v) => v !== 0).length !== 1 && !stairOK(n, centres, minSteps)) return [];
    let d = -Infinity;
    for (const c of centres) d = Math.max(d, dot(c, n));
    return [[...n, d + 0.5 * Math.max(...n.map(Math.abs))] as Plane];
  });
}
/** Sequential coordinate descent. Strict improvement preserves the outermost offset on ties. */
export function tuneOffsets(planes: readonly Plane[], mask: SolidGrid): Plane[] {
  const out = planes.map((p) => [...p] as Plane);
  if (!out.length) return out;
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  forCells(mask.size, (p, i) => {
    if (mask.data[i])
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], p[a] - 1);
        hi[a] = Math.max(hi[a], p[a] + 2);
      }
  });
  const points: Vec3[] = [],
    truth: number[] = [];
  forCells(mask.size, (p, i) => {
    if (!p.every((v, a) => v >= lo[a] && v < hi[a])) return;
    for (const a of [0.25, 0.75])
      for (const b of [0.25, 0.75])
        for (const c of [0.25, 0.75]) {
          points.push([p[0] + a, p[1] + b, p[2] + c]);
          truth.push(mask.data[i] ? 1 : 0);
        }
  });
  out.forEach((n, i) => {
    if (n.slice(0, 3).filter((v) => v !== 0).length === 1) return;
    // Samples rejected by other planes have constant error at every candidate offset.
    const eligible = points.flatMap((p, k) => (out.every((q, j) => j === i || dot(p, q) <= q[3] + 1e-9) ? [k] : []));
    let best = Infinity,
      bestD = n[3];
    const step = 0.25 * Math.max(...n.slice(0, 3).map(Math.abs));
    for (let k = 0; k < 7; k++) {
      const d = n[3] - step * k;
      let err = 0;
      for (const j of eligible) if ((dot(points[j], n) <= d + 1e-9 ? 1 : 0) !== truth[j]) err++;
      if (err < best) {
        best = err;
        bestD = d;
      }
    }
    n[3] = bestD;
  });
  return out;
}
function inRegion(p: Vec3, region: Region, size: SolidGrid['size']): boolean {
  // Match NumPy's clipped half-open slices, including negative bounds.
  return p.every((v, a) => {
    const bound = (n: number) => Math.max(0, Math.min(size[a], n < 0 ? n + size[a] : n));
    return v >= bound(region[0][a]) && v < bound(region[1][a]);
  });
}
export function selectComponent(solid: SolidGrid, colours: Int16Array | undefined, comp: SeedComponent): SolidGrid {
  if (!comp.region) throw new Error('Fitted components require a region');
  if (comp.colours && !colours) throw new Error('Colour filtering requires a colour volume');
  const data = new Uint8Array(solid.data.length);
  forCells(solid.size, (p, i) => {
    if (
      solid.data[i] &&
      inRegion(p, comp.region!, solid.size) &&
      !(comp.exclude ?? []).some((ex) => inRegion(p, ex, solid.size)) &&
      (!comp.colours || (colours![i] >= comp.colours[0] && colours![i] <= comp.colours[1]))
    )
      data[i] = 1;
  });
  return { size: [...solid.size], data };
}
export function fitCylinder(
  centres: readonly Vec3[],
  axis: Axis,
  comp: Pick<SeedComponent, 'c' | 'r' | 'range'>,
): Primitive {
  const a = axisIndex(axis),
    [u, v] = [0, 1, 2].filter((i) => i !== a),
    lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of centres)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], p[k] - 0.5);
      hi[k] = Math.max(hi[k], p[k] + 0.5);
    }
  return {
    type: 'cyl',
    axis,
    c: comp.c ? [...comp.c] : [(lo[u] + hi[u]) / 2, (lo[v] + hi[v]) / 2],
    r: comp.r || Math.max(hi[u] - lo[u], hi[v] - lo[v]) / 2,
    range: comp.range ? [...comp.range] : [lo[a], hi[a]],
  };
}
export function fitHardSurface(
  seed: HardSurfaceSeed,
  grids: Readonly<Record<string, FitGrid>>,
  options: FitOptions = {},
): HardSurfaceSpec {
  const out: HardSurfaceSpec = {
    units: options.units ?? 'voxel lattice, corner coordinates (voxel i spans [i, i+1])',
    parts: {},
  };
  for (const [role, ps] of Object.entries(seed.parts)) {
    const grid = grids[role];
    if (!grid) throw new Error(`Missing voxel part: ${role}`);
    validateGrid(grid);
    if (grid.colours && grid.colours.length !== grid.data.length) throw new Error('Invalid colour volume dimensions');
    const solid = hardSurfaceSolid(grid, options.down),
      primitives: Primitive[] = [];
    for (const comp of ps.components) {
      if (comp.off) continue;
      const base: PrimitiveOptions = {};
      for (const key of ['name', 'paint', 'chamfer', 'mirror', 'centred', 'groove'] as const)
        if (key in comp) Object.assign(base, { [key]: comp[key] });
      if (comp.type === 'raw') {
        if (!comp.prim) throw new Error('Raw components require a primitive');
        primitives.push({ ...structuredClone(comp.prim), ...base });
        continue;
      }
      const m = selectComponent(solid, grid.colours, comp),
        C: Vec3[] = [];
      forCells(m.size, (p, i) => {
        if (m.data[i]) C.push(p.map((v) => v + 0.5) as Vec3);
      });
      if (!C.length) {
        options.onEmpty?.(role, comp);
        continue;
      }
      let q: Primitive;
      if (comp.type === 'cyl') {
        if (!comp.axis) throw new Error('Cylinder components require an axis');
        q = fitCylinder(C, comp.axis, comp);
      } else {
        const dirs = directions(comp.dirs);
        for (const ratio of comp.extra_ratios ?? []) dirs.push(...directions([ratio]).slice(6));
        const planes = [
          ...tuneOffsets(fitPoly(C, dirs, comp.minsteps), m),
          ...(comp.cuts ?? []).map((p) => [...p] as Plane),
        ];
        for (const [key, v] of Object.entries(comp.set ?? {})) {
          const sign = key[0] === '+' ? 1 : -1,
            a = axisIndex(key[1] as Axis);
          for (const p of planes)
            if (p[a] === sign && p.slice(0, 3).filter((n) => n !== 0).length === 1) p[3] = sign * v!;
        }
        q = { type: 'poly', planes };
      }
      primitives.push({ ...q, ...base });
    }
    out.parts[role] = { ...(ps.mirror ? { mirror: { ...ps.mirror } } : {}), primitives };
  }
  return out;
}
