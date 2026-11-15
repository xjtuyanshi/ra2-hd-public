import { directions, fitPoly, tuneOffsets } from './hardSurfaceFit';
import { dot, polytopeFaces, type PartSpec, type Plane, type Primitive } from './hardSurfaceMesh';
import { forCells, gridCell, gridIndex, validateGrid, type Axis, type SolidGrid, type Vec3 } from './hardSurfaceSolid';

export type RegionKind =
  'hull' | 'track-guard' | 'glacis' | 'rear-deck' | 'turret' | 'mantlet' | 'barrel' | 'wheel' | 'drum';
export interface RegionProposal {
  id: string;
  kind: RegionKind;
  bounds: [Vec3, Vec3];
  cells: Vec3[];
  primitive: Primitive;
  /** Centre-sample classification error, used for proposals only; never an acceptance metric. */
  error: number;
  support: number;
}
export interface ProposalOptions {
  category: 'chassis' | 'crown' | 'tube';
  budget: number;
  minSpan?: number;
  minCells?: number;
  extraRatios?: [number, number][];
  chamfer?: number;
  tunePlanes?: boolean;
}
export interface SymmetryEvidence {
  axis: Axis;
  at: number;
  iou: number;
  mismatched: number;
}
export interface RegionModel {
  regions: RegionProposal[];
  symmetry: SymmetryEvidence;
  part: PartSpec;
  options: ProposalOptions;
}

export function occupiedCells(grid: SolidGrid): Vec3[] {
  validateGrid(grid);
  const out: Vec3[] = [];
  forCells(grid.size, (p, i) => {
    if (grid.data[i]) out.push(p);
  });
  if (!out.length) throw new Error('Cannot propose regions for an empty solid');
  return out;
}
export function cellBounds(cells: readonly Vec3[]): [Vec3, Vec3] {
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of cells)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], p[a]);
      hi[a] = Math.max(hi[a], p[a] + 1);
    }
  return [lo, hi];
}
/** Detect lateral lattice reflection without silently replacing asymmetric source detail. */
export function detectMirror(grid: SolidGrid, axis: Axis = 'y'): SymmetryEvidence {
  const cells = occupiedCells(grid),
    a = 'xyz'.indexOf(axis),
    [lo, hi] = cellBounds(cells);
  const centre = (lo[a] + hi[a]) / 2;
  let best: SymmetryEvidence = { axis, at: centre, iou: 0, mismatched: cells.length * 2 };
  for (let shift = -2; shift <= 2; shift++) {
    const at = centre + shift * 0.5;
    let intersection = 0;
    for (const p of cells) {
      const q = [...p] as Vec3;
      q[a] = 2 * at - p[a] - 1;
      if (gridCell(grid, q)) intersection++;
    }
    const union = 2 * cells.length - intersection,
      iou = intersection / union;
    if (iou > best.iou) best = { axis, at, iou, mismatched: 2 * (cells.length - intersection) };
  }
  return best;
}
function semantic(bounds: [Vec3, Vec3], whole: [Vec3, Vec3], options: ProposalOptions): RegionKind {
  if (options.category === 'tube') return 'barrel';
  const [lo, hi] = bounds,
    [a, b] = whole;
  const cx = (lo[0] + hi[0] - 2 * a[0]) / (2 * (b[0] - a[0]));
  if (options.category === 'crown') return cx > 0.8 ? 'mantlet' : 'turret';
  const edge = Math.min(lo[1] - a[1], b[1] - hi[1]) / (b[1] - a[1]);
  if (edge < 0.18 && hi[1] - lo[1] < (b[1] - a[1]) * 0.4) return 'track-guard';
  return cx > 0.68 ? 'glacis' : cx < 0.3 && lo[2] > a[2] + (b[2] - a[2]) * 0.4 ? 'rear-deck' : 'hull';
}
function compactPlanes(planes: Plane[]): Plane[] {
  const { faces } = polytopeFaces(planes);
  return planes.filter((_, i) => faces.some((f) => f.plane === i));
}
function classifyError(
  cells: readonly Vec3[],
  bounds: [Vec3, Vec3],
  primitive: Primitive,
): { error: number; support: number } {
  const occupied = new Set(cells.map((p) => p.join(','))),
    [lo, hi] = bounds;
  let missing = 0,
    extra = 0,
    hit = 0;
  const inside = (p: Vec3) => {
    if (primitive.type === 'poly') return primitive.planes.every((n) => dot(p, n) <= n[3] + 1e-7);
    if (primitive.type === 'box') return p.every((v, a) => v >= primitive.min[a] && v <= primitive.max[a]);
    const a = 'xyz'.indexOf(primitive.axis),
      uv = [0, 1, 2].filter((k) => k !== a);
    return (
      p[a] >= primitive.range[0] &&
      p[a] <= primitive.range[1] &&
      Math.hypot(p[uv[0]] - primitive.c[0], p[uv[1]] - primitive.c[1]) <= primitive.r
    );
  };
  for (let x = lo[0]; x < hi[0]; x++)
    for (let y = lo[1]; y < hi[1]; y++)
      for (let z = lo[2]; z < hi[2]; z++) {
        const truth = occupied.has([x, y, z].join(',')),
          pred = inside([x + 0.5, y + 0.5, z + 0.5]);
        if (truth && pred) hit++;
        if (truth && !pred) missing++;
        if (!truth && pred) extra++;
      }
  return { error: missing + extra, support: hit / Math.max(1, hit + extra) };
}
/** A circle is a hypothesis only when both transverse spans and source support justify it. */
export function cylinderProposal(cells: readonly Vec3[], axis: Axis): Primitive {
  const [lo, hi] = cellBounds(cells),
    a = 'xyz'.indexOf(axis),
    [u, v] = [0, 1, 2].filter((k) => k !== a);
  const c: [number, number] = [(lo[u] + hi[u]) / 2, (lo[v] + hi[v]) / 2];
  // Equal-area circle is a useful start for square low-resolution samples; formal refinement remains necessary.
  const r = Math.sqrt(((hi[u] - lo[u]) * (hi[v] - lo[v])) / Math.PI);
  return { type: 'cyl', axis, c, r, range: [lo[a], hi[a]] };
}
export function fitRegion(cells: Vec3[], id: string, whole: [Vec3, Vec3], options: ProposalOptions): RegionProposal {
  const bounds = cellBounds(cells),
    centres = cells.map((p) => p.map((v) => v + 0.5) as Vec3);
  const dirs = directions('all');
  for (const ratio of options.extraRatios ?? [
    [1, 3],
    [3, 1],
    [1, 4],
    [4, 1],
  ])
    dirs.push(...directions([ratio]).slice(6));
  const [origin, limit] = bounds,
    localSize = limit.map((v, a) => v - origin[a]) as Vec3;
  const data = new Uint8Array(localSize.reduce((a, b) => a * b, 1));
  for (const p of cells) data[gridIndex(localSize, p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])] = 1;
  const localPlanes = fitPoly(
    centres.map((p) => p.map((v, a) => v - origin[a]) as Vec3),
    dirs,
    3,
  );
  const tuned = options.tunePlanes ? tuneOffsets(localPlanes, { size: localSize, data }) : localPlanes;
  const planes = compactPlanes(tuned.map((p) => [p[0], p[1], p[2], p[3] + dot(p, origin)]));
  let primitive: Primitive = { type: 'poly', planes },
    kind = semantic(bounds, whole, options);
  let score = classifyError(cells, bounds, primitive);
  if (options.category === 'tube') {
    primitive = cylinderProposal(cells, 'x');
    score = classifyError(cells, bounds, primitive);
  }
  const [lo, hi] = bounds;
  for (const axis of ['x', 'y', 'z'] as const) {
    const a = 'xyz'.indexOf(axis),
      uv = [0, 1, 2].filter((k) => k !== a),
      spans = uv.map((k) => hi[k] - lo[k]);
    if (Math.min(...spans) < 2 || Math.max(...spans) / Math.min(...spans) > 1.4) continue;
    if (options.category === 'tube' && axis !== 'x') continue;
    if (options.category !== 'tube' && axis === 'x') continue;
    const cyl = cylinderProposal(cells, axis),
      cs = classifyError(cells, bounds, cyl);
    // Do not turn arbitrary convex hulls into drums; require a disk-like region and good occupancy evidence.
    if (
      (options.category === 'tube' || hi[a] - lo[a] <= Math.min(...spans) * 0.8) &&
      cs.error <= Math.max(cells.length * 0.12, score.error * 1.1) &&
      cs.support >= 0.85
    ) {
      primitive = cyl;
      score = cs;
      if (options.category !== 'tube')
        kind = axis === 'y' ? (lo[2] < whole[0][2] + (whole[1][2] - whole[0][2]) * 0.4 ? 'wheel' : 'drum') : 'turret';
      break;
    }
  }
  primitive = {
    ...primitive,
    name: `${kind}_${id}`,
    chamfer: options.chamfer ?? (options.category === 'tube' ? 0.025 : 0.08),
    mirror: false,
  };
  return { id, kind, bounds, cells, primitive, ...score };
}

/** Broad BSP regions, never a voxel cover: children need a substantial span and support. */
export function splitRegion(
  region: RegionProposal,
  whole: [Vec3, Vec3],
  options: ProposalOptions,
): RegionProposal[] | undefined {
  const [lo, hi] = region.bounds,
    minSpan = options.minSpan ?? 3,
    minCells = options.minCells ?? 6;
  let best: RegionProposal[] | undefined,
    bestError = region.error - Math.max(2, region.cells.length * 0.008);
  for (let a = 0; a < 3; a++) {
    if (options.category === 'tube' && a !== 0) continue;
    // At most five broad cuts per axis; no row-by-row candidate cover.
    const cuts = new Set([0.25, 0.4, 0.5, 0.6, 0.75].map((t) => Math.round(lo[a] + (hi[a] - lo[a]) * t)));
    for (const cut of cuts) {
      if (cut - lo[a] < minSpan || hi[a] - cut < minSpan) continue;
      const left = region.cells.filter((p) => p[a] < cut),
        right = region.cells.filter((p) => p[a] >= cut);
      if (left.length < minCells || right.length < minCells) continue;
      const parts = [
        fitRegion(left, region.id + 'a', whole, options),
        fitRegion(right, region.id + 'b', whole, options),
      ];
      if (options.category === 'tube' && region.primitive.type === 'cyl') {
        for (const child of parts)
          if (child.primitive.type === 'cyl') {
            child.primitive.c = [...region.primitive.c];
            Object.assign(child, classifyError(child.cells, child.bounds, child.primitive));
          }
      }
      if (parts.some((p) => p.bounds[1].map((v, k) => v - p.bounds[0][k]).sort((a, b) => b - a)[0] < minSpan)) continue;
      const error = parts.reduce((s, p) => s + p.error, 0);
      if (error < bestError) {
        bestError = error;
        best = parts;
      }
    }
  }
  return best;
}
function tubeLanes(cells: Vec3[]): Vec3[][] {
  // Transverse connected components allow several parallel barrels; no joining across empty lanes.
  const keys = new Set(cells.map((p) => `${p[1]},${p[2]}`)),
    groups: Vec3[][] = [];
  while (keys.size) {
    const first = keys.values().next().value!,
      queue = [first],
      component = new Set([first]);
    keys.delete(first);
    for (let i = 0; i < queue.length; i++) {
      const [y, z] = queue[i].split(',').map(Number);
      for (const [dy, dz] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const k = `${y + dy},${z + dz}`;
        if (keys.delete(k)) {
          component.add(k);
          queue.push(k);
        }
      }
    }
    groups.push(cells.filter((p) => component.has(`${p[1]},${p[2]}`)));
  }
  return groups;
}
export function proposeHardSurface(grid: SolidGrid, options: ProposalOptions): RegionModel {
  if (!Number.isInteger(options.budget) || options.budget < 1) throw new Error('Primitive budget must be positive');
  const cells = occupiedCells(grid),
    whole = cellBounds(cells),
    symmetry = detectMirror(grid);
  const groups = options.category === 'tube' ? tubeLanes(cells) : [cells];
  if (groups.length > options.budget) throw new Error('Disconnected tube lanes exceed the primitive budget');
  const regions = groups.map((g, i) => fitRegion(g, String(i), whole, options));
  // Proposal stage is deliberately small. The acceptance loop owns further complexity growth.
  while (
    regions.length <
    Math.min(options.budget, options.category === 'chassis' ? 12 : options.category === 'crown' ? 8 : options.budget)
  ) {
    const ranked = regions.map((r, i) => ({ r, i })).sort((a, b) => b.r.error - a.r.error);
    let changed = false;
    for (const { r, i } of ranked) {
      const children = splitRegion(r, whole, options);
      if (children) {
        regions.splice(i, 1, ...children);
        changed = true;
        break;
      }
    }
    if (!changed) break;
  }
  return { regions, symmetry, options, part: { primitives: regions.map((r) => r.primitive) } };
}
