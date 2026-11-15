import { directions, fitPoly, tuneOffsets } from './hardSurfaceFit';
import { dot, polytopeFaces, type PartSpec, type Plane, type Primitive } from './hardSurfaceMesh';
import { forCells, gridCell, gridIndex, validateGrid, type Axis, type SolidGrid, type Vec3 } from './hardSurfaceSolid';
import { patchRoundFeature } from './hardSurfaceRoundPatch';
import { buildPart } from './hardSurfaceMesh';
import { tubeLanes, tubeSegments } from './hardSurfaceTube';
import {
  fitDome,
  fitEllipsoidDome,
  fitTaperedTube,
  fitCapsule,
  fitFillet,
  detectRoundFeatures,
} from './hardSurfaceRoundFit';

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
  tubeAxis?: Axis;
  primitivesPerTube?: number;
  minSlopeSteps?: number;
  taperedTubes?: boolean;
  roundPipes?: boolean;
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

/** Pair only exactly reflected source regions. Asymmetric details remain independent. */
export function mirrorRegionPairs(regions: readonly RegionProposal[], symmetry: SymmetryEvidence): Map<number, number> {
  const axis = 'xyz'.indexOf(symmetry.axis),
    out = new Map<number, number>();
  const reflected = (p: Vec3) => p.map((v, a) => (a === axis ? 2 * symmetry.at - v - 1 : v)).join(',');
  for (let i = 0; i < regions.length; i++) {
    if (out.has(i)) continue;
    const r = regions[i],
      cells = new Set(r.cells.map(reflected));
    for (let j = i + 1; j < regions.length; j++) {
      if (out.has(j) || regions[j].cells.length !== cells.size) continue;
      if (regions[j].cells.every((p) => cells.has(p.join(',')))) {
        out.set(i, j);
        out.set(j, i);
        break;
      }
    }
  }
  return out;
}
export function reflectPrimitive(primitive: Primitive, symmetry: SymmetryEvidence): Primitive {
  const q = structuredClone(primitive),
    a = 'xyz'.indexOf(symmetry.axis);
  const reflectPlanes = (planes: Plane[]) =>
    planes.map((p) => {
      const n = [...p] as Plane;
      n[a] = -n[a];
      n[3] -= 2 * symmetry.at * p[a];
      return n;
    });
  if (q.type === 'poly') q.planes = reflectPlanes(q.planes);
  else {
    if (q.cuts) q.cuts = reflectPlanes(q.cuts);
    if (q.type === 'box') [q.min[a], q.max[a]] = [2 * symmetry.at - q.max[a], 2 * symmetry.at - q.min[a]];
    else if ('xyz'.indexOf(q.axis) === a) q.range = [2 * symmetry.at - q.range[1], 2 * symmetry.at - q.range[0]];
    else {
      const k = [0, 1, 2].filter((i) => i !== 'xyz'.indexOf(q.axis)).indexOf(a);
      q.c[k] = 2 * symmetry.at - q.c[k];
    }
  }
  if (q.round) {
    if ('centre' in q.round) q.round.centre[a] = 2 * symmetry.at - q.round.centre[a];
    if ('start' in q.round) {
      q.round.start[a] = 2 * symmetry.at - q.round.start[a];
      q.round.end[a] = 2 * symmetry.at - q.round.end[a];
    }
    if ('axis' in q.round) q.round.axis[a] *= -1;
  }
  return q;
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
  // A short asymmetric base must not displace the axis of the much longer tube.
  // Give every axial slice one vote, rather than weighting its cells or the global bounding box.
  const slices = new Map<number, Vec3[]>();
  for (const p of cells) {
    const row = slices.get(p[a]);
    if (row) row.push(p);
    else slices.set(p[a], [p]);
  }
  const centres = [...slices.values()].map(cellBounds).map(([l, h]) => [(l[u] + h[u]) / 2, (l[v] + h[v]) / 2]);
  const median = (values: number[]) => {
    values.sort((x, y) => x - y);
    return (values[Math.floor((values.length - 1) / 2)] + values[Math.ceil((values.length - 1) / 2)]) / 2;
  };
  const c: [number, number] = [median(centres.map((p) => p[0])), median(centres.map((p) => p[1]))];
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
    options.minSlopeSteps ?? 3,
  );
  const tuned = options.tunePlanes ? tuneOffsets(localPlanes, { size: localSize, data }) : localPlanes;
  const planes = compactPlanes(tuned.map((p) => [p[0], p[1], p[2], p[3] + dot(p, origin)]));
  let primitive: Primitive = { type: 'poly', planes },
    kind = semantic(bounds, whole, options);
  let score = classifyError(cells, bounds, primitive);
  if (options.category === 'tube') {
    primitive = cylinderProposal(cells, options.tubeAxis ?? 'x');
    score = classifyError(cells, bounds, primitive);
  }
  const [lo, hi] = bounds;
  for (const axis of ['x', 'y', 'z'] as const) {
    const a = 'xyz'.indexOf(axis),
      uv = [0, 1, 2].filter((k) => k !== a),
      spans = uv.map((k) => hi[k] - lo[k]);
    if (Math.min(...spans) < 2 || Math.max(...spans) / Math.min(...spans) > 1.4) continue;
    if (options.category === 'tube' && axis !== (options.tubeAxis ?? 'x')) continue;
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
  if (options.category !== 'tube') {
    for (const q of [fitCapsule(cells), fitFillet(cells)]) {
      if (!q) continue;
      const qs = classifyError(cells, bounds, q);
      if (qs.support >= 0.9 && qs.error <= Math.max(score.error * 1.05, cells.length * 0.05)) {
        primitive = q;
        score = qs;
        break;
      }
    }
    for (const axis of ['z', 'x', 'y'] as const) {
      const dome = fitDome(cells, axis) ?? fitEllipsoidDome(cells, axis);
      if (!dome) continue;
      const ds = classifyError(cells, bounds, dome);
      if (ds.support >= 0.9 && ds.error <= Math.max(score.error * 1.05, cells.length * 0.05)) {
        primitive = dome;
        score = ds;
        break;
      }
    }
  }
  primitive = {
    ...primitive,
    name: `${kind}_${id}`,
    chamfer: primitive.round ? 0 : (options.chamfer ?? (options.category === 'tube' ? 0.025 : 0.08)),
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
  return splitRegionCandidates(region, whole, options)[0];
}
/** Alternative broad cuts let the surface gate reject an occupancy-best cut without exhausting the region. */
export function splitRegionCandidates(
  region: RegionProposal,
  whole: [Vec3, Vec3],
  options: ProposalOptions,
  allowTies = false,
): RegionProposal[][] {
  if (region.primitive.round) return [];
  const [lo, hi] = region.bounds,
    minSpan = options.minSpan ?? 3,
    minCells = options.minCells ?? 6;
  const candidates: { parts: RegionProposal[]; error: number; axis: number; cut: number }[] = [];
  const bestError = allowTies ? region.error + 1e-7 : region.error - Math.max(2, region.cells.length * 0.008);
  for (let a = 0; a < 3; a++) {
    if (options.category === 'tube' && a !== 'xyz'.indexOf(options.tubeAxis ?? 'x')) continue;
    // At most five broad cuts per axis; no row-by-row candidate cover.
    const cuts = new Set([0.25, 0.4, 0.5, 0.6, 0.75].map((t) => Math.round(lo[a] + (hi[a] - lo[a]) * t)));
    // A neck/sleeve or a change in plate width need not occur at a fixed fraction of its parent.
    // Rank source profile changes and retain a bounded set of broad cuts.
    const counts = new Map<number, number>();
    if (options.category === 'crown') for (const p of region.cells) counts.set(p[a], (counts.get(p[a]) ?? 0) + 1);
    const transitions = [...counts.keys()]
      .filter((x) => x > lo[a])
      .map((x) => ({
        x,
        score:
          Math.abs((counts.get(x) ?? 0) - (counts.get(x - 1) ?? 0)) /
          Math.max(1, counts.get(x) ?? 0, counts.get(x - 1) ?? 0),
      }))
      .filter((p) => p.score >= 0.2)
      .sort((p, q) => q.score - p.score || p.x - q.x)
      .slice(0, 8);
    for (const { x } of transitions) cuts.add(x);
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
        candidates.push({ parts, error, axis: a, cut });
      }
    }
  }
  const sorted = candidates.sort((a, b) => a.error - b.error || a.axis - b.axis || a.cut - b.cut);
  if (!allowTies) return sorted.map((c) => c.parts);
  // Do not spend the entire formal-evaluation allowance on similar cuts along just one axis.
  const axes = new Set<number>(),
    diverse = sorted.filter((c) => {
      if (axes.has(c.axis)) return false;
      axes.add(c.axis);
      return true;
    });
  return [...diverse, ...sorted.filter((c) => !diverse.includes(c))].map((c) => c.parts);
}
export function proposeHardSurface(grid: SolidGrid, options: ProposalOptions): RegionModel {
  if (!Number.isInteger(options.budget) || options.budget < 1) throw new Error('Primitive budget must be positive');
  const cells = occupiedCells(grid),
    whole = cellBounds(cells),
    symmetry = detectMirror(grid);
  const bilateral =
    options.category === 'crown' && symmetry.iou >= 0.9 && options.budget >= 2 && Number.isInteger(symmetry.at);
  const groups =
    options.category === 'tube'
      ? tubeLanes(cells, options.tubeAxis)
      : bilateral
        ? [cells.filter((p) => p[1] < symmetry.at), cells.filter((p) => p[1] >= symmetry.at)]
        : [cells];
  if (groups.length > options.budget) throw new Error('Disconnected tube lanes exceed the primitive budget');
  const regions = groups.flatMap((g, i) => {
    if (options.category !== 'tube' || !options.primitivesPerTube) return [fitRegion(g, String(i), whole, options)];
    const axis = options.tubeAxis ?? 'x',
      cylinder = cylinderProposal(g, axis);
    if (cylinder.type !== 'cyl') throw new Error('Expected shared tube axis');
    return tubeSegments(g, axis, Math.min(options.primitivesPerTube, Math.floor(options.budget / groups.length))).map(
      (s, j) => {
        const region = fitRegion(s, `${i}_${j}`, whole, options);
        if (region.primitive.type === 'cyl') {
          region.primitive.c = [...cylinder.c];
          if (options.taperedTubes) region.primitive = fitTaperedTube(s, axis, cylinder.c) ?? region.primitive;
          Object.assign(region, classifyError(region.cells, region.bounds, region.primitive));
        }
        return region;
      },
    );
  });
  // Proposal stage is deliberately small. The acceptance loop owns further complexity growth.
  while (
    !(options.category === 'tube' && options.primitivesPerTube) &&
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
  for (const [i, j] of mirrorRegionPairs(regions, symmetry))
    if (i < j) {
      const reflected = reflectPrimitive(regions[i].primitive, symmetry);
      reflected.name = regions[j].primitive.name;
      regions[j].primitive = reflected;
      Object.assign(regions[j], classifyError(regions[j].cells, regions[j].bounds, reflected));
    }
  let part: PartSpec = { primitives: regions.map((r) => r.primitive) };
  if (options.roundPipes && options.category !== 'tube') {
    for (const [i, feature] of detectRoundFeatures(grid).entries()) {
      const name = `pipe_${i}`,
        candidate = patchRoundFeature(part, feature, name);
      if (buildPart(candidate).primitives.length > options.budget) continue;
      part = candidate;
      const a = 'xyz'.indexOf(feature.axis);
      for (const j of feature.primitives.keys()) {
        const l = feature.points[Math.min(j, feature.points.length - 2)]?.[a] ?? feature.bounds[0][a];
        const h = feature.points[Math.min(j + 1, feature.points.length - 1)]?.[a] ?? feature.bounds[1][a];
        const cells = feature.cells.filter((p) => p[a] + 0.5 >= Math.min(l, h) && p[a] + 0.5 <= Math.max(l, h));
        regions.push({
          id: name + '_' + j,
          kind: 'drum',
          bounds: feature.bounds,
          cells: cells.length ? cells : feature.cells,
          primitive: part.primitives[part.primitives.length - feature.primitives.length + j],
          error: 0,
          support: feature.evidence.sourceSupport,
        });
      }
    }
  }
  return { regions, symmetry, options, part };
}
