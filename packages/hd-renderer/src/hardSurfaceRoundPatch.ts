import { sub, unit, scale, dot, buildPart, type PartSpec, type Primitive } from './hardSurfaceMesh';
import { simplifyPath, type RoundFeature } from './hardSurfaceRoundFit';
import { sweptCylinders } from './hardSurfaceRound';
import { gridCell, type SolidGrid, type Axis, type Vec3 } from './hardSurfaceSolid';

/** Replace a source-supported region while preserving every other plate and its metadata.
 * Clipping pieces are charged to the same expanded budget as ordinary reconstruction.
 */
export function patchRoundFeature(
  part: PartSpec,
  feature: RoundFeature,
  name: string,
  replacement = feature.primitives,
): PartSpec {
  const out = structuredClone(part),
    names = replacement.map((_, i) => `${name}_${i}`);
  const primitives: Primitive[] = replacement.map((q, i) => ({ ...structuredClone(q), name: names[i], mirror: false }));
  // A replacement is the already supported source region, rather than a new plate subject to historical local cuts.
  for (const cut of out.carve ?? []) cut.keep = [...(cut.keep ?? []), ...names];
  out.carve = [...(out.carve ?? []), { min: [...feature.bounds[0]], max: [...feature.bounds[1]], keep: names }];
  out.primitives.push(...primitives);
  return out;
}

export function withinExpandedBudget(part: PartSpec, budget: number): boolean {
  return buildPart(part).primitives.length <= budget;
}

export interface CoplanarBoundary {
  key: string;
  members: { primitive: number; plane: number; sign: number; length: number }[];
}
/** Opposite inequalities at the same BSP seam share a canonical boundary and move together. */
export function coplanarBoundaries(part: PartSpec): CoplanarBoundary[] {
  const groups = new Map<string, CoplanarBoundary>();
  part.primitives.forEach((q, i) => {
    if (q.type !== 'poly' || q.round) return;
    q.planes.forEach((p, j) => {
      const length = Math.hypot(p[0], p[1], p[2]),
        sign = p.slice(0, 3).find((v) => Math.abs(v) > 1e-8)! < 0 ? -1 : 1;
      const key = p.map((v) => Math.round((v / length) * sign * 1000)).join(',');
      const group = groups.get(key) ?? { key, members: [] };
      group.members.push({ primitive: i, plane: j, sign, length });
      groups.set(key, group);
    });
  });
  return [...groups.values()];
}
export function shiftCoplanarBoundary(part: PartSpec, boundary: CoplanarBoundary, delta: number): PartSpec {
  const out = structuredClone(part);
  for (const r of boundary.members) {
    const q = out.primitives[r.primitive];
    if (q.type !== 'poly' || q.round) throw new Error('Boundary owner changed');
    q.planes[r.plane][3] += delta * r.sign * r.length;
  }
  return out;
}

/** Extend an established upper pipe track through attached cross sections. A flat unexposed deck cannot seed one. */
export function upperPipeContinuation(
  grid: SolidGrid,
  feature: RoundFeature,
  up: Axis = 'z',
): RoundFeature | undefined {
  const a = 'xyz'.indexOf(feature.axis),
    u = 'xyz'.indexOf(up),
    c = [0, 1, 2].find((k) => k !== a && k !== u);
  if (a === u || c === undefined || feature.points.length < 2) return;
  const segments = feature.points
    .slice(1)
    .map((end, i) => ({ start: feature.points[i], end, length: Math.abs(end[a] - feature.points[i][a]) }))
    .sort((p, q) => q.length - p.length);
  const longest = segments[0];
  if (longest.length < 6 || Math.abs(longest.end[c] - longest.start[c]) > 0.75) return;
  const crossCentre = (longest.start[c] + longest.end[c]) / 2,
    seed = Math.floor((longest.start[a] + longest.end[a]) / 2),
    radius = feature.radius;
  const height = (slice: number, cross: number) => {
    const p: Vec3 = [0, 0, 0];
    p[a] = slice;
    p[c] = Math.floor(cross);
    for (let z = grid.size[u] - 1; z >= 0; z--) {
      p[u] = z;
      if (gridCell(grid, p)) return z + 1;
    }
    return -1;
  };
  const sample = (slice: number): Vec3 | undefined => {
    const h = height(slice, crossCentre);
    if (h < 0) return;
    const p: Vec3 = [0, 0, 0];
    p[a] = slice + 0.5;
    p[c] = crossCentre;
    p[u] = h - radius;
    let support = 0;
    for (let i = 0; i < 12; i++) {
      const angle = (2 * Math.PI * i) / 12,
        q = [...p] as Vec3;
      q[c] += radius * 0.75 * Math.cos(angle);
      q[u] += radius * 0.75 * Math.sin(angle);
      if (gridCell(grid, q.map(Math.floor) as Vec3)) support++;
    }
    if (support < 11) return;
    return p;
  };
  const centre = sample(seed);
  if (!centre) return;
  const walk = (sign: number) => {
    const rows: Vec3[] = [];
    let last = centre,
      hidden = 0;
    for (let slice = seed + sign; slice >= 0 && slice < grid.size[a]; slice += sign) {
      const p = sample(slice);
      if (!p || Math.abs(p[u] - last[u]) > 1.25) break;
      const side = Math.min(height(slice, crossCentre - radius - 1.5), height(slice, crossCentre + radius + 1.5));
      const exposed = height(slice, crossCentre) > side + 0.5;
      hidden = exposed ? 0 : hidden + 1;
      if (hidden > 3) break;
      rows.push(p);
      last = p;
    }
    if (hidden) rows.splice(Math.max(0, rows.length - hidden));
    return rows;
  };
  const path = [...walk(-1).reverse(), centre, ...walk(1)];
  if (path.length < Math.max(8, feature.evidence.slices)) return;
  const smoothed = path.map(
    (p, i) =>
      p.map((v, k) =>
        k !== u
          ? v
          : path.slice(Math.max(0, i - 2), Math.min(path.length, i + 3)).reduce((s, q) => s + q[u], 0) /
            path.slice(Math.max(0, i - 2), Math.min(path.length, i + 3)).length,
      ) as Vec3,
  );
  smoothed[0][a] -= 0.5;
  smoothed[smoothed.length - 1][a] += 0.5;
  const points = simplifyPath(smoothed, 0.2);
  if (points.length > 9) return;
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of points)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], p[k] - (k === a ? 0 : radius));
      hi[k] = Math.max(hi[k], p[k] + (k === a ? 0 : radius));
    }
  const cells: Vec3[] = [];
  for (let x = Math.max(0, Math.floor(lo[0])); x < Math.min(grid.size[0], Math.ceil(hi[0])); x++)
    for (let y = Math.max(0, Math.floor(lo[1])); y < Math.min(grid.size[1], Math.ceil(hi[1])); y++)
      for (let z = Math.max(0, Math.floor(lo[2])); z < Math.min(grid.size[2], Math.ceil(hi[2])); z++)
        if (gridCell(grid, [x, y, z])) cells.push([x, y, z]);
  return {
    ...feature,
    points,
    cells,
    bounds: [lo, hi],
    primitives: sweptCylinders(points, radius),
    evidence: { ...feature.evidence, slices: path.length },
  };
}

/** Only replace material above each centreline plane. The existing parent below it remains flush and whole. */
export function patchUpperPipe(
  part: PartSpec,
  feature: RoundFeature,
  name: string,
  up: Axis = 'z',
  floorOffset = 0,
  floorSlope = 0,
): PartSpec {
  const out = structuredClone(part),
    names = feature.primitives.map((_, i) => `${name}_${i}`),
    a = 'xyz'.indexOf(feature.axis),
    u = 'xyz'.indexOf(up);
  if (a === u) throw new Error('Upper pipe requires a transverse up axis');
  for (const cut of out.carve ?? []) cut.keep = [...(cut.keep ?? []), ...names];
  out.primitives.push(...feature.primitives.map((q, i) => ({ ...structuredClone(q), name: names[i], mirror: false })));
  out.carve ??= [];
  feature.primitives.forEach((_, i) => {
    const start = feature.points[i],
      end = feature.points[i + 1],
      direction = unit(sub(end, start)),
      upVector: Vec3 = [0, 0, 0];
    upVector[u] = 1;
    const crossAxis = [0, 1, 2].find((k) => k !== a && k !== u)!;
    upVector[crossAxis] = -floorSlope;
    const normal = unit(sub(upVector, scale(direction, dot(upVector, direction)))),
      lo = [...feature.bounds[0]] as Vec3,
      hi = [...feature.bounds[1]] as Vec3;
    lo[a] = Math.min(start[a], end[a]);
    hi[a] = Math.max(start[a], end[a]);
    out.carve!.push({
      min: lo,
      max: hi,
      cuts: [[...scale(normal, -1), -dot(normal, start) - floorOffset * normal[u]]],
      keep: names,
    });
  });
  return out;
}
