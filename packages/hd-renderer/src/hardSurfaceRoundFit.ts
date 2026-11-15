/** Source-only round evidence. No model names, palette cues, or caller-supplied coordinates. */
import { gridCell, validateGrid, type SolidGrid, type Vec3, type Axis } from './hardSurfaceSolid';
import {
  ellipsoidCap,
  torusSegments,
  sphericalCap,
  sweptCylinders,
  taperedCylinder,
  capsule as makeCapsule,
  edgeFillet as makeFillet,
} from './hardSurfaceRound';
import {
  sub,
  dot,
  cross,
  add,
  scale,
  unit,
  expand as expandRound,
  insidePrimitive as insideRound,
  type Primitive,
} from './hardSurfaceMesh';

export interface RoundFeature {
  axis: Axis;
  points: Vec3[];
  radius: number;
  cells: Vec3[];
  bounds: [Vec3, Vec3];
  primitives: Primitive[];
  evidence: {
    slices: number;
    exposedArc: number;
    radialResidual: number;
    sourceSupport: number;
    ambiguousSmallCrossSection: boolean;
  };
}
interface Disc {
  p: Vec3;
  r: number;
  error: number;
  arc: number;
  support: number;
}
const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];

/** Cross-section ray curvature separates an exposed circular lobe from a flat plate corner.
 * One attached quadrant may be hidden by a parent solid; the other three must agree on radius.
 */
export function detectRoundFeatures(grid: SolidGrid, maxFeatures = 32): RoundFeature[] {
  validateGrid(grid);
  const features: RoundFeature[] = [];
  const angles = Array.from({ length: 16 }, (_, i) => [Math.cos((i * Math.PI) / 8), Math.sin((i * Math.PI) / 8)]);
  for (let a = 0; a < 3; a++) {
    const uv = [0, 1, 2].filter((k) => k !== a),
      tracks: Disc[][] = [];
    let active: Disc[][] = [];
    for (let slice = 0; slice < grid.size[a]; slice++) {
      const candidates: Disc[] = [];
      const at = (u: number, v: number) => {
        const p: Vec3 = [0, 0, 0];
        p[a] = slice;
        p[uv[0]] = Math.floor(u);
        p[uv[1]] = Math.floor(v);
        return !!gridCell(grid, p);
      };
      for (let u = 0.5; u < grid.size[uv[0]]; u += 0.5)
        for (let v = 0.5; v < grid.size[uv[1]]; v += 0.5) {
          if (!at(u, v)) continue;
          // Cheap rejection: a circular lobe must expose both directions on at least one transverse axis.
          const extent = angles
            .filter((_, i) => i % 4 === 0)
            .map(([du, dv]) => {
              let r = 0.25;
              while (r <= 6 && at(u + du * r, v + dv * r)) r += 0.25;
              return r;
            });
          if (
            Math.min(extent[0], extent[2]) < 0.45 ||
            Math.min(extent[1], extent[3]) < 0.45 ||
            (Math.max(extent[0], extent[2]) > 6 && Math.max(extent[1], extent[3]) > 6)
          )
            continue;
          const rays = angles.map(([du, dv]) => {
            let r = 0.25;
            while (r <= 6 && at(u + du * r, v + dv * r)) r += 0.25;
            return r - 0.125;
          });
          const continuing = active.some((t) => {
            const p = t[t.length - 1];
            return Math.hypot(p.p[uv[0]] - u, p.p[uv[1]] - v) <= 1.25;
          });
          let radius = median([...rays]);
          let agreeing = rays.filter((r) => Math.abs(r - radius) <= Math.max(0.55, radius * 0.22));
          if (continuing && agreeing.length < 10) {
            radius = median([...rays].sort((a, b) => a - b).slice(0, 8));
            agreeing = rays.filter((r) => Math.abs(r - radius) <= Math.max(0.55, radius * 0.22));
          }
          if (radius < 0.45 || radius > 5.5) continue;
          if (agreeing.length < (continuing ? 8 : radius < 0.85 ? 11 : 12)) continue;
          const error = agreeing.reduce((s, r) => s + Math.abs(r - radius), 0) / agreeing.length;
          let inside = 0,
            hit = 0;
          for (let du = -radius; du <= radius; du += 0.5)
            for (let dv = -radius; dv <= radius; dv += 0.5)
              if (Math.hypot(du, dv) < radius - 0.15) {
                inside++;
                if (at(u + du, v + dv)) hit++;
              }
          const support = hit / Math.max(1, inside);
          if (support < 0.94) continue;
          const p: Vec3 = [0, 0, 0];
          p[a] = slice + 0.5;
          p[uv[0]] = u;
          p[uv[1]] = v;
          candidates.push({ p, r: radius, error, arc: agreeing.length / 16, support });
        }
      const continuity = (d: Disc) =>
        active.some((t) => {
          const p = t[t.length - 1];
          return Math.abs(p.r - d.r) <= 0.6 && Math.hypot(p.p[uv[0]] - d.p[uv[0]], p.p[uv[1]] - d.p[uv[1]]) <= 0.75;
        })
          ? 0.4
          : 0;
      candidates.sort((p, q) => q.arc - p.arc + p.error - q.error + continuity(q) - continuity(p) || q.r - p.r);
      const discs: Disc[] = [];
      for (const q of candidates)
        if (!discs.some((p) => Math.hypot(p.p[uv[0]] - q.p[uv[0]], p.p[uv[1]] - q.p[uv[1]]) < p.r + q.r)) discs.push(q);
      const next: Disc[][] = [],
        used = new Set<Disc[]>();
      for (const disc of discs) {
        const matches = active
          .filter((t) => !used.has(t))
          .map((t) => ({ t, p: t[t.length - 1] }))
          .filter(
            ({ p }) =>
              Math.abs(p.r - disc.r) <= 0.6 &&
              Math.hypot(p.p[uv[0]] - disc.p[uv[0]], p.p[uv[1]] - disc.p[uv[1]]) <= 1.25,
          )
          .sort((p, q) => Math.hypot(...sub(p.p.p, disc.p)) - Math.hypot(...sub(q.p.p, disc.p)));
        const track = matches[0]?.t;
        if (track) {
          track.push(disc);
          used.add(track);
          next.push(track);
        } else next.push([disc]);
      }
      tracks.push(...active.filter((t) => !used.has(t)));
      active = next;
    }
    tracks.push(...active);
    const stableTracks: Disc[][] = [];
    for (const track of tracks) {
      let run: Disc[] = [],
        lo = Infinity,
        hi = -Infinity;
      for (const disc of track) {
        const nextLo = Math.min(lo, disc.r),
          nextHi = Math.max(hi, disc.r);
        if (nextHi - nextLo > 1) {
          if (run.length) stableTracks.push(run);
          run = [];
          lo = Infinity;
          hi = -Infinity;
        }
        run.push(disc);
        lo = Math.min(lo, disc.r);
        hi = Math.max(hi, disc.r);
      }
      if (run.length) stableTracks.push(run);
    }
    for (const track of stableTracks) {
      const radius = median(track.map((p) => p.r));
      if (
        track.filter((p) => p.arc >= (p.r < 0.85 ? 0.6875 : 0.75)).length < 6 ||
        track.reduce((s, p) => s + p.arc, 0) / track.length < 0.6
      )
        continue;
      if (track.length < Math.max(6, 2.5 * radius) || Math.max(...track.map((p) => Math.abs(p.r - radius))) > 0.65)
        continue;
      // Remove lattice jitter, then retain only geometrically necessary bend vertices.
      const points = track.map(
        (d, i) =>
          d.p.map((v, k) =>
            k === a ? v : median(track.slice(Math.max(0, i - 1), Math.min(track.length, i + 2)).map((s) => s.p[k])),
          ) as Vec3,
      );
      points[0][a] -= 0.5;
      points[points.length - 1][a] += 0.5;
      const simplified = simplifyPath(points, 0.55);
      if (simplified.length > 9) continue;
      const lo: Vec3 = [Infinity, Infinity, Infinity],
        hi: Vec3 = [-Infinity, -Infinity, -Infinity];
      for (const d of points)
        for (let k = 0; k < 3; k++) {
          lo[k] = Math.min(lo[k], Math.floor(d[k] - (k === a ? 0 : radius)));
          hi[k] = Math.max(hi[k], Math.ceil(d[k] + (k === a ? 0 : radius)));
        }
      const cells: Vec3[] = [];
      for (let x = Math.max(0, lo[0]); x < Math.min(grid.size[0], hi[0]); x++)
        for (let y = Math.max(0, lo[1]); y < Math.min(grid.size[1], hi[1]); y++)
          for (let z = Math.max(0, lo[2]); z < Math.min(grid.size[2], hi[2]); z++) {
            const p: Vec3 = [x, y, z];
            if (gridCell(grid, p)) cells.push(p);
          }
      if (!cells.length) continue;
      features.push({
        axis: 'xyz'[a] as Axis,
        points: simplified,
        radius,
        cells,
        bounds: [lo, hi],
        primitives: fitPipeBend(simplified, radius, grid) ?? sweptCylinders(simplified, radius),
        evidence: {
          slices: track.length,
          exposedArc: track.reduce((s, p) => s + p.arc, 0) / track.length,
          radialResidual: track.reduce((s, p) => s + p.error, 0) / track.length,
          sourceSupport: track.reduce((s, p) => s + p.support, 0) / track.length,
          ambiguousSmallCrossSection: radius < 1.5,
        },
      });
    }
  }
  features.sort((p, q) => q.cells.length - p.cells.length);
  const selected: RoundFeature[] = [],
    occupied = new Set<string>();
  for (const f of features) {
    if (f.cells.filter((p) => occupied.has(p.join(','))).length > f.cells.length * 0.1) continue;
    selected.push(f);
    f.cells.forEach((p) => occupied.add(p.join(',')));
    if (selected.length >= maxFeatures) break;
  }
  return selected;
}

export function simplifyPath(points: readonly Vec3[], tolerance: number): Vec3[] {
  if (points.length <= 2) return points.map((p) => [...p]);
  const a = points[0],
    b = points[points.length - 1],
    axis = sub(b, a),
    length2 = dot(axis, axis);
  let worst = -1,
    distance = tolerance;
  for (let i = 1; i < points.length - 1; i++) {
    const t = length2 ? Math.max(0, Math.min(1, dot(sub(points[i], a), axis) / length2)) : 0;
    const d = Math.hypot(...points[i].map((v, k) => v - a[k] - axis[k] * t));
    if (d > distance) {
      distance = d;
      worst = i;
    }
  }
  if (worst < 0) return [[...a], [...b]];
  return [
    ...simplifyPath(points.slice(0, worst + 1), tolerance).slice(0, -1),
    ...simplifyPath(points.slice(worst), tolerance),
  ];
}

/** Source profile regression: accept a taper only when slice radii change coherently, not at a sleeve step. */
export function fitTaperedTube(cells: readonly Vec3[], axis: Axis, centre: [number, number]): Primitive | undefined {
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    slices = new Map<number, Vec3[]>();
  for (const p of cells) {
    const row = slices.get(p[a]);
    if (row) row.push(p);
    else slices.set(p[a], [p]);
  }
  const xs = [...slices.keys()].sort((x, y) => x - y);
  if (xs.length < 4) return;
  const rs = xs.map((x) => Math.sqrt(slices.get(x)!.length / Math.PI));
  const meanX = xs.reduce((s, x) => s + x, 0) / xs.length,
    meanR = rs.reduce((s, r) => s + r, 0) / rs.length;
  const slope =
    xs.reduce((s, x, i) => s + (x - meanX) * (rs[i] - meanR), 0) / xs.reduce((s, x) => s + (x - meanX) ** 2, 0);
  if (Math.abs(slope) * (xs.length - 1) < 0.45) return;
  const residual = rs.reduce((s, r, i) => s + (r - meanR - slope * (xs[i] - meanX)) ** 2, 0) / rs.length;
  if (Math.sqrt(residual) > 0.25 || Math.max(...rs.slice(1).map((r, i) => Math.abs(r - rs[i]))) > 0.65) return;
  const start: Vec3 = [0, 0, 0],
    end: Vec3 = [0, 0, 0];
  start[a] = xs[0];
  end[a] = xs[xs.length - 1] + 1;
  start[uv[0]] = end[uv[0]] = centre[0];
  start[uv[1]] = end[uv[1]] = centre[1];
  const radii: [number, number] = [meanR + slope * (start[a] - meanX - 0.5), meanR + slope * (end[a] - meanX - 0.5)];
  if (Math.min(...radii) < 0.2) return;
  return taperedCylinder(start, end, radii);
}

/** Fit a spherical cap to a region's circular axial sections; all candidate parameters come from the solid. */
export function fitDome(cells: readonly Vec3[], axis: Axis): Primitive | undefined {
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    xs = [...new Set(cells.map((p) => p[a]))].sort((x, y) => x - y);
  if (xs.length < 3) return;
  const sections = xs.map((x) => cells.filter((p) => p[a] === x));
  const centres = sections.map((s) =>
    uv.map((u) => (Math.min(...s.map((p) => p[u])) + Math.max(...s.map((p) => p[u])) + 1) / 2),
  );
  const c = uv.map((_, k) => median(centres.map((p) => p[k])));
  if (centres.some((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) > 0.5)) return;
  const ys = sections.map((s, i) => s.length / Math.PI + (xs[i] + 0.5) ** 2),
    meanX = xs.reduce((s, x) => s + x + 0.5, 0) / xs.length,
    meanY = ys.reduce((s, y) => s + y, 0) / ys.length;
  const slope =
    xs.reduce((s, x, i) => s + (x + 0.5 - meanX) * (ys[i] - meanY), 0) /
    xs.reduce((s, x) => s + (x + 0.5 - meanX) ** 2, 0);
  const z = slope / 2,
    radius = Math.sqrt(meanY - slope * meanX + z * z),
    top = xs[xs.length - 1] + 1,
    height = top - xs[0];
  if (!Number.isFinite(radius) || radius < 2 || height > radius * 1.6 || Math.abs(top - z - radius) > 1) return;
  if (
    Math.sqrt(ys.reduce((s, y, i) => s + (y - meanY - slope * (xs[i] + 0.5 - meanX)) ** 2, 0) / ys.length) >
    radius * 0.65
  )
    return;
  const centre: Vec3 = [0, 0, 0],
    direction: Vec3 = [0, 0, 0];
  centre[a] = z;
  centre[uv[0]] = c[0];
  centre[uv[1]] = c[1];
  direction[a] = 1;
  return sphericalCap(centre, radius, direction, Math.min(2 * radius, centre[a] + radius - xs[0]));
}

/** Constant-radius rounded ends require a stable circular middle and two shrinking terminal profiles. */
export function fitCapsule(cells: readonly Vec3[]): Primitive | undefined {
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of cells)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], p[a]);
      hi[a] = Math.max(hi[a], p[a] + 1);
    }
  const spans = hi.map((v, a) => v - lo[a]),
    a = spans.indexOf(Math.max(...spans)),
    uv = [0, 1, 2].filter((k) => k !== a);
  const diameter = (spans[uv[0]] + spans[uv[1]]) / 2;
  if (
    diameter < 4 ||
    Math.max(spans[uv[0]], spans[uv[1]]) / Math.min(spans[uv[0]], spans[uv[1]]) > 1.15 ||
    spans[a] < diameter * 1.6
  )
    return;
  const profile = Array.from({ length: spans[a] }, (_, i) => cells.filter((p) => p[a] === lo[a] + i).length);
  const max = Math.max(...profile),
    edge = Math.max(profile[0], profile[profile.length - 1]);
  if (edge > max * 0.75 || profile.filter((n) => n >= max * 0.9).length < 3) return;
  const r = diameter / 2,
    start = lo.map((v, k) => (k === a ? v + r : (v + hi[k]) / 2)) as Vec3,
    end = [...start] as Vec3;
  end[a] = hi[a] - r;
  const q = makeCapsule(start, end, r);
  return supportedRoundCandidate(cells, q, 0.86) ? q : undefined;
}

/** An extruded circular quadrant is a fillet only when the opposite corner is actually absent. */
export function fitFillet(cells: readonly Vec3[]): Primitive | undefined {
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of cells)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], p[a]);
      hi[a] = Math.max(hi[a], p[a] + 1);
    }
  const spans = hi.map((v, a) => v - lo[a]),
    a = spans.indexOf(Math.max(...spans)),
    uv = [0, 1, 2].filter((k) => k !== a);
  if (
    Math.min(...uv.map((k) => spans[k])) < 3 ||
    spans[a] < Math.max(...uv.map((k) => spans[k])) * 1.5 ||
    Math.max(...uv.map((k) => spans[k])) / Math.min(...uv.map((k) => spans[k])) > 1.15
  )
    return;
  const volume = spans.reduce((s, n) => s * n, 1),
    fill = cells.length / volume;
  if (fill < 0.68 || fill > 0.9) return;
  const profiles = Array.from({ length: spans[a] }, (_, i) => cells.filter((p) => p[a] === lo[a] + i).length);
  if (Math.max(...profiles) - Math.min(...profiles) > Math.max(...profiles) * 0.15) return;
  const r = (spans[uv[0]] + spans[uv[1]]) / 2;
  for (const su of [-1, 1])
    for (const sv of [-1, 1]) {
      const start: Vec3 = [0, 0, 0],
        end: Vec3 = [0, 0, 0];
      start[a] = lo[a];
      end[a] = hi[a];
      start[uv[0]] = end[uv[0]] = su === 1 ? lo[uv[0]] : hi[uv[0]];
      start[uv[1]] = end[uv[1]] = sv === 1 ? lo[uv[1]] : hi[uv[1]];
      const clips = uv.map((k, i) => {
        const sign = i ? sv : su,
          n: Vec3 = [0, 0, 0];
        n[k] = -sign;
        return [...n, -sign * start[k]] as import('./hardSurfaceMesh').Plane;
      });
      const q = makeFillet(start, end, r, clips);
      if (supportedRoundCandidate(cells, q, 0.92)) return q;
    }
}

/** Check source centres plus a one-cell halo, so a curved hull cannot hide added volume outside its input bounds. */
function supportedRoundCandidate(cells: readonly Vec3[], q: Primitive, floor: number): boolean {
  const occupied = new Set(cells.map((p) => p.join(','))),
    prim = expandRound(q),
    lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const p of cells)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], p[a] - 1);
      hi[a] = Math.max(hi[a], p[a] + 2);
    }
  let intersection = 0,
    union = 0;
  for (let x = lo[0]; x < hi[0]; x++)
    for (let y = lo[1]; y < hi[1]; y++)
      for (let z = lo[2]; z < hi[2]; z++) {
        const truth = occupied.has([x, y, z].join(',')),
          prediction = insideRound([x + 0.5, y + 0.5, z + 0.5], prim);
        if (truth || prediction) union++;
        if (truth && prediction) intersection++;
      }
  return intersection / union >= floor;
}

/** A circular centreline is used only when the intervening arc is also occupied in the source. */
export function fitPipeBend(points: readonly Vec3[], minor: number, grid: SolidGrid): Primitive[] | undefined {
  if (points.length < 3) return;
  const start = points[0],
    middle = points[Math.floor(points.length / 2)],
    end = points[points.length - 1];
  const u = sub(middle, start),
    v = sub(end, start),
    normal = cross(u, v),
    n2 = dot(normal, normal);
  if (n2 < 1e-5) return;
  const centre = add(
    start,
    scale(add(scale(cross(v, normal), dot(u, u)), scale(cross(normal, u), dot(v, v))), 1 / (2 * n2)),
  );
  const radius = Math.hypot(...sub(start, centre));
  if (radius < minor * 2 || radius > 200) return;
  if (points.some((p) => Math.abs(Math.hypot(...sub(p, centre)) - radius) > 0.45)) return;
  let axis = unit(normal),
    radial = unit(sub(start, centre)),
    side = cross(axis, radial);
  let angle = Math.atan2(dot(sub(end, centre), side), dot(sub(end, centre), radial));
  if (angle < 0) {
    axis = scale(axis, -1);
    side = scale(side, -1);
    angle = -angle;
  }
  const midAngle = Math.atan2(dot(sub(middle, centre), side), dot(sub(middle, centre), radial));
  if (angle > Math.PI || midAngle <= 0 || midAngle >= angle) return;
  let hit = 0,
    total = 0;
  for (let i = 0; i <= 16; i++) {
    const t = (angle * i) / 16,
      rad = add(scale(radial, Math.cos(t)), scale(side, Math.sin(t))),
      p = add(centre, scale(rad, radius));
    for (const [du, dv] of [
      [0, 0],
      [0.6, 0],
      [-0.6, 0],
      [0, 0.6],
      [0, -0.6],
    ]) {
      const q = add(p, add(scale(rad, minor * du), scale(axis, minor * dv))).map(Math.floor) as Vec3;
      total++;
      if (gridCell(grid, q)) hit++;
    }
  }
  if (hit / total < 0.9) return;
  return torusSegments(centre, axis, radial, radius, minor, angle);
}

/** Robust slice centres define one shared straight axis, including shallow source-supported elevation. */
export function fittedTubeAxis(
  cells: readonly Vec3[],
  axis: Axis,
): { start: Vec3; end: Vec3; direction: Vec3; residual: number } {
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    slices = new Map<number, Vec3[]>();
  for (const p of cells) {
    const row = slices.get(p[a]);
    if (row) row.push(p);
    else slices.set(p[a], [p]);
  }
  const xs = [...slices.keys()].sort((a, b) => a - b),
    centres = xs.map((x) =>
      uv.map((k) => {
        const row = slices.get(x)!;
        return (Math.min(...row.map((p) => p[k])) + Math.max(...row.map((p) => p[k])) + 1) / 2;
      }),
    );
  const slopes = uv.map((_, k) => {
    const samples: number[] = [];
    for (let i = 0; i < xs.length; i++)
      for (let j = i + 1; j < xs.length; j++)
        if (xs[j] - xs[i] >= (xs.length - 1) / 3) samples.push((centres[j][k] - centres[i][k]) / (xs[j] - xs[i]));
    const slope = samples.length ? median(samples) : 0;
    return Math.abs(slope) < 0.025 ? 0 : slope;
  });
  const intercept = uv.map((_, k) => median(xs.map((x, i) => centres[i][k] - slopes[k] * (x + 0.5))));
  const point = (x: number) => {
    const p: Vec3 = [0, 0, 0];
    p[a] = x;
    uv.forEach((k, i) => (p[k] = intercept[i] + slopes[i] * x));
    return p;
  };
  const start = point(xs[0]),
    end = point(xs[xs.length - 1] + 1);
  const errors = xs
    .map((x, i) => Math.hypot(...uv.map((_, k) => centres[i][k] - intercept[k] - slopes[k] * (x + 0.5))))
    .sort((a, b) => a - b);
  return { start, end, direction: unit(sub(end, start)), residual: errors[Math.floor((errors.length - 1) * 0.9)] };
}

/** Area is a quadratic of axial position on an ellipsoid. Reject flat, unstable, or poorly supported profiles. */
export function fitEllipsoidDome(cells: readonly Vec3[], axis: Axis): Primitive | undefined {
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    xs = [...new Set(cells.map((p) => p[a]))].sort((x, y) => x - y);
  if (xs.length < 5) return;
  const sections = xs.map((x) => cells.filter((p) => p[a] === x)),
    profiles = sections.map((s) => ({
      c: uv.map((k) => (Math.min(...s.map((p) => p[k])) + Math.max(...s.map((p) => p[k])) + 1) / 2),
      spans: uv.map((k) => Math.max(...s.map((p) => p[k])) - Math.min(...s.map((p) => p[k])) + 1),
      area: s.length / Math.PI,
    }));
  const c = uv.map((_, k) => median(profiles.map((p) => p.c[k])));
  if (
    profiles.some((p) => Math.hypot(p.c[0] - c[0], p.c[1] - c[1]) > 0.85) ||
    profiles[profiles.length - 1].area > profiles[0].area * 0.7
  )
    return;
  const origin = xs[0] + 0.5,
    ts = xs.map((x) => x + 0.5 - origin),
    matrix = Array.from({ length: 3 }, (_, i) =>
      Array.from({ length: 4 }, (_, j) =>
        j < 3 ? ts.reduce((s, t) => s + t ** (i + j), 0) : ts.reduce((s, t, k) => s + t ** i * profiles[k].area, 0),
      ),
    );
  for (let i = 0; i < 3; i++) {
    let pivot = i;
    for (let j = i + 1; j < 3; j++) if (Math.abs(matrix[j][i]) > Math.abs(matrix[pivot][i])) pivot = j;
    [matrix[i], matrix[pivot]] = [matrix[pivot], matrix[i]];
    const scale = matrix[i][i];
    if (Math.abs(scale) < 1e-9) return;
    for (let k = i; k < 4; k++) matrix[i][k] /= scale;
    for (let j = 0; j < 3; j++)
      if (j !== i) {
        const factor = matrix[j][i];
        for (let k = i; k < 4; k++) matrix[j][k] -= factor * matrix[i][k];
      }
  }
  const [constant, linear, quadratic] = matrix.map((row) => row[3]);
  if (quadratic >= -0.1) return;
  const axial = -linear / (2 * quadratic),
    peak = constant - (linear * linear) / (4 * quadratic);
  if (peak <= 0) return;
  const rz = Math.sqrt(-peak / quadratic),
    rxy = Math.sqrt(peak),
    top = xs[xs.length - 1] + 1;
  if (Math.abs(origin + axial + rz - top) > 1.25 || rz < 2 || rz > 40 || rxy < 2 || rxy > 40 || xs.length > rz * 1.6)
    return;
  const residual = Math.sqrt(
    ts.reduce((s, t, k) => s + (constant + linear * t + quadratic * t * t - profiles[k].area) ** 2, 0) / ts.length,
  );
  if (residual > rxy * 0.4) return;
  const aspect = median(profiles.map((p) => p.spans[0] / p.spans[1]));
  if (
    aspect < 0.45 ||
    aspect > 2.2 ||
    profiles.some((p) => Math.min(...p.spans) > 4 && Math.abs(p.spans[0] / p.spans[1] - aspect) > 0.35)
  )
    return;
  const centre: Vec3 = [0, 0, 0],
    radii: Vec3 = [0, 0, 0],
    direction: Vec3 = [0, 0, 0];
  centre[a] = origin + axial;
  centre[uv[0]] = c[0];
  centre[uv[1]] = c[1];
  radii[a] = rz;
  radii[uv[0]] = rxy * Math.sqrt(aspect);
  radii[uv[1]] = rxy / Math.sqrt(aspect);
  direction[a] = 1;
  const q = ellipsoidCap(centre, radii, direction, Math.min(2 * rz, centre[a] + rz - xs[0]));
  return supportedRoundCandidate(cells, q, 0.83) ? q : undefined;
}
