/** Quantization-bounded circular upper ridges, detected from the source height field. */
import { gridCell, type SolidGrid, type Vec3, type Axis } from './hardSurfaceSolid';
import { sweptCylinders } from './hardSurfaceRound';
import { simplifyPath, type RoundFeature } from './hardSurfaceRoundFit';
export interface RidgeFeature extends RoundFeature {
  floorOffset: number;
  floorSlope: number;
  maximumHeightError: number;
}
interface Row {
  slice: number;
  centre: number;
  width: number;
  height: number;
  base: number;
  left: number;
  right: number;
  profile: number[];
}
const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];
export function detectRoundRidges(grid: SolidGrid, up: Axis = 'z'): RidgeFeature[] {
  const u = 'xyz'.indexOf(up),
    out: RidgeFeature[] = [];
  for (const a of [0, 1, 2].filter((k) => k !== u)) {
    const c = [0, 1, 2].find((k) => k !== a && k !== u)!,
      tracks: Row[][] = [];
    let active: Row[][] = [];
    for (let slice = 0; slice < grid.size[a]; slice++) {
      const profile = Array.from({ length: grid.size[c] }, (_, cross) => {
          const p: Vec3 = [0, 0, 0];
          p[a] = slice;
          p[c] = cross;
          for (let h = grid.size[u] - 1; h >= 0; h--) {
            p[u] = h;
            if (gridCell(grid, p)) return h + 1;
          }
          return -1;
        }),
        rows: Row[] = [];
      for (let start = 1; start < profile.length - 1;) {
        let end = start + 1;
        while (end < profile.length && profile[end] === profile[start]) end++;
        const width = end - start,
          height = profile[start];
        if (
          width <= 4 &&
          height >= 2 &&
          end < profile.length &&
          profile[start - 1] >= 0 &&
          profile[end] >= 0 &&
          height > profile[start - 1] &&
          height > profile[end]
        ) {
          const base = Math.max(profile[start - 1], profile[end]);
          if (height - base <= 3)
            rows.push({ slice, centre: (start + end) / 2, width, height, base, left: start, right: end, profile });
        }
        start = end;
      }
      const used = new Set<Row[]>(),
        next: Row[][] = [];
      for (const row of rows) {
        const matches = active
          .filter((t) => !used.has(t))
          .map((t) => ({ t, p: t[t.length - 1] }))
          .filter(
            ({ p }) =>
              Math.abs(p.centre - row.centre) <= 1.5 &&
              Math.abs(p.height - row.height) <= 1.5 &&
              Math.abs(p.width - row.width) <= 2,
          )
          .sort((p, q) => Math.abs(p.p.centre - row.centre) - Math.abs(q.p.centre - row.centre));
        if (matches.length) {
          const t = matches[0].t;
          t.push(row);
          used.add(t);
          next.push(t);
        } else next.push([row]);
      }
      tracks.push(...active.filter((t) => !used.has(t)));
      active = next;
    }
    tracks.push(...active);
    const stable: Row[][] = [];
    for (const t of tracks) {
      let run: Row[] = [],
        lo = Infinity,
        hi = -Infinity;
      for (const row of t) {
        if (Math.max(hi, row.width) - Math.min(lo, row.width) > 2) {
          if (run.length) stable.push(run);
          run = [];
          lo = Infinity;
          hi = -Infinity;
        }
        run.push(row);
        lo = Math.min(lo, row.width);
        hi = Math.max(hi, row.width);
      }
      if (run.length) stable.push(run);
    }
    const errorAt = (row: Row, r: number) => {
      let worst = 0;
      const slope = (row.profile[row.right] - row.profile[row.left - 1]) / (row.width + 1);
      for (let cross = row.left - 1; cross <= row.right; cross++) {
        const base = row.profile[row.left - 1] + (cross - row.left + 1) * slope,
          d = Math.abs(cross + 0.5 - row.centre),
          pred = Math.max(base, d <= r ? row.height - r + Math.sqrt(Math.max(0, r * r - d * d)) : base);
        worst = Math.max(worst, Math.abs(pred - row.profile[cross]));
      }
      return worst;
    };
    const feasible: Row[][] = [];
    for (const track of stable) {
      let run: Row[] = [],
        allowed: number[] = [];
      for (const row of track) {
        const options = Array.from({ length: 23 }, (_, i) => 0.5 + i * 0.125).filter((r) => errorAt(row, r) <= 0.56);
        const shared = run.length ? allowed.filter((r) => options.includes(r)) : options;
        if (!shared.length) {
          if (run.length) feasible.push(run);
          run = [];
          allowed = [];
          if (options.length) {
            run = [row];
            allowed = options;
          }
        } else {
          run.push(row);
          allowed = shared;
        }
      }
      if (run.length) feasible.push(run);
    }
    for (const track of feasible) {
      if (track.length < 6) continue;
      let bestError = Infinity,
        radius = 0;
      const maxWidth = Math.max(...track.map((p) => p.width));
      for (let r = 0.5; r <= (maxWidth + 2) / 2 + 0.25; r += 0.125) {
        let worst = 0;
        for (const row of track)
          for (let cross = row.left - 1; cross <= row.right; cross++) {
            const slope = (row.profile[row.right] - row.profile[row.left - 1]) / (row.width + 1),
              base = row.profile[row.left - 1] + (cross - row.left + 1) * slope;
            const d = Math.abs(cross + 0.5 - row.centre),
              pred = Math.max(base, d <= r ? row.height - r + Math.sqrt(Math.max(0, r * r - d * d)) : base);
            worst = Math.max(worst, Math.abs(pred - row.profile[cross]));
          }
        if (worst < bestError - 1e-8) {
          bestError = worst;
          radius = r;
        }
      }
      if (bestError > 0.56 || !radius) continue;
      // A coarse upper arc does not determine the hidden radius. At least the central circular interior must already be occupied; this is not a new rail above a flat plate.
      let support = 0,
        total = 0;
      for (const row of track)
        for (let angle = 0; angle < 12; angle++) {
          const p: Vec3 = [0, 0, 0];
          p[a] = row.slice + 0.5;
          p[c] = row.centre + radius * 0.45 * Math.cos((angle * Math.PI) / 6);
          p[u] = row.height - radius + radius * 0.45 * Math.sin((angle * Math.PI) / 6);
          total++;
          if (gridCell(grid, p.map(Math.floor) as Vec3)) support++;
        }
      if (support / total < 0.94) continue;
      const raw = track.map((row) => {
        const p: Vec3 = [0, 0, 0];
        p[a] = row.slice + 0.5;
        p[c] = row.centre;
        p[u] = row.height - radius;
        return p;
      });
      const smooth = raw.map(
        (p, i) =>
          p.map((v, k) =>
            k !== u
              ? v
              : raw.slice(Math.max(0, i - 2), Math.min(raw.length, i + 3)).reduce((s, p) => s + p[u], 0) /
                raw.slice(Math.max(0, i - 2), Math.min(raw.length, i + 3)).length,
          ) as Vec3,
      );
      smooth[0][a] -= 0.5;
      smooth[smooth.length - 1][a] += 0.5;
      const points = simplifyPath(smooth, 0.25);
      if (points.length > 9) continue;
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
      const slope = median(track.map((row) => (row.profile[row.right] - row.profile[row.left - 1]) / (row.width + 1)));
      if (Math.abs(slope) > 1.5) continue;
      out.push({
        axis: 'xyz'[a] as Axis,
        points,
        radius,
        cells,
        bounds: [lo, hi],
        primitives: sweptCylinders(points, radius),
        floorOffset: median(
          track.map((row) => radius - row.height + (row.profile[row.left - 1] + row.profile[row.right]) / 2),
        ),
        floorSlope: median(track.map((row) => (row.profile[row.right] - row.profile[row.left - 1]) / (row.width + 1))),
        maximumHeightError: bestError,
        evidence: {
          slices: track.length,
          exposedArc: 0,
          radialResidual: bestError,
          sourceSupport: support / total,
          ambiguousSmallCrossSection: true,
        },
      });
    }
  }
  out.sort((a, b) => b.evidence.slices - a.evidence.slices || b.radius - a.radius);
  return out;
}
