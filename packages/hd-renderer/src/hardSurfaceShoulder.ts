/** Source-supported exposed elliptic shoulders; the parent hides the unobserved part of each section. */
import { gridCell, type SolidGrid, type Axis, type Vec3 } from './hardSurfaceSolid';
import { ellipticCylinder } from './hardSurfaceRound';
import type { RoundFeature } from './hardSurfaceRoundFit';
import type { Plane } from './hardSurfaceMesh';
interface Row {
  slice: number;
  sign: number;
  centre: number;
  top: number;
  radius: number;
  height: number;
  error: number;
  range: [number, number];
  profile: number[];
}
export interface ShoulderFeature extends RoundFeature {
  radii: [number, number];
  profileError: number;
}
const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];
export function detectRoundShoulders(grid: SolidGrid, up: Axis = 'z', maximum = 24): ShoulderFeature[] {
  const u = 'xyz'.indexOf(up),
    out: ShoulderFeature[] = [];
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
        return 0;
      });
      const rows: Row[] = [];
      for (let start = 1; start < profile.length - 1;) {
        let end = start + 1;
        while (end < profile.length && profile[end] === profile[start]) end++;
        const top = profile[start];
        if (end < profile.length && top >= 4 && top > profile[start - 1] && top > profile[end])
          for (const sign of [-1, 1]) {
            const seeds = [(start + end) / 2, sign < 0 ? start + 0.5 : end - 0.5];
            let best: Row | undefined,
              score = Infinity;
            for (const centre of new Set(seeds.flatMap((seed) => [seed - 0.5, seed, seed + 0.5])))
              for (let radius = 2.5; radius <= 10; radius += 0.25) {
                const indices = Array.from({ length: profile.length }, (_, i) => i).filter(
                  (i) => (i + 0.5 - centre) * sign >= -1e-7 && Math.abs(i + 0.5 - centre) <= radius,
                );
                if (indices.length < 4) continue;
                const truth = indices.map((i) => profile[i]);
                if (
                  Math.max(...truth) - Math.min(...truth) < 2 ||
                  truth.some((v, i) => i > 0 && (v - truth[i - 1]) * sign > 0)
                )
                  continue;
                const meanX = indices.reduce((s, v) => s + v, 0) / indices.length,
                  meanY = truth.reduce((s, v) => s + v, 0) / truth.length;
                const slope =
                  indices.reduce((s, v, i) => s + (v - meanX) * (truth[i] - meanY), 0) /
                  indices.reduce((s, v) => s + (v - meanX) ** 2, 0);
                // A plane/chamfer with no observed curvature cannot seed a fillet.
                if (Math.max(...truth.map((v, i) => Math.abs(v - meanY - slope * (indices[i] - meanX)))) < 0.36)
                  continue;
                const arc = indices.map((i) => Math.sqrt(Math.max(0, 1 - ((i + 0.5 - centre) / radius) ** 2)));
                for (let height = 2; height <= 10; height += 0.25) {
                  if (height / radius < 0.5 || height / radius > 2) continue;
                  const error = Math.max(...truth.map((v, i) => Math.abs(top - height + height * arc[i] - v))),
                    cost = error + 0.002 * Math.abs(Math.log(height / radius));
                  if (cost < score) {
                    score = cost;
                    best = {
                      slice,
                      sign,
                      centre,
                      top,
                      radius,
                      height,
                      error,
                      range: [indices[0] + 0.5, indices[indices.length - 1] + 0.5],
                      profile,
                    };
                  }
                }
              }
            if (best && best.error <= 0.65) rows.push(best);
          }
        start = end;
      }
      const next: Row[][] = [],
        used = new Set<Row[]>();
      for (const row of rows) {
        const t = active
          .filter((t) => !used.has(t))
          .filter((t) => {
            const p = t[t.length - 1];
            return (
              p.sign === row.sign &&
              Math.abs(p.centre - row.centre) <= 0.75 &&
              Math.abs(p.top - row.top) <= 0.5 &&
              Math.abs(p.radius - row.radius) <= 0.5 &&
              Math.abs(p.height - row.height) <= 0.75
            );
          })
          .sort(
            (p, q) => Math.abs(p[p.length - 1].centre - row.centre) - Math.abs(q[q.length - 1].centre - row.centre),
          )[0];
        if (t) {
          t.push(row);
          used.add(t);
          next.push(t);
        } else next.push([row]);
      }
      tracks.push(...active.filter((t) => !used.has(t)));
      active = next;
    }
    tracks.push(...active);
    for (const track of tracks) {
      if (track.length < 6) continue;
      const centre = median(track.map((r) => r.centre)),
        top = median(track.map((r) => r.top)),
        radius = median(track.map((r) => r.radius)),
        height = median(track.map((r) => r.height)),
        base = top - height;
      const crossLo = Math.max(...track.map((r) => r.range[0])),
        crossHi = Math.min(...track.map((r) => r.range[1]));
      if (crossHi - crossLo < 2.5) continue;
      let error = 0,
        hit = 0,
        total = 0;
      for (const row of track)
        for (let cross = Math.ceil(crossLo - 0.5); cross + 0.5 <= crossHi; cross++) {
          const predicted = base + height * Math.sqrt(Math.max(0, 1 - ((cross + 0.5 - centre) / radius) ** 2));
          error = Math.max(error, Math.abs(predicted - row.profile[cross]));
          for (let h = base + 0.5; h < predicted - 0.25; h += 0.5) {
            const p: Vec3 = [0, 0, 0];
            p[a] = row.slice;
            p[c] = cross;
            p[u] = Math.floor(h);
            total++;
            if (gridCell(grid, p)) hit++;
          }
        }
      if (error > 0.8 || !total || hit / total < 0.95) continue;
      const start: Vec3 = [0, 0, 0],
        end: Vec3 = [0, 0, 0],
        radial: Vec3 = [0, 0, 0];
      start[a] = track[0].slice;
      end[a] = track[track.length - 1].slice + 1;
      start[c] = end[c] = centre;
      start[u] = end[u] = base;
      radial[c] = 1;
      const lo: Vec3 = [0, 0, 0],
        hi: Vec3 = [0, 0, 0];
      lo[a] = start[a];
      hi[a] = end[a];
      lo[c] = crossLo;
      hi[c] = crossHi;
      lo[u] = base;
      hi[u] = top;
      const primitive = ellipticCylinder(start, end, radial, [radius, height]);
      if (primitive.type !== 'poly') throw new Error('Expected elliptic polytope');
      for (let k = 0; k < 3; k++) {
        const low: Plane = [0, 0, 0, -lo[k]],
          high: Plane = [0, 0, 0, hi[k]];
        low[k] = -1;
        high[k] = 1;
        primitive.planes.push(low, high);
      }
      const cells: Vec3[] = [];
      for (let x = Math.max(0, Math.floor(lo[0])); x < Math.min(grid.size[0], Math.ceil(hi[0])); x++)
        for (let y = Math.max(0, Math.floor(lo[1])); y < Math.min(grid.size[1], Math.ceil(hi[1])); y++)
          for (let z = Math.max(0, Math.floor(lo[2])); z < Math.min(grid.size[2], Math.ceil(hi[2])); z++)
            if (gridCell(grid, [x, y, z])) cells.push([x, y, z]);
      out.push({
        axis: 'xyz'[a] as Axis,
        points: [start, end],
        radius,
        cells,
        bounds: [lo, hi],
        primitives: [primitive],
        radii: [radius, height],
        profileError: error,
        evidence: {
          slices: track.length,
          exposedArc: 0.25,
          radialResidual: error,
          sourceSupport: hit / total,
          ambiguousSmallCrossSection: true,
        },
      });
    }
  }
  return out.sort((p, q) => q.cells.length - p.cells.length).slice(0, maximum);
}
