import { directions, fitPoly } from './hardSurfaceFit';
import { dot, insideAny, polytopeFaces, type Carve, type ExpandedPrimitive } from './hardSurfaceMesh';
import { cellBounds } from './hardSurfacePropose';
import { forCells, gridCell, gridIndex, type SolidGrid, type Vec3 } from './hardSurfaceSolid';

export interface RecessProposal {
  id: string;
  bounds: [Vec3, Vec3];
  cells: number;
  enclosedAxes: number[];
  sourceOverlap: number;
  carve: Carve;
}
/** Continuous source-empty recesses, not a decomposition of every residual cell into boxes. */
export function proposeRecesses(solid: SolidGrid, primitives: readonly ExpandedPrimitive[]): RecessProposal[] {
  const extra = new Set<number>();
  forCells(solid.size, (p, i) => {
    if (!solid.data[i] && insideAny(p.map((v) => v + 0.5) as Vec3, primitives)) extra.add(i);
  });
  const point = (i: number): Vec3 => {
    const z = i % solid.size[2],
      y = Math.floor(i / solid.size[2]) % solid.size[1];
    return [Math.floor(i / (solid.size[1] * solid.size[2])), y, z];
  };
  const out: RecessProposal[] = [];
  function consider(queue: Vec3[], depth: number): void {
    const bounds = cellBounds(queue),
      spans = bounds[1].map((v, a) => v - bounds[0][a]).sort((a, b) => b - a);
    // A large groove/deck pocket is eligible. Tiny dimples and rows of isolated residual cells are not.
    if (queue.length < 12 || spans[0] < 6 || spans[1] < 2) return;
    const enclosedAxes: number[] = [];
    for (let a = 0; a < 3; a++) {
      let enclosed = 0;
      for (const p of queue) {
        const both = [-1, 1].every((sign) => {
          for (let d = 1; d < solid.size[a]; d++) {
            const q = [...p] as Vec3;
            q[a] += sign * d;
            if (q[a] < 0 || q[a] >= solid.size[a]) break;
            if (gridCell(solid, q)) return true;
          }
          return false;
        });
        if (both) enclosed++;
      }
      if (enclosed / queue.length >= 0.25) enclosedAxes.push(a);
    }
    if (!enclosedAxes.length) return;
    const [lo, hi] = bounds;
    let boxOverlap = false;
    for (let x = lo[0]; x < hi[0]; x++)
      for (let y = lo[1]; y < hi[1]; y++)
        for (let z = lo[2]; z < hi[2]; z++) if (gridCell(solid, [x, y, z])) boxOverlap = true;
    // A wholly source-empty rectangular pocket needs six walls, not spurious diagonal corner cuts.
    const planes = fitPoly(
      queue.map((p) => p.map((v) => v + 0.5) as Vec3),
      directions(boxOverlap ? 'all' : 'axis'),
      3,
    );
    const { faces } = polytopeFaces(planes),
      compact = planes.filter((_, i) => faces.some((f) => f.plane === i));
    if (compact.length < 4) return;
    let covered = 0,
      overlap = 0;
    for (let x = lo[0]; x < hi[0]; x++)
      for (let y = lo[1]; y < hi[1]; y++)
        for (let z = lo[2]; z < hi[2]; z++) {
          const p: Vec3 = [x + 0.5, y + 0.5, z + 0.5];
          if (!compact.every((n) => dot(p, n) <= n[3] + 1e-7)) continue;
          covered++;
          if (gridCell(solid, [x, y, z])) overlap++;
        }
    // Source occupancy, rather than dark paint, must justify a subtraction. Formal subvoxel metrics follow.
    const sourceOverlap = overlap / Math.max(1, covered);
    if (sourceOverlap > 0.01) {
      // Connected residual space can wrap around a source-supported island. Its convex
      // envelope is unsafe, but a few broad sub-pockets may still be valid. Never cut
      // source-occupied cells or recurse into dimples/voxel-sized patches.
      if (depth < 3) {
        const axes = [0, 1, 2].sort((a, b) => hi[b] - lo[b] - (hi[a] - lo[a]));
        for (const axis of axes) {
          if (hi[axis] - lo[axis] < 12) continue;
          const cut = Math.floor((lo[axis] + hi[axis]) / 2),
            left = queue.filter((p) => p[axis] < cut),
            right = queue.filter((p) => p[axis] >= cut);
          if (Math.min(left.length, right.length) < 12) continue;
          consider(left, depth + 1);
          consider(right, depth + 1);
          break;
        }
      }
      return;
    }
    out.push({
      id: `${lo.join(',')}:${hi.join(',')}`,
      bounds,
      cells: queue.length,
      enclosedAxes,
      sourceOverlap,
      carve: { type: 'poly', planes: compact },
    });
  }
  while (extra.size) {
    const first = extra.values().next().value!,
      queue = [point(first)];
    extra.delete(first);
    for (let head = 0; head < queue.length; head++)
      for (let a = 0; a < 3; a++)
        for (const sign of [-1, 1]) {
          const q = [...queue[head]] as Vec3;
          q[a] += sign;
          if (q.some((v, a) => v < 0 || v >= solid.size[a])) continue;
          if (extra.delete(gridIndex(solid.size, ...q))) queue.push(q);
        }
    consider(queue, 0);
  }

  return out.sort((a, b) => b.cells - a.cells || a.id.localeCompare(b.id));
}
