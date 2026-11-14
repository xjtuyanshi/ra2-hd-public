import { DataTexture, FloatType, RedFormat } from 'three/webgpu';
import { If, float, floor, ivec2, max, min, select, textureLoad, vec2 } from 'three/tsl';
import type { ShaderNode } from './tsl';

/**
 * Where a view ray meets a terrain height grid (TerrainBuild.heightTexture: world heights on a regular vertex grid,
 * `sub` nodes per world unit), for ground-lying art that is drawn at fixed screen pixels over sloped ground.
 *
 * The grid is read with texel loads and interpolated over the two triangles every terrain mesh of this renderer cuts a
 * grid square into (a-c-b and b-c-d: the diagonal runs from the square's (1, 0) corner to its (0, 1) corner), so the
 * height matches the drawn surface rather than a bilinear one (up to a sixteenth of a square's twist away). Grid nodes
 * at or above GRID_NO_GROUND (no ground there) count as far below every ray.
 */
export const GRID_NO_GROUND = 49;
const NO_GROUND_HEIGHT = -1e4;

/** An empty 1x1 grid (no ground anywhere), bound until a world supplies its height grid. */
export function emptyHeightGrid(): DataTexture {
  const grid = new DataTexture(new Float32Array([NO_GROUND_HEIGHT]), 1, 1, RedFormat, FloatType);
  grid.needsUpdate = true;
  return grid;
}

/** Height of the triangulated grid surface at world x/z: CPU twin of the shader's (see groundRayHitNode). */
export function gridSurfaceHeight(
  grid: ArrayLike<number>,
  columns: number,
  rows: number,
  sub: number,
  x: number,
  z: number,
): number {
  const gx = Math.min(Math.max(x * sub, 0), columns - 1);
  const gz = Math.min(Math.max(z * sub, 0), rows - 1);
  const i = Math.min(Math.floor(gx), Math.max(columns - 2, 0));
  const j = Math.min(Math.floor(gz), Math.max(rows - 2, 0));
  const fx = gx - i;
  const fz = gz - j;
  const at = (c: number, r: number): number => {
    const v = grid[Math.min(r, rows - 1) * columns + Math.min(c, columns - 1)]!;
    return v >= GRID_NO_GROUND ? NO_GROUND_HEIGHT : v;
  };
  const a = at(i, j);
  const b = at(i + 1, j);
  const c = at(i, j + 1);
  const d = at(i + 1, j + 1);
  return fx + fz <= 1 ? a + (b - a) * fx + (c - a) * fz : d + (c - d) * (1 - fx) + (b - d) * (1 - fz);
}

/**
 * A ray search: heights are world y; the ray is followed from `above` steps of `step` over the anchor down to `below`
 * steps under it (on an absolute ladder of heights), then the crossing is refined by `refine` halvings.
 */
export interface GroundRaySearch {
  step: number;
  above: number;
  below: number;
  refine: number;
}

/**
 * CPU twin of groundRayHitNode: the height at which the ray through `point` (towards the camera along `ray`, ray[1] >
 * 0) first meets the grid surface coming from the camera, searched around `anchor` on an absolute ladder of heights
 * (every caller on one pixel whose search reaches the crossing finds exactly the same height), or `anchor` itself when
 * the surface is not met within the search or already blocks the ray at its top.
 */
export function groundRayHit(
  heightAt: (x: number, z: number) => number,
  point: readonly [number, number, number],
  ray: readonly [number, number, number],
  anchor: number,
  { step, above, below: under, refine }: GroundRaySearch,
): number {
  const below = (h: number): boolean => {
    const t = (h - point[1]) / ray[1];
    return heightAt(point[0] + ray[0] * t, point[2] + ray[2] * t) > h + 1e-4;
  };
  // Rungs are whole multiples of the step, so searches from different anchors sample bit-identical heights.
  const rung = Math.floor(anchor / step) + above;
  const top = rung * step;
  if (below(top)) return anchor;
  let lo = top;
  let hi = top;
  let found = false;
  for (let k = 1; k <= above + under && !found; k++) {
    const h = (rung - k) * step;
    if (below(h)) {
      found = true;
      lo = h;
      hi = (rung - k + 1) * step;
    }
  }
  if (!found) return anchor;
  for (let k = 0; k < refine; k++) {
    const mid = (lo + hi) / 2;
    if (below(mid)) lo = mid;
    else hi = mid;
  }
  return hi;
}

/**
 * Shader version of groundRayHit. `grid` is a texture node of the height grid (R32F, read with loads), `gridInfo`
 * (columns, rows, sub), `point` a world point on the pixel's view ray and `ray` the world direction towards the camera.
 */
export function groundRayHitNode(
  grid: ShaderNode,
  gridInfo: ShaderNode,
  point: ShaderNode,
  ray: ShaderNode,
  anchor: ShaderNode,
  { step, above, below: under, refine }: GroundRaySearch,
): ShaderNode {
  const columns: ShaderNode = gridInfo.x;
  const rows: ShaderNode = gridInfo.y;
  const sub: ShaderNode = gridInfo.z;
  const load = (c: ShaderNode, r: ShaderNode): ShaderNode => {
    const v: ShaderNode = textureLoad(
      grid as never,
      ivec2(vec2(min(c, columns.sub(1)), min(r, rows.sub(1))) as ShaderNode),
    ).r;
    return select(v.greaterThanEqual(GRID_NO_GROUND), float(NO_GROUND_HEIGHT), v);
  };
  const heightAt = (xz: ShaderNode): ShaderNode => {
    const g: ShaderNode = min(max(xz.mul(sub), vec2(0, 0)), vec2(columns.sub(1), rows.sub(1)));
    const k: ShaderNode = min(floor(g), max(vec2(columns.sub(2), rows.sub(2)), vec2(0, 0)));
    const f: ShaderNode = g.sub(k);
    const a = load(k.x, k.y);
    const b = load(k.x.add(1), k.y);
    const c = load(k.x, k.y.add(1));
    const d = load(k.x.add(1), k.y.add(1));
    return select(
      f.x.add(f.y).lessThanEqual(1),
      a.add(b.sub(a).mul(f.x)).add(c.sub(a).mul(f.y)),
      d.add(c.sub(d).mul(float(1).sub(f.x))).add(b.sub(d).mul(float(1).sub(f.y))),
    );
  };
  const below = (h: ShaderNode): ShaderNode => {
    const t: ShaderNode = h.sub(point.y).div(ray.y);
    return heightAt(point.xz.add(ray.xz.mul(t))).greaterThan(h.add(1e-4));
  };
  const rung: ShaderNode = floor(anchor.div(step)).add(above).toVar();
  const top: ShaderNode = rung.mul(step).toVar();
  const result: ShaderNode = anchor.toVar();
  const found: ShaderNode = float(0).toVar();
  const lo: ShaderNode = top.toVar();
  const hi: ShaderNode = top.toVar();
  If(below(top).not(), () => {
    for (let k = 1; k <= above + under; k++) {
      If(found.lessThan(0.5), () => {
        const h: ShaderNode = rung.sub(k).mul(step);
        If(below(h), () => {
          found.assign(1);
          lo.assign(h);
          hi.assign(rung.sub(k - 1).mul(step));
        });
      });
    }
    If(found.greaterThan(0.5), () => {
      for (let k = 0; k < refine; k++) {
        const mid: ShaderNode = lo.add(hi).mul(0.5);
        If(below(mid), () => {
          lo.assign(mid);
        }).Else(() => {
          hi.assign(mid);
        });
      }
      result.assign(hi);
    });
  });
  return result;
}
