import { DataTexture, FloatType, NearestFilter, RGBAFormat, RedFormat } from 'three/webgpu';
import {
  If,
  abs,
  float,
  floor,
  ivec2,
  max,
  min,
  mix,
  select,
  step as stepNode,
  textureLoad,
  vec2,
  vec4,
} from 'three/tsl';
import type { ShaderNode } from './tsl';

/**
 * Where a view ray meets a terrain surface, for ground-lying art that is drawn at fixed screen pixels over sloped
 * ground.
 *
 * The surface is a grid of squares (`sub` per world unit), each holding the world heights of its own four corners
 * (RGBA: (0, 0), (1, 0), (0, 1), (1, 1)), interpolated over the two triangles every terrain mesh of this renderer cuts a
 * square into (a-c-b and b-c-d: the diagonal runs from the square's (1, 0) corner to its (0, 1) corner), so the height
 * matches the drawn surface rather than a bilinear one. Squares do not share corners: a terrain whose cells meet at a
 * cliff (flat top, vertical wall, flat foot) keeps each side's own level up to the shared edge, as its mesh draws it.
 * A grid of shared vertex nodes cannot: one node on the edge takes either side's level, so the surface there would
 * ramp over the last square (RA2: half a cell along every camera-facing cliff rim). One load per evaluation.
 * Corners at or above GRID_NO_GROUND (no ground there) count as far below every ray.
 */
export const GRID_NO_GROUND = 49;
/** Height (in terrain levels) of the front box TerrainBuild.surfaceSquares.front covers; searches must stay inside. */
export const GROUND_FRONT_REACH_LEVELS = 5;
const NO_GROUND_HEIGHT = -1e4;

/** An empty 1x1 square grid (no ground anywhere), bound until a world supplies its surface. */
export function emptyHeightGrid(): DataTexture {
  return surfaceSquaresTexture(new Float32Array([1e4, 1e4, 1e4, 1e4]), 1, 1);
}

/** A square grid (4 floats per square, see above) as the texture groundRayHitNode reads with loads. */
export function surfaceSquaresTexture(squares: Float32Array, columns: number, rows: number): DataTexture {
  const texture = new DataTexture(squares, columns, rows, RGBAFormat, FloatType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

/**
 * The square grid of a continuous mesh on a shared vertex grid (`nodeColumns` x `nodeRows` world heights): every
 * square takes its four corner nodes.
 */
export function squaresFromNodeGrid(nodes: ArrayLike<number>, nodeColumns: number, nodeRows: number): Float32Array {
  const columns = Math.max(1, nodeColumns - 1);
  const rows = Math.max(1, nodeRows - 1);
  const node = (c: number, r: number): number =>
    nodes[Math.min(r, nodeRows - 1) * nodeColumns + Math.min(c, nodeColumns - 1)]!;
  const squares = new Float32Array(columns * rows * 4);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < columns; c++)
      squares.set([node(c, r), node(c + 1, r), node(c, r + 1), node(c + 1, r + 1)], (r * columns + c) * 4);
  return squares;
}

/**
 * How far in front of a point (towards the camera) a ray search may look, as a height above that point: the box of
 * squares `frontMaxFromSquares` covers. World.ts's IsoCamera looks down 30 degrees from the +x +z side, so a ray rising
 * by `h` from a point moves h / tan(30 deg) / sqrt(2) along +x and along +z.
 */
export function frontReachSquares(height: number, sub: number): number {
  return Math.ceil(((height / Math.tan(Math.PI / 6)) * Math.SQRT1_2 + 1e-6) * sub) + 1;
}

/**
 * For every square: the highest corner of the squares from it up to `reach` squares towards +x and +z (missing ground
 * excluded; -1e4 when there is none). A view ray leaving a point of a square towards the camera stays inside that box
 * until it has risen by the height frontReachSquares was given, so a flat square whose front maximum is its own level
 * is the first surface the ray meets (groundRayHitNode's shortcut).
 */
export function frontMaxFromSquares(
  squares: ArrayLike<number>,
  columns: number,
  rows: number,
  reach: number,
): Float32Array {
  const own = new Float32Array(columns * rows);
  for (let k = 0; k < columns * rows; k++) {
    let m = NO_GROUND_HEIGHT;
    for (let c = 0; c < 4; c++) {
      const v = squares[k * 4 + c]!;
      if (v < GRID_NO_GROUND && v > m) m = v;
    }
    own[k] = m;
  }
  // Separable running maximum over [i, i + reach] along x, then along z.
  const alongX = new Float32Array(columns * rows);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < columns; c++) {
      let m = NO_GROUND_HEIGHT;
      for (let a = 0; a <= reach && c + a < columns; a++) m = Math.max(m, own[r * columns + c + a]!);
      alongX[r * columns + c] = m;
    }
  const front = new Float32Array(columns * rows);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < columns; c++) {
      let m = NO_GROUND_HEIGHT;
      for (let b = 0; b <= reach && r + b < rows; b++) m = Math.max(m, alongX[(r + b) * columns + c]!);
      front[r * columns + c] = m;
    }
  return front;
}

/** A front maximum grid as the R32F texture groundRayHitNode reads with loads. */
export function frontMaxTexture(front: Float32Array, columns: number, rows: number): DataTexture {
  const texture = new DataTexture(front, columns, rows, RedFormat, FloatType);
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

/** Height of the square grid's surface at world x/z: CPU twin of the shader's (see groundRayHitNode). */
export function gridSurfaceHeight(
  squares: ArrayLike<number>,
  columns: number,
  rows: number,
  sub: number,
  x: number,
  z: number,
): number {
  const gx = Math.min(Math.max(x * sub, 0), columns);
  const gz = Math.min(Math.max(z * sub, 0), rows);
  const i = Math.min(Math.floor(gx), columns - 1);
  const j = Math.min(Math.floor(gz), rows - 1);
  const fx = gx - i;
  const fz = gz - j;
  const o = (j * columns + i) * 4;
  const at = (k: number): number => {
    const v = squares[o + k]!;
    return v >= GRID_NO_GROUND ? NO_GROUND_HEIGHT : v;
  };
  const a = at(0);
  const b = at(1);
  const c = at(2);
  const d = at(3);
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
  shortcut?: (x: number, z: number) => boolean,
): number {
  if (shortcut) {
    // groundRayHitNode's flat-ground shortcut: where the ray meets the anchor's plane on a square flat at that level
    // with nothing higher in front, the search would end on that rung exactly.
    const t = (anchor - point[1]) / ray[1];
    if (shortcut(point[0] + ray[0] * t, point[2] + ray[2] * t)) return Math.round(anchor / step) * step;
  }
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
 * Shader version of groundRayHit. `grid` is a texture node of the square grid (RGBA32F, read with loads), `gridInfo`
 * (columns, rows, sub), `point` a world point on the pixel's view ray and `ray` the world direction towards the camera.
 */
export function groundRayHitNode(
  grid: ShaderNode,
  gridInfo: ShaderNode,
  point: ShaderNode,
  ray: ShaderNode,
  anchor: ShaderNode,
  { step, above, below: under, refine }: GroundRaySearch,
  front?: ShaderNode,
): ShaderNode {
  const columns: ShaderNode = gridInfo.x;
  const rows: ShaderNode = gridInfo.y;
  const sub: ShaderNode = gridInfo.z;
  const heightAt = (xz: ShaderNode): ShaderNode => {
    const g: ShaderNode = min(max(xz.mul(sub), vec2(0, 0)), vec2(columns, rows));
    const k: ShaderNode = min(floor(g), vec2(columns.sub(1), rows.sub(1)));
    const f: ShaderNode = g.sub(k);
    const raw: ShaderNode = textureLoad(grid as never, ivec2(k as ShaderNode));
    // Missing ground (corner >= GRID_NO_GROUND) becomes far below every ray; mix by a 0/1 step is exact.
    const corners: ShaderNode = mix(raw, vec4(NO_GROUND_HEIGHT), stepNode(vec4(GRID_NO_GROUND), raw));
    const a: ShaderNode = corners.x;
    const b: ShaderNode = corners.y;
    const c: ShaderNode = corners.z;
    const d: ShaderNode = corners.w;
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
  // Flat-ground shortcut (two loads instead of up to 20 evaluations): where the ray meets the anchor's plane on a square
  // flat at that level and nothing in front of it (towards the camera, within the search's reach) is higher, that
  // point is the first surface the ray meets, and the search would end exactly on its rung.
  const searching: ShaderNode = float(1).toVar();
  if (front) {
    const t0: ShaderNode = anchor.sub(point.y).div(ray.y);
    const g: ShaderNode = min(max(point.xz.add(ray.xz.mul(t0)).mul(sub), vec2(0, 0)), vec2(columns, rows));
    const k: ShaderNode = ivec2(min(floor(g), vec2(columns.sub(1), rows.sub(1))) as ShaderNode);
    const own: ShaderNode = textureLoad(grid as never, k);
    const spread: ShaderNode = max(
      max(abs(own.x.sub(anchor)), abs(own.y.sub(anchor))),
      max(abs(own.z.sub(anchor)), abs(own.w.sub(anchor))),
    );
    const ahead: ShaderNode = textureLoad(front as never, k).r;
    If(spread.lessThan(1e-4).and(ahead.lessThanEqual(anchor.add(1e-4))), () => {
      searching.assign(0);
      result.assign(floor(anchor.div(step).add(0.5)).mul(step));
    });
  }
  If(searching.greaterThan(0.5).and(below(top).not()), () => {
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
