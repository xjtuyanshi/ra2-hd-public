import { describe, expect, it } from 'vitest';
import { GRID_NO_GROUND, groundRayHit, gridSurfaceHeight } from '../src/groundRay';
import { SHADOW_GROUND_SEARCH, shadowGroundPull } from '../src/ra2/sprites';
import { LEVEL_HEIGHT } from '../src/world';

// The game view: 30 degrees down, looking along +x +z (towards the camera: -x -z, up).
const RAY: [number, number, number] = (() => {
  const h = Math.cos(Math.PI / 6) / Math.SQRT2;
  return [-h, Math.sin(Math.PI / 6), -h];
})();

/** A 2-node-per-cell grid of `cells` x `cells` cells whose height rises one level per cell along +x from cell `from`. */
function ramp(cells: number, from: number): { grid: Float32Array; columns: number } {
  const columns = cells * 2 + 1;
  const grid = new Float32Array(columns * columns);
  for (let j = 0; j < columns; j++)
    for (let i = 0; i < columns; i++) grid[j * columns + i] = Math.max(0, i / 2 - from) * LEVEL_HEIGHT;
  return { grid, columns };
}

describe('ground under a ground-lying pixel', () => {
  it('interpolates the grid over the terrain mesh triangles and treats missing ground as far below', () => {
    const columns = 3;
    // One square of a 2x2-node grid... with a third column so the clamp is exercised.
    const grid = new Float32Array([0, 1, 0, 0, 0, 0, 0, 0, 0]);
    // Triangle a-c-b: b (1, 0) raised by 1, diagonal from (1, 0) to (0, 1).
    expect(gridSurfaceHeight(grid, columns, 3, 1, 0.5, 0.25)).toBeCloseTo(0.5, 12);
    expect(gridSurfaceHeight(grid, columns, 3, 1, 0.75, 0.75)).toBeCloseTo(0.25, 12);
    const holes = new Float32Array([GRID_NO_GROUND + 1, 0, 0, 0]);
    expect(gridSurfaceHeight(holes, 2, 2, 1, 0, 0)).toBeLessThan(-1000);
  });

  it('lies on flat ground at the anchor and keeps the anchor plane over ground far below it', () => {
    const { grid, columns } = ramp(16, 100);
    const at = (x: number, z: number) => gridSurfaceHeight(grid, columns, columns, 2, x, z);
    expect(groundRayHit(at, [8, 0, 8], RAY, 0, SHADOW_GROUND_SEARCH)).toBeCloseTo(0, 3);
    // A deck three levels above the ground: the ground is beyond the search, the shadow stays on the deck's plane.
    const deck = LEVEL_HEIGHT * 3;
    expect(groundRayHit(at, [8, deck, 8], RAY, deck, SHADOW_GROUND_SEARCH)).toBe(deck);
  });

  it('puts overlapping shadows of anchors one level apart on the same ground', () => {
    const { grid, columns } = ramp(24, 8);
    const at = (x: number, z: number) => gridSurfaceHeight(grid, columns, columns, 2, x, z);
    const point: [number, number, number] = [10.5, 0, 12];
    const low = groundRayHit(at, point, RAY, LEVEL_HEIGHT, SHADOW_GROUND_SEARCH);
    const high = groundRayHit(at, point, RAY, LEVEL_HEIGHT * 2, SHADOW_GROUND_SEARCH);
    expect(high).toBe(low);
  });

  it('finds the drawn ramp surface above the anchor plane within the shadow pull', () => {
    const { grid, columns } = ramp(24, 8);
    const at = (x: number, z: number) => gridSurfaceHeight(grid, columns, columns, 2, x, z);
    // The pixel's ray through the anchor plane (height 0) at x = 10..12 meets the ramp (rising from x = 8) above it.
    for (const x of [9, 10, 10.6, 11.5, 12]) {
      const point: [number, number, number] = [x, 0, 12];
      const hit = groundRayHit(at, point, RAY, 0, SHADOW_GROUND_SEARCH);
      expect(hit).toBeGreaterThan(0);
      // The surface height where the ray is at `hit`: the two agree to well inside the smallest shadow pull.
      const t = (hit - point[1]) / RAY[1];
      const ground = at(point[0] + RAY[0] * t, point[2] + RAY[2] * t);
      expect(Math.abs(hit - ground) * Math.sin(Math.PI / 6)).toBeLessThan(shadowGroundPull(0) / 4);
      expect(hit).toBeGreaterThanOrEqual(ground);
    }
  });

  it('samples the same heights for every anchor, so overlapping shadows on one pixel agree', () => {
    const { grid, columns } = ramp(24, 8);
    const at = (x: number, z: number) => gridSurfaceHeight(grid, columns, columns, 2, x, z);
    const a = groundRayHit(at, [11, 0, 12], RAY, 0, SHADOW_GROUND_SEARCH);
    // Another anchor a little higher, the same pixel's ray (moved along it).
    const lift = LEVEL_HEIGHT * 0.25;
    const t = lift / RAY[1];
    const b = groundRayHit(at, [11 + RAY[0] * t, lift, 12 + RAY[2] * t], RAY, lift, SHADOW_GROUND_SEARCH);
    expect(b).toBe(a);
  });
});
