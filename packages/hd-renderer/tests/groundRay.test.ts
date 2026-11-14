import { describe, expect, it } from 'vitest';
import { DataTexture } from 'three/webgpu';
import {
  GROUND_FRONT_REACH_LEVELS,
  GRID_NO_GROUND,
  frontMaxFromSquares,
  frontReachSquares,
  groundRayHit,
  gridSurfaceHeight,
  squaresFromNodeGrid,
} from '../src/groundRay';
import { AtmosphereNodes } from '../src/atmosphere';
import type { Ra2Cache } from '../src/ra2/cache';
import { buildRa2Terrain } from '../src/ra2/terrain';
import { SHADOW_GROUND_SEARCH, shadowGroundPull } from '../src/ra2/sprites';
import { LEVEL_HEIGHT } from '../src/world';

// A view 30 degrees down from the -x -z side (towards the camera: -x -z, up), facing ramps that rise along +x.
const RAY: [number, number, number] = (() => {
  const h = Math.cos(Math.PI / 6) / Math.SQRT2;
  return [-h, Math.sin(Math.PI / 6), -h];
})();
// The game view (world.ts IsoCamera): the camera sits on the +x +z side, so cliffs facing +x / +z are the visible ones.
const GAME_RAY: [number, number, number] = [-RAY[0], RAY[1], -RAY[2]];

/** Squares (2 per cell) of `cells` x `cells` cells whose height rises one level per cell along +x from cell `from`. */
function ramp(cells: number, from: number): { grid: Float32Array; columns: number } {
  const nodes = cells * 2 + 1;
  const heights = new Float32Array(nodes * nodes);
  for (let j = 0; j < nodes; j++)
    for (let i = 0; i < nodes; i++) heights[j * nodes + i] = Math.max(0, i / 2 - from) * LEVEL_HEIGHT;
  return { grid: squaresFromNodeGrid(heights, nodes, nodes), columns: cells * 2 };
}

/** One row of RA2 cells at these levels (flat tiles: a level change between two cells is a cliff). */
function ra2Row(levels: number[]) {
  const width = levels.length;
  const texture = new DataTexture(new Uint8Array([32, 64, 96, 255]), 1, 1);
  const nodes = new AtmosphereNodes();
  const cache = {
    nodes,
    exists: new Uint8Array(width).fill(1),
    height: new Uint8Array(levels),
    tile: new Uint16Array(width),
    mapLight: null,
    manifest: {
      grid: { width, height: 1 },
      tiles: [{ terrainType: 0, rampType: 0, file: 'clear01.tem', rect: [0, 0, 1, 1], anchor: [0, 0] }],
    },
    terrainMap: {
      texture,
      waterMask: texture,
      shore: texture,
      cliff: null,
      waterBed: false,
      waterDetail: texture,
      waterSurface: null,
      ref: { file: 'terrain.png', width: 128, height: 64, scale: 1, originX: 0, originY: 0 },
    },
  } as unknown as Ra2Cache;
  const build = buildRa2Terrain(cache);
  const dispose = () => {
    const materials = Array.isArray(build.mesh.material) ? build.mesh.material : [build.mesh.material];
    for (const material of materials) material.dispose();
    build.mesh.geometry.dispose();
    build.heightTexture.dispose();
    build.surfaceSquares?.texture.dispose();
    build.surfaceSquares?.front.dispose();
    texture.dispose();
    nodes.dispose();
  };
  return { build, dispose };
}

describe('ground under a ground-lying pixel', () => {
  it('interpolates the grid over the terrain mesh triangles and treats missing ground as far below', () => {
    const columns = 3;
    // One square of a 2x2-node grid... with a third column so the clamp is exercised.
    const grid = squaresFromNodeGrid(new Float32Array([0, 1, 0, 0, 0, 0, 0, 0, 0]), columns, 3);
    // Triangle a-c-b: b (1, 0) raised by 1, diagonal from (1, 0) to (0, 1).
    expect(gridSurfaceHeight(grid, 2, 2, 1, 0.5, 0.25)).toBeCloseTo(0.5, 12);
    expect(gridSurfaceHeight(grid, 2, 2, 1, 0.75, 0.75)).toBeCloseTo(0.25, 12);
    const holes = new Float32Array([GRID_NO_GROUND + 1, 0, 0, 0]);
    expect(gridSurfaceHeight(holes, 1, 1, 1, 0, 0)).toBeLessThan(-1000);
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

  it('keeps a cliff top at its own level up to the rim, as the terrain mesh draws it', () => {
    // Plateau cells 0-1 at level 3, then a camera-facing cliff down to level 0 (cells 2-3, built after the plateau).
    const { build, dispose } = ra2Row([3, 3, 0, 0]);
    try {
      const squares = build.surfaceSquares!;
      const { width: columns, height: rows } = squares.texture.image as { width: number; height: number };
      const data = squares.texture.image.data as Float32Array;
      const at = (x: number, z: number) => gridSurfaceHeight(data, columns, rows, squares.sub, x, z);
      const top = 3 * LEVEL_HEIGHT;
      for (const x of [1.2, 1.5, 1.6, 1.8, 1.95, 1.999])
        expect(at(x, 0.5)).toBeCloseTo(build.surfaceHeightAt!(x, 0.5), 6);
      expect(at(2.001, 0.5)).toBeCloseTo(0, 6);
      // The shared vertex grid ramps the plateau down over its last half cell (the lower cell writes the rim's nodes).
      const nodeGrid = build.heightTexture.image as { data: Float32Array; width: number; height: number };
      const shared = squaresFromNodeGrid(nodeGrid.data, nodeGrid.width, nodeGrid.height);
      expect(gridSurfaceHeight(shared, nodeGrid.width - 1, nodeGrid.height - 1, 2, 1.75, 0.5)).toBeCloseTo(top / 2, 6);
      // A shadow pixel drawn on the plateau next to the rim, from a tree on the plateau or at the cliff's foot: the
      // ground found is the plateau's (within the smallest pull), never the foot behind it.
      for (const x of [1.55, 1.7, 1.85, 1.98])
        for (const anchor of [top, 0]) {
          const hit = groundRayHit(at, [x, top, 0.5], GAME_RAY, anchor, SHADOW_GROUND_SEARCH);
          expect(Math.abs(hit - top) * Math.sin(Math.PI / 6)).toBeLessThan(shadowGroundPull(0) / 4);
        }
      // A pixel on the cliff wall (the ray crosses the rim below the top): the wall, where the ray crosses x = 2.
      const wallPoint: [number, number, number] = [2, top / 2, 0.5];
      const wall = groundRayHit(at, wallPoint, GAME_RAY, 0, SHADOW_GROUND_SEARCH);
      expect(Math.abs(wall - top / 2) * Math.sin(Math.PI / 6)).toBeLessThan(shadowGroundPull(0) / 4);
    } finally {
      dispose();
    }
  });

  it('takes the flat-ground shortcut only where the full search ends on the anchor plane', () => {
    expect(SHADOW_GROUND_SEARCH.step * SHADOW_GROUND_SEARCH.above).toBeLessThanOrEqual(
      GROUND_FRONT_REACH_LEVELS * LEVEL_HEIGHT,
    );
    // Flat level 2, a plateau at level 4 behind a camera-facing cliff, then flat level 0 and a ramp back up.
    const levels = [2, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0, 2, 2, 2];
    const { build, dispose } = ra2Row(levels);
    try {
      const squares = build.surfaceSquares!;
      const { width: columns, height: rows } = squares.texture.image as { width: number; height: number };
      const data = squares.texture.image.data as Float32Array;
      const front = frontMaxFromSquares(
        data,
        columns,
        rows,
        frontReachSquares(GROUND_FRONT_REACH_LEVELS * LEVEL_HEIGHT, 2),
      );
      expect(front).toEqual(squares.front.image.data);
      const at = (x: number, z: number) => gridSurfaceHeight(data, columns, rows, squares.sub, x, z);
      const shortcutAt = (anchor: number) => (x: number, z: number) => {
        const i = Math.min(columns - 1, Math.max(0, Math.floor(x * 2)));
        const j = Math.min(rows - 1, Math.max(0, Math.floor(z * 2)));
        const o = (j * columns + i) * 4;
        const flat = [0, 1, 2, 3].every((c) => Math.abs(data[o + c]! - anchor) < 1e-4);
        return flat && front[j * columns + i]! <= anchor + 1e-4;
      };
      let taken = 0;
      for (let x = 0.05; x < levels.length; x += 0.1)
        for (const anchorLevel of [0, 2, 4]) {
          const anchor = anchorLevel * LEVEL_HEIGHT;
          const point: [number, number, number] = [x, anchor, 0.5];
          const full = groundRayHit(at, point, GAME_RAY, anchor, SHADOW_GROUND_SEARCH);
          const fast = groundRayHit(at, point, GAME_RAY, anchor, SHADOW_GROUND_SEARCH, shortcutAt(anchor));
          if (fast !== full) throw new Error(`x ${x.toFixed(2)} anchor ${anchorLevel}: ${fast} != ${full}`);
          if (shortcutAt(anchor)(x, 0.5)) taken++;
        }
      expect(taken).toBeGreaterThan(40);
    } finally {
      dispose();
    }
  });
});
