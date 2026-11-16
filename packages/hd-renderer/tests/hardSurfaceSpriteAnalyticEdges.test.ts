import { describe, it, expect } from 'vitest';
import { buildPart, type Primitive } from '../src/hardSurfaceMesh';
import {
  analyticSpriteEdges,
  type SpriteAnalyticMesh,
  type SpriteAnalyticTriangle,
} from '../src/hardSurfaceSpriteAnalyticEdges';
import type { SpriteCamera } from '../src/hardSurfaceSpriteCamera';
const camera: SpriteCamera = {
  right: [1, 0, 0],
  up: [0, 1, 0],
  towards: [0, 0, 1],
  origin: [4, 28],
  pixelsPerWorld: 8,
};
function mesh(primitives: Primitive[]): SpriteAnalyticMesh {
  const triangles: SpriteAnalyticTriangle[] = [];
  for (const [primitive, p] of primitives.entries()) {
    const m = buildPart({ primitives: [p] }).meshes[0];
    for (let i = 0; i < m.indices.length; i += 3) {
      const indices = Array.from(m.indices.slice(i, i + 3)),
        vertices = indices.map((v) => Array.from(m.positions.slice(v * 3, v * 3 + 3)) as [number, number, number]);
      triangles.push({
        vertices: vertices as SpriteAnalyticTriangle['vertices'],
        face: (primitive + 1) * 1024 + m.face[indices[0]] + 1,
        primitive: primitive + 1,
        chamfer: !!m.chamfer[indices[0]],
      });
    }
  }
  return { triangles };
}
function coverage() {
  return Float64Array.from(
    { length: 32 * 32 },
    (_, i) => +(i % 32 >= 4 && i % 32 < 28 && Math.floor(i / 32) >= 12 && Math.floor(i / 32) < 28),
  );
}
describe('analytic visible union edges', () => {
  it('has no internal seam between overlapping coplanar faces of different solids', () => {
    const solids = mesh([
      { type: 'box', min: [0, 0, 0], max: [2, 2, 1], chamfer: 0 },
      { type: 'box', min: [1, 0, 0], max: [3, 2, 1], chamfer: 0 },
    ]);
    const result = analyticSpriteEdges(solids, camera, 32, 32, coverage());
    expect(result.interior.some(Boolean)).toBe(false);
    expect(result.allEdges.some(Boolean)).toBe(true);
  });
  it('keeps the visible depth step but removes the covered edge behind the union', () => {
    const solids = mesh([
      { type: 'box', min: [0, 0, 0], max: [2, 2, 1], chamfer: 0 },
      { type: 'box', min: [1, 0, 0], max: [3, 2, 1.5], chamfer: 0 },
    ]);
    const result = analyticSpriteEdges(solids, camera, 32, 32, coverage());
    expect(result.interior[18 * 32 + 12]).toBe(1);
    expect(result.interior[18 * 32 + 20]).toBe(0);
    expect(Array.from(result.interior).every((v, i) => !v || i % 32 === 12)).toBe(true);
  });
  it('represents bevel strips by host-edge midlines', () => {
    const result = analyticSpriteEdges(
      mesh([{ type: 'box', min: [0, 0, 0], max: [2, 2, 1], chamfer: 0.1 }]),
      camera,
      32,
      32,
      coverage(),
    );
    expect(result.segments.some((s) => s.kind === 'chamfer-midline')).toBe(true);
    expect(result.segments.every((s) => s.visibleIntervals.every(([a, b]) => a >= 0 && b <= 1 && b > a))).toBe(true);
  });
});
