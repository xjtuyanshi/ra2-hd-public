import { describe, expect, it } from 'vitest';
import { grid } from './hardSurfaceFixtures';
import { occupiedCells, proposeHardSurface } from '../src/hardSurfacePropose';
import { tubeLanes, tubeSegments } from '../src/hardSurfaceTube';

describe('source cross-section tube chain', () => {
  it('allocates three source-step segments independently to each parallel tube', () => {
    const source = grid([22, 18, 9], ([x, y, z]) => {
      const radius = x < 4 ? 3.1 : x < 16 ? 1.7 : 2.4;
      return Math.hypot(y - 4, z - 4) < radius || Math.hypot(y - 13, z - 4) < radius;
    });
    const cells = occupiedCells(source),
      lanes = tubeLanes(cells);
    expect(lanes).toHaveLength(2);
    const ranges = tubeSegments(lanes[0], 'x', 3).map((g) => [
      Math.min(...g.map((p) => p[0])),
      Math.max(...g.map((p) => p[0])) + 1,
    ]);
    expect(ranges).toEqual([
      [0, 4],
      [4, 16],
      [16, 22],
    ]);
    const model = proposeHardSurface(source, { category: 'tube', budget: 6, primitivesPerTube: 3 });
    expect(model.regions).toHaveLength(6);
    for (const lane of ['0_', '1_']) {
      const primitives = model.regions.filter((r) => r.id.startsWith(lane)).map((r) => r.primitive);
      expect(new Set(primitives.map((p) => p.type === 'cyl' && p.c.join(','))).size).toBe(1);
      expect(primitives.every((p) => p.type === 'cyl')).toBe(true);
    }
  });
  it('supports an injected axis and preserves a uniform tube as one continuous segment', () => {
    const cells = occupiedCells(grid([5, 5, 12], ([x, y]) => Math.hypot(x - 2, y - 2) < 1.7));
    expect(tubeLanes(cells, 'z')).toHaveLength(1);
    expect(tubeSegments(cells, 'z', 3)).toHaveLength(1);
    expect(() => tubeSegments(cells, 'z', 0)).toThrow('budget');
  });
  it('separates tubes with a short shared root and never bridges an axial source gap', () => {
    const joined = occupiedCells(
      grid(
        [18, 13, 5],
        ([x, y, z]) => z >= 1 && z < 4 && ((y >= 1 && y < 4) || (y >= 9 && y < 12) || (x < 2 && y >= 4 && y < 9)),
      ),
    );
    const lanes = tubeLanes(joined);
    expect(lanes).toHaveLength(2);
    expect(lanes.flat().length).toBe(joined.length);
    const gap = occupiedCells(grid([14, 3, 3], ([x]) => x < 5 || x >= 8));
    const segments = tubeSegments(gap, 'x', 3);
    expect(segments).toHaveLength(2);
    expect(segments.every((g) => Math.max(...g.map((p) => p[0])) < 5 || Math.min(...g.map((p) => p[0])) >= 8)).toBe(
      true,
    );
    expect(() => tubeSegments(gap, 'x', 1)).toThrow('axial runs');
  });
});
