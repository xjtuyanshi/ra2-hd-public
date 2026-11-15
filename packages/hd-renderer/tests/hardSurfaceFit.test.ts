import { describe, expect, it } from 'vitest';
import {
  directions,
  fitHardSurface,
  fitPoly,
  robustTop,
  selectComponent,
  stairOK,
  tuneOffsets,
  type HardSurfaceSeed,
} from '../src/hardSurfaceFit';
import { forCells, type Vec3 } from '../src/hardSurfaceSolid';
import { grid } from './hardSurfaceFixtures';
describe('hard surface fitting', () => {
  it('finds a 1:2 staircase and keeps boxes free of accidental slope planes', () => {
    const C: Vec3[] = [];
    forCells([12, 4, 6], (p) => {
      if (p[0] + 0.5 + 2 * (p[2] + 0.5) <= 12.5) C.push(p.map((v) => v + 0.5) as Vec3);
    });
    const planes = fitPoly(C, directions());
    expect(planes).toContainEqual([1, 0, 2, 13.5]);
    expect(stairOK([1, 0, 2], C)).toBe(true);
    const box: Vec3[] = [];
    forCells([5, 4, 3], (p) => box.push(p.map((v) => v + 0.5) as Vec3));
    expect(fitPoly(box, directions())).toHaveLength(6);
    expect(directions('axis')).toHaveLength(6);
    expect(directions('45')).toHaveLength(18);
    expect(directions('all')).toHaveLength(42);
  });
  it('rejects stray high detail in robust_top when a lower real row supports the stair', () => {
    const C: Vec3[] = [];
    for (let y = 0; y < 16; y++) {
      C.push([0.5, y + 0.5, 4.5], [2.5, y + 0.5, 3.5], [4.5, y + 0.5, 2.5]);
    }
    C.push([8.5, 0.5, 4.5]);
    expect(robustTop([1, 0, 2], C)).toBe(9.5);
    expect(stairOK([1, 0, 2], C)).toBe(true);
  });
  it('tunes offsets sequentially and selects the outermost candidate on ties', () => {
    const m = grid([4, 4, 4]),
      planes = fitPoly(
        [
          [0.5, 0.5, 0.5],
          [3.5, 3.5, 3.5],
        ],
        directions('axis'),
      );
    const a = tuneOffsets([...planes, [1, 1, 0, 100]], m);
    expect(a.at(-1)).toEqual([1, 1, 0, 100]);
    expect(planes).toEqual(
      fitPoly(
        [
          [0.5, 0.5, 0.5],
          [3.5, 3.5, 3.5],
        ],
        directions('axis'),
      ),
    );
  });
  it('adds a requested 1:3 slope and reproduces the quarter-step tuned offset', () => {
    const s = grid([18, 4, 6], (p) => p[0] + 0.5 + 3 * (p[2] + 0.5) <= 18.5);
    const fit = (extra_ratios: [number, number][]) =>
      fitHardSurface(
        {
          parts: {
            wedge: {
              components: [
                {
                  region: [
                    [0, 0, 0],
                    [18, 4, 6],
                  ],
                  extra_ratios,
                  chamfer: 0,
                },
              ],
            },
          },
        },
        { wedge: s },
      ).parts.wedge.primitives[0];
    const standard = fit([]),
      extended = fit([[1, 3]]);
    if (standard.type !== 'poly' || extended.type !== 'poly') throw new Error('Expected fitted polytopes');
    expect(standard.planes.some((p) => p[0] === 1 && p[1] === 0 && p[2] === 3)).toBe(false);
    // Synthetic Python reference: outer envelope 19.5, chosen offset 18.75.
    expect(extended.planes).toContainEqual([1, 0, 3, 18.75]);
  });
  it('applies excludes, inclusive colour filters, cuts, overrides, raw primitives and off', () => {
    const m = grid([8, 4, 4]),
      colours = new Int16Array(m.data.length).fill(9);
    forCells(m.size, (p, i) => {
      if (p[0] < 4) colours[i] = 20;
    });
    const selected = selectComponent(m, colours, {
      region: [
        [0, 0, 0],
        [8, 4, 4],
      ],
      exclude: [
        [
          [0, 0, 0],
          [1, 4, 4],
        ],
      ],
      colours: [20, 20],
    });
    expect(selected.data.filter(Boolean)).toHaveLength(3 * 4 * 4);
    expect(() =>
      selectComponent(m, undefined, {
        region: [
          [0, 0, 0],
          [8, 4, 4],
        ],
        colours: [1, 2],
      }),
    ).toThrow('colour');
    const seed: HardSurfaceSeed = {
      parts: {
        sample: {
          mirror: { axis: 'y', at: 2 },
          components: [
            {
              name: 'panel',
              region: [
                [0, 0, 0],
                [8, 4, 4],
              ],
              dirs: 'axis',
              extra_ratios: [[1, 3]],
              colours: [20, 20],
              cuts: [[1, 1, 0, 8]],
              set: { '+z': 3, '-x': 1 },
              paint: 'custom',
              chamfer: 0,
              centred: true,
              groove: { width: 0.1 },
            },
            {
              type: 'raw',
              prim: { type: 'box', min: [0, 0, 0], max: [1, 1, 1], paint: 'old' },
              paint: 'new',
              mirror: false,
            },
            {
              type: 'cyl',
              axis: 'y',
              region: [
                [0, 0, 0],
                [8, 4, 4],
              ],
            },
            {
              off: true,
              region: [
                [0, 0, 0],
                [8, 4, 4],
              ],
            },
            {
              name: 'empty',
              region: [
                [8, 0, 0],
                [8, 4, 4],
              ],
            },
          ],
        },
      },
    };
    const empty: string[] = [],
      before = structuredClone(seed),
      spec = fitHardSurface(seed, { sample: { ...m, colours } }, { onEmpty: (_, c) => empty.push(c.name!) });
    expect(spec.parts.sample.primitives).toHaveLength(3);
    expect(empty).toEqual(['empty']);
    const p = spec.parts.sample.primitives[0];
    expect(p.type).toBe('poly');
    if (p.type === 'poly') {
      expect(p.planes).toContainEqual([0, 0, 1, 3]);
      expect(p.planes).toContainEqual([-1, 0, 0, -1]);
      expect(p.planes.at(-1)).toEqual([1, 1, 0, 8]);
    }
    expect(p.paint).toBe('custom');
    expect(spec.parts.sample.primitives[1].paint).toBe('new');
    expect(spec.parts.sample.primitives[2]).toMatchObject({ type: 'cyl', axis: 'y', c: [4, 2], r: 4, range: [0, 4] });
    expect(seed).toEqual(before);
    expect(fitHardSurface(seed, { sample: { ...m, colours } })).toEqual(spec);
  });
});
