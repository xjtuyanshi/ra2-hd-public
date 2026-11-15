import { describe, expect, it } from 'vitest';
import {
  assignDetailCharts,
  expandVertexAttribute,
  parseDetailChartTable,
  type DetailChartTable,
} from '../src/hardSurfaceDetailAtlas';

// Chart of the top face z = 1 of a unit square at 4 texels per unit, placed at atlas (10, 20) with a 1-texel margin.
const table: DetailChartTable = {
  res: 4,
  size: [64, 32],
  faces: { '7': [10, 20, 6, 6, -0.25, -0.25, 1, 1, 0, 0, 0, 1, 0] },
};

describe('hard-surface detail atlas', () => {
  it('validates chart tables', () => {
    expect(parseDetailChartTable(JSON.parse(JSON.stringify(table))).size).toEqual([64, 32]);
    expect(() => parseDetailChartTable({ res: 4, size: [0, 1], faces: {} })).toThrow();
    expect(() => parseDetailChartTable({ res: 4, size: [4, 4], faces: { a: [1, 2] } })).toThrow();
  });

  it('maps coplanar triangles into their chart and leaves other planes neutral', () => {
    // Two top triangles sharing an edge, and one side triangle (y = 0 plane) sharing vertex 0 with the top.
    const position = new Float32Array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]);
    const indices = new Uint32Array([0, 1, 2, 0, 2, 3, 0, 4, 5]);
    const a = assignDetailCharts(position, indices, table);
    expect(a.matched).toBe(2);
    // Vertex 0 is split: the side triangle gets its own neutral copy.
    expect(a.source.length).toBe(7);
    expect(a.indices[6]).toBe(6);
    expect(a.source[6]).toBe(0);
    expect(Array.from(a.uv.subarray(0, 2))).toEqual([11, 21]);
    expect(Array.from(a.uv.subarray(4, 6))).toEqual([15, 25]);
    expect(Array.from(a.uv.subarray(12, 14))).toEqual([-1, -1]);
    expect(Array.from(expandVertexAttribute(new Float32Array([0, 1, 2, 3, 4, 5]), 1, a.source))).toEqual([
      0, 1, 2, 3, 4, 5, 0,
    ]);
  });

  it('rejects a parallel plane and a triangle outside the chart', () => {
    const position = new Float32Array([0, 0, 1.5, 1, 0, 1.5, 1, 1, 1.5, 3, 0, 1, 4, 0, 1, 4, 1, 1]);
    const a = assignDetailCharts(position, new Uint32Array([0, 1, 2, 3, 4, 5]), table);
    expect(a.matched).toBe(0);
    expect(a.uv.every((v) => v === -1)).toBe(true);
  });
});
