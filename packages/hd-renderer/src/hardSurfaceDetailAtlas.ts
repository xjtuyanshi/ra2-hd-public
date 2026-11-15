/**
 * Baked per-face detail atlases for hard-surface meshes.
 *
 * An offline baker charts every planar face of a part in its own plane (origin o, orthonormal in-plane axes u and
 * v = n x u, `res` texels per lattice unit) and packs the charts into one RGBA8 atlas. The live mesh does not share
 * face ids with the baker, so each triangle finds its chart geometrically: same plane (normal and offset within a
 * tolerance) and all three vertices inside the chart rectangle. Triangles without a chart (bevels the baker did not
 * chart, downward faces) get the sentinel -1 and render neutral. A vertex shared by triangles of different charts is
 * split so every triangle interpolates its own chart's coordinates.
 */

/** [ox, oy, w, h, origin xyz, u xyz, v xyz] in atlas texels and lattice units. */
export type DetailChart = readonly number[];
export interface DetailChartTable {
  res: number;
  size: readonly [number, number];
  faces: Readonly<Record<string, DetailChart>>;
}
export interface DetailAssignment {
  /** Per output vertex, atlas texel coordinates; (-1, -1) where the triangle has no chart. */
  uv: Float32Array;
  /** Output vertex -> source vertex. */
  source: Uint32Array;
  indices: Uint32Array;
  triangles: number;
  matched: number;
}

export function parseDetailChartTable(value: unknown): DetailChartTable {
  const v = value as Partial<DetailChartTable> | null;
  if (!v || typeof v !== 'object' || !(Number(v.res) > 0) || !Array.isArray(v.size) || v.size.length !== 2)
    throw new Error('Invalid detail chart table');
  const [w, h] = v.size.map(Number);
  if (!(w > 0 && h > 0 && w <= 16384 && h <= 16384)) throw new Error('Invalid detail atlas size');
  if (!v.faces || typeof v.faces !== 'object') throw new Error('Invalid detail chart faces');
  for (const chart of Object.values(v.faces))
    if (!Array.isArray(chart) || chart.length !== 13 || !chart.every(Number.isFinite))
      throw new Error('Invalid detail chart');
  return { res: Number(v.res), size: [w, h], faces: v.faces };
}

type Vec3 = [number, number, number];
interface Chart {
  ox: number;
  oy: number;
  w: number;
  h: number;
  o: Vec3;
  u: Vec3;
  v: Vec3;
  n: Vec3;
  d: number;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Assign each triangle of an indexed mesh (lattice-space positions) to a chart and emit per-vertex atlas
 * coordinates. `cos` and `plane` bound the normal and plane-offset mismatch; `slack` (texels) lets vertices sit on
 * the chart border.
 */
export function assignDetailCharts(
  position: ArrayLike<number>,
  indices: ArrayLike<number>,
  table: DetailChartTable,
  { cos = 0.995, plane = 0.06, slack = 0.5 } = {},
): DetailAssignment {
  const charts: Chart[] = Object.values(table.faces).map((c) => {
    const o: Vec3 = [c[4], c[5], c[6]],
      u: Vec3 = [c[7], c[8], c[9]],
      v: Vec3 = [c[10], c[11], c[12]];
    const n: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1;
    n[0] /= l;
    n[1] /= l;
    n[2] /= l;
    return { ox: c[0], oy: c[1], w: c[2], h: c[3], o, u, v, n, d: dot(o, n) };
  });
  // Group charts by (near-)identical normals so each triangle tests only coplanar candidates.
  const groups: { n: Vec3; charts: number[] }[] = [];
  charts.forEach((chart, i) => {
    const group = groups.find((g) => dot(g.n, chart.n) > 0.99995);
    if (group) group.charts.push(i);
    else groups.push({ n: chart.n, charts: [i] });
  });
  const res = table.res,
    triangles = Math.floor(indices.length / 3);
  const chartOf = new Int32Array(triangles).fill(-1);
  const p: Vec3[] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const r: Vec3 = [0, 0, 0];
  let matched = 0;
  for (let t = 0; t < triangles; t++) {
    for (let k = 0; k < 3; k++) {
      const i = indices[t * 3 + k] * 3;
      p[k][0] = position[i];
      p[k][1] = position[i + 1];
      p[k][2] = position[i + 2];
    }
    const e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]],
      e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    const n: Vec3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]);
    if (l < 1e-9) continue;
    n[0] /= l;
    n[1] /= l;
    n[2] /= l;
    const c: Vec3 = [0, 1, 2].map((a) => (p[0][a] + p[1][a] + p[2][a]) / 3) as Vec3;
    let best = -1,
      score = -Infinity;
    for (const group of groups) {
      if (dot(group.n, n) < cos) continue;
      for (const ci of group.charts) {
        const chart = charts[ci];
        if (Math.abs(dot(c, chart.n) - chart.d) > plane) continue;
        let margin = Infinity;
        for (const q of p) {
          r[0] = q[0] - chart.o[0];
          r[1] = q[1] - chart.o[1];
          r[2] = q[2] - chart.o[2];
          const s = dot(r, chart.u) * res,
            y = dot(r, chart.v) * res;
          margin = Math.min(margin, s, chart.w - s, y, chart.h - y);
        }
        if (margin >= -slack && margin > score) {
          score = margin;
          best = ci;
        }
      }
    }
    chartOf[t] = best;
    if (best >= 0) matched++;
  }
  // Split vertices shared across charts, keyed by (source vertex, chart).
  const vertexCount = Math.floor(position.length / 3);
  const first = new Int32Array(vertexCount).fill(-2);
  const extra = new Map<number, number>();
  const source: number[] = [],
    chartAt: number[] = [];
  const out = new Uint32Array(triangles * 3);
  for (let t = 0; t < triangles; t++) {
    const ci = chartOf[t];
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k];
      let index = v;
      if (first[v] === -2) first[v] = ci;
      else if (first[v] !== ci) {
        const key = (ci + 1) * vertexCount + v;
        let e = extra.get(key);
        if (e === undefined) {
          e = vertexCount + source.length;
          extra.set(key, e);
          source.push(v);
          chartAt.push(ci);
        }
        index = e;
      }
      out[t * 3 + k] = index;
    }
  }
  const total = vertexCount + source.length;
  const src = new Uint32Array(total),
    uv = new Float32Array(total * 2).fill(-1);
  for (let v = 0; v < total; v++) {
    const s = v < vertexCount ? v : source[v - vertexCount],
      ci = v < vertexCount ? first[v] : chartAt[v - vertexCount];
    src[v] = s;
    if (ci < 0) continue;
    const chart = charts[ci];
    r[0] = position[s * 3] - chart.o[0];
    r[1] = position[s * 3 + 1] - chart.o[1];
    r[2] = position[s * 3 + 2] - chart.o[2];
    uv[v * 2] = chart.ox + dot(r, chart.u) * res;
    uv[v * 2 + 1] = chart.oy + dot(r, chart.v) * res;
  }
  return { uv, source: src, indices: out, triangles, matched };
}

/** Gather per-vertex attribute rows through an assignment's source map. */
export function expandVertexAttribute<T extends Float32Array | Uint32Array | Int32Array | Uint16Array | Uint8Array>(
  data: T,
  itemSize: number,
  source: Uint32Array,
): T {
  const out = new (data.constructor as new (n: number) => T)(source.length * itemSize);
  for (let v = 0; v < source.length; v++)
    for (let a = 0; a < itemSize; a++) out[v * itemSize + a] = data[source[v] * itemSize + a];
  return out;
}
