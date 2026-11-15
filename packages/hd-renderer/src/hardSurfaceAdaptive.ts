import { Earcut } from 'three/src/extras/Earcut.js';
import { cross, dot, sub, type HardSurfaceMesh, type Vec3 } from './hardSurfaceMesh';

export interface SurfacePaint {
  index: number;
  shade: number;
}
export type SurfaceSampler = (p: readonly number[], n: readonly number[]) => SurfacePaint;
const SHADE_ERROR = 0.5;
const AO_ERROR = 1 / 8;
const TRIANGLE_BUDGET = 2_000_000;
type Vertex = { p: Vec3; n: Vec3; u: number; v: number; paint?: SurfacePaint; ao?: number };
type Triangle = { vertices: Vertex[]; index: number; shade: number[]; ao: number[]; start: number };
export interface PaintedSurfaceMesh extends HardSurfaceMesh {
  pindex: Float32Array;
  pshade: Float32Array;
  occlusion: Float32Array;
}

function clip(polygon: Vertex[], axis: 'u' | 'v', at: number, sign: number): Vertex[] {
  const out: Vertex[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i],
      b = polygon[(i + 1) % polygon.length];
    const da = (a[axis] - at) * sign,
      db = (b[axis] - at) * sign;
    if (da >= -1e-9) out.push(a);
    if ((da > 1e-9 && db < -1e-9) || (da < -1e-9 && db > 1e-9)) {
      const t = da / (da - db),
        lerp = (x: number, y: number) => x + (y - x) * t;
      out.push({
        p: a.p.map((x, j) => lerp(x, b.p[j])) as Vec3,
        n: a.n.map((x, j) => lerp(x, b.n[j])) as Vec3,
        u: lerp(a.u, b.u),
        v: lerp(a.v, b.v),
      });
    }
  }
  return out;
}

/**
 * Sample the reference fine triangles, union identical palette regions, then triangulate their contours/holes.
 * Retain samples where interpolation differs by more than half a palette entry or one AO ray (1/8).
 * Material ownership never interpolates across a region boundary. maxEdge controls the reference sampling lattice,
 * not the size of uniform output triangles. Rectangular/cylinder strips share samples instead of repeating AO work.
 */
export function adaptiveHardSurfaceMesh(
  mesh: HardSurfaceMesh,
  maxEdge: number,
  paint: SurfaceSampler,
  occlusion: (p: Vec3, n: Vec3) => number,
  covered: (points: Vec3[], self: number, normal: Vec3) => boolean = () => false,
): PaintedSurfaceMesh {
  if (!Number.isFinite(maxEdge) || maxEdge <= 0) throw new Error('Invalid CAD edge length');
  const cell = maxEdge / Math.SQRT2,
    triangles: Triangle[] = [];

  let fineCount = 0;
  const position = (i: number) => Array.from(mesh.positions.subarray(i * 3, i * 3 + 3)) as Vec3;
  for (let start = 0; start < mesh.chamfer.length;) {
    let end = start + 1;
    while (
      end < mesh.chamfer.length &&
      mesh.face[end] === mesh.face[start] &&
      mesh.primitive[end] === mesh.primitive[start]
    )
      end++;
    const n = cross(sub(position(start + 1), position(start)), sub(position(start + 2), position(start)));
    const nl = Math.hypot(...n);
    for (let a = 0; a < 3; a++) n[a] /= nl;
    const u = cross(n, Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
    const ul = Math.hypot(...u);
    for (let a = 0; a < 3; a++) u[a] /= ul;
    const v = cross(n, u),
      polygon: Vertex[] = [];
    for (let i = start; i < end; i++) {
      const p = position(i);
      polygon.push({ p, n: Array.from(mesh.normals.subarray(i * 3, i * 3 + 3)) as Vec3, u: dot(p, u), v: dot(p, v) });
    }
    if (
      covered(
        polygon.map((p) => p.p),
        mesh.primitive[start],
        n,
      )
    ) {
      start = end;
      continue;
    }
    const xs = polygon.map((p) => p.u),
      ys = polygon.map((p) => p.v);
    const x0 = Math.floor(Math.min(...xs) / cell),
      x1 = Math.ceil(Math.max(...xs) / cell);
    const y0 = Math.floor(Math.min(...ys) / cell),
      y1 = Math.ceil(Math.max(...ys) / cell);
    const width = y1 - y0,
      count = (x1 - x0) * width;
    if (count > 4_000_000) throw new Error('CAD face exceeds subdivision budget');
    const cells = new Map<number, Triangle[]>(),
      samples = new Map<string, Vertex>();
    const vertex = (p: Vertex): Vertex => {
      const key = `${Math.round(p.u * 1e7)},${Math.round(p.v * 1e7)}`;
      const old = samples.get(key);
      if (old) return old;
      const len = Math.hypot(...p.n);
      p.n = p.n.map((x) => x / len) as Vec3;
      const fp = p.p.map(Math.fround) as Vec3,
        fn = p.n.map(Math.fround) as Vec3;
      p.paint = paint(fp, fn);
      p.ao = occlusion(fp, fn);
      samples.set(key, p);
      return p;
    };
    const rectangular =
      polygon.length === 4 &&
      polygon.every(
        (p, i) => Math.abs(p.u - polygon[(i + 1) % 4].u) < 1e-9 || Math.abs(p.v - polygon[(i + 1) % 4].v) < 1e-9,
      );
    const minU = Math.min(...xs),
      maxU = Math.max(...xs),
      minV = Math.min(...ys),
      maxV = Math.max(...ys);
    const corners = rectangular
      ? [
          [minU, minV],
          [maxU, minV],
          [maxU, maxV],
          [minU, maxV],
        ].map(([x, y]) =>
          polygon.reduce((best, p) => (Math.hypot(p.u - x, p.v - y) < Math.hypot(best.u - x, best.v - y) ? p : best)),
        )
      : [];
    const rectangularVertex = (x: number, y: number): Vertex => {
      const key = `${Math.round(x * 1e7)},${Math.round(y * 1e7)}`,
        old = samples.get(key);
      if (old) return old;
      const sx = (x - minU) / (maxU - minU),
        sy = (y - minV) / (maxV - minV),
        weights = [(1 - sx) * (1 - sy), sx * (1 - sy), sx * sy, (1 - sx) * sy];
      return vertex({
        p: [0, 1, 2].map((a) => corners.reduce((s, p, i) => s + p.p[a] * weights[i], 0)) as Vec3,
        n: [0, 1, 2].map((a) => corners.reduce((s, p, i) => s + p.n[a] * weights[i], 0)) as Vec3,
        u: x,
        v: y,
      });
    };
    const rotations = new Map<number, number>();
    const clipped = (xa: number, xb: number, ya: number, yb: number): Vertex[] => {
      if (rectangular) {
        xa = Math.max(minU, xa);
        xb = Math.min(maxU, xb);
        ya = Math.max(minV, ya);
        yb = Math.min(maxV, yb);
        if (xa >= xb - 1e-9 || ya >= yb - 1e-9) return [];
        const pattern = (xa === minU ? 1 : 0) | (xb === maxU ? 2 : 0) | (ya === minV ? 4 : 0) | (yb === maxV ? 8 : 0);
        let rotation = rotations.get(pattern);
        if (rotation === undefined) {
          // Preserve the reference clip order/diagonal using UV-only clipping, without allocating 3D interpolants.
          type UV = { u: number; v: number };
          let raw: UV[] = polygon;
          const cut = (input: UV[], axis: 'u' | 'v', at: number, sign: number) => {
            const out: UV[] = [];
            for (let i = 0; i < input.length; i++) {
              const a = input[i],
                b = input[(i + 1) % input.length],
                da = (a[axis] - at) * sign,
                db = (b[axis] - at) * sign;
              if (da >= -1e-9) out.push(a);
              if ((da > 1e-9 && db < -1e-9) || (da < -1e-9 && db > 1e-9)) {
                const t = da / (da - db);
                out.push({ u: a.u + (b.u - a.u) * t, v: a.v + (b.v - a.v) * t });
              }
            }
            return out;
          };
          raw = cut(raw, 'u', xa, 1);
          raw = cut(raw, 'u', xb, -1);
          raw = cut(raw, 'v', ya, 1);
          raw = cut(raw, 'v', yb, -1);
          const p = raw[0];
          rotation =
            Math.abs(p.u - xa) < 1e-8 ? (Math.abs(p.v - ya) < 1e-8 ? 0 : 3) : Math.abs(p.v - ya) < 1e-8 ? 1 : 2;
          rotations.set(pattern, rotation);
        }
        const coords = [
          [xa, ya],
          [xb, ya],
          [xb, yb],
          [xa, yb],
        ];
        return coords.map((_, i) => {
          const p = coords[(i + rotation!) % 4];
          return rectangularVertex(p[0], p[1]);
        });
      }
      return clip(clip(clip(clip(polygon, 'u', xa, 1), 'u', xb, -1), 'v', ya, 1), 'v', yb, -1);
    };
    for (let x = x0; x < x1; x++)
      for (let y = y0; y < y1; y++) {
        const clippedCell = clipped(x * cell, (x + 1) * cell, y * cell, (y + 1) * cell);
        if (
          !clippedCell.length ||
          covered(
            clippedCell.map((p) => p.p),
            mesh.primitive[start],
            n,
          )
        )
          continue;
        const poly = clippedCell.map(vertex);
        const ts: Triangle[] = [];
        for (let k = 1; k < poly.length - 1; k++) {
          const vertices = [poly[0], poly[k], poly[k + 1]];
          if (Math.hypot(...cross(sub(vertices[1].p, vertices[0].p), sub(vertices[2].p, vertices[0].p))) < 1e-10)
            continue;
          const centre: Vec3 = [0, 0, 0],
            normal: Vec3 = [0, 0, 0];
          for (const p of vertices)
            for (let a = 0; a < 3; a++) {
              centre[a] += Math.fround(p.p[a]) / 3;
              normal[a] += Math.fround(p.n[a]) / 3;
            }
          const owner = paint(centre, normal),
            ramp = Math.floor(owner.index / 16) * 16;
          const shade = vertices.map((p) =>
            Math.floor(p.paint!.index / 16) * 16 === ramp
              ? Math.max(ramp, Math.min(ramp + 15, p.paint!.shade))
              : owner.shade,
          );
          const ao = vertices.map((p) => p.ao!);
          ts.push({ vertices, index: owner.index, shade, ao, start });
        }
        fineCount += ts.length;
        if (fineCount > TRIANGLE_BUDGET) throw new Error('CAD mesh exceeds triangle budget');
        if (!ts.length) continue;
        const i = (x - x0) * width + y - y0;
        cells.set(i, ts);
      }
    for (const triangle of mergeRegions([...cells.values()].flat())) triangles.push(triangle);
    start = end;
  }
  return stitchSurface(mesh, triangles);
}

function mergeRegions(fine: Triangle[]): Triangle[] {
  const groups = new Map<number, Triangle[]>();
  for (const t of fine) {
    let g = groups.get(t.index);
    if (!g) {
      g = [];
      groups.set(t.index, g);
    }
    g.push(t);
  }
  const output: Triangle[] = [];
  for (const group of groups.values()) {
    const vertices: Vertex[] = [],
      ids = new Map<Vertex, number>(),
      attrs = new Map<number, { shade: number; ao: number }>();
    const id = (v: Vertex) => {
      let i = ids.get(v);
      if (i === undefined) {
        i = vertices.length;
        ids.set(v, i);
        vertices.push(v);
      }
      return i;
    };
    const edges = new Map<string, [number, number]>();
    for (const t of group)
      for (let j = 0; j < 3; j++) {
        const a = id(t.vertices[j]),
          b = id(t.vertices[(j + 1) % 3]);
        attrs.set(a, { shade: t.shade[j], ao: t.ao[j] });
        const reverse = `${b},${a}`;
        if (edges.has(reverse)) edges.delete(reverse);
        else edges.set(`${a},${b}`, [a, b]);
      }
    const next = new Map<number, Set<number>>();
    for (const [a, b] of edges.values()) {
      if (!next.has(a)) next.set(a, new Set());
      next.get(a)!.add(b);
    }
    const loops: number[][] = [];
    for (const [a, bs] of next)
      while (bs.size) {
        const loop = [a];
        let current = a,
          previous = -1;
        do {
          const options = next.get(current);
          if (!options?.size) break;
          let b = [...options][0];
          if (options.size > 1 && previous >= 0) {
            const p = vertices[previous],
              q = vertices[current];
            let best = -Infinity;
            for (const candidate of options) {
              const r = vertices[candidate];
              const angle = Math.atan2(
                (q.u - p.u) * (r.v - q.v) - (q.v - p.v) * (r.u - q.u),
                (q.u - p.u) * (r.u - q.u) + (q.v - p.v) * (r.v - q.v),
              );
              if (angle > best) {
                best = angle;
                b = candidate;
              }
            }
          }
          options.delete(b);
          previous = current;
          current = b;
          if (current !== a) loop.push(current);
        } while (current !== a && loop.length <= edges.size);
        if (current !== a) throw new Error('Open CAD paint contour');
        if (loop.length >= 3) loops.push(loop);
      }
    const simplify = (loop: number[]) => {
      let changed = true;
      while (changed && loop.length > 3) {
        changed = false;
        for (let j = 0; j < loop.length && loop.length > 3; j++) {
          const ia = loop[(j + loop.length - 1) % loop.length],
            ib = loop[j],
            ic = loop[(j + 1) % loop.length],
            a = vertices[ia],
            b = vertices[ib],
            c = vertices[ic];
          const dx = c.u - a.u,
            dy = c.v - a.v,
            t = ((b.u - a.u) * dx + (b.v - a.v) * dy) / (dx * dx + dy * dy);
          if (t <= 0 || t >= 1 || Math.abs(dx * (b.v - a.v) - dy * (b.u - a.u)) > 1e-7) continue;
          const aa = attrs.get(ia)!,
            ab = attrs.get(ib)!,
            ac = attrs.get(ic)!;
          if (
            Math.abs(ab.shade - (aa.shade + (ac.shade - aa.shade) * t)) > SHADE_ERROR ||
            Math.abs(ab.ao - (aa.ao + (ac.ao - aa.ao) * t)) > AO_ERROR
          )
            continue;
          loop.splice(j--, 1);
          changed = true;
        }
      }
      return loop;
    };
    const area = (loop: number[]) =>
      loop.reduce((sum, i, j) => {
        const a = vertices[i],
          b = vertices[loop[(j + 1) % loop.length]];
        return sum + a.u * b.v - b.u * a.v;
      }, 0) / 2;
    const outers = loops.filter((l) => area(l) > 1e-9).map(simplify),
      holes = loops.filter((l) => area(l) < -1e-9).map(simplify);
    const contains = (p: Vertex, loop: number[]) => {
      let inside = false;
      for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
        const a = vertices[loop[i]],
          b = vertices[loop[j]];
        if (a.v > p.v !== b.v > p.v && p.u < ((b.u - a.u) * (p.v - a.v)) / (b.v - a.v) + a.u) inside = !inside;
      }
      return inside;
    };
    const coarse: Triangle[] = [];
    for (const outer of outers) {
      const rings = [outer, ...holes.filter((h) => contains(vertices[h[0]], outer))],
        list = rings.flat(),
        offsets: number[] = [];
      let count = 0;
      for (let k = 0; k < rings.length; k++) {
        if (k) offsets.push(count);
        count += rings[k].length;
      }
      const indices = Earcut.triangulate(
        list.flatMap((i) => [vertices[i].u, vertices[i].v]),
        offsets,
      );
      for (let k = 0; k < indices.length; k += 3) {
        const ids = indices.slice(k, k + 3).map((i) => list[i]);
        coarse.push({
          vertices: ids.map((i) => vertices[i]),
          index: group[0].index,
          shade: ids.map((i) => attrs.get(i)!.shade),
          ao: ids.map((i) => attrs.get(i)!.ao),
          start: group[0].start,
        });
      }
    }
    // Split at the worst residual first. Sequential insertion creates long thin triangles and redundant AO vertices.
    type Sample = { p: Vertex; shade: number; ao: number };
    type Field = { triangle: Triangle; au: number; av: number; bu: number; bv: number; u: number; v: number };
    const field = (triangle: Triangle): Field => {
      const [a, b, c] = triangle.vertices,
        det = (b.v - c.v) * (a.u - c.u) + (c.u - b.u) * (a.v - c.v);
      return {
        triangle,
        au: (b.v - c.v) / det,
        av: (c.u - b.u) / det,
        bu: (c.v - a.v) / det,
        bv: (a.u - c.u) / det,
        u: c.u,
        v: c.v,
      };
    };
    const containsSample = (f: Field, p: Vertex) => {
      const du = p.u - f.u,
        dv = p.v - f.v,
        a = f.au * du + f.av * dv,
        b = f.bu * du + f.bv * dv;
      return a >= -1e-7 && b >= -1e-7 && a + b <= 1 + 1e-7;
    };
    const fields = coarse.map(field);
    const samples = coarse.map(() => [] as Sample[]);
    for (const [i, attr] of attrs) {
      const p = vertices[i];
      for (let j = 0; j < coarse.length; j++)
        if (containsSample(fields[j], p)) {
          samples[j].push({ p, ...attr });
          break;
        }
    }
    const refine = (f: Field, samples: Sample[], depth = 0): void => {
      const t = f.triangle;
      let worst: Sample | undefined,
        error = 1;
      for (const s of samples) {
        const du = s.p.u - f.u,
          dv = s.p.v - f.v,
          wa = f.au * du + f.av * dv,
          wb = f.bu * du + f.bv * dv;
        const e = Math.max(
          Math.abs(t.shade[2] + (t.shade[0] - t.shade[2]) * wa + (t.shade[1] - t.shade[2]) * wb - s.shade) /
            SHADE_ERROR,
          Math.abs(t.ao[2] + (t.ao[0] - t.ao[2]) * wa + (t.ao[1] - t.ao[2]) * wb - s.ao) / AO_ERROR,
        );
        if (e > error + 1e-6) {
          error = e;
          worst = s;
        }
      }
      if (!worst) {
        output.push(t);
        return;
      }
      if (depth > 128) throw new Error('CAD attribute refinement exceeds budget');
      for (let e = 0; e < 3; e++) {
        const f = (e + 1) % 3,
          va = t.vertices[e],
          vb = t.vertices[f],
          p = worst.p;
        if (Math.abs((vb.u - va.u) * (p.v - va.v) - (vb.v - va.v) * (p.u - va.u)) < 1e-9) continue;
        const child = {
          ...t,
          vertices: [va, vb, p],
          shade: [t.shade[e], t.shade[f], worst.shade],
          ao: [t.ao[e], t.ao[f], worst.ao],
        };
        const cf = field(child);
        refine(
          cf,
          samples.filter((s) => containsSample(cf, s.p)),
          depth + 1,
        );
      }
    };
    fields.forEach((f, i) => refine(f, samples[i]));
  }
  return output;
}

/** Split every collinear edge at its neighbours' vertices, including edges between different plates/cylinder facets.
 * Interpolate attributes on the owning triangle, so a house boundary never blends unrelated palette ramps.
 */
function stitchSurface(source: HardSurfaceMesh, triangles: Triangle[]): PaintedSurfaceMesh {
  type Line = { direction: Vec3; points: Map<number, Vec3>; sorted: [number, Vec3][] };
  const lines = new Map<string, Line>();
  const edgeLine = (a: Vec3, b: Vec3): [string, Vec3] => {
    let d = sub(b, a),
      len = Math.hypot(...d);
    d = d.map((x) => x / len) as Vec3;
    const axis = d.findIndex((x) => Math.abs(x) > 1e-5);
    if (d[axis] < 0) d = d.map((x) => -x) as Vec3;
    const moment = cross(a, d);
    return [[...d, ...moment].map((x) => Math.round(x * 1e5)).join(','), d];
  };
  const edges: Line[][] = [];
  for (const tri of triangles) {
    const row: Line[] = [];
    for (let j = 0; j < 3; j++) {
      const a = tri.vertices[j].p,
        b = tri.vertices[(j + 1) % 3].p,
        [key, direction] = edgeLine(a, b);
      let line = lines.get(key);
      if (!line) {
        line = { direction, points: new Map(), sorted: [] };
        lines.set(key, line);
      }
      for (const p of [a, b]) line.points.set(Math.round(dot(p, line.direction) * 1e7), p);
      row.push(line);
    }
    edges.push(row);
  }
  for (const line of lines.values()) line.sorted = [...line.points].sort((a, b) => a[0] - b[0]);
  const P: number[] = [],
    N: number[] = [],
    C: number[] = [],
    F: number[] = [],
    R: number[] = [],
    I: number[] = [],
    S: number[] = [],
    A: number[] = [];
  const emit = (tri: Triangle, vertices: Vertex[], shade: number[], ao: number[]) => {
    if (C.length / 3 >= TRIANGLE_BUDGET) throw new Error('CAD mesh exceeds triangle budget');
    for (let j = 0; j < 3; j++) {
      P.push(...vertices[j].p);
      N.push(...vertices[j].n);
      C.push(source.chamfer[tri.start]);
      F.push(source.face[tri.start]);
      R.push(source.primitive[tri.start]);
      I.push(tri.index);
      S.push(shade[j]);
      A.push(ao[j]);
    }
  };
  triangles.forEach((tri, ti) => {
    const poly: Vertex[] = [],
      shade: number[] = [],
      ao: number[] = [];
    for (let j = 0; j < 3; j++) {
      const a = tri.vertices[j],
        b = tri.vertices[(j + 1) % 3],
        line = edges[ti][j];
      const ta = dot(a.p, line.direction) * 1e7,
        tb = dot(b.p, line.direction) * 1e7;
      poly.push(a);
      shade.push(tri.shade[j]);
      ao.push(tri.ao[j]);
      const inner = line.sorted.filter(([t]) => t > Math.min(ta, tb) + 10 && t < Math.max(ta, tb) - 10);
      if (tb < ta) inner.reverse();
      for (const [, p] of inner) {
        if (tri.vertices.some((v) => v.p.every((x, k) => x === p[k]))) continue;
        const t = (dot(p, line.direction) * 1e7 - ta) / (tb - ta);
        const n = a.n.map((v, k) => v + (b.n[k] - v) * t) as Vec3;
        poly.push({ p, n, u: 0, v: 0 });
        shade.push(tri.shade[j] + (tri.shade[(j + 1) % 3] - tri.shade[j]) * t);
        ao.push(tri.ao[j] + (tri.ao[(j + 1) % 3] - tri.ao[j]) * t);
      }
    }
    if (poly.length === 3) {
      emit(tri, poly, shade, ao);
      return;
    }
    // Retain near-collinear boundary vertices. Squared areas avoid temporary vectors and square roots in this loop.
    const ring = poly.map((_, i) => i);
    const earArea = (j: number) => {
      const a = poly[ring[(j + ring.length - 1) % ring.length]].p,
        b = poly[ring[j]].p,
        c = poly[ring[(j + 1) % ring.length]].p;
      const bx = b[0] - a[0],
        by = b[1] - a[1],
        bz = b[2] - a[2],
        cx = c[0] - a[0],
        cy = c[1] - a[1],
        cz = c[2] - a[2];
      const nx = by * cz - bz * cy,
        ny = bz * cx - bx * cz,
        nz = bx * cy - by * cx;
      return { area: nx * nx + ny * ny + nz * nz, chord: cx * cx + cy * cy + cz * cz };
    };
    while (ring.length > 3) {
      let ear = -1,
        best = Infinity,
        fallback = -1,
        fallbackArea = Infinity;
      for (let j = 0; j < ring.length; j++) {
        const { area, chord } = earArea(j);
        if (area > 1e-20 && area < fallbackArea) {
          fallback = j;
          fallbackArea = area;
        }
        if (area > 1e-20 && area > 4e-10 * chord && area < best) {
          best = area;
          ear = j;
        }
      }
      if (ear < 0) ear = fallback;
      if (ear < 0) break;
      const ids = [ring[(ear + ring.length - 1) % ring.length], ring[ear], ring[(ear + 1) % ring.length]];
      emit(
        tri,
        ids.map((i) => poly[i]),
        ids.map((i) => shade[i]),
        ids.map((i) => ao[i]),
      );
      ring.splice(ear, 1);
    }
    if (ring.length === 3)
      emit(
        tri,
        ring.map((i) => poly[i]),
        ring.map((i) => shade[i]),
        ring.map((i) => ao[i]),
      );
  });
  return {
    positions: Float32Array.from(P),
    normals: Float32Array.from(N),
    chamfer: Uint8Array.from(C),
    face: Uint32Array.from(F),
    primitive: Uint32Array.from(R),
    indices: Uint32Array.from(C, (_, i) => i),
    pindex: Float32Array.from(I),
    pshade: Float32Array.from(S),
    occlusion: Float32Array.from(A),
  };
}
