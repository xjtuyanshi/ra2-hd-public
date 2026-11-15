import { axisIndex, type Axis, type Vec3 } from './hardSurfaceSolid';
export type { Axis, Vec3 } from './hardSurfaceSolid';
export type Plane = [number, number, number, number];
export interface PrimitiveOptions {
  name?: string;
  centred?: boolean;
  mirror?: boolean;
  chamfer?: number;
  /** Opaque caller-owned paint policy; geometry does not interpret it. */
  paint?: string;
  groove?: unknown;
  off?: boolean;
}
export type Primitive = PrimitiveOptions &
  (
    | { type: 'box'; min: Vec3; max: Vec3; cuts?: Plane[] }
    | { type: 'poly'; planes: Plane[] }
    | { type: 'cyl'; axis: Axis; c: [number, number]; r: number; range: [number, number]; cuts?: Plane[] }
  );
export interface PartSpec {
  mirror?: { axis: Axis; at: number };
  primitives: Primitive[];
}
export interface HardSurfaceSpec {
  units: string;
  parts: Record<string, PartSpec>;
}
export interface ExpandedPrimitive extends PrimitiveOptions {
  planes: Plane[];
  cylinder?: { axis: Vec3; centre: Vec3 };
}
export interface PolytopeFace {
  plane: number;
  vertices: number[];
  area: number;
}
export interface HardSurfaceMesh {
  positions: Float32Array;
  normals: Float32Array;
  chamfer: Uint8Array;
  primitive: Uint32Array;
  face: Uint32Array;
  indices: Uint32Array;
}
export const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a: readonly number[], b: readonly number[]): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const unit = (a: Vec3): Vec3 => scale(a, 1 / Math.hypot(...a));
export function normalisePlanes(planes: readonly Plane[]): Plane[] {
  return planes.map((p) => {
    const l = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (!l || !p.every(Number.isFinite)) throw new Error('Invalid halfspace');
    return p.map((v) => v / l) as Plane;
  });
}
/** Compensated product/add for the reference BLAS's fused accumulation near a halfspace boundary.
 * Lattice coordinates and normalised normals are small finite values (no subnormal/overflow arithmetic).
 */
function fusedMultiplyAdd(a: number, b: number, c: number): number {
  const product = a * b,
    sum = product + c,
    split = 134217729;
  const as = split * a,
    bs = split * b,
    ah = as - (as - a),
    bh = bs - (bs - b),
    al = a - ah,
    bl = b - bh;
  const productError = ah * bh - product + ah * bl + al * bh + al * bl;
  const sumError = Math.abs(product) >= Math.abs(c) ? product - sum + c : c - sum + product;
  return sum + (productError + sumError);
}
export const insidePrimitive = (p: Vec3, q: ExpandedPrimitive, tol = 0): boolean =>
  q.planes.every((n) => {
    let v = dot(n, p);
    const d = n[3] + tol;
    if (
      Math.abs(v - d) <=
      8 * Number.EPSILON * (Math.abs(n[0] * p[0]) + Math.abs(n[1] * p[1]) + Math.abs(n[2] * p[2]) + Math.abs(d))
    )
      v = fusedMultiplyAdd(p[2], n[2], fusedMultiplyAdd(p[1], n[1], p[0] * n[0]));
    return v <= d;
  });
export function insideAny(p: Vec3, prims: readonly ExpandedPrimitive[], skip = -1, tol = 0): boolean {
  return prims.some((q, i) => i !== skip && insidePrimitive(p, q, tol));
}
// NumPy rounds quantisation keys to nearest even, including negative half ties.
export function roundEven(v: number): number {
  const f = Math.floor(v),
    r = v - f;
  return r === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v);
}
export function polytopeFaces(planes: readonly Plane[]): { vertices: Vec3[]; faces: PolytopeFace[] } {
  const vertices: Vec3[] = [],
    seen = new Set<string>();
  for (let i = 0; i < planes.length; i++)
    for (let j = i + 1; j < planes.length; j++)
      for (let k = j + 1; k < planes.length; k++) {
        const a = planes[i],
          b = planes[j],
          c = planes[k],
          bc = cross(b, c),
          det = dot(a, bc);
        if (Math.abs(det) <= 1e-9) continue;
        const p = scale(add(add(scale(bc, a[3]), scale(cross(c, a), b[3])), scale(cross(a, b), c[3])), 1 / det);
        if (!planes.every((n) => dot(n, p) <= n[3] + 1e-6)) continue;
        const key = p.map((v) => roundEven(v / 1e-5)).join(',');
        if (!seen.has(key)) {
          seen.add(key);
          vertices.push(p);
        }
      }
  const faces: PolytopeFace[] = [];
  planes.forEach((pl, i) => {
    const on = vertices.flatMap((p, k) => (Math.abs(dot(pl, p) - pl[3]) < 1e-5 ? [k] : []));
    if (on.length < 3) return;
    const n = unit(pl.slice(0, 3) as Vec3),
      c = scale(
        on.reduce((s, k) => add(s, vertices[k]), [0, 0, 0] as Vec3),
        1 / on.length,
      );
    const a = unit(cross(n, Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0])),
      b = cross(n, a);
    on.sort(
      (i, j) =>
        Math.atan2(dot(sub(vertices[i], c), b), dot(sub(vertices[i], c), a)) -
          Math.atan2(dot(sub(vertices[j], c), b), dot(sub(vertices[j], c), a)) || i - j,
    );
    let sum: Vec3 = [0, 0, 0];
    for (let k = 0; k < on.length; k++)
      sum = add(sum, cross(sub(vertices[on[k]], c), sub(vertices[on[(k + 1) % on.length]], c)));
    const area = 0.5 * Math.hypot(...sum);
    if (area >= 1e-6) faces.push({ plane: i, vertices: on, area });
  });
  return { vertices, faces };
}
/** Input normals must be unit length, as in the reference build pipeline. */
export function chamferPlanes(
  planes: readonly Plane[],
  width: number,
  prims?: readonly ExpandedPrimitive[],
  self = -1,
): Plane[] {
  const { vertices, faces } = polytopeFaces(planes),
    out: Plane[] = [];
  for (let a = 0; a < faces.length; a++)
    for (let b = a + 1; b < faces.length; b++) {
      const f = faces[a],
        g = faces[b],
        shared = f.vertices.filter((k) => g.vertices.includes(k)).sort((a, b) => a - b);
      if (shared.length < 2) continue;
      const ni = planes[f.plane].slice(0, 3) as Vec3,
        nj = planes[g.plane].slice(0, 3) as Vec3,
        cos = dot(ni, nj);
      if ((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI < 20) continue;
      const e0 = vertices[shared[0]],
        e1 = vertices[shared[shared.length - 1]],
        n = unit(add(ni, nj));
      if (
        prims &&
        [0.25, 0.5, 0.75].some((t) => insideAny(add(add(e0, scale(sub(e1, e0), t)), scale(n, 0.12)), prims, self, 0.02))
      )
        continue;
      const t1 = unit(scale(sub(nj, scale(ni, cos)), -1));
      out.push([...n, dot(n, add(e0, scale(t1, width)))]);
    }
  return out;
}
export function cylinderPlanes(
  axis: Axis,
  centre: readonly [number, number],
  radius: number,
  range: readonly [number, number],
): Plane[] {
  const a = axisIndex(axis),
    [u, v] = [0, 1, 2].filter((i) => i !== a),
    out: Plane[] = [];
  const rr = radius * Math.sqrt((2 * Math.PI) / 48 / Math.sin((2 * Math.PI) / 48));
  for (let k = 0; k < 48; k++) {
    const t = (2 * Math.PI * (k + 0.5)) / 48,
      n: Vec3 = [0, 0, 0];
    n[u] = Math.cos(t);
    n[v] = Math.sin(t);
    out.push([...n, n[u] * centre[0] + n[v] * centre[1] + rr * Math.cos(Math.PI / 48)]);
  }
  const n0: Vec3 = [0, 0, 0],
    n1: Vec3 = [0, 0, 0];
  n0[a] = -1;
  n1[a] = 1;
  return [...out, [...n0, -range[0]], [...n1, range[1]]];
}
export function expand(prim: Primitive): ExpandedPrimitive {
  let planes: Plane[], cylinder: ExpandedPrimitive['cylinder'];
  if (prim.type === 'box') {
    const [x, y, z] = prim.min,
      [X, Y, Z] = prim.max;
    planes = [
      [-1, 0, 0, -x],
      [1, 0, 0, X],
      [0, -1, 0, -y],
      [0, 1, 0, Y],
      [0, 0, -1, -z],
      [0, 0, 1, Z],
      ...(prim.cuts ?? []),
    ];
  } else if (prim.type === 'poly') planes = prim.planes;
  else {
    planes = [...cylinderPlanes(prim.axis, prim.c, prim.r, prim.range), ...(prim.cuts ?? [])];
    const a = axisIndex(prim.axis),
      [u, v] = [0, 1, 2].filter((i) => i !== a),
      axis: Vec3 = [0, 0, 0],
      centre: Vec3 = [0, 0, 0];
    axis[a] = 1;
    centre[u] = prim.c[0];
    centre[v] = prim.c[1];
    cylinder = { axis, centre };
  }
  return {
    ...prim,
    planes: normalisePlanes(planes),
    cylinder,
    chamfer: prim.chamfer ?? (prim.type === 'cyl' ? 0.12 : 0.15),
  };
}
export function mirror(prim: ExpandedPrimitive, axis: Axis, at: number): ExpandedPrimitive {
  const a = axisIndex(axis),
    planes = prim.planes.map((p) => {
      const q = [...p] as Plane;
      q[a] = -q[a];
      q[3] = p[3] - 2 * at * p[a];
      return q;
    });
  let cylinder = prim.cylinder;
  if (cylinder) {
    const centre = [...cylinder.centre] as Vec3;
    centre[a] = 2 * at - centre[a];
    cylinder = { axis: [...cylinder.axis], centre };
  }
  return { ...prim, name: (prim.name ?? '') + '_m', planes, cylinder };
}
export function partPrimitives(part: PartSpec): ExpandedPrimitive[] {
  const out: ExpandedPrimitive[] = [];
  for (const p of part.primitives) {
    if (p.off) continue;
    const q = expand(p);
    out.push(q);
    if (part.mirror && p.mirror !== false && !p.centred) out.push(mirror(q, part.mirror.axis, part.mirror.at));
  }
  return out;
}
export function buildPoly(prim: ExpandedPrimitive, prims?: readonly ExpandedPrimitive[], self = 0): HardSurfaceMesh {
  const base = prim.planes,
    planes = [...base, ...((prim.chamfer ?? 0.15) > 0 ? chamferPlanes(base, prim.chamfer ?? 0.15, prims, self) : [])];
  const { vertices, faces } = polytopeFaces(planes);
  const P: number[] = [],
    N: number[] = [],
    C: number[] = [],
    F: number[] = [],
    I: number[] = [];
  faces.forEach((f, fi) => {
    const start = P.length / 3,
      ch = f.plane >= base.length,
      n = planes[f.plane].slice(0, 3) as Vec3;
    for (const k of f.vertices) {
      const p = vertices[k];
      let nv = n;
      if (prim.cylinder && !ch && Math.abs(dot(n, prim.cylinder.axis)) < 0.2) {
        const r = sub(p, prim.cylinder.centre),
          rad = sub(r, scale(prim.cylinder.axis, dot(r, prim.cylinder.axis)));
        nv = scale(rad, 1 / (Math.hypot(...rad) + 1e-12));
      }
      P.push(...p);
      N.push(...nv);
      C.push(ch ? 1 : 0);
      F.push(fi + 1000 * self);
    }
    for (let k = 1; k < f.vertices.length - 1; k++) I.push(start, start + k, start + k + 1);
  });
  return {
    positions: new Float32Array(P),
    normals: new Float32Array(N),
    chamfer: new Uint8Array(C),
    primitive: new Uint32Array(C.length).fill(self),
    face: new Uint32Array(F),
    indices: new Uint32Array(I),
  };
}
export function mergeMeshes(meshes: readonly HardSurfaceMesh[]): HardSurfaceMesh {
  const count = meshes.reduce((s, m) => s + m.chamfer.length, 0),
    nidx = meshes.reduce((s, m) => s + m.indices.length, 0);
  const out: HardSurfaceMesh = {
    positions: new Float32Array(count * 3),
    normals: new Float32Array(count * 3),
    chamfer: new Uint8Array(count),
    primitive: new Uint32Array(count),
    face: new Uint32Array(count),
    indices: new Uint32Array(nidx),
  };
  let v = 0,
    t = 0;
  for (const m of meshes) {
    out.positions.set(m.positions, v * 3);
    out.normals.set(m.normals, v * 3);
    out.chamfer.set(m.chamfer, v);
    out.primitive.set(m.primitive, v);
    out.face.set(m.face, v);
    for (const i of m.indices) out.indices[t++] = i + v;
    v += m.chamfer.length;
  }
  return out;
}
export function buildPart(part: PartSpec): {
  primitives: ExpandedPrimitive[];
  meshes: HardSurfaceMesh[];
  mesh: HardSurfaceMesh;
} {
  const primitives = partPrimitives(part),
    meshes = primitives.map((p, i) => buildPoly(p, primitives, i));
  return { primitives, meshes, mesh: mergeMeshes(meshes) };
}
