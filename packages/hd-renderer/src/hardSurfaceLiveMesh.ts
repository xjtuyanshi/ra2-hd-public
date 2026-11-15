import { adaptiveHardSurfaceMesh, type SurfaceSampler } from './hardSurfaceAdaptive';
import {
  buildPart,
  insidePrimitive,
  type HardSurfaceMesh,
  type HardSurfaceSpec,
  type PartSpec,
  type Plane,
  type Vec3,
} from './hardSurfaceMesh';
import { forCells, gridCell, gridIndex, validateGrid, type SolidGrid } from './hardSurfaceSolid';

export interface VoxelPaint extends SolidGrid {
  /** -1 means unpainted; palette index zero is valid paint. */
  index: Int16Array;
  shade: Float32Array;
}

/** Exact separable squared-distance transform, retaining source indices. Face centres shift half a cell outward. */
function nearestFaces(grid: SolidGrid, paint: VoxelPaint, axis: number, sign: number): Int32Array {
  const { size } = grid;
  let distance = new Float64Array(grid.data.length).fill(Infinity);
  let owner = new Int32Array(grid.data.length).fill(-1);
  forCells(size, (p, i) => {
    const q = [...p];
    q[axis] += sign;
    if (grid.data[i] && paint.index[i] >= 0 && !gridCell(grid, q)) {
      distance[i] = 0;
      owner[i] = i;
    }
  });
  const length = Math.max(...size);
  const envelope = new Int32Array(length),
    boundaries = new Float64Array(length + 1);
  for (let a = 0; a < 3; a++) {
    const next = new Float64Array(distance.length).fill(Infinity);
    const ids = new Int32Array(owner.length).fill(-1);
    const shift = a === axis ? sign * 0.5 : 0;
    const stride = a === 0 ? size[1] * size[2] : a === 1 ? size[2] : 1;
    forCells(size, (p, base) => {
      if (p[a]) return;
      let k = -1;
      for (let q = 0; q < size[a]; q++) {
        const fq = distance[base + q * stride];
        if (!Number.isFinite(fq)) continue;
        let cut = -Infinity;
        while (k >= 0) {
          const v = envelope[k];
          cut = (fq + (q + shift) ** 2 - distance[base + v * stride] - (v + shift) ** 2) / (2 * (q - v));
          if (cut > boundaries[k]) break;
          k--;
        }
        envelope[++k] = q;
        boundaries[k] = k === 0 ? -Infinity : cut;
        boundaries[k + 1] = Infinity;
      }
      if (k < 0) return;
      let at = 0;
      for (let q = 0; q < size[a]; q++) {
        while (at < k && boundaries[at + 1] < q) at++;
        const v = envelope[at],
          dst = base + q * stride,
          src = base + v * stride;
        next[dst] = distance[src] + (q - v - shift) ** 2;
        ids[dst] = owner[src];
      }
    });
    distance = next;
    owner = ids;
  }
  return owner;
}

/** Port of directional nearest_maps: O(6 * cells), never vertices times painted voxels. */
export function hardSurfacePaintSampler(solid: SolidGrid, paint: VoxelPaint) {
  validateGrid(solid);
  validateGrid(paint);
  if (
    solid.size.some((n, a) => n !== paint.size[a]) ||
    paint.index.length !== solid.data.length ||
    paint.shade.length !== solid.data.length
  )
    throw new Error('Paint dimensions differ from solid');
  // Match the reference's three-cell halo: surfaces beyond source bounds still query the correct nearest face.
  const padding = 3;
  const size = solid.size.map((n) => n + padding * 2) as Vec3;
  const count = size.reduce((a, b) => a * b, 1);
  const padded: VoxelPaint = {
    size,
    data: new Uint8Array(count),
    index: new Int16Array(count).fill(-1),
    shade: new Float32Array(count),
  };
  const source = new Int32Array(count).fill(-1);
  forCells(solid.size, (p, i) => {
    const q = p.map((v) => v + padding) as Vec3;
    const j = gridIndex(size, ...q);
    padded.data[j] = solid.data[i];
    padded.index[j] = paint.index[i];
    source[j] = i;
  });
  const maps = [0, 1, 2].flatMap((axis) => [
    nearestFaces(padded, padded, axis, 1),
    nearestFaces(padded, padded, axis, -1),
  ]);
  if (!paint.index.some((index) => index >= 0)) throw new Error('Empty paint grid');
  return (p: readonly number[], normal: readonly number[]): { index: number; shade: number; cell: number } => {
    let a = 0;
    for (let j = 1; j < 3; j++) if (Math.abs(normal[j]) > Math.abs(normal[a])) a = j;
    // The reference lookup is quantised at voxel centres; face-plane bias is included in the EDT above.
    const x = Math.max(0, Math.min(size[0] - 1, Math.floor(p[0]) + padding));
    const y = Math.max(0, Math.min(size[1] - 1, Math.floor(p[1]) + padding));
    const z = Math.max(0, Math.min(size[2] - 1, Math.floor(p[2]) + padding));
    const near = maps[a * 2 + (normal[a] < 0 ? 1 : 0)][(x * size[1] + y) * size[2] + z];
    if (near < 0) throw new Error('No exposed paint in CAD surface direction');
    const cell = source[near];
    return { index: paint.index[cell], shade: paint.shade[cell], cell };
  };
}

/** Clip each convex plate to an orthonormal square lattice: even clipped fan diagonals obey maxEdge. */
export function subdivideHardSurfaceMesh(mesh: HardSurfaceMesh, maxEdge = 0.5): HardSurfaceMesh {
  if (!Number.isFinite(maxEdge) || maxEdge <= 0) throw new Error('Invalid CAD edge length');
  const positions: number[] = [],
    normals: number[] = [],
    chamfer: number[] = [],
    primitive: number[] = [],
    face: number[] = [];
  type Vertex = { p: Vec3; n: Vec3; u: number; v: number };
  const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
  const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const cell = maxEdge / Math.SQRT2;
  const clip = (polygon: Vertex[], axis: 'u' | 'v', at: number, sign: number): Vertex[] => {
    const out: Vertex[] = [];
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i],
        b = polygon[(i + 1) % polygon.length],
        da = (a[axis] - at) * sign,
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
  };
  for (let start = 0; start < mesh.chamfer.length;) {
    let end = start + 1;
    while (
      end < mesh.chamfer.length &&
      mesh.face[end] === mesh.face[start] &&
      mesh.primitive[end] === mesh.primitive[start]
    )
      end++;
    const p = (i: number) => Array.from(mesh.positions.subarray(i * 3, i * 3 + 3)) as Vec3;
    const p0 = p(start),
      p1 = p(start + 1),
      p2 = p(start + 2);
    const edge = p1.map((x, a) => x - p0[a]) as Vec3,
      other = p2.map((x, a) => x - p0[a]) as Vec3;
    const normal = cross(edge, other),
      nl = Math.hypot(...normal) || 1;
    const n = normal.map((x) => x / nl) as Vec3;
    const u = cross(n, Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]),
      ul = Math.hypot(...u) || 1;
    for (let a = 0; a < 3; a++) u[a] /= ul;
    const v = cross(n, u),
      polygon: Vertex[] = [];
    for (let i = start; i < end; i++) {
      const position = p(i);
      polygon.push({
        p: position,
        n: Array.from(mesh.normals.subarray(i * 3, i * 3 + 3)) as Vec3,
        u: dot(position, u),
        v: dot(position, v),
      });
    }
    const us = polygon.map((p) => p.u),
      vs = polygon.map((p) => p.v);
    const u0 = Math.floor(Math.min(...us) / cell),
      u1 = Math.ceil(Math.max(...us) / cell),
      v0 = Math.floor(Math.min(...vs) / cell),
      v1 = Math.ceil(Math.max(...vs) / cell);
    if ((u1 - u0) * (v1 - v0) > 4_000_000) throw new Error('CAD face exceeds subdivision budget');
    for (let x = u0; x < u1; x++)
      for (let y = v0; y < v1; y++) {
        let clipped = clip(polygon, 'u', x * cell, 1);
        clipped = clip(clipped, 'u', (x + 1) * cell, -1);
        clipped = clip(clipped, 'v', y * cell, 1);
        clipped = clip(clipped, 'v', (y + 1) * cell, -1);
        for (let i = 1; i < clipped.length - 1; i++) {
          const tri = [clipped[0], clipped[i], clipped[i + 1]];
          const a = tri[1].p.map((x, j) => x - tri[0].p[j]) as Vec3,
            b = tri[2].p.map((x, j) => x - tri[0].p[j]) as Vec3;
          if (Math.hypot(...cross(a, b)) < 1e-10) continue;
          if (chamfer.length > 6_000_000) throw new Error('CAD mesh exceeds vertex budget');
          for (const vertex of tri) {
            const len = Math.hypot(...vertex.n) || 1;
            positions.push(...vertex.p);
            normals.push(...vertex.n.map((x) => x / len));
            chamfer.push(mesh.chamfer[start]);
            primitive.push(mesh.primitive[start]);
            face.push(mesh.face[start]);
          }
        }
      }
    start = end;
  }
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    chamfer: Uint8Array.from(chamfer),
    primitive: Uint32Array.from(primitive),
    face: Uint32Array.from(face),
    indices: Uint32Array.from(chamfer, (_, i) => i),
  };
}

/** Adaptive painted plates with cached deterministic hemisphere AO, including grooves and undersides. */
export function buildHardSurfaceLiveMesh(part: PartSpec, maxEdge = 0.5, paint?: SurfaceSampler) {
  const built = buildPart(part);
  const mesh = built.mesh;
  if (!mesh.positions.length) throw new Error('Empty CAD mesh');
  const origin: Vec3 = [Infinity, Infinity, Infinity],
    end: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i++) {
    const a = i % 3;
    origin[a] = Math.min(origin[a], Math.floor(mesh.positions[i]) - 1);
    end[a] = Math.max(end[a], Math.ceil(mesh.positions[i]) + 1);
  }
  const size = end.map((v, a) => v - origin[a]) as Vec3;
  if (size.reduce((a, b) => a * b, 1) > 4_000_000) throw new Error('CAD solid exceeds cell budget');
  const solid: SolidGrid = { size, data: new Uint8Array(size.reduce((a, b) => a * b, 1)) };
  const bounds = built.meshes.map((mesh) => {
    const min = [Infinity, Infinity, Infinity],
      max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i++) {
      const a = i % 3;
      min[a] = Math.min(min[a], mesh.positions[i]);
      max[a] = Math.max(max[a], mesh.positions[i]);
    }
    return { min, max };
  });
  // Rasterise each convex primitive only inside its AABB; testing the union at every cell scales with unrelated parts.
  built.primitives.forEach((prim, i) => {
    const lo = bounds[i].min.map((v, a) => Math.max(0, Math.ceil(v - origin[a] - 0.5)));
    const hi = bounds[i].max.map((v, a) => Math.min(size[a] - 1, Math.floor(v - origin[a] - 0.5)));
    for (let x = lo[0]; x <= hi[0]; x++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let z = lo[2]; z <= hi[2]; z++) {
          const index = (x * size[1] + y) * size[2] + z;
          if (
            !solid.data[index] &&
            insidePrimitive([x + origin[0] + 0.5, y + origin[1] + 0.5, z + origin[2] + 0.5], prim)
          )
            solid.data[index] = 1;
        }
  });
  // Culling must use the emitted chamfered hull, not the unchamfered AO union (which would erase exposed bevels).
  const hulls = built.meshes.map((mesh, i) => {
    const planes: Plane[] = [];
    for (let start = 0; start < mesh.chamfer.length;) {
      let end = start + 1;
      while (end < mesh.chamfer.length && mesh.face[end] === mesh.face[start]) end++;
      const p = Array.from(mesh.positions.subarray(start * 3, start * 3 + 3)),
        a = Array.from(mesh.positions.subarray((start + 1) * 3, (start + 1) * 3 + 3)),
        b = Array.from(mesh.positions.subarray((start + 2) * 3, (start + 2) * 3 + 3));
      const u = a.map((x, k) => x - p[k]),
        v = b.map((x, k) => x - p[k]),
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]],
        l = Math.hypot(...n);
      planes.push([n[0] / l, n[1] / l, n[2] / l, n.reduce((s, x, k) => s + x * p[k], 0) / l]);
      start = end;
    }
    return { ...built.primitives[i], planes };
  });
  const overlaps = bounds.map((self, i) =>
    hulls.flatMap((prim, j) =>
      i !== j && bounds[j].min.every((v, a) => v <= self.max[a] + 1e-4 && bounds[j].max[a] >= self.min[a] - 1e-4)
        ? [{ prim, bounds: bounds[j] }]
        : [],
    ),
  );
  const directions = new Map<string, Float64Array>();
  const sampleOcclusion = (p: Vec3, n: Vec3): number => {
    const key = `${Math.round(n[0] * 1e6)},${Math.round(n[1] * 1e6)},${Math.round(n[2] * 1e6)}`;
    let rays = directions.get(key);
    if (!rays) {
      rays = new Float64Array(24);
      const tl = Math.hypot(n[0], n[1]),
        t = tl > 1e-6 ? [-n[1] / tl, n[0] / tl, 0] : [1, 0, 0];
      const u = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
      for (let r = 0; r < 8; r++) {
        const z = Math.sqrt(1 - (r + 0.5) / 8),
          radius = Math.sqrt((r + 0.5) / 8),
          phi = r * 2.399963229728653;
        for (let a = 0; a < 3; a++) rays[r * 3 + a] = n[a] * z + radius * (t[a] * Math.cos(phi) + u[a] * Math.sin(phi));
      }
      directions.set(key, rays);
    }
    const px = p[0] + n[0] * 0.6 - origin[0],
      py = p[1] + n[1] * 0.6 - origin[1],
      pz = p[2] + n[2] * 0.6 - origin[2];
    let ao = 0;
    for (let r = 0; r < 24; r += 3)
      for (let l = 0.5; l < 5; l += 0.5) {
        const x = Math.floor(px + rays[r] * l),
          y = Math.floor(py + rays[r + 1] * l),
          z = Math.floor(pz + rays[r + 2] * l);
        if (
          x >= 0 &&
          y >= 0 &&
          z >= 0 &&
          x < size[0] &&
          y < size[1] &&
          z < size[2] &&
          solid.data[(x * size[1] + y) * size[2] + z]
        ) {
          ao += Math.sqrt(1 - l / 5) / 8;
          break;
        }
      }
    return ao;
  };
  const adaptive = adaptiveHardSurfaceMesh(
    mesh,
    maxEdge,
    paint ?? (() => ({ index: 0, shade: 0 })),
    sampleOcclusion,
    (points, self, n) => {
      const dx = n[0] * 1e-4,
        dy = n[1] * 1e-4,
        dz = n[2] * 1e-4;
      candidate: for (const { prim, bounds: b } of overlaps[self]) {
        for (const p of points) {
          const x = p[0] + dx,
            y = p[1] + dy,
            z = p[2] + dz;
          if (
            x < b.min[0] - 1e-6 ||
            x > b.max[0] + 1e-6 ||
            y < b.min[1] - 1e-6 ||
            y > b.max[1] + 1e-6 ||
            z < b.min[2] - 1e-6 ||
            z > b.max[2] + 1e-6
          )
            continue candidate;
          for (const plane of prim.planes)
            if (x * plane[0] + y * plane[1] + z * plane[2] > plane[3] - 1e-6) continue candidate;
        }
        return true;
      }
      return false;
    },
  );
  return { ...adaptive, lattice: adaptive.positions.slice() };
}

/** Reject malformed JSON before expansion/allocation. Paint labels are intentionally opaque metadata. */
export function parseHardSurfaceSpec(value: unknown): HardSurfaceSpec {
  const fail = (): never => {
    throw new Error('Invalid CAD spec');
  };
  const obj = (v: unknown): Record<string, unknown> =>
    v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : fail();
  const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 4096;
  const vector = (v: unknown, n: number): boolean => Array.isArray(v) && v.length === n && v.every(num);
  const planes = (v: unknown): boolean =>
    Array.isArray(v) &&
    v.length >= 1 &&
    v.length <= 128 &&
    v.every((p) => vector(p, 4) && Math.hypot(p[0], p[1], p[2]) > 0);
  const region = (q: Record<string, unknown>, carve = false): void => {
    if (q.type === 'poly') {
      if (!planes(q.planes)) fail();
    } else if (q.type === 'cyl' && !carve) {
      if (
        !['x', 'y', 'z'].includes(String(q.axis)) ||
        !vector(q.c, 2) ||
        !num(q.r) ||
        q.r <= 0 ||
        !vector(q.range, 2) ||
        (q.range as number[])[1] <= (q.range as number[])[0]
      )
        fail();
    } else if (q.type === 'box' || (carve && q.type === undefined)) {
      if (!vector(q.min, 3) || !vector(q.max, 3) || (q.min as number[]).some((v, a) => v >= (q.max as number[])[a]))
        fail();
    } else fail();
    if (q.cuts !== undefined && !planes(q.cuts)) fail();
    if (q.chamfer !== undefined && (!num(q.chamfer) || q.chamfer < 0)) fail();
    for (const key of ['mirror', 'centred', 'off']) if (q[key] !== undefined && typeof q[key] !== 'boolean') fail();
    for (const key of ['name', 'paint']) if (q[key] !== undefined && typeof q[key] !== 'string') fail();
  };
  const root = obj(value),
    parts = obj(root.parts);
  if (typeof root.units !== 'string' || !Object.keys(parts).length || Object.keys(parts).length > 16) fail();
  for (const part of Object.values(parts)) {
    const p = obj(part);
    if (p.mirror !== undefined) {
      const m = obj(p.mirror);
      if (!['x', 'y', 'z'].includes(String(m.axis)) || !num(m.at)) fail();
    }
    if (!Array.isArray(p.primitives) || !p.primitives.length || p.primitives.length > 256) fail();
    for (const prim of p.primitives as unknown[]) region(obj(prim));
    if (p.carve !== undefined) {
      if (!Array.isArray(p.carve) || p.carve.length > 32) fail();
      for (const c of p.carve as unknown[]) {
        const q = obj(c);
        region(q, true);
        if (q.keep !== undefined && (!Array.isArray(q.keep) || !q.keep.every((v) => typeof v === 'string'))) fail();
      }
    }
  }
  return value as HardSurfaceSpec;
}
