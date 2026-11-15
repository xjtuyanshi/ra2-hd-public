import type { VoxelShapeMesh } from './voxelShapeMesh';

type Shape = Pick<VoxelShapeMesh, 'positions' | 'normals' | 'indices'>;
export interface VoxelShapeRelaxOptions {
  /** Caller-defined small round bodies, one boolean/byte per emitted vertex; welded copies are ORed. */
  smallVertices?: ArrayLike<number | boolean>;
  small?: number;
  stair?: number;
  bevel?: number;
  iterations?: number;
  /** Physical voxel edge length; by default the median first edge of the input triangles. */
  voxelSize?: number;
}
export interface VoxelShapeRelaxResult {
  positions: Float32Array;
  /** Unit face-averaged normals per emitted vertex, oriented like the input normals (not welded shading). */
  normals: Float32Array;
  zone: Uint8Array;
  stats: { maxDisplacementVoxels: number; pinned: number; flips: number };
}

/** NumPy's ties-to-even rounding, including negative half-integers. */
function roundEven(x: number): number {
  const lo = Math.floor(x);
  return x - lo === 0.5 ? lo + (lo % 2 === 0 ? 0 : 1) : Math.round(x);
}

export function voxelMeshEdgeSize(mesh: Shape): number {
  const e = new Float64Array(mesh.indices.length / 3);
  let count = 0;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const a = mesh.indices[t]! * 3,
      b = mesh.indices[t + 1]! * 3;
    const d = Math.hypot(
      mesh.positions[b]! - mesh.positions[a]!,
      mesh.positions[b + 1]! - mesh.positions[a + 1]!,
      mesh.positions[b + 2]! - mesh.positions[a + 2]!,
    );
    if (d > 1e-6) e[count++] = d;
  }
  const sorted = e.subarray(0, count).sort();
  return count ? (sorted[(count - 1) >> 1]! + sorted[count >> 1]!) / 2 : 1;
}

/** Unnormalised triangle cross products, or unit normals when requested. */
function faceNormals(p: ArrayLike<number>, idx: Uint32Array, unit: boolean): Float64Array {
  const out = new Float64Array(idx.length);
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t]! * 3,
      b = idx[t + 1]! * 3,
      c = idx[t + 2]! * 3;
    const ux = p[b]! - p[a]!,
      uy = p[b + 1]! - p[a + 1]!,
      uz = p[b + 2]! - p[a + 2]!;
    const vx = p[c]! - p[a]!,
      vy = p[c + 1]! - p[a + 1]!,
      vz = p[c + 2]! - p[a + 2]!;
    const x = uy * vz - uz * vy,
      y = uz * vx - ux * vz,
      z = ux * vy - uy * vx;
    const d = unit ? Math.hypot(x, y, z) + 1e-12 : 1;
    out[t] = x / d;
    out[t + 1] = y / d;
    out[t + 2] = z / d;
  }
  return out;
}

/** Shape-locked Taubin relaxation. Scratch storage is linear; iteration loops allocate no vertex objects. */
export function relaxVoxelShapeMesh(mesh: Shape, opts: VoxelShapeRelaxOptions = {}): VoxelShapeRelaxResult {
  const { positions: p, normals: nrm, indices: idx } = mesh;
  const count = p.length / 3;
  const small = opts.small ?? 0.62,
    stair = opts.stair ?? 0.35,
    bevel = opts.bevel ?? 0.2;
  const iterations = opts.iterations ?? 12,
    vox = opts.voxelSize ?? voxelMeshEdgeSize(mesh);
  if (
    ![small, stair, bevel].every((x) => Number.isFinite(x) && x >= 0) ||
    !Number.isInteger(iterations) ||
    iterations < 0 ||
    !(vox > 0 && Number.isFinite(vox))
  )
    throw new RangeError('Invalid relaxation options');
  if (opts.smallVertices && opts.smallVertices.length !== count)
    throw new RangeError('Small-vertex mask length mismatch');
  const inv = new Uint32Array(count),
    counts = new Uint32Array(count);
  const original = new Float64Array(p.length),
    nw = new Float64Array(p.length);
  const zoneW = new Uint8Array(count),
    weld = new Map<string, number>();
  let vertices = 0;
  for (let v = 0; v < count; v++) {
    const i = v * 3;
    const key = `${roundEven(p[i]! / 1e-5)},${roundEven(p[i + 1]! / 1e-5)},${roundEven(p[i + 2]! / 1e-5)}`;
    let w = weld.get(key);
    if (w === undefined) {
      w = vertices++;
      weld.set(key, w);
    }
    inv[v] = w;
    counts[w]++;
    if (opts.smallVertices?.[v]) zoneW[w] = 2;
    for (let k = 0; k < 3; k++) {
      original[w * 3 + k] += p[i + k]!;
      nw[w * 3 + k] += nrm[i + k]!;
    }
  }
  for (let w = 0; w < vertices; w++) {
    const i = w * 3,
      len = Math.hypot(nw[i]!, nw[i + 1]!, nw[i + 2]!) + 1e-12;
    for (let k = 0; k < 3; k++) {
      original[i + k] /= counts[w]!;
      nw[i + k] /= len;
    }
  }
  const tri = Uint32Array.from(idx, (v) => inv[v]!);
  const ea = new Uint32Array(idx.length),
    eb = new Uint32Array(idx.length),
    uses = new Uint32Array(idx.length);
  const edgeMap = new Map<string, number>();
  let edges = 0;
  for (let t = 0; t < tri.length; t += 3)
    for (let j = 0; j < 3; j++) {
      const u = tri[t + j]!,
        v = tri[t + ((j + 1) % 3)]!;
      const a = Math.min(u, v),
        b = Math.max(u, v),
        key = `${a},${b}`;
      let e = edgeMap.get(key);
      if (e === undefined) {
        e = edges++;
        edgeMap.set(key, e);
        ea[e] = a;
        eb[e] = b;
      }
      uses[e]++;
    }
  const pin = new Uint8Array(vertices),
    degree = new Uint32Array(vertices),
    sum = new Float64Array(vertices * 3);
  const gather = (x: Float64Array) => {
    sum.fill(0);
    for (let e = 0; e < edges; e++) {
      const a = ea[e]!,
        b = eb[e]!;
      if (a === b) continue;
      for (let k = 0; k < 3; k++) {
        sum[a * 3 + k] += x[b * 3 + k]!;
        sum[b * 3 + k] += x[a * 3 + k]!;
      }
    }
  };
  for (let e = 0; e < edges; e++) {
    const a = ea[e]!,
      b = eb[e]!;
    if (a !== b) {
      degree[a]++;
      degree[b]++;
    }
    if (uses[e]! > 2) {
      pin[a] = 1;
      pin[b] = 1;
    }
  }
  gather(original);
  const concave = new Uint8Array(vertices),
    near = new Uint8Array(vertices);
  for (let w = 0; w < vertices; w++) {
    let dot = 0;
    for (let k = 0; k < 3; k++)
      dot += (sum[w * 3 + k]! / Math.max(degree[w]!, 1) - original[w * 3 + k]!) * nw[w * 3 + k]!;
    near[w] = concave[w] = dot > 0.12 * vox ? 1 : 0;
  }
  for (let e = 0; e < edges; e++) {
    if (concave[ea[e]!]) near[eb[e]!] = 1;
    if (concave[eb[e]!]) near[ea[e]!] = 1;
  }
  const limit = new Float64Array(vertices);
  for (let w = 0; w < vertices; w++) {
    if (zoneW[w] !== 2) zoneW[w] = near[w] ? 1 : 0;
    limit[w] = (zoneW[w] === 2 ? small : zoneW[w] === 1 ? stair : bevel) * vox;
  }
  const f0 = faceNormals(original, tri, true);
  // An initially degenerate face has no orientation to compare, so pin it before any relaxation step.
  for (let t = 0; t < tri.length; t += 3) {
    if (Math.hypot(f0[t]!, f0[t + 1]!, f0[t + 2]!) <= 1e-9) for (let j = 0; j < 3; j++) pin[tri[t + j]!] = 1;
  }
  const x = original.slice(0, vertices * 3);
  let flips = 0;
  for (let attempt = 0; attempt < 4; attempt++) {
    x.set(original.subarray(0, x.length));
    for (let it = 0; it < iterations; it++)
      for (const f of [0.55, -0.58]) {
        gather(x);
        for (let w = 0; w < vertices; w++) {
          const i = w * 3,
            deg = Math.max(degree[w]!, 1);
          let dx = x[i]! + f * (sum[i]! / deg - x[i]!) - original[i]!;
          let dy = x[i + 1]! + f * (sum[i + 1]! / deg - x[i + 1]!) - original[i + 1]!;
          let dz = x[i + 2]! + f * (sum[i + 2]! / deg - x[i + 2]!) - original[i + 2]!;
          const scale = pin[w] ? 0 : Math.min(1, limit[w]! / Math.max(Math.hypot(dx, dy, dz), 1e-12));
          dx *= scale;
          dy *= scale;
          dz *= scale;
          x[i] = original[i]! + dx;
          x[i + 1] = original[i + 1]! + dy;
          x[i + 2] = original[i + 2]! + dz;
        }
      }
    const f1 = faceNormals(x, tri, true);
    flips = 0;
    for (let t = 0; t < tri.length; t += 3) {
      // Initially degenerate faces cannot define a flip; keep their vertices fixed instead.
      const valid = Math.hypot(f0[t]!, f0[t + 1]!, f0[t + 2]!) > 1e-9;
      const bad = f0[t]! * f1[t]! + f0[t + 1]! * f1[t + 1]! + f0[t + 2]! * f1[t + 2]! < 0.25;
      if (!valid || bad) {
        for (let j = 0; j < 3; j++) pin[tri[t + j]!] = 1;
        if (valid) flips++;
      }
    }
    if (!flips) break;
    // Unlike the research tool, never return a last-attempt flip: conservatively restore the input.
    if (attempt === 3) {
      x.set(original.subarray(0, x.length));
      pin.fill(1);
      flips = 0;
    }
  }
  const positions = new Float32Array(p.length),
    zone = new Uint8Array(count);
  let maxDisplacementVoxels = 0,
    pinned = 0;
  for (let w = 0; w < vertices; w++) {
    const i = w * 3;
    maxDisplacementVoxels = Math.max(
      maxDisplacementVoxels,
      Math.hypot(x[i]! - original[i]!, x[i + 1]! - original[i + 1]!, x[i + 2]! - original[i + 2]!) / vox,
    );
    pinned += pin[w]!;
  }
  for (let v = 0; v < count; v++) {
    const w = inv[v]!;
    for (let k = 0; k < 3; k++) positions[v * 3 + k] = x[w * 3 + k]!;
    zone[v] = zoneW[w]!;
  }
  const fn = faceNormals(positions, idx, true),
    vn = new Float64Array(p.length),
    normals = new Float32Array(p.length);
  for (let t = 0; t < idx.length; t += 3)
    for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) vn[idx[t + j]! * 3 + k] += fn[t + k]!;
  for (let v = 0; v < count; v++) {
    const i = v * 3,
      len = Math.hypot(vn[i]!, vn[i + 1]!, vn[i + 2]!);
    const sign = vn[i]! * nrm[i]! + vn[i + 1]! * nrm[i + 1]! + vn[i + 2]! * nrm[i + 2]! < 0 ? -1 : 1;
    for (let k = 0; k < 3; k++) normals[i + k] = len > 1e-9 ? (sign * vn[i + k]!) / len : nrm[i + k]!;
  }
  return { positions, normals, zone, stats: { maxDisplacementVoxels, pinned, flips } };
}

/** Gaussian area-weighted shading normals; the orientation gate avoids bleeding through thin plates. */
export function smoothVoxelShapeNormals(mesh: Shape, radiusVoxels = 1.8): Float32Array {
  const radius = radiusVoxels * voxelMeshEdgeSize(mesh),
    sig2 = (radius * radius) / 4;
  if (!(radius > 0)) throw new RangeError('Normal radius must be positive');
  const { positions: p, normals: n, indices: idx } = mesh;
  const fn = faceNormals(p, idx, false),
    centres = new Float64Array(idx.length);
  const cells = new Map<string, number>(),
    next = new Int32Array(idx.length / 3).fill(-1);
  for (let t = 0; t < idx.length; t += 3) {
    for (let k = 0; k < 3; k++)
      centres[t + k] = (p[idx[t]! * 3 + k]! + p[idx[t + 1]! * 3 + k]! + p[idx[t + 2]! * 3 + k]!) / 3;
    const key = `${Math.floor(centres[t]! / radius)},${Math.floor(centres[t + 1]! / radius)},${Math.floor(centres[t + 2]! / radius)}`;
    next[t / 3] = cells.get(key) ?? -1;
    cells.set(key, t / 3);
  }
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    const cx = Math.floor(p[i]! / radius),
      cy = Math.floor(p[i + 1]! / radius),
      cz = Math.floor(p[i + 2]! / radius);
    let sx = 0,
      sy = 0,
      sz = 0;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          let face = cells.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? -1;
          while (face >= 0) {
            const t = face * 3,
              len = Math.hypot(fn[t]!, fn[t + 1]!, fn[t + 2]!) + 1e-12;
            const d2 =
              (p[i]! - centres[t]!) ** 2 + (p[i + 1]! - centres[t + 1]!) ** 2 + (p[i + 2]! - centres[t + 2]!) ** 2;
            if (
              d2 < radius * radius &&
              (fn[t]! * n[i]! + fn[t + 1]! * n[i + 1]! + fn[t + 2]! * n[i + 2]!) / len > -0.05
            ) {
              const w = Math.exp(-d2 / (2 * sig2));
              sx += w * fn[t]!;
              sy += w * fn[t + 1]!;
              sz += w * fn[t + 2]!;
            }
            face = next[face]!;
          }
        }
    const len = Math.hypot(sx, sy, sz);
    out[i] = len > 1e-12 ? sx / len : n[i]!;
    out[i + 1] = len > 1e-12 ? sy / len : n[i + 1]!;
    out[i + 2] = len > 1e-12 ? sz / len : n[i + 2]!;
  }
  return out;
}
