import type { VoxelSection } from '../../../ra2-assets/src/manifest';
import { lookupNormal, normalTableFor } from '../../../ra2-formats/src/vxl/vxlNormals';
import { sectionNormalsMode } from '../../../ra2-assets/src/voxelNormals';
import { buildSmoothSection, smoothVoxelOptions, type SmoothMeshData } from './voxelSmooth';
import { ra2VoxelClass, VOXEL_CLASS_ID } from './voxelClasses';

/** A separate geometry policy: the existing mesher still enforces its signed voxel-hull band. */
export const VEHICLE_MESH_OPTIONS = smoothVoxelOptions({
  supersample: 3,
  shape: 'imls',
  imlsRadius: 1.1,
  bandIn: 0,
  bandOut: 0,
  storedNormals: 0,
  shadeRadius: 0,
  aoRays: 24,
});
const COS_CREASE = Math.cos((35 * Math.PI) / 180);
const RADIUS = 2.5;
type V3 = [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a: V3): V3 => {
  const d = Math.hypot(...a) || 1;
  return a.map((x) => x / d) as V3;
};
const at = (a: Float32Array, i: number): V3 => [a[i * 3]!, a[i * 3 + 1]!, a[i * 3 + 2]!];

/** Least-squares plate normal. Reject lines, curved regions and fits inconsistent with the original artist normals. */
export function vehiclePlateNormal(points: V3[], reference: V3): V3 | null {
  if (points.length < 6) return null;
  const centre = [0, 0, 0];
  for (const p of points) for (let a = 0; a < 3; a++) centre[a]! += p[a]! / points.length;
  const covariance = Array.from({ length: 3 }, () => [0, 0, 0]);
  const basis = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (const p of points)
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) covariance[a]![b]! += ((p[a]! - centre[a]!) * (p[b]! - centre[b]!)) / points.length;
  // Symmetric Jacobi eigensolver: the least-variance axis is the plane normal, in original voxel coordinates.
  for (let iteration = 0; iteration < 16; iteration++) {
    let a = 0,
      b = 1;
    for (const [i, j] of [
      [0, 2],
      [1, 2],
    ])
      if (Math.abs(covariance[i!]![j!]!) > Math.abs(covariance[a]![b]!)) {
        a = i!;
        b = j!;
      }
    if (Math.abs(covariance[a]![b]!) < 1e-9) break;
    const angle = 0.5 * Math.atan2(2 * covariance[a]![b]!, covariance[b]![b]! - covariance[a]![a]!);
    const c = Math.cos(angle),
      s = Math.sin(angle);
    for (let k = 0; k < 3; k++) {
      const x = covariance[k]![a]!,
        y = covariance[k]![b]!;
      covariance[k]![a] = c * x - s * y;
      covariance[k]![b] = s * x + c * y;
      const u = basis[k]![a]!,
        v = basis[k]![b]!;
      basis[k]![a] = c * u - s * v;
      basis[k]![b] = s * u + c * v;
    }
    for (let k = 0; k < 3; k++) {
      const x = covariance[a]![k]!,
        y = covariance[b]![k]!;
      covariance[a]![k] = c * x - s * y;
      covariance[b]![k] = s * x + c * y;
    }
  }
  const order = [0, 1, 2].sort((a, b) => covariance[a]![a]! - covariance[b]![b]!);
  const axis = order[0]!;
  if (covariance[axis]![axis]! > 0.16 || covariance[order[1]!]![order[1]!]! < 0.5) return null;
  let normal: V3 = [basis[0]![axis]!, basis[1]![axis]!, basis[2]![axis]!];
  if (dot(normal, reference) < 0) normal = normal.map((v) => -v) as V3;
  return dot(normal, reference) > 0.85 ? unit(normal) : null;
}
/** Exact faction mask and palette-ramp boundaries; no RGB inference or invented glass classification. */
export const vehicleRamp = (index: number): number => Math.floor(index / 16);
export const vehicleRegion = (index: number): number =>
  vehicleRamp(index) * 8 + VOXEL_CLASS_ID[ra2VoxelClass(index, 0, 0, 0).kind];

/**
 * Original artist normals supply the intended planes. Bilateral filtering never crosses a 35-degree crease or a
 * palette ramp; the owning house index is never filtered. Geometry positions are not changed by this pass.
 */
export function refineVehicleSection(section: VoxelSection, mesh: SmoothMeshData): SmoothMeshData {
  const raw = Uint8Array.from(atob(section.voxels), (c) => c.charCodeAt(0));
  const table = normalTableFor(sectionNormalsMode(section, raw));
  const [sx, sy, sz] = section.size;
  const cells = new Int32Array(sx * sy * sz).fill(-1);
  const key = (x: number, y: number, z: number) => x + sx * (y + sy * z);
  const normals: V3[] = [];
  const originalNormals: V3[] = [];
  const m = section.transform;
  const transformNormal = (n: V3): V3 =>
    unit([
      m[0]! * n[0] + m[1]! * n[1] + m[2]! * n[2],
      m[8]! * n[0] + m[9]! * n[1] + m[10]! * n[2],
      -(m[4]! * n[0] + m[5]! * n[1] + m[6]! * n[2]),
    ]);
  for (let j = 0; j < raw.length; j += 5) {
    cells[key(raw[j]!, raw[j + 1]!, raw[j + 2]!)] = j;
    const n = lookupNormal(table, raw[j + 4]!);
    originalNormals[j / 5] = [...n] as V3;
    normals[j / 5] = transformNormal([...n] as V3);
  }
  const exposed = new Uint8Array(raw.length / 5);
  const adjacent = (j: number): number[] => {
    const x = raw[j]!,
      y = raw[j + 1]!,
      z = raw[j + 2]!;
    return [
      [x - 1, y, z],
      [x + 1, y, z],
      [x, y - 1, z],
      [x, y + 1, z],
      [x, y, z - 1],
      [x, y, z + 1],
    ].map(([a, b, c]) =>
      a! < 0 || a! >= sx || b! < 0 || b! >= sy || c! < 0 || c! >= sz ? -1 : cells[key(a!, b!, c!)]!,
    );
  };
  for (let j = 0; j < raw.length; j += 5) exposed[j / 5] = adjacent(j).some((n) => n < 0) ? 1 : 0;
  // Only a genuinely isolated voxel may be denoised. Two-voxel hatches, diagonal rings, aligned rivets and repeated
  // track teeth are source detail, even when their exact shade indices never form a four-cell component.
  const visited = new Uint8Array(exposed.length),
    marked = new Uint8Array(exposed.length);
  for (let j = 0; j < raw.length; j += 5) {
    if (!exposed[j / 5] || visited[j / 5]) continue;
    const component = [j];
    visited[j / 5] = 1;
    for (let head = 0; head < component.length; head++)
      for (const n of adjacent(component[head]!)) {
        if (n < 0 || !exposed[n / 5] || visited[n / 5] || raw[n + 3] !== raw[j + 3]) continue;
        visited[n / 5] = 1;
        component.push(n);
      }
    if (component.length >= 2) for (const n of component) marked[n / 5] = 1;
  }
  const nearby = (j: number, radius: number): number[] => {
    const found: number[] = [];
    for (let z = Math.max(0, raw[j + 2]! - radius); z <= Math.min(sz - 1, raw[j + 2]! + radius); z++)
      for (let y = Math.max(0, raw[j + 1]! - radius); y <= Math.min(sy - 1, raw[j + 1]! + radius); y++)
        for (let x = Math.max(0, raw[j]! - radius); x <= Math.min(sx - 1, raw[j]! + radius); x++) {
          const n = cells[key(x, y, z)]!;
          if (n >= 0 && n !== j && exposed[n / 5]) found.push(n);
        }
    return found;
  };
  for (let j = 0; j < raw.length; j += 5)
    if (exposed[j / 5] && !marked[j / 5] && nearby(j, 3).some((n) => raw[n + 3] === raw[j + 3])) marked[j / 5] = 1;

  // Fit connected, near-coplanar artist regions once, independently of the alternating hull gradients on stairs.
  // Failed/curved fits retain bilateral artist normals; positions and the signed hull band are untouched.
  const planar = new Map<number, V3>();
  visited.fill(0);
  for (let j = 0; j < raw.length; j += 5) {
    if (!exposed[j / 5] || visited[j / 5]) continue;
    const component = [j];
    visited[j / 5] = 1;
    for (let head = 0; head < component.length; head++)
      for (const n of nearby(component[head]!, 1)) {
        if (
          visited[n / 5] ||
          vehicleRegion(raw[n + 3]!) !== vehicleRegion(raw[j + 3]!) ||
          dot(normals[n / 5]!, normals[j / 5]!) < 0.94
        )
          continue;
        visited[n / 5] = 1;
        component.push(n);
      }
    const fit = vehiclePlateNormal(
      component.map((n) => [raw[n]! + 0.5, raw[n + 1]! + 0.5, raw[n + 2]! + 0.5]),
      originalNormals[j / 5]!,
    );
    if (fit) for (const n of component) planar.set(n, transformNormal(fit));
  }
  const count = mesh.pindex.length;
  const cache = new Map<string, { normal: V3; shade: number }>();
  for (let i = 0; i < count; i++) {
    const p = at(mesh.vpos, i);
    const own = mesh.pindex[i]!;
    const ref = at(mesh.normal, i);
    const ck = `${p.map((v) => v.toFixed(5)).join(',')}:${own}`;
    let filtered = cache.get(ck);
    if (!filtered) {
      const samples: { index: number; normal: V3; weight: number }[] = [];
      // Pick the closest compatible stored normal before the bilateral pass. Field normals at rounded edges can
      // lie between two artist planes; choosing one region prevents averaging across that edge into a blob.
      let reference = ref;
      let plate: V3 | undefined;
      let best = Infinity;
      let ownDistance = Infinity;
      let preserveMark = false;
      for (let z = Math.max(0, Math.floor(p[2] - RADIUS)); z <= Math.min(sz - 1, p[2] + RADIUS); z++)
        for (let y = Math.max(0, Math.floor(p[1] - RADIUS)); y <= Math.min(sy - 1, p[1] + RADIUS); y++)
          for (let x = Math.max(0, Math.floor(p[0] - RADIUS)); x <= Math.min(sx - 1, p[0] + RADIUS); x++) {
            const j = cells[key(x, y, z)]!;
            if (j < 0 || !exposed[j / 5] || vehicleRegion(raw[j + 3]!) !== vehicleRegion(own)) continue;
            const d2 = (x + 0.5 - p[0]) ** 2 + (y + 0.5 - p[1]) ** 2 + (z + 0.5 - p[2]) ** 2;
            if (d2 > RADIUS * RADIUS) continue;
            if (raw[j + 3] === own && d2 < ownDistance) {
              ownDistance = d2;
              preserveMark = marked[j / 5] !== 0;
            }
            const n = normals[j / 5]!;
            if (dot(n, ref) < 0.3) continue;
            samples.push({ index: raw[j + 3]!, normal: n, weight: Math.exp(-d2 / 3) });
            if (d2 < best) {
              best = d2;
              reference = n;
              plate = planar.get(j);
            }
          }
      const sum: V3 = [0, 0, 0];
      const histogram = new Float64Array(256);
      let weight = 0;
      for (const sample of samples) {
        const similarity = dot(reference, sample.normal);
        if (similarity < COS_CREASE) continue;
        const w = sample.weight * similarity ** 8;
        for (let a = 0; a < 3; a++) sum[a] = sum[a]! + sample.normal[a]! * w;
        histogram[sample.index]! += w;
        weight += w;
      }
      let shade = own;
      if (weight && vehicleRamp(own) !== 1 && !preserveMark && Math.max(...histogram) / weight >= 0.75) {
        let cumulative = 0;
        for (let c = vehicleRamp(own) * 16; c < vehicleRamp(own) * 16 + 16; c++) {
          cumulative += histogram[c]!;
          if (cumulative >= weight / 2) {
            shade = c;
            break;
          }
        }
      }
      filtered = { normal: plate ?? (weight ? unit(sum) : ref), shade };
      cache.set(ck, filtered);
    }
    mesh.normal.set(filtered.normal, i * 3);
    mesh.pshade[i] = filtered.shade;
  }
  mesh.materialClass = Float32Array.from(mesh.pindex, (c) => VOXEL_CLASS_ID[ra2VoxelClass(c, 0, 0, 0).kind]);
  featureDistances(mesh);
  return mesh;
}

/** Edge graph on welded lattice positions, kept separate across palette regions in the actual draw mesh. */
function featureDistances(mesh: SmoothMeshData): void {
  const count = mesh.pindex.length;
  const idOf = new Map<string, number>();
  const ids = new Int32Array(count);
  const points: V3[] = [];
  for (let i = 0; i < count; i++) {
    const p = at(mesh.vpos, i);
    const k = p.map((v) => v.toFixed(5)).join(',');
    let id = idOf.get(k);
    if (id === undefined) {
      id = points.length;
      idOf.set(k, id);
      points.push(p);
    }
    ids[i] = id;
  }
  type Face = { normal: V3; centre: V3; ramp: number; shade: number };
  const edges = new Map<string, { a: number; b: number; face: Face }>();
  const bevel = new Set<number>();
  const seam = new Set<number>();
  for (let q = 0; q < count; q += 4) {
    const normal: V3 = [0, 0, 0];
    const centre: V3 = [0, 0, 0];
    for (let v = q; v < q + 4; v++)
      for (let a = 0; a < 3; a++) {
        normal[a] = normal[a]! + mesh.normal[v * 3 + a]! / 4;
        centre[a] = centre[a]! + mesh.vpos[v * 3 + a]! / 4;
      }
    const face = { normal: unit(normal), centre, ramp: vehicleRegion(mesh.pindex[q]!), shade: mesh.pshade[q]! };
    for (let e = 0; e < 4; e++) {
      const a = ids[q + e]!;
      const b = ids[q + ((e + 1) % 4)]!;
      const k = a < b ? `${a}:${b}` : `${b}:${a}`;
      const other = edges.get(k);
      if (!other) {
        edges.set(k, { a, b, face });
        continue;
      }
      const colourBreak = face.ramp !== other.face.ramp || Math.abs(face.shade - other.face.shade) >= 2;
      const crease = dot(face.normal, other.face.normal) < COS_CREASE;
      if (!crease && !colourBreak) continue;
      const delta = face.centre.map((v, i) => v - other.face.centre[i]!) as V3;
      const convex = dot(other.face.normal, delta) <= 0;
      if (crease && convex) {
        bevel.add(a);
        bevel.add(b);
      }
      if (colourBreak || (crease && !convex)) {
        seam.add(a);
        seam.add(b);
      }
    }
  }
  // Multi-source graph distances run on the surface, so a thin barrel cannot pick up a seam across empty space.
  const graph: { id: number; length: number }[][] = points.map(() => []);
  for (const { a, b } of edges.values()) {
    const length = Math.hypot(...points[a]!.map((v, i) => v - points[b]![i]!));
    graph[a]!.push({ id: b, length });
    graph[b]!.push({ id: a, length });
  }
  const distances = (sources: Set<number>) => {
    const d = new Float32Array(points.length).fill(4);
    const queue = [...sources];
    for (const i of sources) d[i] = 0;
    for (let head = 0; head < queue.length; head++) {
      const i = queue[head]!;
      for (const next of graph[i]!) {
        const distance = d[i]! + next.length;
        if (distance + 1e-5 >= d[next.id]!) continue;
        d[next.id] = distance;
        queue.push(next.id);
      }
    }
    return Float32Array.from(ids, (i) => d[i]!);
  };
  mesh.bevelDistance = distances(bevel);
  mesh.seamDistance = distances(seam);
}

export function buildVehicleSection(section: VoxelSection): SmoothMeshData {
  return refineVehicleSection(section, buildSmoothSection(section, VEHICLE_MESH_OPTIONS));
}
