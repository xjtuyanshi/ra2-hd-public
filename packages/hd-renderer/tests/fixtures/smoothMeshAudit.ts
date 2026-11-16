import type { SmoothMeshData } from '../../src/ra2/voxelSmooth';

/** Weld duplicated material vertices for topology only; zero tolerance means exact Float32 equality. */
export function auditSmoothMesh(mesh: SmoothMeshData, supersample: number, tolerance = 0) {
  const p = mesh.vpos;
  const points = new Map<string, number>();
  const vertex = Array.from({ length: p.length / 3 }, (_, i) => {
    const key = Array.from(p.subarray(i * 3, i * 3 + 3), (x) => (tolerance ? Math.round(x / tolerance) : x)).join(',');
    if (!points.has(key)) points.set(key, points.size);
    return points.get(key)!;
  });
  const edges = new Map<string, { uses: number; direction: number }>();
  let degenerateTriangles = 0;
  let shadingOpposedTriangles = 0;
  const cross = (positions: Float32Array, a: number, b: number, c: number) => {
    const u = [0, 1, 2].map((j) => positions[b * 3 + j]! - positions[a * 3 + j]!);
    const v = [0, 1, 2].map((j) => positions[c * 3 + j]! - positions[a * 3 + j]!);
    return [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
  };
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const t = Array.from(mesh.indices.subarray(i, i + 3));
    if (Math.hypot(...cross(p, t[0]!, t[1]!, t[2]!)) < 1e-10) degenerateTriangles++;
    const n = cross(mesh.position, t[0]!, t[1]!, t[2]!);
    if (n.reduce((sum, x, j) => sum + x * t.reduce((s, v) => s + mesh.normal[v * 3 + j]!, 0), 0) < 0)
      shadingOpposedTriangles++;
    for (let j = 0; j < 3; j++) {
      const a = vertex[t[j]!]!;
      const b = vertex[t[(j + 1) % 3]!]!;
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      const edge = edges.get(key) ?? { uses: 0, direction: 0 };
      edge.uses++;
      edge.direction += a < b ? 1 : -1;
      edges.set(key, edge);
    }
  }
  let invertedTriangles = 0;
  let ambiguousQuads = 0;
  for (let q = 0; q < p.length / 12; q++) {
    // The primal-edge axis is the one lattice-cell coordinate shared by all four dual vertices.
    const axes = [0, 1, 2].filter(
      (axis) => new Set([0, 1, 2, 3].map((j) => Math.floor(p[(q * 4 + j) * 3 + axis]! * supersample))).size === 1,
    );
    if (axes.length !== 1) {
      ambiguousQuads++;
      continue;
    }
    const axis = axes[0]!;
    const components = [0, 3].map((offset) => {
      const i = q * 6 + offset;
      return cross(p, mesh.indices[i]!, mesh.indices[i + 1]!, mesh.indices[i + 2]!)[axis]!;
    });
    const sign = Math.sign(components[0]! + components[1]!);
    invertedTriangles += components.filter((n) => n * sign < 0).length;
  }
  return {
    triangles: mesh.indices.length / 3,
    boundaryEdges: [...edges.values()].filter((e) => e.uses === 1).length,
    nonManifoldEdges: [...edges.values()].filter((e) => e.uses > 2).length,
    windingConflicts: [...edges.values()].filter((e) => e.uses === 2 && e.direction !== 0).length,
    degenerateTriangles,
    invertedTriangles,
    ambiguousQuads,
    // Authored shading normals are not a reliable winding oracle; keep this diagnostic separate.
    shadingOpposedTriangles,
  };
}

/** Orthographic silhouette area, triangle union at a fixed global pixel lattice (no GPU, no depth shader). */
export function projectedMeshArea(mesh: SmoothMeshData, yaw: number, pixelsPerWorldUnit: number): number {
  const p = mesh.position;
  const c = Math.cos(yaw),
    s = Math.sin(yaw);
  const projected = Array.from({ length: p.length / 3 }, (_, i) => {
    const x = c * p[i * 3]! + s * p[i * 3 + 2]!;
    const z = -s * p[i * 3]! + c * p[i * 3 + 2]!;
    return [
      (x - z) * Math.SQRT1_2 * pixelsPerWorldUnit,
      ((x + z) * Math.SQRT1_2 * 0.5 - (p[i * 3 + 1]! * Math.sqrt(3)) / 2) * pixelsPerWorldUnit,
    ];
  });
  const xmin = Math.floor(projected.reduce((n, v) => Math.min(n, v[0]!), Infinity));
  const ymin = Math.floor(projected.reduce((n, v) => Math.min(n, v[1]!), Infinity));
  const width = Math.ceil(projected.reduce((n, v) => Math.max(n, v[0]!), -Infinity)) - xmin + 1;
  const height = Math.ceil(projected.reduce((n, v) => Math.max(n, v[1]!), -Infinity)) - ymin + 1;
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const [a, b, c] = Array.from(mesh.indices.subarray(i, i + 3), (v) => projected[v]!);
    const edge = (a: number[], b: number[], x: number, y: number) =>
      (b[0]! - a[0]!) * (y - a[1]!) - (b[1]! - a[1]!) * (x - a[0]!);
    const sign = Math.sign(edge(a!, b!, c![0]!, c![1]!));
    if (!sign) continue;
    for (let y = Math.floor(Math.min(a![1]!, b![1]!, c![1]!)); y <= Math.ceil(Math.max(a![1]!, b![1]!, c![1]!)); y++)
      for (
        let x = Math.floor(Math.min(a![0]!, b![0]!, c![0]!));
        x <= Math.ceil(Math.max(a![0]!, b![0]!, c![0]!));
        x++
      ) {
        if (
          edge(a!, b!, x + 0.5, y + 0.5) * sign >= 0 &&
          edge(b!, c!, x + 0.5, y + 0.5) * sign >= 0 &&
          edge(c!, a!, x + 0.5, y + 0.5) * sign >= 0
        )
          mask[x - xmin + (y - ymin) * width] = 1;
      }
  }
  return mask.reduce((sum, n) => sum + n, 0) / pixelsPerWorldUnit ** 2;
}
