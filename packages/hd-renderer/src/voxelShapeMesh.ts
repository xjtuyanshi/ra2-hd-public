/**
 * Shape-locked surface for a voxel grid: a dual-contouring mesh that keeps the union of cubes' silhouette and parts
 * while replacing the cube staircases of slopes and curves with planes, so a renderer can shade it as a clean solid.
 *
 * Generic mechanism (no game knowledge): input is a lattice of occupied cells with a material byte each; output is
 * an indexed triangle mesh in the grid's physical frame. The rules that make the result "shape-locked":
 *   - Topology is the exposed-face mesh of the cubes: one quad per exposed cube face, one vertex per lattice corner
 *     and solid component around that corner (two cubes that only share an edge or a corner keep separate vertices).
 *   - Every vertex is the minimiser of a QEF over the planes of its incident faces, biased to the lattice corner and
 *     clamped to `maxOffset` lattice units from it. Each mesh triangle is the image of part of a cube face under that
 *     vertex move, so the two-sided distance between the mesh and the cube surface is at most the largest move.
 *   - A face's plane keeps the cube face's own normal (flat faces stay exactly flat, box edges and corners stay
 *     exactly sharp, one-voxel parts keep their full width) unless the face belongs to a staircase: three or more
 *     parallel levels of the same face direction, one to `stairMaxRise` cells apart, fitting a plane within its
 *     geodesic neighbourhood. Such a face takes the fitted plane's normal, so slopes and curved shells become smooth.
 *     Isolated one-cell steps (raised hatches, grooves, ledges) are real detail and stay hard.
 *   - Shading normals average the plane normals of incident faces within `creaseDegrees`; across sharper (convex)
 *     edges a `bevel` share of the other side's normal tilts the vertex normal, giving a one-cell bevel highlight
 *     without moving geometry.
 * Deterministic: fixed iteration orders, no randomness, no shared state; safe in a Worker.
 */

export interface VoxelShapeOptions {
  /** Largest vertex move from its lattice corner, in lattice units (the silhouette tolerance). */
  maxOffset: number;
  /** Geodesic radius, in face steps, of the neighbourhood that decides and fits a staircase. */
  stairRadius: number;
  /** Distinct parallel levels a staircase needs (2 would also round single hatches and grooves). */
  stairLevels: number;
  /** Largest level gap, in cells, inside one staircase (taller jumps are walls). */
  stairMaxRise: number;
  /** Minimum coefficient of determination of the level plane fit. */
  stairFit: number;
  /** Fitted normals within this angle of the face axis snap to it (near-flat runs stay exactly flat). */
  snapDegrees: number;
  /** Incident faces whose plane normals differ by more than this keep separate shading normals. */
  creaseDegrees: number;
  /** Share of the far side's normal mixed into a vertex normal across a convex crease (0 = knife edge). */
  bevel: number;
  /** QEF pull towards the lattice corner, per incident face. */
  bias: number;
}

export const DEFAULT_VOXEL_SHAPE: Readonly<VoxelShapeOptions> = Object.freeze({
  maxOffset: 0.5,
  stairRadius: 6,
  stairLevels: 3,
  stairMaxRise: 2,
  stairFit: 0.85,
  snapDegrees: 10,
  creaseDegrees: 40,
  bevel: 0.35,
  bias: 0.05,
});

export interface VoxelShapeGrid {
  size: readonly [number, number, number];
  /** Occupied cells as (x, y, z, material, extra) records of `stride` bytes (default 5); `extra` is ignored. */
  cells: Uint8Array;
  stride?: number;
  /** Physical size of one cell per axis and the physical position of lattice corner (0, 0, 0). */
  step?: readonly [number, number, number];
  origin?: readonly [number, number, number];
}

export interface VoxelShapeStats {
  cells: number;
  /** Exposed cube faces = quads. */
  faces: number;
  triangles: number;
  /** Emitted vertices (four per quad, so every quad can carry its own material and normals). */
  vertices: number;
  /** Distinct (position, normal) pairs: the vertex count of a welded index buffer. */
  weldedVertices: number;
  /** Lattice-corner vertices (positions only). */
  cornerVertices: number;
  stairFaces: number;
  /** Largest and mean vertex move from the lattice corner, lattice units (bound on the mesh/cube surface distance). */
  maxOffset: number;
  meanOffset: number;
}

export interface VoxelShapeMesh {
  /** Physical positions (origin + lattice * step), 3 per vertex. */
  positions: Float32Array;
  /** Unit shading normals in the physical frame, 3 per vertex. */
  normals: Float32Array;
  /** Lattice position of the vertex, 3 per vertex (for material and detail lookups). */
  lattice: Float32Array;
  /** Material byte of the cell owning the quad, 1 per vertex. */
  material: Uint8Array;
  /** Face direction (0..5 = +x, -x, +y, -y, +z, -z) of the quad's cube face, 1 per vertex. */
  direction: Uint8Array;
  /** 1 where the quad's plane came from a staircase fit, 1 per vertex. */
  stair: Uint8Array;
  indices: Uint32Array;
  stats: VoxelShapeStats;
}

export function voxelShapeOptions(options: Partial<VoxelShapeOptions> = {}): Readonly<VoxelShapeOptions> {
  const merged = { ...DEFAULT_VOXEL_SHAPE, ...options };
  const check = (ok: boolean, name: string) => {
    if (!ok) throw new RangeError(`voxel shape option ${name} out of range`);
  };
  check(merged.maxOffset >= 0 && merged.maxOffset <= 0.5, 'maxOffset');
  check(Number.isInteger(merged.stairRadius) && merged.stairRadius >= 1 && merged.stairRadius <= 16, 'stairRadius');
  check(Number.isInteger(merged.stairLevels) && merged.stairLevels >= 2, 'stairLevels');
  check(Number.isInteger(merged.stairMaxRise) && merged.stairMaxRise >= 1, 'stairMaxRise');
  check(merged.stairFit >= 0 && merged.stairFit <= 1, 'stairFit');
  check(merged.snapDegrees >= 0 && merged.snapDegrees < 45, 'snapDegrees');
  check(merged.creaseDegrees > 0 && merged.creaseDegrees < 180, 'creaseDegrees');
  check(merged.bevel >= 0 && merged.bevel <= 1, 'bevel');
  check(merged.bias > 0, 'bias');
  return Object.freeze(merged);
}

/** Direction d: axis d >> 1, sign + for even d. */
const DIR = new Int8Array([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]);

/** For a 2x2x2 occupancy mask, the lowest octant of each octant's face-connected component (255 when empty). */
const COMPONENT = (() => {
  const table = new Uint8Array(256 * 8).fill(255);
  for (let mask = 1; mask < 256; mask++) {
    const label = new Uint8Array(8).fill(255);
    for (let o = 0; o < 8; o++) {
      if (!(mask & (1 << o)) || label[o] !== 255) continue;
      const stack = [o];
      label[o] = o;
      while (stack.length) {
        const p = stack.pop()!;
        for (const bit of [1, 2, 4]) {
          const q = p ^ bit;
          if (mask & (1 << q) && label[q] === 255) {
            label[q] = o;
            stack.push(q);
          }
        }
      }
    }
    table.set(label, mask * 8);
  }
  return table;
})();

export function buildVoxelShapeMesh(
  grid: VoxelShapeGrid,
  options: Readonly<VoxelShapeOptions> = DEFAULT_VOXEL_SHAPE,
): VoxelShapeMesh {
  const [nx, ny, nz] = grid.size;
  const stride = grid.stride ?? 5;
  const step = grid.step ?? [1, 1, 1];
  const origin = grid.origin ?? [0, 0, 0];
  const raw = grid.cells;
  const count = Math.floor(raw.length / stride);

  // Padded lattice of cell ids (-1 = empty); one empty border makes every neighbour lookup in-bounds.
  const X = nx + 2;
  const Y = ny + 2;
  const Z = nz + 2;
  const sxy = X * Y;
  const id = new Int32Array(X * Y * Z).fill(-1);
  const cellKey = new Int32Array(count);
  let cellCount = 0;
  for (let c = 0; c < count; c++) {
    const i = c * stride;
    const x = raw[i]!;
    const y = raw[i + 1]!;
    const z = raw[i + 2]!;
    if (x >= nx || y >= ny || z >= nz) throw new RangeError(`voxel cell (${x}, ${y}, ${z}) outside ${nx}x${ny}x${nz}`);
    const k = x + 1 + X * (y + 1) + sxy * (z + 1);
    // Duplicate records: the last one wins, like a painter writing the same cell twice.
    if (id[k]! < 0) cellKey[cellCount++] = k;
    id[k] = c;
  }
  const offset = (d: number) => DIR[d * 3]! + X * DIR[d * 3 + 1]! + sxy * DIR[d * 3 + 2]!;
  const OFF = Int32Array.from({ length: 6 }, (_, d) => offset(d));

  // Exposed faces, in cell order then direction order.
  const faceOf = new Int32Array(cellCount * 6).fill(-1);
  const slotOfKey = new Map<number, number>();
  for (let s = 0; s < cellCount; s++) slotOfKey.set(cellKey[s]!, s);
  const faceKeyList: number[] = [];
  const faceDirList: number[] = [];
  for (let s = 0; s < cellCount; s++) {
    const k = cellKey[s]!;
    for (let d = 0; d < 6; d++) {
      if (id[k + OFF[d]!]! >= 0) continue;
      faceOf[s * 6 + d] = faceKeyList.length;
      faceKeyList.push(k);
      faceDirList.push(d);
    }
  }
  const faces = faceKeyList.length;
  const faceKey = Int32Array.from(faceKeyList);
  const faceDir = Uint8Array.from(faceDirList);
  const coord = (k: number, axis: number) =>
    axis === 0 ? (k % X) - 1 : axis === 1 ? (Math.floor(k / X) % Y) - 1 : Math.floor(k / sxy) - 1;
  const faceAt = (k: number, d: number) => {
    const s = slotOfKey.get(k);
    return s === undefined ? -1 : faceOf[s * 6 + d]!;
  };

  // Edge neighbours on the surface (concave turn, flat continuation, convex turn, in that priority).
  const adjacency = new Int32Array(faces * 4).fill(-1);
  for (let f = 0; f < faces; f++) {
    const k = faceKey[f]!;
    const d = faceDir[f]!;
    let slot = 0;
    for (let t = 0; t < 6; t++) {
      if (t >> 1 === d >> 1) continue;
      const up = k + OFF[t]! + OFF[d]!;
      let g: number;
      if (id[up]! >= 0) g = faceAt(up, t ^ 1);
      else if (id[k + OFF[t]!]! >= 0) g = faceAt(k + OFF[t]!, d);
      else g = faceAt(k, t);
      adjacency[f * 4 + slot++] = g;
    }
  }

  // Face plane normals in lattice units; staircase faces get the fitted level plane.
  const plane = new Float32Array(faces * 3);
  const stair = new Uint8Array(faces);
  for (let f = 0; f < faces; f++) plane[f * 3 + (faceDir[f]! >> 1)] = faceDir[f]! & 1 ? -1 : 1;
  const cosSnap = Math.cos((options.snapDegrees * Math.PI) / 180);
  {
    const stamp = new Int32Array(faces).fill(-1);
    const hop = new Uint8Array(faces);
    const queue = new Int32Array(faces);
    const levels: number[] = [];
    for (let f0 = 0; f0 < faces; f0++) {
      const d0 = faceDir[f0]!;
      const a = d0 >> 1;
      const u = (a + 1) % 3;
      const w = (a + 2) % 3;
      let head = 0;
      let tail = 0;
      queue[tail++] = f0;
      stamp[f0] = f0;
      hop[f0] = 0;
      levels.length = 0;
      while (head < tail) {
        const f = queue[head++]!;
        if (faceDir[f] === d0) levels.push(coord(faceKey[f]!, a));
        if (hop[f]! >= options.stairRadius) continue;
        for (let j = 0; j < 4; j++) {
          const g = adjacency[f * 4 + j]!;
          if (g < 0 || stamp[g] === f0 || faceDir[g] === (d0 ^ 1)) continue;
          stamp[g] = f0;
          hop[g] = hop[f]! + 1;
          queue[tail++] = g;
        }
      }
      // The run of levels around this face's own level with gaps of at most stairMaxRise.
      const own = coord(faceKey[f0]!, a);
      const unique = [...new Set(levels)].sort((p, q) => p - q);
      const at = unique.indexOf(own);
      let lo = at;
      let hi = at;
      while (lo > 0 && unique[lo]! - unique[lo - 1]! <= options.stairMaxRise) lo--;
      while (hi < unique.length - 1 && unique[hi + 1]! - unique[hi]! <= options.stairMaxRise) hi++;
      if (hi - lo + 1 < options.stairLevels) continue;
      const minLevel = unique[lo]!;
      const maxLevel = unique[hi]!;
      // Weighted least squares level = c0 + c1 u + c2 w over the run's faces, nearer faces weigh more.
      let sw = 0,
        su = 0,
        sv = 0,
        sl = 0,
        suu = 0,
        suv = 0,
        svv = 0,
        sul = 0,
        svl = 0,
        sll = 0;
      for (let q = 0; q < tail; q++) {
        const f = queue[q]!;
        if (faceDir[f] !== d0) continue;
        const k = faceKey[f]!;
        const l = coord(k, a);
        if (l < minLevel || l > maxLevel) continue;
        const wt = 1 / (1 + hop[f]!);
        const pu = coord(k, u) + 0.5;
        const pv = coord(k, w) + 0.5;
        sw += wt;
        su += wt * pu;
        sv += wt * pv;
        sl += wt * l;
        suu += wt * pu * pu;
        suv += wt * pu * pv;
        svv += wt * pv * pv;
        sul += wt * pu * l;
        svl += wt * pv * l;
        sll += wt * l * l;
      }
      // Centred normal equations for the slopes.
      const mu = su / sw,
        mv = sv / sw,
        ml = sl / sw;
      const cuu = suu / sw - mu * mu,
        cuv = suv / sw - mu * mv,
        cvv = svv / sw - mv * mv;
      const cul = sul / sw - mu * ml,
        cvl = svl / sw - mv * ml,
        cll = sll / sw - ml * ml;
      const det = cuu * cvv - cuv * cuv;
      if (!(det > 1e-9) || !(cll > 1e-9)) continue;
      const bu = (cul * cvv - cvl * cuv) / det;
      const bv = (cvl * cuu - cul * cuv) / det;
      const explained = (bu * cul + bv * cvl) / cll;
      if (explained < options.stairFit) continue;
      // Level surface a = c + bu u + bv w: outward normal (+-1 along a, minus the slopes), then physical units.
      const sgn = d0 & 1 ? -1 : 1;
      const n = [0, 0, 0];
      n[a] = sgn / step[a]!;
      n[u] = (-sgn * bu) / step[u]!;
      n[w] = (-sgn * bv) / step[w]!;
      const len = Math.hypot(n[0]!, n[1]!, n[2]!);
      if (Math.abs(n[a]!) / len >= cosSnap) continue;
      stair[f0] = 1;
      // Stored back in lattice units so the QEF and the clamp share one frame.
      for (let c = 0; c < 3; c++) plane[f0 * 3 + c] = (n[c]! * step[c]!) / len;
    }
    // Risers between two staircase faces of a perpendicular direction take their mean plane.
    const riser: number[] = [];
    for (let f = 0; f < faces; f++) {
      if (stair[f]) continue;
      let m = 0;
      const n = [0, 0, 0];
      for (let j = 0; j < 4; j++) {
        const g = adjacency[f * 4 + j]!;
        if (g < 0 || stair[g] !== 1 || faceDir[g]! >> 1 === faceDir[f]! >> 1) continue;
        m++;
        for (let c = 0; c < 3; c++) n[c] += plane[g * 3 + c]!;
      }
      const axis = faceDir[f]! >> 1;
      const sgn = faceDir[f]! & 1 ? -1 : 1;
      // The mean must still face out through this face.
      if (m < 2 || n[axis]! * sgn <= 0) continue;
      const len = Math.hypot(n[0]!, n[1]!, n[2]!);
      riser.push(f, n[0]! / len, n[1]! / len, n[2]! / len);
    }
    for (let r = 0; r < riser.length; r += 4) {
      const f = riser[r]!;
      stair[f] = 2;
      plane.set([riser[r + 1]!, riser[r + 2]!, riser[r + 3]!], f * 3);
    }
  }
  const physical = (out: number[], n: ArrayLike<number>, i: number) => {
    // Lattice-unit normal to physical: divide by step, renormalise.
    const x = n[i]! / step[0]!,
      y = n[i + 1]! / step[1]!,
      z = n[i + 2]! / step[2]!;
    const len = Math.hypot(x, y, z) || 1;
    out[0] = x / len;
    out[1] = y / len;
    out[2] = z / len;
  };

  // Corner vertices: one per lattice corner and solid component; accumulate the QEF in physical units.
  const CX = nx + 1;
  const CXY = CX * (ny + 1);
  const vertexOf = new Map<number, number>();
  const qef: number[] = []; // per vertex: A (6), b (3), count
  const vertexCorner: number[] = [];
  const quadVertex = new Int32Array(faces * 4);
  const p = [0, 0, 0];
  const n = [0, 0, 0];
  for (let f = 0; f < faces; f++) {
    const k = faceKey[f]!;
    const d = faceDir[f]!;
    const a = d >> 1;
    const u = (a + 1) % 3;
    const w = (a + 2) % 3;
    const cell = [coord(k, 0), coord(k, 1), coord(k, 2)];
    const sgn = d & 1 ? -1 : 1;
    physical(n, plane, f * 3);
    // Face centre in physical units.
    const centre = [0, 1, 2].map((c) => (cell[c]! + 0.5 + (c === a ? 0.5 * sgn : 0)) * step[c]!);
    // Corners counter-clockwise seen from outside.
    const order = sgn > 0 ? [0, 1, 2, 3] : [0, 3, 2, 1];
    for (let j = 0; j < 4; j++) {
      const corner = order[j]!;
      const cu = corner === 1 || corner === 2 ? 1 : 0;
      const cw = corner >= 2 ? 1 : 0;
      const c = [0, 0, 0];
      c[a] = cell[a]! + (sgn > 0 ? 1 : 0);
      c[u] = cell[u]! + cu;
      c[w] = cell[w]! + cw;
      let mask = 0;
      for (let o = 0; o < 8; o++) {
        const kk = c[0]! + (o & 1) + X * (c[1]! + ((o >> 1) & 1)) + sxy * (c[2]! + ((o >> 2) & 1));
        if (id[kk]! >= 0) mask |= 1 << o;
      }
      const own = cell[0]! - c[0]! + 1 + 2 * (cell[1]! - c[1]! + 1) + 4 * (cell[2]! - c[2]! + 1);
      const ck = c[0]! + CX * c[1]! + CXY * c[2]!;
      const key = ck * 8 + COMPONENT[mask * 8 + own]!;
      let v = vertexOf.get(key);
      if (v === undefined) {
        v = vertexCorner.length / 3;
        vertexOf.set(key, v);
        vertexCorner.push(c[0]!, c[1]!, c[2]!);
        qef.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      }
      quadVertex[f * 4 + j] = v;
      // Plane through the face centre, relative to this corner.
      for (let e = 0; e < 3; e++) p[e] = centre[e]! - c[e]! * step[e]!;
      const dist = n[0]! * p[0]! + n[1]! * p[1]! + n[2]! * p[2]!;
      const q = v * 10;
      qef[q] += n[0]! * n[0]!;
      qef[q + 1] += n[0]! * n[1]!;
      qef[q + 2] += n[0]! * n[2]!;
      qef[q + 3] += n[1]! * n[1]!;
      qef[q + 4] += n[1]! * n[2]!;
      qef[q + 5] += n[2]! * n[2]!;
      qef[q + 6] += n[0]! * dist;
      qef[q + 7] += n[1]! * dist;
      qef[q + 8] += n[2]! * dist;
      qef[q + 9] += 1;
    }
  }
  const cornerVertices = vertexCorner.length / 3;
  const vertexLattice = new Float64Array(cornerVertices * 3);
  let maxMove = 0;
  let sumMove = 0;
  for (let v = 0; v < cornerVertices; v++) {
    const q = v * 10;
    const lambda = options.bias * qef[q + 9]!;
    const a00 = qef[q]! + lambda,
      a01 = qef[q + 1]!,
      a02 = qef[q + 2]!;
    const a11 = qef[q + 3]! + lambda,
      a12 = qef[q + 4]!,
      a22 = qef[q + 5]! + lambda;
    const b0 = qef[q + 6]!,
      b1 = qef[q + 7]!,
      b2 = qef[q + 8]!;
    const c00 = a11 * a22 - a12 * a12,
      c01 = a02 * a12 - a01 * a22,
      c02 = a01 * a12 - a02 * a11;
    const det = a00 * c00 + a01 * c01 + a02 * c02;
    const c11 = a00 * a22 - a02 * a02,
      c12 = a01 * a02 - a00 * a12,
      c22 = a00 * a11 - a01 * a01;
    let x = (c00 * b0 + c01 * b1 + c02 * b2) / det;
    let y = (c01 * b0 + c11 * b1 + c12 * b2) / det;
    let z = (c02 * b0 + c12 * b1 + c22 * b2) / det;
    // To lattice units, then clamp the move's length.
    x /= step[0]!;
    y /= step[1]!;
    z /= step[2]!;
    let move = Math.hypot(x, y, z);
    if (move > options.maxOffset) {
      const s = options.maxOffset / move;
      x *= s;
      y *= s;
      z *= s;
      move = options.maxOffset;
    }
    maxMove = Math.max(maxMove, move);
    sumMove += move;
    vertexLattice[v * 3] = vertexCorner[v * 3]! + x;
    vertexLattice[v * 3 + 1] = vertexCorner[v * 3 + 1]! + y;
    vertexLattice[v * 3 + 2] = vertexCorner[v * 3 + 2]! + z;
  }

  // Shading normals per (quad, corner): plane normals of the corner's incident faces within the crease angle,
  // plus a bevel share of those beyond it across convex edges.
  const incident: number[][] = Array.from({ length: cornerVertices }, () => []);
  for (let f = 0; f < faces; f++) for (let j = 0; j < 4; j++) incident[quadVertex[f * 4 + j]!]!.push(f);
  const cosCrease = Math.cos((options.creaseDegrees * Math.PI) / 180);
  const physicalPlane = new Float32Array(faces * 3);
  const centrePhysical = new Float32Array(faces * 3);
  for (let f = 0; f < faces; f++) {
    physical(n, plane, f * 3);
    physicalPlane.set(n, f * 3);
    const k = faceKey[f]!;
    const d = faceDir[f]!;
    for (let c = 0; c < 3; c++)
      centrePhysical[f * 3 + c] = (coord(k, c) + 0.5 + (c === d >> 1 ? 0.5 * (d & 1 ? -1 : 1) : 0)) * step[c]!;
  }

  const vertices = faces * 4;
  const positions = new Float32Array(vertices * 3);
  const normals = new Float32Array(vertices * 3);
  const lattice = new Float32Array(vertices * 3);
  const material = new Uint8Array(vertices);
  const direction = new Uint8Array(vertices);
  const stairOut = new Uint8Array(vertices);
  const indices = new Uint32Array(faces * 6);
  const welded = new Set<string>();
  let stairFaces = 0;
  for (let f = 0; f < faces; f++) {
    if (stair[f]) stairFaces++;
    const fx = physicalPlane[f * 3]!,
      fy = physicalPlane[f * 3 + 1]!,
      fz = physicalPlane[f * 3 + 2]!;
    const mat = raw[id[faceKey[f]!]! * stride + 3]!;
    for (let j = 0; j < 4; j++) {
      const v = quadVertex[f * 4 + j]!;
      let sx = 0,
        sy = 0,
        sz = 0,
        hx = 0,
        hy = 0,
        hz = 0,
        hard = 0;
      for (const g of incident[v]!) {
        const gx = physicalPlane[g * 3]!,
          gy = physicalPlane[g * 3 + 1]!,
          gz = physicalPlane[g * 3 + 2]!;
        if (fx * gx + fy * gy + fz * gz >= cosCrease) {
          sx += gx;
          sy += gy;
          sz += gz;
          continue;
        }
        // Convex when the other face's centre lies behind this face's plane.
        const ox = centrePhysical[g * 3]! - centrePhysical[f * 3]!;
        const oy = centrePhysical[g * 3 + 1]! - centrePhysical[f * 3 + 1]!;
        const oz = centrePhysical[g * 3 + 2]! - centrePhysical[f * 3 + 2]!;
        if (fx * ox + fy * oy + fz * oz >= -1e-6) continue;
        hx += gx;
        hy += gy;
        hz += gz;
        hard++;
      }
      let len = Math.hypot(sx, sy, sz) || 1;
      sx /= len;
      sy /= len;
      sz /= len;
      if (hard && options.bevel > 0) {
        len = Math.hypot(hx, hy, hz) || 1;
        sx += (options.bevel * hx) / len;
        sy += (options.bevel * hy) / len;
        sz += (options.bevel * hz) / len;
        len = Math.hypot(sx, sy, sz) || 1;
        sx /= len;
        sy /= len;
        sz /= len;
      }
      const o = f * 4 + j;
      const lx = vertexLattice[v * 3]!,
        ly = vertexLattice[v * 3 + 1]!,
        lz = vertexLattice[v * 3 + 2]!;
      lattice[o * 3] = lx;
      lattice[o * 3 + 1] = ly;
      lattice[o * 3 + 2] = lz;
      positions[o * 3] = origin[0]! + lx * step[0]!;
      positions[o * 3 + 1] = origin[1]! + ly * step[1]!;
      positions[o * 3 + 2] = origin[2]! + lz * step[2]!;
      normals[o * 3] = sx;
      normals[o * 3 + 1] = sy;
      normals[o * 3 + 2] = sz;
      material[o] = mat;
      direction[o] = faceDir[f]!;
      stairOut[o] = stair[f] ? 1 : 0;
      welded.add(`${v}:${Math.round(sx * 1e4)},${Math.round(sy * 1e4)},${Math.round(sz * 1e4)}`);
    }
    // Split along the diagonal whose two triangles are closer to coplanar (ties: 0-2).
    const b = f * 4;
    const flat = (i0: number, i1: number, i2: number, i3: number) => {
      const t0 = triangleNormal(positions, b + i0, b + i1, b + i2);
      const t1 = triangleNormal(positions, b + i0, b + i2, b + i3);
      return t0[0] * t1[0] + t0[1] * t1[1] + t0[2] * t1[2];
    };
    const alternate = flat(1, 2, 3, 0) > flat(0, 1, 2, 3) + 1e-6;
    const tri = alternate ? [1, 2, 3, 1, 3, 0] : [0, 1, 2, 0, 2, 3];
    for (let t = 0; t < 6; t++) indices[f * 6 + t] = b + tri[t]!;
  }
  return {
    positions,
    normals,
    lattice,
    material,
    direction,
    stair: stairOut,
    indices,
    stats: {
      cells: cellCount,
      faces,
      triangles: faces * 2,
      vertices,
      weldedVertices: welded.size,
      cornerVertices,
      stairFaces,
      maxOffset: maxMove,
      meanOffset: cornerVertices ? sumMove / cornerVertices : 0,
    },
  };
}

function triangleNormal(p: Float32Array, a: number, b: number, c: number): [number, number, number] {
  const ux = p[b * 3]! - p[a * 3]!,
    uy = p[b * 3 + 1]! - p[a * 3 + 1]!,
    uz = p[b * 3 + 2]! - p[a * 3 + 2]!;
  const vx = p[c * 3]! - p[a * 3]!,
    vy = p[c * 3 + 1]! - p[a * 3 + 1]!,
    vz = p[c * 3 + 2]! - p[a * 3 + 2]!;
  const x = uy * vz - uz * vy,
    y = uz * vx - ux * vz,
    z = ux * vy - uy * vx;
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}
