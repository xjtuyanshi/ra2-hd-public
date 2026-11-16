/** Visible union crease lines, derived from mesh geometry rather than raster normal/face changes. */
import { projectSpritePoint, validateSpriteCamera, type SpriteCamera } from './hardSurfaceSpriteCamera';
import { erodeMask, thinSpriteEdges } from './hardSurfaceSpriteGates';
type V = [number, number, number];
export interface SpriteAnalyticTriangle {
  vertices: [V, V, V];
  face: number;
  primitive: number;
  chamfer?: boolean;
}
export interface SpriteAnalyticMesh {
  triangles: SpriteAnalyticTriangle[];
}
interface Triangle extends SpriteAnalyticTriangle {
  normal: V;
  d: number;
  screen: V[];
  min: V;
  max: V;
}
interface Candidate {
  a: V;
  b: V;
  faces: number[];
  primitives: number[];
  kind: string;
  offset: number;
}
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V, k: number): V => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: V) => Math.hypot(...a);
const midpoint = (a: V, b: V) => scale(add(a, b), 0.5);
function inside(p: V, t: Triangle, epsilon: number): boolean {
  const a = sub(t.vertices[1], t.vertices[0]),
    b = sub(t.vertices[2], t.vertices[0]),
    q = sub(p, t.vertices[0]);
  const aa = dot(a, a),
    ab = dot(a, b),
    bb = dot(b, b),
    qa = dot(q, a),
    qb = dot(q, b),
    den = aa * bb - ab * ab;
  if (Math.abs(den) < 1e-18) return false;
  const u = (bb * qa - ab * qb) / den,
    v = (aa * qb - ab * qa) / den;
  return u >= -epsilon && v >= -epsilon && u + v <= 1 + epsilon;
}
function intersection(a: Triangle, b: Triangle, epsilon: number): [V, V] | undefined {
  if (length(cross(a.normal, b.normal)) < 1e-6) return;
  const points: V[] = [];
  for (const [one, two] of [
    [a, b],
    [b, a],
  ])
    for (let i = 0; i < 3; i++) {
      const p = one.vertices[i],
        q = one.vertices[(i + 1) % 3],
        direction = sub(q, p),
        den = dot(two.normal, direction);
      if (Math.abs(den) < 1e-12) continue;
      const f = (two.d - dot(two.normal, p)) / den;
      if (f < -epsilon || f > 1 + epsilon) continue;
      const hit = add(p, scale(direction, f));
      if (inside(hit, two, epsilon) && !points.some((x) => length(sub(x, hit)) < epsilon)) points.push(hit);
    }
  let best = 0,
    result: [V, V] | undefined;
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      const d = length(sub(points[i], points[j]));
      if (d > best) {
        best = d;
        result = [points[i], points[j]];
      }
    }
  return best > epsilon ? result : undefined;
}
/**
 * Primitive crease/silhouette adjacency and cross-primitive triangle intersections supply analytic candidates.
 * Continuous ray queries clip them against the entire union. Coincident coplanar hits form one surface.
 * A chamfer strip contributes the midline between its two non-chamfer host boundaries, never two bands.
 */
export function analyticSpriteEdges(
  mesh: SpriteAnalyticMesh,
  camera: SpriteCamera,
  width: number,
  height: number,
  coverage: ArrayLike<number>,
  options: { normalAngleDegrees?: number; weldWorld?: number; visibilityStepPixels?: number } = {},
) {
  validateSpriteCamera(camera);
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    coverage.length !== width * height
  )
    throw new Error('Invalid analytic sprite raster');
  const epsilon = options.weldWorld ?? 1e-6,
    step = options.visibilityStepPixels ?? 0.125,
    angle = options.normalAngleDegrees ?? 20;
  if (!(epsilon > 0 && step > 0 && angle > 0 && angle < 180)) throw new Error('Invalid analytic edge options');
  const cosine = Math.cos((angle * Math.PI) / 180),
    towards = camera.towards,
    triangles: Triangle[] = [];
  for (const input of mesh.triangles) {
    if (input.vertices.length !== 3 || input.vertices.some((v) => v.length !== 3 || !v.every(Number.isFinite)))
      throw new Error('Invalid analytic triangle');
    const raw = cross(sub(input.vertices[1], input.vertices[0]), sub(input.vertices[2], input.vertices[0])),
      size = length(raw);
    if (size < 1e-12) continue;
    const normal = scale(raw, 1 / size);
    triangles.push({
      ...input,
      normal,
      d: dot(normal, input.vertices[0]),
      screen: input.vertices.map((v) => projectSpritePoint(v, camera)),
      min: [0, 1, 2].map((k) => Math.min(...input.vertices.map((v) => v[k]))) as V,
      max: [0, 1, 2].map((k) => Math.max(...input.vertices.map((v) => v[k]))) as V,
    });
  }
  const pointKey = (p: V) => p.map((v) => Math.round(v / epsilon)).join(','),
    edges = new Map<string, { a: V; b: V; triangles: number[] }>();
  triangles.forEach((t, i) => {
    for (let k = 0; k < 3; k++) {
      const a = t.vertices[k],
        b = t.vertices[(k + 1) % 3],
        keys = [pointKey(a), pointKey(b)].sort(),
        key = `${t.primitive}:${keys.join('|')}`;
      const edge = edges.get(key);
      if (edge) edge.triangles.push(i);
      else edges.set(key, { a, b, triangles: [i] });
    }
  });
  const candidates: Candidate[] = [],
    chamferHosts = new Map<string, { a: V; b: V; host: Triangle; bevel: Triangle }[]>();
  for (const edge of edges.values()) {
    const faces = edge.triangles.map((i) => triangles[i]);
    if (faces.length === 2 && !!faces[0].chamfer !== !!faces[1].chamfer) {
      const bevel = faces.find((t) => t.chamfer)!,
        host = faces.find((t) => !t.chamfer)!,
        key = `${bevel.primitive}:${bevel.face}`;
      const list = chamferHosts.get(key) ?? [];
      list.push({ a: edge.a, b: edge.b, host, bevel });
      chamferHosts.set(key, list);
      continue;
    }
    if (faces.some((t) => t.chamfer)) continue;
    const crease = faces.length === 1 || (faces.length === 2 && dot(faces[0].normal, faces[1].normal) < cosine);
    const silhouette = faces.length === 2 && dot(faces[0].normal, towards) > 0 !== dot(faces[1].normal, towards) > 0;
    if (crease || silhouette)
      candidates.push({
        a: edge.a,
        b: edge.b,
        faces: faces.map((t) => t.face),
        primitives: [...new Set(faces.map((t) => t.primitive))],
        kind: crease ? 'crease' : 'silhouette',
        offset: 0.15,
      });
  }
  for (const hosts of chamferHosts.values()) {
    if (hosts.length < 2) continue;
    let chosen: [(typeof hosts)[number], (typeof hosts)[number]] | undefined,
      best = -1;
    for (let i = 0; i < hosts.length; i++)
      for (let j = i + 1; j < hosts.length; j++) {
        if (dot(hosts[i].host.normal, hosts[j].host.normal) >= cosine) continue;
        const score = length(sub(hosts[i].a, hosts[i].b)) + length(sub(hosts[j].a, hosts[j].b));
        if (score > best) {
          best = score;
          chosen = [hosts[i], hosts[j]];
        }
      }
    if (!chosen) continue;
    const [a, b] = chosen;
    const same = length(sub(a.a, b.a)) + length(sub(a.b, b.b)) <= length(sub(a.a, b.b)) + length(sub(a.b, b.a));
    const start = midpoint(a.a, same ? b.a : b.b),
      end = midpoint(a.b, same ? b.b : b.a);
    const sa = projectSpritePoint(midpoint(a.a, a.b), camera),
      sb = projectSpritePoint(midpoint(b.a, b.b), camera);
    candidates.push({
      a: start,
      b: end,
      faces: [a.host.face, b.host.face],
      primitives: [a.host.primitive],
      kind: 'chamfer-midline',
      offset: Math.max(0.15, Math.hypot(sa[0] - sb[0], sa[1] - sb[1]) * 0.5 + 0.1),
    });
  }
  // Spatial buckets bound triangle-intersection work; no raster edge/normal buffer participates.
  const cell = 4 / camera.pixelsPerWorld,
    buckets = new Map<string, number[]>(),
    pairs = new Set<string>();
  triangles.forEach((t, i) => {
    const lo = t.min.map((v) => Math.floor((v - epsilon) / cell)),
      hi = t.max.map((v) => Math.floor((v + epsilon) / cell));
    for (let x = lo[0]; x <= hi[0]; x++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let z = lo[2]; z <= hi[2]; z++) {
          const key = `${x}:${y}:${z}`,
            bucket = buckets.get(key) ?? [];
          for (const j of bucket) if (t.primitive !== triangles[j].primitive) pairs.add(`${j}:${i}`);
          bucket.push(i);
          buckets.set(key, bucket);
        }
  });
  for (const pair of pairs) {
    const [i, j] = pair.split(':').map(Number),
      a = triangles[i],
      b = triangles[j];
    if (
      dot(a.normal, b.normal) >= cosine ||
      a.min.some((v, k) => v > b.max[k] + epsilon || a.max[k] < b.min[k] - epsilon)
    )
      continue;
    const hit = intersection(a, b, epsilon);
    if (hit)
      candidates.push({
        a: hit[0],
        b: hit[1],
        faces: [a.face, b.face],
        primitives: [a.primitive, b.primitive],
        kind: 'union-intersection',
        offset: 0.15,
      });
  }
  const screenBuckets = new Map<string, number[]>(),
    screenCell = 3;
  triangles.forEach((t, i) => {
    if (dot(t.normal, towards) <= 1e-9) return;
    const xs = t.screen.map((p) => p[0]),
      ys = t.screen.map((p) => p[1]);
    const x0 = Math.max(-1, Math.floor(Math.min(...xs) / screenCell)),
      x1 = Math.min(Math.ceil(width / screenCell), Math.floor(Math.max(...xs) / screenCell));
    const y0 = Math.max(-1, Math.floor(Math.min(...ys) / screenCell)),
      y1 = Math.min(Math.ceil(height / screenCell), Math.floor(Math.max(...ys) / screenCell));
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const key = `${x}:${y}`,
          list = screenBuckets.get(key) ?? [];
        list.push(i);
        screenBuckets.set(key, list);
      }
  });
  const ray = (x: number, y: number) => {
    const origin = add(
      scale(camera.right as V, (x - camera.origin[0]) / camera.pixelsPerWorld),
      scale(camera.up as V, -(y - camera.origin[1]) / camera.pixelsPerWorld),
    );
    let depth = -Infinity,
      hit: Triangle | undefined;
    for (const index of screenBuckets.get(`${Math.floor(x / screenCell)}:${Math.floor(y / screenCell)}`) ?? []) {
      const t = triangles[index],
        z = (t.d - dot(t.normal, origin)) / dot(t.normal, towards);
      if (z <= depth + epsilon * 0.1) continue;
      const p = add(origin, scale(towards as V, z));
      if (inside(p, t, epsilon * 5)) {
        depth = z;
        hit = t;
      }
    }
    return hit ? { depth, triangle: hit } : undefined;
  };
  const raw = new Uint8Array(width * height),
    sx = new Float64Array(raw.length),
    sy = new Float64Array(raw.length),
    groups = new Map<string, { faces: [number, number]; pixels: Set<number> }>();
  const segments: {
    id: number;
    kind: string;
    faces: number[];
    primitives: number[];
    world: V[];
    screen: number[][];
    visibleIntervals: number[][];
  }[] = [];
  for (const c of candidates) {
    const a = projectSpritePoint(c.a, camera),
      b = projectSpritePoint(c.b, camera),
      dx = b[0] - a[0],
      dy = b[1] - a[1],
      span = Math.hypot(dx, dy);
    if (span < 0.05) continue;
    const nx = -dy / span,
      ny = dx / span,
      normal = Math.atan2(ny, nx),
      count = Math.max(1, Math.ceil(span / step));
    const intervals: number[][] = [];
    let begin: number | undefined;
    const faces = [c.faces[0] ?? 0, c.faces[1] ?? c.faces[0] ?? 0].sort((a, b) => a - b) as [number, number],
      key = faces.join(':');
    for (let k = 0; k < count; k++) {
      const t = (k + 0.5) / count,
        x = a[0] + dx * t,
        y = a[1] + dy * t,
        z = a[2] + (b[2] - a[2]) * t;
      const centre = ray(x, y),
        left = ray(x + nx * c.offset, y + ny * c.offset),
        right = ray(x - nx * c.offset, y - ny * c.offset);
      const coplanar =
        left &&
        right &&
        dot(left.triangle.normal, right.triangle.normal) > 1 - 1e-7 &&
        Math.abs(left.triangle.d - right.triangle.d) < epsilon * 10;
      let changed = !left || !right;
      if (left && right) {
        const direction = add(scale(camera.right as V, -nx), scale(camera.up as V, ny)),
          distance = (2 * c.offset) / camera.pixelsPerWorld;
        const da = (-distance * dot(left.triangle.normal, direction)) / dot(left.triangle.normal, towards),
          db = (-distance * dot(right.triangle.normal, direction)) / dot(right.triangle.normal, towards),
          dz = right.depth - left.depth;
        changed =
          dot(left.triangle.normal, right.triangle.normal) < cosine ||
          dz < Math.min(da, db) - 1e-5 ||
          dz > Math.max(da, db) + 1e-5;
      }
      const visible = !!centre && Math.abs(centre.depth - z) < Math.max(epsilon * 20, 1e-5) && !coplanar && changed;
      if (visible) {
        if (begin === undefined) begin = k / count;
        const xx = Math.floor(x),
          yy = Math.floor(y);
        if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
          const i = yy * width + xx;
          raw[i] = 1;
          sx[i] += Math.cos(2 * normal);
          sy[i] += Math.sin(2 * normal);
          const group = groups.get(key) ?? { faces, pixels: new Set<number>() };
          group.pixels.add(i);
          groups.set(key, group);
        }
      } else if (begin !== undefined) {
        intervals.push([begin, k / count]);
        begin = undefined;
      }
    }
    if (begin !== undefined) intervals.push([begin, 1]);
    if (intervals.length)
      segments.push({
        id: segments.length + 1,
        kind: c.kind,
        faces: c.faces,
        primitives: c.primitives,
        world: [c.a, c.b],
        screen: [
          [a[0], a[1]],
          [b[0], b[1]],
        ],
        visibleIntervals: intervals,
      });
  }
  const allEdges = thinSpriteEdges(raw, width, height),
    insideMask = erodeMask(
      Uint8Array.from(coverage, (v) => +(v >= 0.5)),
      width,
      height,
    ),
    interior = Uint8Array.from(allEdges, (v, i) => +(!!v && !!insideMask[i]));
  const normals = Float64Array.from(raw, (_, i) =>
    Math.hypot(sx[i], sy[i]) > 1e-9 ? (Math.atan2(sy[i], sx[i]) / 2 + Math.PI) % Math.PI : NaN,
  );
  return {
    interior,
    allEdges,
    normals,
    segments,
    triangleCount: triangles.length,
    candidateCount: candidates.length,
    creases: [...groups.values()]
      .map((g) => ({ faces: g.faces, pixels: [...g.pixels].filter((i) => interior[i]) }))
      .filter((g) => g.pixels.length),
  };
}
