/** Orthographic sprite projection and geometric edge extraction. Coordinate policy belongs to the caller. */
import { erodeMask, type PixelMask } from './hardSurfaceSpriteGates';
export type SpriteVector = readonly [number, number, number];
export interface SpriteCamera {
  right: SpriteVector;
  up: SpriteVector;
  towards: SpriteVector;
  origin: readonly [number, number];
  pixelsPerWorld: number;
}
const dot = (a: SpriteVector, b: SpriteVector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export function validateSpriteCamera(camera: SpriteCamera): void {
  const axes = [camera.right, camera.up, camera.towards];
  if (
    !(camera.pixelsPerWorld > 0) ||
    !Number.isFinite(camera.pixelsPerWorld) ||
    !camera.origin.every(Number.isFinite) ||
    axes.some((a) => a.length !== 3 || !a.every(Number.isFinite) || Math.abs(dot(a, a) - 1) > 1e-5) ||
    Math.abs(dot(axes[0], axes[1])) > 1e-5 ||
    Math.abs(dot(axes[0], axes[2])) > 1e-5 ||
    Math.abs(dot(axes[1], axes[2])) > 1e-5
  )
    throw new Error('Camera must have finite orthonormal axes and positive scale');
}
export function projectSpritePoint(point: SpriteVector, camera: SpriteCamera): [number, number, number] {
  return [
    camera.origin[0] + dot(point, camera.right) * camera.pixelsPerWorld,
    camera.origin[1] - dot(point, camera.up) * camera.pixelsPerWorld,
    dot(point, camera.towards),
  ];
}
export interface SpriteGeometryBuffer {
  width: number;
  height: number;
  supersampling: number;
  coverage: ArrayLike<number>;
  /** Geometric face normals (not interpolated lighting normals) and view depth, at sample resolution. */
  normals: ArrayLike<number>;
  depth: ArrayLike<number>;
  /** Optional sample-resolution IDs for the evidence table; never used to decide whether an edge exists. */
  faceIds?: ArrayLike<number>;
}
/**
 * Compare adjacent supersamples before reducing edges by OR. A depth jump must exceed both incident
 * tangent-plane prediction interval: ordinary sloping planes, connected creases and smoothly shaded
 * cylinders are not depth discontinuities. The world tolerance is numerical, not an image-distance gate.
 * The one-native-pixel inner silhouette band is excluded after reduction.
 */
export function spriteGeometryEdges(
  buffer: SpriteGeometryBuffer,
  camera: SpriteCamera,
  options: { normalAngleDegrees: number; depthJumpWorld: number },
): { interior: PixelMask; coverage: Float64Array; creases: { faces: [number, number]; pixels: number[] }[] } {
  validateSpriteCamera(camera);
  const { width: w, height: h, supersampling: ss, coverage: cov, normals: n, depth: z } = buffer;
  if (
    !Number.isInteger(w) ||
    !Number.isInteger(h) ||
    w < 1 ||
    h < 1 ||
    !Number.isInteger(ss) ||
    ss < 1 ||
    cov.length !== w * h ||
    z.length !== w * h ||
    n.length !== w * h * 3 ||
    (buffer.faceIds !== undefined && buffer.faceIds.length !== w * h) ||
    w % ss ||
    h % ss ||
    !(options.normalAngleDegrees > 0 && options.normalAngleDegrees < 180) ||
    !(options.depthJumpWorld > 0) ||
    !Number.isFinite(options.depthJumpWorld)
  )
    throw new Error('Invalid geometry buffers');
  const nw = w / ss,
    nh = h / ss,
    coverage = new Float64Array(nw * nh),
    edge = new Uint8Array(nw * nh);
  const cosine = Math.cos((options.normalAngleDegrees * Math.PI) / 180),
    step = 1 / (camera.pixelsPerWorld * ss);
  const normal = (i: number): [number, number, number] => [n[3 * i], n[3 * i + 1], n[3 * i + 2]];
  const pairs = new Map<string, { faces: [number, number]; pixels: Set<number> }>();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x,
        ni = Math.floor(y / ss) * nw + Math.floor(x / ss);
      if (!Number.isFinite(cov[i]) || cov[i] < 0 || cov[i] > 1) throw new Error('Invalid supersample coverage');
      coverage[ni] += cov[i] / (ss * ss);
      if (cov[i] < 0.5) continue;
      if (!Number.isFinite(z[i])) throw new Error('Nonfinite depth at a covered sample');
      const a = normal(i);
      if (!a.every(Number.isFinite) || Math.abs(dot(a, a) - 1) > 0.02) throw new Error('Invalid covered normal');
      for (const [j, dir] of [
        [x + 1 < w ? i + 1 : -1, 0],
        [y + 1 < h ? i + w : -1, 1],
      ]) {
        if (j < 0 || cov[j] < 0.5) continue;
        const b = normal(j),
          dz = z[j] - z[i];
        const prediction = (v: SpriteVector) => {
          const denom = dot(v, camera.towards);
          return Math.abs(denom) < 1e-5 ? NaN : (step * (dir ? dot(v, camera.up) : -dot(v, camera.right))) / denom;
        };
        const da = prediction(a),
          db = prediction(b);
        const jump =
          Number.isFinite(da) &&
          Number.isFinite(db) &&
          (dz < Math.min(da, db) - options.depthJumpWorld || dz > Math.max(da, db) + options.depthJumpWorld);
        if (dot(a, b) < cosine || jump) {
          // The boundary is at the midpoint between sample centres; put it in its containing native pixel.
          const ex = (x + 0.5 + (dir ? 0 : 0.5)) / ss,
            ey = (y + 0.5 + (dir ? 0.5 : 0)) / ss;
          const pixel = Math.min(nh - 1, Math.floor(ey)) * nw + Math.min(nw - 1, Math.floor(ex));
          edge[pixel] = 1;
          if (buffer.faceIds) {
            const faces = [buffer.faceIds[i], buffer.faceIds[j]].sort((a, b) => a - b) as [number, number];
            const key = faces.join(':');
            const pair = pairs.get(key);
            if (pair) pair.pixels.add(pixel);
            else pairs.set(key, { faces, pixels: new Set([pixel]) });
          }
        }
      }
    }
  const inside = erodeMask(
    Uint8Array.from(coverage, (v) => +(v >= 0.5)),
    nw,
    nh,
  );
  return {
    interior: Uint8Array.from(edge, (v, i) => +(!!v && !!inside[i])),
    coverage,
    creases: Array.from(pairs.values(), ({ faces, pixels }) => ({
      faces,
      pixels: [...pixels].filter((i) => inside[i]),
    })).filter((pair) => pair.pixels.length > 0),
  };
}
