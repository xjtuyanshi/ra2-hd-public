import {
  combParallelSeams,
  hatchRingProfile,
  hoopProfile,
  isStaircaseSeam,
  panelLineProfile,
  type LookVector,
  type PanelLineOptions,
  type SeamCombOptions,
  type SeamSegment,
} from './hardSurfaceLook';
import { cross, dot, sub, unit, type Vec3 } from './hardSurfaceMesh';

export interface SeamVoxel {
  position: LookVector;
  strength: number;
  surface: boolean;
  kind: string;
}
export interface VectorSeams {
  segments: SeamSegment[];
  dots: LookVector[];
}
const DIRECTIONS: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 0],
  [1, -1, 0],
  [1, 0, 1],
  [1, 0, -1],
  [0, 1, 1],
  [0, 1, -1],
];
/** Greedy straight runs from exposed line features; ties follow input order, without using palette or game policy. */
export function vectoriseSeams(voxels: readonly SeamVoxel[], minLength = 2): VectorSeams {
  const points = new Map(
    voxels.filter((v) => v.surface && v.kind === 'line' && v.strength > 0.15).map((v) => [v.position.join(','), v]),
  );
  const used = new Set<string>(),
    segments: SeamSegment[] = [];
  while (true) {
    let best: SeamVoxel[] = [],
      bestNew = 0;
    for (const p of points.values())
      for (const d of DIRECTIONS) {
        if (points.has(p.position.map((v, a) => v - d[a]).join(','))) continue;
        const run: SeamVoxel[] = [p];
        let q = p.position;
        while (true) {
          q = q.map((v, a) => v + d[a]) as Vec3;
          const next = points.get(q.join(','));
          if (!next) break;
          run.push(next);
        }
        const fresh = run.filter((v) => !used.has(v.position.join(','))).length;
        if (
          run.length >= minLength &&
          fresh >= 2 &&
          (fresh > bestNew || (fresh === bestNew && run.length > best.length))
        ) {
          best = run;
          bestNew = fresh;
        }
      }
    if (!best.length) break;
    for (const p of best) used.add(p.position.join(','));
    const centre = (p: SeamVoxel): Vec3 => p.position.map((v) => v + 0.5) as Vec3;
    segments.push({
      start: centre(best[0]),
      end: centre(best.at(-1)!),
      strength: best.reduce((s, p) => s + p.strength, 0) / best.length,
    });
  }
  return {
    segments,
    dots: [...points.values()]
      .filter((p) => !used.has(p.position.join(',')))
      .map((p) => p.position.map((v) => v + 0.5) as Vec3),
  };
}

/** Cylinder metadata is evaluated only while baking; the shader never interprets primitive geometry. */
export interface LookCylinder {
  centre: Vec3;
  axis: Vec3;
  range: [number, number];
  profile: 'wheel' | 'hoop' | 'ring';
}
export interface LookVolume {
  origin: Vec3;
  size: Vec3;
  res: number;
  data: Uint8Array;
  /** Unique texels touched by the surface rasteriser, including zero-valued ones. */
  surfaceTexels: number;
  grooveTexels: number;
  seams: number;
  dots: number;
}
export interface LookVolumeMesh {
  lattice: Float32Array;
  indices: Uint16Array | Uint32Array;
  primitive: Uint32Array;
}
export interface LookVolumeOptions {
  res?: number;
  lines: PanelLineOptions;
  comb: SeamCombOptions;
  seams: VectorSeams;
  /** Indexed by the same expanded primitive id as the mesh. */
  cylinders?: ReadonlyMap<number, LookCylinder>;
}
const clamp = (v: number) => Math.max(0, Math.min(1, v));
/** RGBA8, x-fastest (GPU texture order). Bounds enclose the part with one lattice voxel of padding. */
export function bakeLookVolume(mesh: LookVolumeMesh, options: LookVolumeOptions): LookVolume {
  const res = options.res ?? 4;
  if (!Number.isInteger(res) || res < 1 || res > 16 || !mesh.lattice.length)
    throw new Error('Invalid look volume resolution/mesh');
  const origin: Vec3 = [Infinity, Infinity, Infinity],
    end: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.lattice.length; i++) {
    const a = i % 3;
    origin[a] = Math.min(origin[a], Math.floor(mesh.lattice[i]) - 1);
    end[a] = Math.max(end[a], Math.ceil(mesh.lattice[i]) + 1);
  }
  const size = end.map((v, a) => Math.ceil((v - origin[a]) * res)) as Vec3;
  const count = size[0] * size[1] * size[2];
  if (count > 16_000_000) throw new Error('Look volume exceeds texel budget');
  const data = new Uint8Array(count * 4),
    touched = new Uint8Array(count);
  const seams = combParallelSeams(options.seams.segments, options.comb).map((s) => {
    const delta = sub([...s.end], [...s.start]),
      len = Math.hypot(...delta),
      u = delta.map((v) => v / len) as Vec3;
    return { ...s, u, start: s.start.map((v, a) => v - u[a] * 0.35) as Vec3, len: len + 0.7 };
  });
  const vertex = (i: number): Vec3 => [mesh.lattice[i * 3], mesh.lattice[i * 3 + 1], mesh.lattice[i * 3 + 2]];
  const tolerance = 0.75 / res;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const ids = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]],
      [a, b, c] = ids.map(vertex);
    const ab = sub(b, a),
      ac = sub(c, a),
      normal = cross(ab, ac);
    if (Math.hypot(...normal) < 1e-10) continue;
    const n = unit(normal),
      d00 = dot(ab, ab),
      d01 = dot(ab, ac),
      d11 = dot(ac, ac),
      det = d00 * d11 - d01 * d01;
    const lo = a.map((v, k) =>
      Math.max(0, Math.floor((Math.min(v, b[k], c[k]) - tolerance - origin[k]) * res)),
    ) as Vec3;
    const hi = a.map((v, k) =>
      Math.min(size[k] - 1, Math.floor((Math.max(v, b[k], c[k]) + tolerance - origin[k]) * res)),
    ) as Vec3;
    const nearSeams = seams.filter(
      (s) =>
        !isStaircaseSeam(s.u, n) &&
        a.every(
          (_, k) =>
            Math.max(s.start[k], s.start[k] + s.u[k] * s.len) + 2 > origin[k] + lo[k] / res &&
            Math.min(s.start[k], s.start[k] + s.u[k] * s.len) - 2 < origin[k] + (hi[k] + 1) / res,
        ),
    );
    const nearDots = options.seams.dots.filter((p) =>
      p.every((v, k) => v + 1.6 > origin[k] + lo[k] / res && v - 1.6 < origin[k] + (hi[k] + 1) / res),
    );
    const cylinder = options.cylinders?.get(mesh.primitive[ids[0]]);
    // A small physical edge margin fills shared-edge cracks without reaching neighbouring parallel plates.
    const margin = 0.03 / res / Math.max(1e-6, Math.min(Math.sqrt(d00), Math.sqrt(d11)));
    for (let z = lo[2]; z <= hi[2]; z++)
      for (let y = lo[1]; y <= hi[1]; y++)
        for (let x = lo[0]; x <= hi[0]; x++) {
          let p: Vec3 = [origin[0] + (x + 0.5) / res, origin[1] + (y + 0.5) / res, origin[2] + (z + 0.5) / res];
          const plane = dot(sub(p, a), n);
          if (Math.abs(plane) > tolerance) continue;
          p = p.map((v, k) => v - plane * n[k]) as Vec3;
          const ap = sub(p, a),
            d20 = dot(ap, ab),
            d21 = dot(ap, ac),
            v = (d11 * d20 - d01 * d21) / det,
            w = (d00 * d21 - d01 * d20) / det;
          if (v < -margin || w < -margin || v + w > 1 + margin) continue;
          let dark = 0,
            halo = 0,
            code = 0;
          for (const s of nearSeams) {
            const q = sub(p, s.start),
              at = Math.max(0, Math.min(s.len, dot(q, s.u))),
              off = q.map((v, k) => v - at * s.u[k]) as Vec3,
              depth = dot(off, n);
            if (Math.abs(depth) >= 1.2) continue;
            const dist = Math.hypot(...off.map((v, k) => v - depth * n[k]));
            const profile = panelLineProfile(dist, s.strength, options.lines);
            dark = Math.max(dark, profile.dark);
            halo = Math.max(halo, profile.halo);
          }
          for (const q of nearDots) {
            const off = sub(p, [...q]),
              depth = dot(off, n);
            if (Math.abs(depth) < 1)
              dark = Math.max(
                dark,
                (clamp((0.3 - Math.hypot(...off.map((v, k) => v - depth * n[k]))) / 0.08) * 0.45) / 0.92,
              );
          }
          if (cylinder) {
            const q = sub(p, cylinder.centre),
              axial = dot(q, cylinder.axis),
              radial = Math.hypot(...q.map((v, k) => v - axial * cylinder.axis[k]));
            if (cylinder.profile === 'wheel') {
              code = radial > 1.45 ? 0.25 : radial < 0.62 ? 0.75 : 0.5;
              if (radial > 1.05 && radial < 1.22) dark = Math.max(dark, 0.4 / 0.92);
            }
            if (cylinder.profile === 'hoop')
              dark = Math.max(dark, (hoopProfile(axial, cylinder.range, 1.25, 0.2) * 0.45) / 0.92);
            if (cylinder.profile === 'ring' && dot(n, cylinder.axis) > 0.9)
              dark = Math.max(dark, (hatchRingProfile(radial, 2.7, 0.16) * 0.6) / 0.92);
          }
          const i = x + size[0] * (y + size[1] * z);
          touched[i] = 1;
          for (const [ch, value] of [dark, halo, code].entries())
            data[i * 4 + ch] = Math.max(data[i * 4 + ch], Math.round(clamp(value) * 255));
        }
  }
  let surfaceTexels = 0,
    grooveTexels = 0;
  for (let i = 0; i < count; i++) {
    surfaceTexels += touched[i];
    if (touched[i] && data[i * 4] > 0.3 * 255) grooveTexels++;
  }
  return { origin, size, res, data, surfaceTexels, grooveTexels, seams: seams.length, dots: options.seams.dots.length };
}
