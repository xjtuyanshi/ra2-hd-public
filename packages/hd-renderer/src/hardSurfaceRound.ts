/** Convex curved patches with explicit halfspaces. Tessellation pieces consume the ordinary expanded budget. */
import {
  add,
  sub,
  scale,
  cross,
  dot,
  unit,
  type Vec3,
  type Plane,
  type Primitive,
  type RoundSurface,
} from './hardSurfaceMesh';

const sphereDirections = (): Vec3[] => {
  const out: Vec3[] = [
    [0, 0, -1],
    [0, 0, 1],
  ];
  for (let j = 1; j < 6; j++) {
    const phi = (Math.PI * j) / 6;
    for (let i = 0; i < 12; i++) {
      const theta = (2 * Math.PI * i) / 12;
      out.push([Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi)]);
    }
  }
  return out;
};
const positive = (r: number) => {
  if (!Number.isFinite(r) || r <= 0) throw new Error('Positive finite round radius required');
};
const patch = (planes: Plane[], round: RoundSurface): Primitive => ({
  type: 'poly',
  planes,
  round,
  chamfer: 0,
  mirror: false,
});

/** A straight circular frustum; equal radii give an arbitrarily oriented cylinder. */
export function taperedCylinder(start: Vec3, end: Vec3, radii: [number, number]): Primitive {
  radii.forEach(positive);
  const length = Math.hypot(...sub(end, start));
  if (!Number.isFinite(length) || !(length > 1e-6)) throw new Error('Distinct finite pipe endpoints required');
  const axis = unit(sub(end, start)),
    u = unit(cross(axis, Math.abs(axis[2]) < 0.8 ? [0, 0, 1] : [0, 1, 0])),
    v = cross(axis, u);
  const slope = (radii[1] - radii[0]) / length,
    planes: Plane[] = [];
  // Match the established cylinder's area correction and angular resolution.
  const correction = Math.sqrt((2 * Math.PI) / 48 / Math.sin((2 * Math.PI) / 48)) * Math.cos(Math.PI / 48);
  for (let i = 0; i < 48; i++) {
    const theta = (2 * Math.PI * (i + 0.5)) / 48;
    const normal = sub(add(scale(u, Math.cos(theta)), scale(v, Math.sin(theta))), scale(axis, slope * correction));
    planes.push([...normal, dot(normal, start) + radii[0] * correction]);
  }
  planes.push([...scale(axis, -1), -dot(axis, start)], [...axis, dot(axis, end)]);
  return patch(planes, { type: 'taper', start: [...start], end: [...end], radii: [...radii] });
}

/** Polyline sweep with flush bisector joints, not overlapping discs or per-voxel boxes. */
export function sweptCylinders(points: readonly Vec3[], radius: number): Primitive[] {
  positive(radius);
  if (points.length < 2 || points.length > 65) throw new Error('Pipe path requires 2..65 points');
  const dirs = points.slice(1).map((p, i) => unit(sub(p, points[i])));
  return dirs.map((axis, i) => {
    const q = taperedCylinder(points[i], points[i + 1], [radius, radius]);
    if (q.type !== 'poly') throw new Error('Expected convex patch');
    if (i) {
      const n = unit(add(dirs[i - 1], axis));
      if (!n.every(Number.isFinite)) throw new Error('Pipe path reverses direction');
      q.planes[q.planes.length - 2] = [...scale(n, -1), -dot(n, points[i])];
    }
    if (i + 1 < dirs.length) {
      const n = unit(add(axis, dirs[i + 1]));
      if (!n.every(Number.isFinite)) throw new Error('Pipe path reverses direction');
      q.planes[q.planes.length - 1] = [...n, dot(n, points[i + 1])];
    }
    return q;
  });
}

/** A spherical cap clipped at an injected plane; height may range from a tiny cap to a full sphere. */
export function sphericalCap(centre: Vec3, radius: number, axis: Vec3, height: number): Primitive {
  positive(radius);
  if (!(height > 0 && height <= 2 * radius)) throw new Error('Cap height must lie in (0, 2r]');
  const n = unit(axis);
  if (!n.every(Number.isFinite)) throw new Error('Nonzero cap axis required');
  const planes: Plane[] = sphereDirections().map((d) => [...d, dot(d, centre) + radius]);
  planes.push([...scale(n, -1), -dot(n, centre) - radius + height]);
  return patch(planes, { type: 'sphere', centre: [...centre] });
}

/** Convex sphere-ended segment. Endpoints are sphere centres, not the outside tips. */
export function capsule(start: Vec3, end: Vec3, radius: number): Primitive {
  positive(radius);
  if (!Math.hypot(...sub(end, start))) return sphericalCap(start, radius, [0, 0, 1], 2 * radius);
  return patch(
    sphereDirections().map((n): Plane => [...n, Math.max(dot(n, start), dot(n, end)) + radius]),
    { type: 'capsule', start: [...start], end: [...end] },
  );
}

/** Bends are bounded sweeps. Each convex chord is counted, so a bend cannot hide arbitrary complexity. */
export function torusSegments(
  centre: Vec3,
  axis: Vec3,
  radial: Vec3,
  major: number,
  minor: number,
  angle: number,
): Primitive[] {
  positive(major);
  positive(minor);
  if (minor >= major || !Number.isFinite(angle) || Math.abs(angle) < 1e-6 || Math.abs(angle) > 2 * Math.PI)
    throw new Error('Invalid torus arc');
  const n = unit(axis),
    u = unit(sub(radial, scale(n, dot(radial, n)))),
    v = cross(n, u);
  if (![...n, ...u].every(Number.isFinite)) throw new Error('Independent torus axes required');
  const step = Math.min(Math.PI / 12, 2 * Math.acos(Math.max(-1, 1 - 0.125 / major)));
  const steps = Math.ceil(Math.abs(angle) / step);
  const points = Array.from({ length: steps + 1 }, (_, i) =>
    add(centre, scale(add(scale(u, Math.cos((angle * i) / steps)), scale(v, Math.sin((angle * i) / steps))), major)),
  );
  return sweptCylinders(points, minor).map((q) => ({
    ...q,
    round: { type: 'torus', centre: [...centre], axis: n, radius: major },
  }));
}

/** A source-supported convex edge fillet, clipped to its two adjoining plate halfspaces. */
export function edgeFillet(start: Vec3, end: Vec3, radius: number, clips: readonly Plane[]): Primitive {
  const q = taperedCylinder(start, end, [radius, radius]);
  if (q.type === 'poly') q.planes.push(...clips.map((p) => [...p] as Plane));
  return q;
}

/** An anisotropic cap uses source-supported quadric radii, not a distorted normal field on a sphere. */
export function ellipsoidCap(centre: Vec3, radii: Vec3, axis: Vec3, height: number): Primitive {
  radii.forEach(positive);
  const n = unit(axis),
    extent = Math.hypot(...n.map((v, a) => v * radii[a]));
  if (!n.every(Number.isFinite) || !(height > 0 && height <= 2 * extent)) throw new Error('Invalid ellipsoid cap');
  const planes: Plane[] = sphereDirections().map((d) => [
    ...d,
    dot(d, centre) + Math.hypot(...d.map((v, a) => v * radii[a])),
  ]);
  planes.push([...scale(n, -1), -dot(n, centre) - extent + height]);
  return patch(planes, { type: 'ellipsoid', centre: [...centre], radii: [...radii] });
}
