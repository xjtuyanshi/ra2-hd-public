import { buildPart, type PartSpec, type Primitive } from './hardSurfaceMesh';
import type { RoundFeature } from './hardSurfaceRoundFit';

/** Replace a source-supported region while preserving every other plate and its metadata.
 * Clipping pieces are charged to the same expanded budget as ordinary reconstruction.
 */
export function patchRoundFeature(
  part: PartSpec,
  feature: RoundFeature,
  name: string,
  replacement = feature.primitives,
): PartSpec {
  const out = structuredClone(part),
    names = replacement.map((_, i) => `${name}_${i}`);
  const primitives: Primitive[] = replacement.map((q, i) => ({ ...structuredClone(q), name: names[i], mirror: false }));
  // A replacement is the already supported source region, rather than a new plate subject to historical local cuts.
  for (const cut of out.carve ?? []) cut.keep = [...(cut.keep ?? []), ...names];
  out.carve = [...(out.carve ?? []), { min: [...feature.bounds[0]], max: [...feature.bounds[1]], keep: names }];
  out.primitives.push(...primitives);
  return out;
}

export function withinExpandedBudget(part: PartSpec, budget: number): boolean {
  return buildPart(part).primitives.length <= budget;
}

export interface CoplanarBoundary {
  key: string;
  members: { primitive: number; plane: number; sign: number; length: number }[];
}
/** Opposite inequalities at the same BSP seam share a canonical boundary and move together. */
export function coplanarBoundaries(part: PartSpec): CoplanarBoundary[] {
  const groups = new Map<string, CoplanarBoundary>();
  part.primitives.forEach((q, i) => {
    if (q.type !== 'poly' || q.round) return;
    q.planes.forEach((p, j) => {
      const length = Math.hypot(p[0], p[1], p[2]),
        sign = p.slice(0, 3).find((v) => Math.abs(v) > 1e-8)! < 0 ? -1 : 1;
      const key = p.map((v) => Math.round((v / length) * sign * 1000)).join(',');
      const group = groups.get(key) ?? { key, members: [] };
      group.members.push({ primitive: i, plane: j, sign, length });
      groups.set(key, group);
    });
  });
  return [...groups.values()];
}
export function shiftCoplanarBoundary(part: PartSpec, boundary: CoplanarBoundary, delta: number): PartSpec {
  const out = structuredClone(part);
  for (const r of boundary.members) {
    const q = out.primitives[r.primitive];
    if (q.type !== 'poly' || q.round) throw new Error('Boundary owner changed');
    q.planes[r.plane][3] += delta * r.sign * r.length;
  }
  return out;
}
