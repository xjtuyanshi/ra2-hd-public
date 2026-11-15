/** Merge only compatible convex neighbours whose original halfspace union is unchanged. */
import {
  buildPart,
  polytopeFaces,
  dot,
  type PartSpec,
  type Primitive,
  type Plane,
  type ExpandedPrimitive,
} from './hardSurfaceMesh';
const tolerance = 1e-7;
const same = (a: Plane, b: Plane) => a.every((v, k) => Math.abs(v - b[k]) < tolerance);
export function coalescePart(part: PartSpec): { part: PartSpec; merges: { left: string[]; right: string[] }[] } {
  const built = buildPart(part),
    flats = built.primitives.filter((q) => !q.round && !q.cylinder),
    keepRaw = part.primitives.filter((q) => q.type === 'cyl' || q.round);
  let groups = flats.map((q, i) => ({
    q: structuredClone(q),
    names: [q.name ?? String(i)],
    vertices: polytopeFaces(q.planes).vertices,
  }));
  const merges: { left: string[]; right: string[] }[] = [];
  const compatible = (p: ExpandedPrimitive, q: ExpandedPrimitive) =>
    JSON.stringify([p.paint, p.groove, p.chamfer]) === JSON.stringify([q.paint, q.groove, q.chamfer]);
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < groups.length; i++)
      for (let j = i + 1; j < groups.length; j++) {
        const p = groups[i],
          q = groups[j];
        if (!compatible(p.q, q.q)) continue;
        let shared: [number, number] | undefined;
        for (let a = 0; a < p.q.planes.length && !shared; a++)
          for (let b = 0; b < q.q.planes.length; b++)
            if (same(p.q.planes[a], q.q.planes[b].map((v) => -v) as Plane)) {
              shared = [a, b];
              break;
            }
        if (!shared) continue;
        const [a, b] = shared,
          pp = p.q.planes.filter((_, k) => k !== a),
          qp = q.q.planes.filter((_, k) => k !== b);
        if (
          pp.some((n) => q.vertices.some((v) => dot(n, v) > n[3] + tolerance)) ||
          qp.some((n) => p.vertices.some((v) => dot(n, v) > n[3] + tolerance))
        )
          continue;
        const planes = [...pp, ...qp].filter((p, i, all) => !all.slice(0, i).some((q) => same(p, q)));
        const faces = polytopeFaces(planes);
        if (!faces.vertices.length) continue;
        const compact = planes.filter((_, i) => faces.faces.some((f) => f.plane === i));
        merges.push({ left: p.names, right: q.names });
        groups[i] = { q: { ...p.q, planes: compact }, names: [...p.names, ...q.names], vertices: faces.vertices };
        groups.splice(j, 1);
        changed = true;
        break outer;
      }
  }
  if (!merges.length) return { part: structuredClone(part), merges };
  const flat: Primitive[] = groups.map((g, i) => ({
    type: 'poly',
    name: `coalesced_${i}`,
    mirror: false,
    planes: g.q.planes,
    chamfer: g.q.chamfer,
    ...(g.q.paint !== undefined ? { paint: g.q.paint } : {}),
    ...(g.q.groove !== undefined ? { groove: g.q.groove } : {}),
  }));
  const names = flat.map((q) => q.name!);
  const out = structuredClone(part);
  out.primitives = [...keepRaw, ...flat];
  for (const c of out.carve ?? []) c.keep = [...(c.keep ?? []), ...names];
  return { part: out, merges };
}
