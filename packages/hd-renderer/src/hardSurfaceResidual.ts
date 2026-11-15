/** Independent exact triangle-surface diagnostic. Acceptance still uses hardSurfaceMetrics unchanged. */
import { dot, sub, add, scale, type Vec3 } from './hardSurfaceMesh';

function segmentDistance(p: Vec3, a: Vec3, b: Vec3): number {
  const ab = sub(b, a),
    denominator = dot(ab, ab),
    t = denominator ? Math.max(0, Math.min(1, dot(sub(p, a), ab) / denominator)) : 0,
    delta = sub(p, add(a, scale(ab, t)));
  return dot(delta, delta);
}
export function triangleDistanceSquared(p: Vec3, a: Vec3, b: Vec3, c: Vec3): number {
  const ab = sub(b, a),
    ac = sub(c, a),
    ap = sub(p, a),
    d1 = dot(ab, ap),
    d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return dot(ap, ap);
  const bp = sub(p, b),
    d3 = dot(ab, bp),
    d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return dot(bp, bp);
  const cp = sub(p, c),
    d5 = dot(ab, cp),
    d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return dot(cp, cp);
  const vc = d1 * d4 - d3 * d2,
    vb = d5 * d2 - d1 * d6,
    va = d3 * d6 - d5 * d4;
  const denominator = va + vb + vc;
  if (Math.abs(denominator) < 1e-20)
    return Math.min(segmentDistance(p, a, b), segmentDistance(p, b, c), segmentDistance(p, c, a));
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return segmentDistance(p, a, b);
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return segmentDistance(p, a, c);
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return segmentDistance(p, b, c);
  const delta = sub(p, add(a, add(scale(ab, vb / denominator), scale(ac, vc / denominator))));
  return dot(delta, delta);
}
type Triangle = [Vec3, Vec3, Vec3];
interface Node {
  lo: Vec3;
  hi: Vec3;
  triangles?: Triangle[];
  children?: [Node, Node];
}
function tree(triangles: Triangle[]): Node {
  const lo: Vec3 = [Infinity, Infinity, Infinity],
    hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const triangle of triangles)
    for (const p of triangle)
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], p[a]);
        hi[a] = Math.max(hi[a], p[a]);
      }
  if (triangles.length <= 12) return { lo, hi, triangles };
  const axis = hi.map((v, a) => v - lo[a]).reduce((best, v, a, sizes) => (v > sizes[best] ? a : best), 0);
  triangles.sort((p, q) => p.reduce((s, v) => s + v[axis], 0) - q.reduce((s, v) => s + v[axis], 0));
  const middle = Math.floor(triangles.length / 2);
  return { lo, hi, children: [tree(triangles.slice(0, middle)), tree(triangles.slice(middle))] };
}
function boxDistance(p: Vec3, node: Node): number {
  return p.reduce((s, v, a) => s + Math.max(0, node.lo[a] - v, v - node.hi[a]) ** 2, 0);
}
/** Build once, then query exact distances to all supplied triangles using AABB pruning. */
export function surfaceDistanceQuery(
  meshes: readonly { positions: ArrayLike<number>; indices: ArrayLike<number> }[],
): (point: Vec3) => number {
  const triangles: Triangle[] = [];
  for (const m of meshes)
    for (let i = 0; i < m.indices.length; i += 3)
      triangles.push(
        [0, 1, 2].map((k) => [0, 1, 2].map((a) => m.positions[m.indices[i + k] * 3 + a]) as Vec3) as Triangle,
      );
  if (!triangles.length) throw new Error('Cannot query an empty surface');
  const root = tree(triangles);
  return (p) => {
    let best = Infinity;
    const visit = (node: Node) => {
      if (boxDistance(p, node) >= best) return;
      if (node.triangles)
        for (const [a, b, c] of node.triangles) best = Math.min(best, triangleDistanceSquared(p, a, b, c));
      if (node.children) {
        const [a, b] = node.children;
        if (boxDistance(p, a) < boxDistance(p, b)) {
          visit(a);
          visit(b);
        } else {
          visit(b);
          visit(a);
        }
      }
    };
    visit(root);
    return Math.sqrt(best);
  };
}
