import {
  Fn,
  If,
  dot,
  exp,
  float,
  floor,
  int,
  ivec2,
  clamp,
  textureSize,
  uv,
  vec3,
  vec4,
  max,
  abs,
  mod,
  mix,
} from 'three/tsl';
import type { ShaderNode } from './tsl';

export interface SurfacePolish {
  category: number;
  sigma: number;
  range: number;
}
/** A small display-space bilateral gather. Exact object codes protect silhouettes and neighbouring materials. */
export function polishSurface(
  picture: ShaderNode,
  codes: ShaderNode,
  policy: SurfacePolish,
  amount: ShaderNode,
): ShaderNode {
  return Fn(() => {
    const size: ShaderNode = (textureSize(codes) as ShaderNode).xy;
    const point: ShaderNode = floor(uv().mul(size));
    const p: ShaderNode = ivec2(int(point.x), int(point.y));
    const at = (q: ShaderNode) => (clamp as ShaderNode)(q, ivec2(0), ivec2(size).sub(1));
    const c = codes.load(at(p)).yz.toVar();
    const centre = picture.load(at(p)).toVar();
    const result = centre.toVar();
    const category = mod(floor(c.y.mul(256).add(0.5)), 8);
    If(category.equal(policy.category).and(amount.greaterThan(0)), () => {
      const sum = vec3(0).toVar(),
        norm = float(0).toVar();
      const lum = dot(centre.rgb, vec3(0.2126, 0.7152, 0.0722));
      for (let y = -2; y <= 2; y++)
        for (let x = -2; x <= 2; x++) {
          const q = at(p.add(ivec2(x, y))),
            code = codes.load(q).yz;
          const same = max(abs(code.x.sub(c.x)), abs(code.y.sub(c.y))).lessThan(0.5 / 256);
          const colour = picture.load(q).rgb;
          const difference = dot(colour, vec3(0.2126, 0.7152, 0.0722)).sub(lum);
          const w = float(same)
            .mul(Math.exp(-(x * x + y * y) / (2 * policy.sigma ** 2)))
            .mul(exp(difference.pow(2).mul(-0.5 / policy.range ** 2)));
          sum.addAssign(colour.mul(w));
          norm.addAssign(w);
        }
      result.assign(vec4(mix(centre.rgb, sum.div(norm.max(1e-6)), amount), centre.a));
    });
    return result;
  })();
}

/** Scalar-code CPU reference for noise, seam and excluded-category checks. */
export function polishSurfaceCPU(
  rgb: readonly (readonly number[])[],
  codes: readonly number[],
  width: number,
  x: number,
  y: number,
  policy: SurfacePolish,
) {
  const k = y * width + x,
    centre = rgb[k]!;
  if (codes[k]! % 8 !== policy.category) return [...centre];
  const height = rgb.length / width,
    sum = [0, 0, 0];
  let norm = 0;
  const luma = (c: readonly number[]) => c[0]! * 0.2126 + c[1]! * 0.7152 + c[2]! * 0.0722;
  for (let dy = -2; dy <= 2; dy++)
    for (let dx = -2; dx <= 2; dx++) {
      const q = Math.max(0, Math.min(height - 1, y + dy)) * width + Math.max(0, Math.min(width - 1, x + dx));
      if (codes[q] !== codes[k]) continue;
      const difference = luma(rgb[q]!) - luma(centre);
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * policy.sigma ** 2) - difference ** 2 / (2 * policy.range ** 2));
      for (let i = 0; i < 3; i++) sum[i]! += rgb[q]![i]! * w;
      norm += w;
    }
  return sum.map((v) => v / norm);
}
