import {
  Fn,
  If,
  abs,
  clamp,
  dot,
  exp,
  float,
  floor,
  int,
  ivec2,
  max,
  mix,
  mod,
  textureSize,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import SharpenNode from 'three/addons/tsl/display/SharpenNode.js';
import type { NodeBuilder, NodeMaterial, TextureNode } from 'three/webgpu';
import type { ShaderNode } from './tsl';
import type { SpriteIntegration } from './spriteIntegration';

const LUMA = [0.2126, 0.7152, 0.0722] as const;
const CODE_EPSILON = 0.5 / 256;

/** Physical-depth codes are discrete data. Never interpolate them, or gather colour across a code boundary. */
function footprint(codes: ShaderNode, protectedCategory = 0) {
  const size: ShaderNode = (textureSize(codes) as ShaderNode).xy;
  const point: ShaderNode = floor(uv().mul(size));
  const p: ShaderNode = ivec2(int(point.x), int(point.y));
  const at = (q: ShaderNode): ShaderNode => (clamp as ShaderNode)(q, ivec2(0), ivec2(size).sub(1));
  const code = (q: ShaderNode): ShaderNode => codes.load(at(q)).yz;
  const same = (a: ShaderNode, b: ShaderNode): ShaderNode =>
    max(abs(a.x.sub(b.x)), abs(a.y.sub(b.y))).lessThan(CODE_EPSILON);
  const protectedCode = (c: ShaderNode): ShaderNode =>
    protectedCategory ? mod(floor(c.y.mul(256).add(0.5)), 8).equal(protectedCategory) : float(0).greaterThan(0);
  const object = (c: ShaderNode): ShaderNode => max(c.x, c.y).greaterThan(CODE_EPSILON).and(protectedCode(c).not());
  return { p, at, code, same, object, protectedCode };
}

/** One-device-pixel dilation with a feathered outer ring: centre 1, axial neighbours 1/2, diagonals 1/4. */
export function objectPostMask(codes: ShaderNode, protectedCategory = 0): ShaderNode {
  return Fn(() => {
    const { p, code, object, protectedCode } = footprint(codes, protectedCategory);
    const mask: ShaderNode = float(object(code(p))).toVar('objectPostMask');
    for (let y = -1; y <= 1; y++)
      for (let x = -1; x <= 1; x++) {
        if (!x && !y) continue;
        mask.assign(max(mask, float(object(code(p.add(ivec2(x, y))))).mul(x && y ? 0.25 : 0.5)));
      }
    return protectedCode(code(p)).select(float(0), mask);
  })();
}

/** Same-code discrete Gaussian gathers. Small = sigma .7 CSS px; broad = sigma 3 CSS px (truncated at 2 sigma). */
export function objectSurfaceFilter(
  picture: ShaderNode,
  codes: ShaderNode,
  group: SpriteIntegration,
  pixelScale: ShaderNode,
): ShaderNode {
  const t = group.uniforms;
  return Fn(() => {
    const { p, at, code, same, object } = footprint(codes, group.protectedCategory);
    const centre: ShaderNode = picture.load(at(p)).toVar();
    const result: ShaderNode = centre.toVar();
    const c: ShaderNode = code(p).toVar();
    If(t.on.greaterThan(0).and(object(c)).and(t.objSoft.add(t.objContrast).greaterThan(0)), () => {
      const gather = (radius: number, spacing: number, sigma: number): ShaderNode => {
        const sum: ShaderNode = vec3(0).toVar();
        const norm: ShaderNode = float(0).toVar();
        for (let y = -radius; y <= radius; y++)
          for (let x = -radius; x <= radius; x++) {
            const q: ShaderNode = p.add(
              ivec2(
                int(floor(pixelScale.mul(Math.abs(x) * spacing).add(0.5)).mul(Math.sign(x))),
                int(floor(pixelScale.mul(Math.abs(y) * spacing).add(0.5)).mul(Math.sign(y))),
              ),
            );
            const w: ShaderNode = float(same(c, code(q))).mul(
              Math.exp((-(x * x + y * y) * spacing * spacing) / (2 * sigma * sigma)),
            );
            sum.addAssign(picture.load(at(q)).rgb.mul(w));
            norm.addAssign(w);
          }
        return sum.div(norm.max(1e-6));
      };
      const fine: ShaderNode = gather(1, 1, 0.7);
      const broad: ShaderNode = gather(2, 3, 3);
      const lum: ShaderNode = dot(centre.rgb, vec3(...LUMA));
      const target: ShaderNode = mix(lum, dot(broad, vec3(...LUMA)), t.on.mul(t.objContrast).mul(0.5));
      const contrasted: ShaderNode = centre.rgb.mul(target.div(lum.max(1e-6)));
      // The soft gather also takes the centre's contrast scale: no intermediate full-size blur target is needed.
      result.assign(vec4(mix(contrasted, fine.mul(target.div(lum.max(1e-6))), t.on.mul(t.objSoft)), centre.a));
    });
    return result;
  })();
}

/** Symmetric device-pixel Gaussian across different object/ground codes; only this stage may cross those boundaries. */
export function objectOutlineFeather(
  picture: ShaderNode,
  codes: ShaderNode,
  group: SpriteIntegration,
  pixel?: ShaderNode,
): ShaderNode {
  const t = group.uniforms;
  return Fn(() => {
    const f = footprint(codes, group.protectedCategory);
    const { at, code, same } = f;
    const p: ShaderNode = pixel ?? f.p;
    const centre: ShaderNode = picture.load(at(p)).toVar();
    const result: ShaderNode = centre.toVar();
    If(
      t.on
        .greaterThan(0)
        .and(t.outlineFeather.greaterThan(0))
        .and(f.protectedCode(code(p)).not()),
      () => {
        const c: ShaderNode = code(p).toVar();
        const taps: { q: ShaderNode; w: ShaderNode }[] = [];
        const different: ShaderNode = float(0).toVar();
        const norm: ShaderNode = float(0).toVar();
        const inverseSigma2: ShaderNode = float(-0.5).div(t.outlineFeather.mul(0.6).max(0.5).pow(2)).toVar();
        const weights = new Map<number, ShaderNode>();
        for (let y = -3; y <= 3; y++)
          for (let x = -3; x <= 3; x++) {
            const d2 = x * x + y * y;
            if (d2 > 9) continue;
            const q: ShaderNode = p.add(ivec2(x, y));
            let w = weights.get(d2);
            if (!w) {
              w = exp(inverseSigma2.mul(d2))
                .mul(t.outlineFeather.add(1 - Math.sqrt(d2)).clamp())
                .toVar();
              weights.set(d2, w);
            }
            norm.addAssign(w);
            taps.push({ q, w });
            different.assign(
              max(
                different,
                float(same(c, code(q)))
                  .oneMinus()
                  .mul(w.greaterThan(0)),
              ),
            );
          }
        If(different.greaterThan(0), () => {
          const sum: ShaderNode = vec3(0).toVar();
          for (const { q, w } of taps) sum.addAssign(picture.load(at(q)).rgb.mul(w));
          result.assign(vec4(mix(centre.rgb, sum.div(norm), t.on.mul(t.outlineFeather.min(1))), centre.a));
        });
      },
    );
    return result;
  })();
}

/** CPU twin of the discrete gathers/mask/feather, with scalar code IDs standing for the exact code pairs. */
export function objectPostCPU(
  pixels: readonly (readonly [number, number, number])[],
  codes: readonly number[],
  width: number,
  x: number,
  y: number,
  t: { on: number; objSoft: number; objContrast: number; outlineFeather: number },
  pixelScale = 1,
  protectedCode = 0,
) {
  const height = pixels.length / width;
  const at = (dx: number, dy: number) =>
    Math.max(0, Math.min(height - 1, dy)) * width + Math.max(0, Math.min(width - 1, dx));
  const k = at(x, y),
    c = codes[k]!,
    centre = pixels[k]!;
  const eligible = (value: number) => value !== 0 && (!protectedCode || value !== protectedCode);
  let mask = eligible(c) ? 1 : 0;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++)
      if ((dx || dy) && eligible(codes[at(x + dx, y + dy)]!)) mask = Math.max(mask, dx && dy ? 0.25 : 0.5);
  const gather = (radius: number, spacing: number, sigma: number) => {
    const sum = [0, 0, 0];
    let norm = 0;
    for (let dy = -radius; dy <= radius; dy++)
      for (let dx = -radius; dx <= radius; dx++) {
        const round = (n: number) => Math.sign(n) * Math.floor(Math.abs(n) * spacing + 0.5);
        const q = at(x + round(dx), y + round(dy));
        if (codes[q] !== c) continue;
        const w = Math.exp((-(dx * dx + dy * dy) * spacing * spacing) / (2 * sigma * sigma));
        norm += w;
        for (let i = 0; i < 3; i++) sum[i]! += pixels[q]![i]! * w;
      }
    return sum.map((v) => v / norm);
  };
  let surface: readonly number[] = centre;
  if (t.on > 0 && c && t.objSoft + t.objContrast > 0) {
    const fine = gather(1, pixelScale, 0.7 * pixelScale);
    const broad = gather(2, 3 * pixelScale, 3 * pixelScale);
    const lum = centre.reduce((a, v, i) => a + v * LUMA[i]!, 0);
    const blurred = broad.reduce((a, v, i) => a + v * LUMA[i]!, 0);
    const scale = (lum + (blurred - lum) * t.on * t.objContrast * 0.5) / Math.max(lum, 1e-6);
    surface = centre.map((v, i) => (v + (fine[i]! - v) * t.on * t.objSoft) * scale);
  }
  let feather: readonly number[] = centre;
  if (t.on > 0 && t.outlineFeather > 0) {
    const sum = [0, 0, 0];
    let norm = 0,
      different = false;
    const sigma = Math.max(t.outlineFeather * 0.6, 0.5);
    for (let dy = -3; dy <= 3; dy++)
      for (let dx = -3; dx <= 3; dx++) {
        const d2 = dx * dx + dy * dy;
        if (d2 > 9) continue;
        const w = Math.exp(-d2 / (2 * sigma * sigma)) * Math.max(0, Math.min(1, t.outlineFeather + 1 - Math.sqrt(d2)));
        const q = at(x + dx, y + dy);
        norm += w;
        different ||= codes[q] !== c && w > 0;
        for (let i = 0; i < 3; i++) sum[i]! += pixels[q]![i]! * w;
      }
    if (different) feather = centre.map((v, i) => v + (sum[i]! / norm - v) * t.on * Math.min(t.outlineFeather, 1));
  }
  if (protectedCode && c === protectedCode) return { mask: 0, surface: centre, feather: centre };
  return { mask, surface, feather };
}

/** Reuse r186's RCAS pass/target; mask its fragment result inside that pass instead of copying the mixed output. */
export class ObjectMaskedSharpen extends SharpenNode {
  constructor(
    picture: TextureNode,
    sharpness: number,
    private codes: ShaderNode,
    private group: SpriteIntegration,
  ) {
    super(picture, sharpness);
  }
  override setup(builder: NodeBuilder): TextureNode {
    const output = super.setup(builder) as TextureNode;
    // r186's setup owns this material. Keep the upstream RCAS arithmetic and lifecycle, modifying only its resolve.
    const material = (this as unknown as { _material: NodeMaterial })._material;
    material.fragmentNode = mix(
      material.fragmentNode as ShaderNode,
      this.textureNode,
      objectPostMask(this.codes, this.group.protectedCategory)
        .mul(this.group.uniforms.on)
        .mul(this.group.uniforms.objSharp),
    );
    return output;
  }
}
