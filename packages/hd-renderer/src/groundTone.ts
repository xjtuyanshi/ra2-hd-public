import { Vector3 } from 'three/webgpu';
import { cos, dot, float, floor, Fn, fract, If, int, mix, sin, smoothstep, uint, uniform, vec2, vec3 } from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Linear-light, world-anchored ground dressing. Distances are in world xz units; angles are degrees. */
export interface GroundToneParams {
  strength: number;
  scale: number;
  detailScale: number;
  detailMix: number;
  stretch: number;
  angle: number;
  coverage: number;
  softness: number;
  depth: number;
  /** Depth of non-green ground (sand, rock, paving), which the owner's reference keeps bright and warm. */
  bareDepth: number;
  /** Extra chroma gain on non-green ground. */
  bareChroma: number;
  /**
   * Already-dark ground (a jungle floor) keeps most of its light: below albedo luminance darkLo only darkKeep of the
   * gain change applies, ramping to the full change at darkHi. darkKeep 1 disables it.
   */
  darkLo: number;
  darkHi: number;
  darkKeep: number;
  sunGain: number;
  shadeGain: number;
  sunTint: [number, number, number];
  shadeTint: [number, number, number];
  grassShadeTint: [number, number, number];
  sunChroma: number;
  shadeChroma: number;
  soften: number;
}
const W = [0.2126, 0.7152, 0.0722] as const;
const RAD = Math.PI / 180;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const blend = (a: number, b: number, t: number) => a * (1 - t) + b * t;
const smooth = (a: number, b: number, v: number) => {
  const t = clamp((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const luma = (v: readonly number[]) => v.reduce((s, x, i) => s + x * W[i]!, 0);

// Unsigned wrapping arithmetic is shared with WGSL, including negative lattice coordinates. Keep only 24 bits
// before conversion to float: every integer involved in the conversion is exactly representable by an f32.
function hash(x: number, z: number): number {
  let h = (Math.imul(x, 0x1f123bb5) ^ Math.imul(z, 0x5f356495)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return ((h ^ (h >>> 16)) >>> 8) / 16777216;
}
function noise(x: number, z: number): number {
  const ix = Math.floor(x),
    iz = Math.floor(z);
  const u = smooth(0, 1, x - ix),
    v = smooth(0, 1, z - iz);
  return blend(blend(hash(ix, iz), hash(ix + 1, iz), u), blend(hash(ix, iz + 1), hash(ix + 1, iz + 1), u), v);
}
/** Approximate area coverage of the centred value-noise distribution; exact all-shade/all-sun at the endpoints. */
export function groundPatchField(x: number, z: number, p: GroundToneParams): number {
  if (p.coverage <= 0) return 0;
  if (p.coverage >= 1) return 1;
  const a = p.angle * RAD;
  const u = (x * Math.cos(a) + z * Math.sin(a)) / Math.max(1, p.stretch);
  const v = -x * Math.sin(a) + z * Math.cos(a);
  const n = blend(
    noise(u / Math.max(p.scale, 1e-4), v / Math.max(p.scale, 1e-4)),
    noise(u / Math.max(p.detailScale, 1e-4) + 19.25, v / Math.max(p.detailScale, 1e-4) - 7.75),
    p.detailMix,
  );
  const threshold = 0.5 + (0.5 - p.coverage) * (0.75 * Math.sqrt((1 - p.detailMix) ** 2 + p.detailMix ** 2));
  const half = Math.max(p.softness, 1e-4) * 0.5;
  return smooth(threshold - half, threshold + half, n);
}
export function groundGreenness(rgb: readonly [number, number, number]): number {
  const y = Math.max(luma(rgb), 0.02);
  return smooth(-0.25, -0.08, (rgb[1] - rgb[0]) / y) * smooth(0.15, 0.35, (rgb[1] - rgb[2]) / y);
}
/** JS twin. Optional greenness lets callers retain classification from the unsoftened albedo. */
export function groundToneRgb(
  rgb: readonly [number, number, number],
  lit: number,
  p: GroundToneParams,
  night = 0,
  greenness = groundGreenness(rgb),
): [number, number, number] {
  const amount = p.strength * (1 - night);
  if (amount === 0) return [...rgb];
  const tint = p.shadeTint.map(
    (v, i) => blend(v, p.sunTint[i]!, lit) * blend(1, p.grassShadeTint[i]!, (1 - lit) * greenness),
  );
  const tinted = rgb.map((v, i) => v * tint[i]!);
  const y = luma(rgb);
  const norm = y / Math.max(luma(tinted), 1e-6);
  const chroma = blend(p.shadeChroma, p.sunChroma, lit) * blend(p.bareChroma, 1, greenness);
  const keep = blend(p.darkKeep, 1, smooth(p.darkLo, Math.max(p.darkHi, p.darkLo + 1e-4), y));
  const gain = blend(1, blend(p.bareDepth, p.depth, greenness) * blend(p.shadeGain, p.sunGain, lit), keep);
  return tinted.map((v, i) => blend(rgb[i]!, Math.max(0, (y + (v * norm - y) * chroma) * gain), amount)) as [
    number,
    number,
    number,
  ];
}

export class GroundToneUniforms {
  readonly strength = uniform(0);
  readonly scale = uniform(1);
  readonly detailScale = uniform(1);
  readonly detailMix = uniform(0);
  readonly stretch = uniform(1);
  readonly angle = uniform(0);
  readonly coverage = uniform(0.5);
  readonly softness = uniform(0.35);
  readonly depth = uniform(1);
  readonly bareDepth = uniform(1);
  readonly bareChroma = uniform(1);
  readonly darkLo = uniform(0);
  readonly darkHi = uniform(0.1);
  readonly darkKeep = uniform(1);
  readonly sunGain = uniform(1);
  readonly shadeGain = uniform(1);
  readonly sunTint = uniform(new Vector3(1, 1, 1));
  readonly shadeTint = uniform(new Vector3(1, 1, 1));
  readonly grassShadeTint = uniform(new Vector3(1, 1, 1));
  readonly sunChroma = uniform(1);
  readonly shadeChroma = uniform(1);
  readonly soften = uniform(0);

  set(p: GroundToneParams | null): void {
    if (!p) {
      this.strength.value = 0;
      return;
    }
    for (const key of Object.keys(p) as (keyof GroundToneParams)[]) {
      // Callers may carry extra policy data (e.g. grain), which is not a ground uniform.
      if (!Object.hasOwn(this, key)) continue;
      const u = this[key];
      if (typeof u.value === 'number') u.value = p[key] as number;
      else u.value.fromArray(p[key] as [number, number, number]);
    }
  }
  copyValuesFrom(other: GroundToneUniforms): void {
    for (const key of Object.keys(this) as (keyof GroundToneParams)[]) {
      const u = this[key],
        v = other[key].value;
      if (typeof u.value === 'number') u.value = v as number;
      else u.value.copy(v as Vector3);
    }
  }
}

function hashNode(x: ShaderNode, z: ShaderNode): ShaderNode {
  let h: ShaderNode = uint(int(x))
    .mul(uint(0x1f123bb5))
    .bitXor(uint(int(z)).mul(uint(0x5f356495)));
  h = h
    .bitXor(h.shiftRight(uint(16)))
    .mul(uint(0x7feb352d))
    .toVar();
  h = h
    .bitXor(h.shiftRight(uint(15)))
    .mul(uint(0x846ca68b))
    .toVar();
  return float(h.bitXor(h.shiftRight(uint(16))).shiftRight(uint(8))).div(16777216);
}
function noiseNode(p: ShaderNode): ShaderNode {
  const i: ShaderNode = floor(p).toVar(),
    f: ShaderNode = fract(p).toVar();
  const w: ShaderNode = f.mul(f).mul(float(3).sub(f.mul(2)));
  return mix(
    mix(hashNode(i.x, i.y), hashNode(i.x.add(1), i.y), w.x),
    mix(hashNode(i.x, i.y.add(1)), hashNode(i.x.add(1), i.y.add(1)), w.x),
    w.y,
  );
}
export function groundPatchNode(xz: ShaderNode, p: GroundToneUniforms): ShaderNode {
  const a = p.angle.mul(RAD),
    c = cos(a),
    s = sin(a);
  const uv = vec2(xz.x.mul(c).add(xz.y.mul(s)).div(p.stretch.max(1)), xz.y.mul(c).sub(xz.x.mul(s))).toVar();
  const n = mix(
    noiseNode(uv.div(p.scale.max(1e-4))),
    noiseNode(uv.div(p.detailScale.max(1e-4)).add(vec2(19.25, -7.75))),
    p.detailMix,
  );
  const threshold = float(0.5).add(
    float(0.5)
      .sub(p.coverage)
      .mul(float(1).sub(p.detailMix).pow(2).add(p.detailMix.pow(2)).sqrt().mul(0.75)),
  );
  const half = p.softness.max(1e-4).mul(0.5);
  return p.coverage
    .lessThanEqual(0)
    .select(
      float(0),
      p.coverage.greaterThanEqual(1).select(float(1), smoothstep(threshold.sub(half), threshold.add(half), n)),
    );
}
/** Uniform branch keeps disabled materials an exact identity without evaluating noise or colour work. */
export function groundToneNode(
  albedo: ShaderNode,
  xz: ShaderNode,
  p: GroundToneUniforms,
  night: ShaderNode,
): ShaderNode {
  return Fn(() => {
    const out = vec3(albedo).toVar();
    If(p.strength.greaterThan(0).and(night.lessThan(1)), () => {
      const lit = groundPatchNode(xz, p).toVar();
      const y = dot(albedo, vec3(...W)).toVar();
      // Grass, yellow-green as well as green, so the art's alternating yellow/green tile stripes are one class (a
      // narrower gate deepened only the green stripes and drew the tile lattice as diagonal bands); sand (red above
      // green) and grey rock or paving (no green over blue) are bare ground.
      const yk = y.max(0.02);
      const green = smoothstep(-0.25, -0.08, albedo.g.sub(albedo.r).div(yk))
        .mul(smoothstep(0.15, 0.35, albedo.g.sub(albedo.b).div(yk)))
        .toVar();
      const tint = mix(p.shadeTint, p.sunTint, lit).mul(mix(vec3(1), p.grassShadeTint, float(1).sub(lit).mul(green)));
      const tinted = albedo.mul(tint).toVar();
      // Normalise against the actual albedo, so even strongly coloured ground keeps its luminance before gain.
      const normalized = tinted.mul(y.div(dot(tinted, vec3(...W)).max(1e-6)));
      const chroma = mix(p.shadeChroma, p.sunChroma, lit).mul(mix(p.bareChroma, float(1), green));
      const keep = mix(p.darkKeep, float(1), smoothstep(p.darkLo, p.darkHi.max(p.darkLo.add(1e-4)), y));
      const gain = mix(float(1), mix(p.bareDepth, p.depth, green).mul(mix(p.shadeGain, p.sunGain, lit)), keep);
      const transformed = mix(vec3(y), normalized, chroma).mul(gain).max(0);
      out.assign(mix(albedo, transformed, p.strength.mul(float(1).sub(night))));
    });
    return out;
  })();
}
