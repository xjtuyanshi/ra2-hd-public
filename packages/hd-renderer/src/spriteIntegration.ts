import { clamp, exp, float, length, max, min, mix, sRGBTransferOETF, uniform, vec3 } from 'three/tsl';
import type { ShaderNode } from './tsl';

export const INTEGRATION_KEYS = [
  'on',
  'edgeSigma',
  'edgeSoft',
  'thinKeep',
  'paint',
  'deHalo',
  'soften',
  'highlightRoll',
  'relight',
  'specular',
  'normalSmooth',
  'rimFade',
  'shadowSoft',
  'objSharp',
  'objClarity',
  'objSoft',
  'objContrast',
  'outlineFeather',
  'vSharpen',
  'vShine',
  'vCut',
  'treeDetail',
] as const;
export type IntegrationKey = (typeof INTEGRATION_KEYS)[number];
export type IntegrationValues = Record<IntegrationKey, number>;

/** One group per renderer context, shared by all its atlas pages; never changes graph topology. */
export class SpriteIntegration {
  compiled = false;
  readonly uniforms = Object.fromEntries(INTEGRATION_KEYS.map((key) => [key, uniform(0)])) as Record<
    IntegrationKey,
    ShaderNode
  >;
  get(): IntegrationValues {
    return Object.fromEntries(INTEGRATION_KEYS.map((key) => [key, this.uniforms[key].value])) as IntegrationValues;
  }
  set(values: Partial<IntegrationValues>): void {
    for (const key of INTEGRATION_KEYS) {
      const value = values[key];
      if (typeof value === 'number' && Number.isFinite(value)) this.uniforms[key].value = value;
    }
  }
  ab(enabled: boolean): void {
    this.uniforms.on.value = enabled ? 1 : 0;
  }
}
const groups = new WeakMap<object, SpriteIntegration>();
export function spriteIntegrationFor(owner: object): SpriteIntegration {
  let group = groups.get(owner);
  if (!group) groups.set(owner, (group = new SpriteIntegration()));
  return group;
}

/** d is tap centre minus reconstruction point. Both numerator and normaliser derivatives are required. */
export function gaussianWeightNode(d: ShaderNode, sigma: ShaderNode): ShaderNode {
  return exp(d.dot(d).div(sigma.mul(sigma).mul(-2)));
}
export function quotientGradientNode(
  sum: ShaderNode,
  gradient: ShaderNode,
  norm: ShaderNode,
  normGradient: ShaderNode,
): ShaderNode {
  return gradient
    .mul(norm)
    .sub(normGradient.mul(sum))
    .div(max(norm.mul(norm), 1e-8));
}
export function edgeAANode(
  c: ShaderNode,
  threshold: ShaderNode | number,
  gradient: ShaderNode,
  px: ShaderNode,
  soft: ShaderNode,
): ShaderNode {
  return clamp(
    c
      .sub(threshold)
      .div(max(soft.mul(length(gradient)).div(px), 1e-4))
      .add(0.5),
    0,
    1,
  );
}
/** Protect both narrow solid details and narrow empty gaps with the same AA footprint. */
export function integratedCoverageNode(
  wide: ShaderNode,
  narrow: ShaderNode,
  gradWide: ShaderNode,
  gradNarrow: ShaderNode,
  px: ShaderNode,
  soft: ShaderNode,
  thinKeep: ShaderNode,
): ShaderNode {
  return clamp(
    edgeAANode(wide, 0.5, gradWide, px, soft),
    edgeAANode(narrow, thinKeep, gradNarrow, px, soft),
    edgeAANode(narrow, float(1).sub(thinKeep), gradNarrow, px, soft),
  );
}
/** Classify individual linear-light samples, allowing one sRGB code value of channel noise. */
export function nonGreyNode(rgb: ShaderNode): ShaderNode {
  const srgb: ShaderNode = sRGBTransferOETF(max(rgb, vec3(0)));
  const c: ShaderNode = srgb.toVar();
  return float(
    max(c.r, max(c.g, c.b))
      .sub(min(c.r, min(c.g, c.b)))
      .greaterThan(1 / 255),
  );
}
export function softRangeNode(value: ShaderNode, lo: ShaderNode, hi: ShaderNode): ShaderNode {
  const above = max(value.sub(hi), 0);
  const below = max(lo.sub(value), 0);
  return clamp(value, lo, hi)
    .add(above.div(float(1).add(above.mul(4).div(max(hi, 1e-3)))))
    .sub(below.div(float(1).add(below.mul(4).div(max(lo, 1e-3)))));
}
export function highlightRollNode(value: ShaderNode, knee: ShaderNode, strength: ShaderNode): ShaderNode {
  const excess = max(value.sub(knee), 0);
  return value.sub(excess).add(excess.div(float(1).add(strength.mul(3).mul(excess))));
}
export function quadraticWeightNode(t: ShaderNode, offset: number): ShaderNode {
  return offset === 0 ? float(0.75).sub(t.mul(t)) : float(0.5).add(t.mul(offset)).pow(2).mul(0.5);
}
export function quadraticGradientNode(t: ShaderNode, offset: number): ShaderNode {
  return offset === 0 ? t.mul(-2) : float(0.5).add(t.mul(offset)).mul(offset);
}
export function colourFilterWeightNode(t: ShaderNode, offset: number, softness: ShaderNode): ShaderNode {
  const tent = offset === 0 ? float(1).sub(t.abs()) : max(t.mul(offset), 0);
  return mix(tent, quadraticWeightNode(t, offset), softness);
}

// CPU twins use the same 25 taps, including empty taps in the normaliser.
export function gaussianWeight(dx: number, dy: number, sigma: number): number {
  return Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
}
export function quadraticWeight(t: number, offset: number): number {
  return offset === 0 ? 0.75 - t * t : 0.5 * (0.5 + offset * t) ** 2;
}
export function quadraticGradient(t: number, offset: number): number {
  return offset === 0 ? -2 * t : offset * (0.5 + offset * t);
}
export function colourFilterWeight(t: number, offset: number, softness: number): number {
  const tent = offset === 0 ? 1 - Math.abs(t) : Math.max(t * offset, 0);
  return tent * (1 - softness) + quadraticWeight(t, offset) * softness;
}
export function quotientGradient(
  sum: number,
  gradient: readonly number[],
  norm: number,
  normGradient: readonly number[],
): number[] {
  return gradient.map((g, axis) => (g * norm - sum * normGradient[axis]!) / Math.max(norm * norm, 1e-8));
}
export function coverageAt(x: number, y: number, occupied: (x: number, y: number) => number, sigma: number) {
  const kx = Math.floor(x),
    ky = Math.floor(y),
    tx = x - kx - 0.5,
    ty = y - ky - 0.5;
  let sum = 0,
    norm = 0,
    gx = 0,
    gy = 0,
    nx = 0,
    ny = 0,
    narrow = 0,
    ngx = 0,
    ngy = 0;
  for (let oy = -2; oy <= 2; oy++)
    for (let ox = -2; ox <= 2; ox++) {
      const dx = kx + ox + 0.5 - x,
        dy = ky + oy + 0.5 - y;
      const g = gaussianWeight(dx, dy, sigma),
        a = occupied(kx + ox, ky + oy),
        gdx = (g * dx) / (sigma * sigma),
        gdy = (g * dy) / (sigma * sigma);
      norm += g;
      sum += g * a;
      gx += gdx * a;
      gy += gdy * a;
      nx += gdx;
      ny += gdy;
      if (Math.abs(ox) <= 1 && Math.abs(oy) <= 1) {
        narrow += quadraticWeight(tx, ox) * quadraticWeight(ty, oy) * a;
        ngx += quadraticGradient(tx, ox) * quadraticWeight(ty, oy) * a;
        ngy += quadraticWeight(tx, ox) * quadraticGradient(ty, oy) * a;
      }
    }
  return {
    wide: sum / norm,
    gradWide: [(gx * norm - sum * nx) / norm ** 2, (gy * norm - sum * ny) / norm ** 2] as const,
    narrow,
    gradNarrow: [ngx, ngy] as const,
  };
}
export function edgeAA(c: number, threshold: number, gradient: readonly number[], px: number, soft: number): number {
  return Math.min(1, Math.max(0, (c - threshold) / Math.max((soft * Math.hypot(...gradient)) / px, 1e-4) + 0.5));
}
export function integratedAlpha(
  c: { wide: number; narrow: number; gradWide: readonly number[]; gradNarrow: readonly number[] },
  px: number,
  soft: number,
  thinKeep: number,
): number {
  return Math.min(
    edgeAA(c.narrow, 1 - thinKeep, c.gradNarrow, px, soft),
    Math.max(edgeAA(c.wide, 0.5, c.gradWide, px, soft), edgeAA(c.narrow, thinKeep, c.gradNarrow, px, soft)),
  );
}
export function nonGreyFraction(taps: readonly { rgb: readonly number[]; weight: number }[]): number {
  let nonGrey = 0,
    weight = 0;
  for (const tap of taps) {
    // Match three's sRGBTransferOETF constants, including its shader exponent.
    const srgb = tap.rgb.map((v) => (v <= 0.0031308 ? Math.max(v, 0) * 12.92 : 1.055 * v ** 0.41666 - 0.055));
    weight += tap.weight;
    if (Math.max(...srgb) - Math.min(...srgb) > 1 / 255) nonGrey += tap.weight;
  }
  return nonGrey / Math.max(weight, 1e-8);
}
export function softRange(value: number, lo: number, hi: number): number {
  if (value > hi) return hi + (value - hi) / (1 + (4 * (value - hi)) / Math.max(hi, 1e-3));
  if (value < lo) return lo - (lo - value) / (1 + (4 * (lo - value)) / Math.max(lo, 1e-3));
  return value;
}
export function highlightRoll(value: number, knee: number, strength: number): number {
  const excess = Math.max(value - knee, 0);
  return value - excess + excess / (1 + strength * 3 * excess);
}
