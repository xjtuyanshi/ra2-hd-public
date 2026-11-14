import { Vector3, Vector4 } from 'three/webgpu';
import { uniform, vec3, dot, mix, smoothstep, Fn, If, float, sRGBTransferOETF } from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Multiplicative surface dressing. Policy supplies all artistic values; no source pixels are averaged into output. */
export interface CloudToneParams {
  strength: number;
  depth: number;
  bareDepth: number;
  chroma: number;
  bareChroma: number;
  cloudTintShare: number;
  snowWhite: number;
  neutralGradeLuma: number;
  neutralGradeTint: [number, number, number];
  snowGate: [number, number, number, number];
  shadeHue: number;
  gold: number;
  unitMatch: number;
  contact: number;
  wet: number;
  snowBlue: number;
  snow: number;
  nightKeep: number;
  sunGain: number;
  shadeTint: [number, number, number];
  moonTint: [number, number, number];
  grassShadeTint: [number, number, number];
  grassSunTint: [number, number, number];
  goldTint: [number, number, number];
  snowShadeTint: [number, number, number];
  wetTint: [number, number, number];
  wetDepth: number;
  wetChroma: number;
  highlightGate: [number, number, number, number];
}
export class CloudToneUniforms {
  readonly strength = uniform(0);
  readonly depth = uniform(1);
  readonly bareDepth = uniform(1);
  readonly chroma = uniform(1);
  readonly bareChroma = uniform(1);
  readonly cloudTintShare = uniform(1);
  readonly snowWhite = uniform(0);
  readonly neutralGradeLuma = uniform(1);
  readonly neutralGradeTint = uniform(new Vector3(1, 1, 1));
  readonly snowGate = uniform(new Vector4());
  readonly shadeHue = uniform(0);
  readonly gold = uniform(0);
  readonly unitMatch = uniform(0);
  readonly contact = uniform(0);
  readonly wet = uniform(0);
  readonly snowBlue = uniform(0);
  readonly snow = uniform(0);
  readonly nightKeep = uniform(1);
  readonly sunGain = uniform(1);
  readonly shadeTint = uniform(new Vector3(1, 1, 1));
  readonly moonTint = uniform(new Vector3(1, 1, 1));
  readonly grassShadeTint = uniform(new Vector3(1, 1, 1));
  readonly grassSunTint = uniform(new Vector3(1, 1, 1));
  readonly goldTint = uniform(new Vector3(1, 1, 1));
  readonly snowShadeTint = uniform(new Vector3(1, 1, 1));
  readonly wetTint = uniform(new Vector3(1, 1, 1));
  readonly wetDepth = uniform(0);
  readonly wetChroma = uniform(1);
  readonly highlightGate = uniform(new Vector4());
  set(p: CloudToneParams | null): void {
    if (!p) {
      this.strength.value = 0;
      return;
    }
    for (const key of Object.keys(this) as (keyof CloudToneParams)[]) {
      const u = this[key];
      if (typeof u.value === 'number') u.value = p[key] as number;
      else u.value.fromArray(p[key] as number[]);
    }
  }
  copyValuesFrom(p: CloudToneUniforms): void {
    for (const key of Object.keys(this) as (keyof CloudToneParams)[]) {
      const u = this[key],
        v = p[key].value;
      if (typeof u.value === 'number') u.value = v as number;
      else u.value.fromArray((v as Vector3 | Vector4).toArray());
    }
  }
}
const W = [0.2126, 0.7152, 0.0722] as const;
const blend = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const luma = (rgb: readonly number[]) => rgb.reduce((s, x, i) => s + x * W[i]!, 0);
/** Inverse chromatic response of a downstream display grade, with no luminance lift. */
export function cloudNeutralGradeTint(
  reference: number,
  graded: readonly number[],
  share: number,
): [number, number, number] {
  const response = graded.map((v) => blend(reference, v, share));
  const y = luma(response);
  return response.map((v) => y / Math.max(v, 1e-6)) as [number, number, number];
}
const displayRgb = (rgb: readonly number[]) =>
  rgb.map((value) => {
    const v = Math.max(value, 0);
    return v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
  });
const displayHsv = (rgb: readonly number[]) => {
  const [r, g, b] = displayRgb(rgb) as [number, number, number];
  const max = Math.max(r, g, b),
    min = Math.min(r, g, b),
    c = max - min;
  const divisor = Math.max(c, 1e-6);
  const h = max === r ? (g - b) / divisor : max === g ? (b - r) / divisor + 2 : (r - g) / divisor + 4;
  return { hue: (h * 60 + 360) % 360, sat: c / Math.max(max, 1e-6) };
};
/** Classify the broad linear-light mean in sRGB HSV; source hue is only a wide sand/cyan fade. */
export function cloudGreenness(mean: readonly number[], source?: readonly number[]): number {
  const { hue, sat } = displayHsv(mean);
  const pixel = source ? displayHsv(source) : { hue, sat };
  // Hue is undefined at grey; trust the broad mean there rather than inventing a source-pixel boundary.
  const pixelFade = source
    ? blend(1, smooth(40, 60, pixel.hue) * (1 - smooth(175, 215, pixel.hue)), smooth(0.02, 0.08, pixel.sat))
    : 1;
  return smooth(48, 62, hue) * (1 - smooth(175, 195, hue)) * smooth(0.08, 0.22, sat) * pixelFade;
}
/** CPU twin of cloudToneNode, including the highlight gate and the exact disabled identity. */
export function cloudToneRgb(
  rgb: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
  mean: readonly [number, number, number] = rgb,
  object = false,
  wet = 0,
): [number, number, number] {
  const amount = p.strength * blend(1, p.nightKeep, night) * (object ? p.unitMatch : 1);
  if (amount === 0) return [...rgb];
  // Broad taps own material eligibility; a wide source-hue fade protects sand without vetoing pale lawn detail.
  const green = object || wet > 0 ? 0 : cloudGreenness(mean, rgb) * (1 - p.snow);
  const lit = 1 - shade;
  const y = luma(rgb);
  // Absolute AND relative gates: only isolated already-bright lit grass can acquire gold.
  const gold =
    p.gold *
    green *
    lit *
    lit *
    (1 - night) *
    smooth(p.highlightGate[0], p.highlightGate[1], y) *
    smooth(p.highlightGate[2], p.highlightGate[3], y / Math.max(luma(mean), 0.02));
  const display = displayRgb(rgb);
  const displayMax = Math.max(...display);
  const snow = object
    ? 0
    : p.snow *
      (1 - night) *
      smooth(p.snowGate[0], p.snowGate[1], luma(display)) *
      (1 - smooth(p.snowGate[2], p.snowGate[3], (displayMax - Math.min(...display)) / Math.max(displayMax, 1e-6)));
  const wetShare = object ? 0 : p.wet * wet;
  const tint = p.shadeTint.map((v, i) => blend(1, v, shade * (object ? 1 : green)) * blend(1, p.moonTint[i]!, night));
  const grassTint = p.grassShadeTint;
  const sunTint = p.grassSunTint;
  const goldTint = p.goldTint;
  const snowTint = p.snowShadeTint.map((v) => blend(1, v, p.snowBlue * shade));
  const tinted = rgb.map(
    (v, i) =>
      v *
      tint[i]! *
      blend(1, blend(grassTint[i]!, sunTint[i]!, lit), green * (1 - night)) *
      blend(1, goldTint[i]!, gold) *
      blend(1, p.wetTint[i]!, wetShare),
  );
  // Chroma is a linear-light gain around the tinted luminance, never an object operation.
  const tintedY = luma(tinted);
  const chroma = object
    ? 1
    : blend(
        1,
        blend(blend(p.bareChroma, p.chroma, green) * blend(1, p.wetChroma, wetShare), 1 - p.snowWhite, snow),
        1 - night,
      );
  // Pale grass still receives tint and chroma. Ease positive chroma gain towards the nonnegative gamut
  // boundary instead of abruptly clipping a channel; rich grass, bare ground, wet banks and snow are unchanged.
  const pale = green * (1 - smooth(0.45, 0.6, displayHsv(rgb).sat));
  const extra = Math.max(0, chroma - 1);
  const room = Math.max(1e-6, tintedY / Math.max(tintedY - Math.min(...tinted), 1e-6) - 1);
  const softChroma = chroma - (pale * extra * extra) / (room + extra);
  const saturated = tinted.map(
    (v, i) =>
      Math.max(0, tintedY + (v - tintedY) * softChroma) *
      blend(1, snowTint[i]!, snow) *
      blend(1, p.neutralGradeTint[i]!, snow),
  );
  // A single scalar normalisation keeps chroma instead of mixing towards grey. Gains never lift dark pixels.
  const norm = y / Math.max(luma(saturated), 1e-6);
  const depth = object ? p.depth : blend(p.bareDepth, p.depth, green);
  // Whitening preserves luminance. Only compensated highlights need a smooth shoulder to avoid the
  // downstream grade's input clamp destroying the inverse tint; ordinary snow stays below this shoulder.
  const excess = Math.max(0, y - p.neutralGradeLuma);
  const compensated = p.neutralGradeTint.some((v) => Math.abs(v - 1) > 1e-6);
  const headroom = compensated
    ? blend(1, 1 - (excess * excess) / ((1 - p.neutralGradeLuma + excess) * Math.max(y, 1e-6)), snow)
    : 1;
  const gain =
    headroom * depth * blend(1, p.sunGain, lit * (1 - night)) * (1 - p.wetDepth * p.wet * wet * (object ? 0 : 1));
  const graded = saturated.map((v, i) => Math.max(0, blend(rgb[i]!, v * norm * gain, amount)));
  // Extrapolated strength can clip a channel below zero. Cap the resulting luminance so that clipping cannot lift it.
  const ceiling = y * blend(1, p.sunGain, lit * green * (1 - night));
  const limit = Math.min(1, ceiling / Math.max(luma(graded), 1e-6));
  return graded.map((v) => v * limit) as [number, number, number];
}

/** Coverage is shared with the cloud darkening node. Objects skip every grass-only operation. */
export function cloudToneNode(
  source: ShaderNode,
  mean: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
  object = false,
  wet: ShaderNode = float(0),
): ShaderNode {
  return Fn(() => {
    const result = source.toVar();
    const amount = p.strength
      .mul(mix(float(1), p.nightKeep, night))
      .mul(object ? p.unitMatch : 1)
      .toVar();
    If(amount.greaterThan(0), () => {
      const y = dot(source, vec3(...W)).toVar();
      const my = dot(mean, vec3(...W)).max(0.02);
      const hsv = (rgb: ShaderNode) => {
        const display: ShaderNode = sRGBTransferOETF(rgb.max(0));
        const max = display.r.max(display.g).max(display.b);
        const c = max.sub(display.r.min(display.g).min(display.b));
        const divisor = c.max(1e-6);
        const h = max
          .equal(display.r)
          .select(
            display.g.sub(display.b).div(divisor),
            max
              .equal(display.g)
              .select(display.b.sub(display.r).div(divisor).add(2), display.r.sub(display.g).div(divisor).add(4)),
          );
        return { hue: h.mul(60).add(360).mod(360), sat: c.div(max.max(1e-6)) };
      };
      const meanHsv = hsv(mean),
        pixelHsv = hsv(source);
      const green = object
        ? float(0)
        : smoothstep(48, 62, meanHsv.hue)
            .mul(float(1).sub(smoothstep(175, 195, meanHsv.hue)))
            .mul(smoothstep(0.08, 0.22, meanHsv.sat))
            .mul(
              mix(
                float(1),
                smoothstep(40, 60, pixelHsv.hue).mul(float(1).sub(smoothstep(175, 215, pixelHsv.hue))),
                smoothstep(0.02, 0.08, pixelHsv.sat),
              ),
            )
            .mul(float(1).sub(p.snow))
            .mul(wet.greaterThan(0).select(float(0), float(1)));
      const lit = float(1).sub(shade);
      const gold = p.gold
        .mul(green)
        .mul(lit.pow(2))
        .mul(float(1).sub(night))
        .mul(smoothstep(p.highlightGate.x, p.highlightGate.y, y))
        .mul(smoothstep(p.highlightGate.z, p.highlightGate.w, y.div(my)));
      // Snow eligibility is measured in sRGB luma and HSV saturation, independent of the grass classifier.
      const display: ShaderNode = sRGBTransferOETF(source.max(0));
      const displayMax = display.r.max(display.g).max(display.b);
      const pixelChroma = displayMax.sub(display.r.min(display.g).min(display.b));
      const snow = object
        ? float(0)
        : p.snow
            .mul(float(1).sub(night))
            .mul(smoothstep(p.snowGate.x, p.snowGate.y, dot(display, vec3(...W))))
            .mul(float(1).sub(smoothstep(p.snowGate.z, p.snowGate.w, pixelChroma.div(displayMax.max(1e-6)))));
      const wetShare = p.wet.mul(wet).mul(object ? 0 : 1);
      const tint = mix(vec3(1), p.shadeTint, shade.mul(object ? float(1) : green)).mul(mix(vec3(1), p.moonTint, night));
      const grassTint = p.grassShadeTint;
      const snowTint = mix(vec3(1), p.snowShadeTint, p.snowBlue.mul(shade));
      const tinted = source
        .mul(tint)
        .mul(mix(vec3(1), mix(grassTint, p.grassSunTint, lit), green.mul(float(1).sub(night))))
        .mul(mix(vec3(1), p.goldTint, gold));
      const wetTinted = tinted.mul(mix(vec3(1), p.wetTint, wetShare));
      const tintedY = dot(wetTinted, vec3(...W));
      const chroma = object
        ? float(1)
        : mix(
            float(1),
            mix(
              mix(p.bareChroma, p.chroma, green).mul(mix(float(1), p.wetChroma, wetShare)),
              float(1).sub(p.snowWhite),
              snow,
            ),
            float(1).sub(night),
          );
      const pale = green.mul(float(1).sub(smoothstep(0.45, 0.6, pixelHsv.sat)));
      const extra = chroma.sub(1).max(0);
      const room = tintedY
        .div(tintedY.sub(wetTinted.r.min(wetTinted.g).min(wetTinted.b)).max(1e-6))
        .sub(1)
        .max(1e-6);
      const softChroma = chroma.sub(pale.mul(extra.pow(2)).div(room.add(extra)));
      const saturated = vec3(tintedY)
        .add(wetTinted.sub(tintedY).mul(softChroma))
        .max(0)
        .mul(mix(vec3(1), snowTint, snow))
        .mul(mix(vec3(1), p.neutralGradeTint, snow));
      const norm = y.div(dot(saturated, vec3(...W)).max(1e-6));
      const depth = object ? p.depth : mix(p.bareDepth, p.depth, green);
      const excess = y.sub(p.neutralGradeLuma).max(0);
      const response = p.neutralGradeTint.sub(1).abs();
      const compensated = response.x.max(response.y).max(response.z).greaterThan(1e-6);
      const shoulder = float(1).sub(excess.pow(2).div(float(1).sub(p.neutralGradeLuma).add(excess).mul(y.max(1e-6))));
      const headroom = compensated.select(mix(float(1), shoulder, snow), float(1));
      const gain = depth
        .mul(headroom)
        .mul(mix(float(1), p.sunGain, lit.mul(float(1).sub(night))))
        .mul(
          float(1).sub(
            p.wet
              .mul(wet)
              .mul(object ? 0 : 1)
              .mul(p.wetDepth),
          ),
        );
      const graded = vec3(mix(source, saturated.mul(norm).mul(gain), amount))
        .max(0)
        .toVar();
      const ceiling = y.mul(mix(float(1), p.sunGain, lit.mul(green).mul(float(1).sub(night))));
      const limit = ceiling.div(dot(graded, vec3(...W)).max(1e-6)).min(1);
      result.assign(graded.mul(limit));
    });
    return result;
  })();
}
