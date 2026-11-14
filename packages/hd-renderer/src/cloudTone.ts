import { Vector2, Vector3, Vector4, type Texture } from 'three/webgpu';
import {
  uniform,
  vec3,
  dot,
  mix,
  smoothstep,
  Fn,
  If,
  float,
  sRGBTransferOETF,
  sRGBTransferEOTF,
  texture,
  vec2,
  clamp,
} from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Mean-only surface grading, with multiplicative source detail. Artistic values belong to the caller. */
export interface CloudToneParams {
  detail: number;
  channelFloor: number;
  wetTint: [number, number, number];
  objectShoulder: number;
  objectHighlight: number;
  bounce: number;
  bounceHeight: number;
  greenHue: [number, number, number, number];
  greenSat: [number, number];
  contextHue: [number, number, number, number];
  bareHue: [number, number];
  strength: number;
  depth: number;
  bareDepth: number;
  chroma: number;
  bareChroma: number;
  contextChroma: number;
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
  snowShadeTint: [number, number, number];
  wetDepth: number;
  wetChroma: number;
  highlightGate: [number, number, number, number];
}
export class CloudToneUniforms {
  readonly detail = uniform(1);
  readonly channelFloor = uniform(0);
  readonly wetTint = uniform(new Vector3(1, 1, 1));
  readonly objectShoulder = uniform(0);
  readonly objectHighlight = uniform(1);
  readonly bounce = uniform(0);
  readonly bounceHeight = uniform(1);
  readonly greenHue = uniform(new Vector4());
  readonly greenSat = uniform(new Vector2());
  readonly contextHue = uniform(new Vector4());
  readonly bareHue = uniform(new Vector2());
  readonly strength = uniform(0);
  readonly depth = uniform(1);
  readonly bareDepth = uniform(1);
  readonly chroma = uniform(1);
  readonly bareChroma = uniform(1);
  readonly contextChroma = uniform(1);
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
  readonly snowShadeTint = uniform(new Vector3(1, 1, 1));
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
      else u.value.fromArray((v as Vector2 | Vector3 | Vector4).toArray());
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
/** Smooth mean-material eligibility; the wet shoreline never classifies source pixels. */
export function cloudGreenness(mean: readonly number[], p: Pick<CloudToneParams, 'greenHue' | 'greenSat'>): number {
  const { hue, sat } = displayHsv(mean);
  return (
    smooth(p.greenHue[0], p.greenHue[1], hue) *
    (1 - smooth(p.greenHue[2], p.greenHue[3], hue)) *
    smooth(p.greenSat[0], p.greenSat[1], sat)
  );
}
const linearRgb = (rgb: readonly number[]) => rgb.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
/** Smooth minimum avoids a hard channel clamp at the chroma boundary. */
const softMin = (a: number, b: number, width = 0.02) => 0.5 * (a + b - Math.sqrt((a - b) ** 2 + width ** 2));

/** Two antithetic, rotated trilinear mip samples. No level-zero stripe lattice enters the mean. */
export function cloudToneMeanNode(
  map: Texture,
  uv: ShaderNode,
  texel: ShaderNode,
  blockWidth: number,
  bounds: ShaderNode,
): ShaderNode {
  const offset = vec2(0.46, 0.19).mul(blockWidth).mul(texel);
  const sample = (sign: number) =>
    texture(map, clamp(uv.add(offset.mul(sign)), bounds.xy, bounds.zw)).level(float(Math.log2(blockWidth))).rgb;
  return sample(1).add(sample(-1)).mul(0.5);
}

/** CPU twin. RGB/mean are linear; classification and the chroma safety boundary are display-sRGB. */
export function cloudToneRgb(
  rgb: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
  mean: readonly [number, number, number] = rgb,
  object = false,
  wet = 0,
  height = 0,
  context: readonly [number, number, number] = mean,
): [number, number, number] {
  const amount = p.strength * blend(1, p.nightKeep, night) * (object ? p.unitMatch : 1);
  if (amount === 0) return [...rgb];
  const y = luma(rgb),
    my = Math.max(luma(mean), 1e-6),
    lit = 1 - shade;
  if (object) {
    const tinted = rgb.map(
      (v, i) =>
        v *
        blend(1, p.shadeTint[i]!, shade) *
        blend(1, p.moonTint[i]!, night) *
        blend(1, p.grassSunTint[i]! ** 0.25, p.bounce * (1 - smooth(0, p.bounceHeight, height)) * (1 - night)),
    );
    const shoulder = 1 - (p.objectShoulder * Math.max(0, y - p.objectHighlight)) / Math.max(y, 1e-6);
    return tinted.map((v, i) =>
      Math.max(0, blend(rgb[i]!, ((v * y) / Math.max(luma(tinted), 1e-6)) * p.depth * shoulder, amount)),
    ) as [number, number, number];
  }
  if (p.snow > 0) {
    const display = displayRgb(rgb),
      hsv = displayHsv(rgb);
    const snow =
      p.snow *
      (1 - night) *
      smooth(p.snowGate[0], p.snowGate[1], luma(display)) *
      (1 - smooth(p.snowGate[2], p.snowGate[3], hsv.sat));
    const tinted = rgb.map((v, i) => v * blend(1, p.moonTint[i]!, night));
    const ty = luma(tinted);
    const saturated = tinted.map(
      (v, i) =>
        (ty + (v - ty) * blend(1, 1 - p.snowWhite, snow)) *
        blend(1, blend(1, p.snowShadeTint[i]!, p.snowBlue * shade), snow) *
        blend(1, p.neutralGradeTint[i]!, snow),
    );
    const excess = Math.max(0, y - p.neutralGradeLuma);
    const compensated = p.neutralGradeTint.some((v) => Math.abs(v - 1) > 1e-6);
    const headroom = compensated
      ? blend(1, 1 - excess ** 2 / ((1 - p.neutralGradeLuma + excess) * Math.max(y, 1e-6)), snow)
      : 1;
    return saturated.map((v, i) =>
      Math.max(0, blend(rgb[i]!, ((v * y) / Math.max(luma(saturated), 1e-6)) * p.bareDepth * headroom, amount)),
    ) as [number, number, number];
  }
  const display = displayRgb(mean),
    hsv = displayHsv(mean);
  const materialGreen = cloudGreenness(mean, p);
  const contextGreen = cloudGreenness(context, { ...p, greenHue: p.contextHue });
  const wetGate = smooth(0, 0.3, wet);
  const green = materialGreen * (1 - wetGate);
  const fringe =
    smooth(p.bareHue[0], p.bareHue[1], hsv.hue) *
    (1 - smooth(p.greenHue[2], p.greenHue[3], hsv.hue)) *
    smooth(p.greenSat[0], p.greenSat[1], hsv.sat);
  const tinted = display.map(
    (v, i) =>
      v *
      blend(1, p.shadeTint[i]!, shade * green) *
      blend(1, blend(p.grassShadeTint[i]!, p.grassSunTint[i]!, lit), green * (1 - night)),
  );
  const ty = luma(tinted),
    low = Math.min(...tinted);
  const chroma = blend(1, blend(blend(p.bareChroma, p.contextChroma, contextGreen), p.chroma, green), 1 - night);
  // Limit mean chroma smoothly; the post-depth mean boundary below also protects the sRGB toe.
  const bound = (ty * (1 - p.channelFloor)) / Math.max(ty - low, 1e-6);
  const c = blend(chroma, Math.max(0, softMin(chroma, bound)), green * (1 - night));
  let graded = linearRgb(tinted.map((v) => ty + (v - ty) * c));
  const sandDepth = blend(1, blend(p.bareDepth, p.depth, materialGreen), fringe);
  const depth = blend(blend(sandDepth, p.depth, contextGreen), 1, wetGate);
  graded = graded.map((v, i) => v * blend(1, p.moonTint[i]!, night));
  graded = graded.map((v) => ((v * my) / Math.max(luma(graded), 1e-6)) * depth);
  // OETF is concave: display luma <= OETF(linear luma). This conservative mean-only floor
  // guarantees the display-space share without classifying or clipping source pixels.
  const gy = luma(graded),
    floor = linearRgb([p.channelFloor * displayRgb([gy])[0]!])[0]!;
  const safeChroma = Math.max(0, softMin(1, (gy - floor) / Math.max(gy - Math.min(...graded), 1e-6), 0.001));
  graded = graded.map((v) => gy + (v - gy) * blend(1, safeChroma, materialGreen * (1 - night)));
  const k = blend(1, p.detail, materialGreen * (1 - night));
  let detailed = graded.map((v, i) => {
    const ratio = mean[i]! <= 1e-6 && rgb[i]! <= 1e-6 ? 1 : Math.max(rgb[i]!, 0) / Math.max(mean[i]!, 1e-6);
    return v * ratio ** k;
  });
  // Context dirt shares lawn depth without inheriting its channel-dependent detail gain.
  const bareShare = contextGreen * (1 - materialGreen);
  const dy = Math.max(luma(detailed), 1e-6);
  detailed = detailed.map((v) => v * blend(1, (y * depth) / dy, bareShare));
  // One spatial gain for every hue. Fade the dry grade to source art before applying the shared wet recipe.
  const wetShare = p.wet * wet * (1 - night);
  const tintY = Math.max(luma(p.wetTint), 1e-6);
  detailed = detailed.map(
    (v, i) => blend(v, rgb[i]!, wetGate) * blend(1, (p.wetTint[i]! * (1 - p.wetDepth)) / tintY, wetShare),
  );
  const wetY = luma(detailed);
  detailed = detailed.map((v) => wetY + (v - wetY) * blend(1, p.wetChroma, wetShare));
  // Gold uses source highlights; its small scalar gain cannot change local hue/saturation.
  const gold =
    p.gold *
    green *
    lit ** 2 *
    (1 - night) *
    smooth(p.highlightGate[0], p.highlightGate[1], y) *
    smooth(p.highlightGate[2], p.highlightGate[3], y / my);
  return detailed.map((v, i) => Math.max(0, blend(rgb[i]!, v * blend(1, p.sunGain, gold), amount))) as [
    number,
    number,
    number,
  ];
}

/** Uniform-gated paths keep A/B graph identities stable and avoid the snow work on other surfaces. */
export function cloudToneNode(
  source: ShaderNode,
  mean: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
  object = false,
  wet: ShaderNode = float(0),
  height: ShaderNode = float(0),
  context: ShaderNode = mean,
): ShaderNode {
  return Fn(() => {
    const result = source.toVar();
    const amount = p.strength
      .mul(mix(float(1), p.nightKeep, night))
      .mul(object ? p.unitMatch : 1)
      .toVar();
    If(amount.greaterThan(0), () => {
      const y = dot(source, vec3(...W)).toVar();
      if (object) {
        const bounce = p.bounce.mul(float(1).sub(smoothstep(0, p.bounceHeight, height))).mul(float(1).sub(night));
        const tinted = source
          .mul(mix(vec3(1), p.shadeTint, shade))
          .mul(mix(vec3(1), p.moonTint, night))
          .mul(mix(vec3(1), p.grassSunTint.pow(0.25), bounce))
          .toVar();
        const shoulder = float(1).sub(p.objectShoulder.mul(y.sub(p.objectHighlight).max(0)).div(y.max(1e-6)));
        result.assign(
          mix(
            source,
            tinted
              .mul(y.div(dot(tinted, vec3(...W)).max(1e-6)))
              .mul(p.depth)
              .mul(shoulder),
            amount,
          ).max(0),
        );
      } else {
        If(p.snow.greaterThan(0), () => {
          const display: ShaderNode = sRGBTransferOETF(source.max(0));
          const high = display.r.max(display.g).max(display.b),
            low = display.r.min(display.g).min(display.b);
          const snow = p.snow
            .mul(float(1).sub(night))
            .mul(smoothstep(p.snowGate.x, p.snowGate.y, dot(display, vec3(...W))))
            .mul(float(1).sub(smoothstep(p.snowGate.z, p.snowGate.w, high.sub(low).div(high.max(1e-6)))));
          const tinted = source.mul(mix(vec3(1), p.moonTint, night));
          const ty = dot(tinted, vec3(...W));
          const saturated = vec3(ty)
            .add(tinted.sub(ty).mul(mix(float(1), float(1).sub(p.snowWhite), snow)))
            .mul(mix(vec3(1), mix(vec3(1), p.snowShadeTint, p.snowBlue.mul(shade)), snow))
            .mul(mix(vec3(1), p.neutralGradeTint, snow))
            .toVar();
          const excess = y.sub(p.neutralGradeLuma).max(0),
            response = p.neutralGradeTint.sub(1).abs();
          const compensated = response.x.max(response.y).max(response.z).greaterThan(1e-6);
          const shoulder = float(1).sub(
            excess.pow(2).div(float(1).sub(p.neutralGradeLuma).add(excess).mul(y.max(1e-6))),
          );
          const headroom = compensated.select(mix(float(1), shoulder, snow), float(1));
          result.assign(
            mix(
              source,
              saturated
                .mul(y.div(dot(saturated, vec3(...W)).max(1e-6)))
                .mul(p.bareDepth)
                .mul(headroom),
              amount,
            ).max(0),
          );
        }).Else(() => {
          const my = dot(mean, vec3(...W))
            .max(1e-6)
            .toVar();
          const display: ShaderNode = (sRGBTransferOETF(mean.max(0)) as ShaderNode).toVar();
          const high = display.r.max(display.g).max(display.b),
            low = display.r.min(display.g).min(display.b);
          const delta = high.sub(low),
            divisor = delta.max(1e-6);
          const h = high
            .equal(display.r)
            .select(
              display.g.sub(display.b).div(divisor),
              high
                .equal(display.g)
                .select(display.b.sub(display.r).div(divisor).add(2), display.r.sub(display.g).div(divisor).add(4)),
            );
          const hue = h.mul(60).add(360).mod(360).toVar(),
            sat = delta.div(high.max(1e-6));
          const eligibility = float(1)
            .sub(smoothstep(p.greenHue.z, p.greenHue.w, hue))
            .mul(smoothstep(p.greenSat.x, p.greenSat.y, sat));
          const materialGreen = smoothstep(p.greenHue.x, p.greenHue.y, hue).mul(eligibility).toVar();
          const hsvOf = (linear: ShaderNode) => {
            const d: ShaderNode = (sRGBTransferOETF(linear.max(0)) as ShaderNode).toVar();
            const hi = d.r.max(d.g).max(d.b),
              lo = d.r.min(d.g).min(d.b),
              delta = hi.sub(lo);
            const h = hi
              .equal(d.r)
              .select(
                d.g.sub(d.b).div(delta.max(1e-6)),
                hi
                  .equal(d.g)
                  .select(d.b.sub(d.r).div(delta.max(1e-6)).add(2), d.r.sub(d.g).div(delta.max(1e-6)).add(4)),
              );
            return { hue: h.mul(60).add(360).mod(360), sat: delta.div(hi.max(1e-6)), high: hi };
          };
          const ch = hsvOf(context);
          const contextGreen = smoothstep(p.contextHue.x, p.contextHue.y, ch.hue)
            .mul(float(1).sub(smoothstep(p.contextHue.z, p.contextHue.w, ch.hue)))
            .mul(smoothstep(p.greenSat.x, p.greenSat.y, ch.sat))
            .toVar();
          const wetGate = smoothstep(0, 0.3, wet);
          const green = materialGreen.mul(float(1).sub(wetGate));
          const fringe = smoothstep(p.bareHue.x, p.bareHue.y, hue).mul(eligibility);
          const lit = float(1).sub(shade);
          const tinted = display
            .mul(mix(vec3(1), p.shadeTint, shade.mul(green)))
            .mul(mix(vec3(1), mix(p.grassShadeTint, p.grassSunTint, lit), green.mul(float(1).sub(night))))
            .toVar();
          const ty = dot(tinted, vec3(...W));
          const chroma = mix(
            float(1),
            mix(mix(p.bareChroma, p.contextChroma, contextGreen), p.chroma, green),
            float(1).sub(night),
          );
          const bound = ty
            .mul(float(1).sub(p.channelFloor))
            .div(ty.sub(tinted.r.min(tinted.g).min(tinted.b)).max(1e-6));
          const c = mix(
            chroma,
            chroma.add(bound).sub(chroma.sub(bound).pow(2).add(0.0004).sqrt()).mul(0.5).max(0),
            green.mul(float(1).sub(night)),
          );
          const linear: ShaderNode = sRGBTransferEOTF(vec3(ty).add(tinted.sub(ty).mul(c)));
          const sandDepth = mix(float(1), mix(p.bareDepth, p.depth, materialGreen), fringe);
          const depth = mix(mix(sandDepth, p.depth, contextGreen), float(1), wetGate);
          const moonlit = linear.mul(mix(vec3(1), p.moonTint, night));
          const graded = moonlit
            .mul(my.div(dot(moonlit, vec3(...W)).max(1e-6)))
            .mul(depth)
            .toVar();
          const gy = dot(graded, vec3(...W))
            .max(0)
            .toVar();
          // Scalar transfer functions avoid running a three-channel conversion for this luma boundary.
          const displayY = gy.lessThanEqual(0.0031308).select(
            gy.mul(12.92),
            gy
              .pow(1 / 2.4)
              .mul(1.055)
              .sub(0.055),
          );
          const displayFloor = displayY.mul(p.channelFloor).toVar();
          const floor = displayFloor
            .lessThanEqual(0.04045)
            .select(displayFloor.div(12.92), displayFloor.add(0.055).div(1.055).pow(2.4));
          const boundAfterDepth = gy.sub(floor).div(gy.sub(graded.r.min(graded.g).min(graded.b)).max(1e-6));
          const safeChroma = float(1)
            .add(boundAfterDepth)
            .sub(float(1).sub(boundAfterDepth).pow(2).add(0.000001).sqrt())
            .mul(0.5)
            .max(0);
          graded.assign(
            vec3(gy).add(graded.sub(gy).mul(mix(float(1), safeChroma, materialGreen.mul(float(1).sub(night))))),
          );
          const k = mix(float(1), p.detail, materialGreen.mul(float(1).sub(night)));
          const ratio = (s: ShaderNode, m: ShaderNode) =>
            m
              .lessThanEqual(1e-6)
              .and(s.lessThanEqual(1e-6))
              .select(float(1), s.max(0).div(m.max(1e-6)));
          const detailRatio = vec3(ratio(source.r, mean.r), ratio(source.g, mean.g), ratio(source.b, mean.b));
          const detailed = graded.mul(detailRatio).toVar();
          If(p.detail.notEqual(1), () => {
            detailed.assign(graded.mul(detailRatio.pow(k)));
          });
          const bareShare = contextGreen.mul(float(1).sub(materialGreen));
          detailed.mulAssign(mix(float(1), y.mul(depth).div(dot(detailed, vec3(...W)).max(1e-6)), bareShare));
          // No pixel hue/saturation gates: neighbours share the same gain before their own-luma chroma step.
          const wetShare = p.wet.mul(wet).mul(float(1).sub(night));
          const wetGain = mix(
            vec3(1),
            p.wetTint.mul(float(1).sub(p.wetDepth)).div(dot(p.wetTint, vec3(...W)).max(1e-6)),
            wetShare,
          );
          detailed.assign(mix(detailed, source, wetGate).mul(wetGain));
          const wetY = dot(detailed, vec3(...W)).toVar();
          detailed.assign(vec3(wetY).add(detailed.sub(wetY).mul(mix(float(1), p.wetChroma, wetShare))));
          const gold = float(0).toVar();
          If(p.gold.greaterThan(0), () => {
            gold.assign(
              p.gold
                .mul(green)
                .mul(lit.pow(2))
                .mul(float(1).sub(night))
                .mul(smoothstep(p.highlightGate.x, p.highlightGate.y, y))
                .mul(smoothstep(p.highlightGate.z, p.highlightGate.w, y.div(my))),
            );
          });
          result.assign(mix(source, detailed.mul(mix(float(1), p.sunGain, gold)), amount).max(0));
        });
      }
    });
    return result;
  })();
}
