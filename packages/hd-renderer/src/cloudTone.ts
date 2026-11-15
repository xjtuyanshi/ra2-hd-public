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
  cos,
  sin,
  texture,
  vec2,
  clamp,
} from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Mean-only surface grading, with multiplicative source detail. Artistic values belong to the caller. */
export interface CloudToneParams {
  grassLight: number;
  grassHue: number;
  grassChroma: number;
  shadeChroma: number;
  bareLight: number;
  objectShare: number;
  objectFloor: number;
  snowLift: number;
  detail: number;
  channelFloor: number;
  goldTint: [number, number, number];
  wetTint: [number, number, number];
  objectShoulder: number;
  objectHighlight: number;
  bounce: number;
  bounceHeight: number;
  greenHue: [number, number, number, number];
  greenSat: [number, number];
  contextHue: [number, number, number, number];
  strength: number;
  bareDepth: number;
  cloudTintShare: number;
  snowWhite: number;
  neutralGradeLuma: number;
  neutralGradeTint: [number, number, number];
  snowGate: [number, number, number, number];
  shadeHue: number;
  gold: number;
  contact: number;
  wet: number;
  snowBlue: number;
  snow: number;
  nightKeep: number;
  sunGain: number;
  shadeTint: [number, number, number];
  moonTint: [number, number, number];
  grassSunTint: [number, number, number];
  snowShadeTint: [number, number, number];
  wetDepth: number;
  wetChroma: number;
  wetSand: number;
  wetSandTint: [number, number, number];
  wetSandDepth: number;
  wetSandFloor: number;
  wetSheen: number;
  wetSheenGain: number;
  wetSheenStart: number;
  /** Wet-band share (0..1 of the shore mask) where wet sand begins; 0 darkens across the whole band. */
  wetSandStart: number;
  /** Painted shallow water over sand inside the band takes wetShallowTint instead of becoming wet sand. */
  wetShallow: number;
  wetShallowTint: [number, number, number];
  wetShallowLight: number;
  /** Source-pixel water classification on (b - r / 2) / luma: [sand edge, water edge]. */
  wetPaintQ: [number, number];
  /** Pale swash line where wet sand meets the painted shallows: desaturate, then lift by wetLipLift. */
  wetLip: number;
  wetLipLift: number;
  /** Pale water film over the last stretch of wet sand before the waterline (from wetFilmStart), times wetLip. */
  wetFilm: number;
  wetFilmStart: number;
  treeBlend: number;
  treeReference: [number, number, number];
  treeContact: number;
  treeContactHeight: number;
  highlightGate: [number, number, number, number];
}
export class CloudToneUniforms {
  readonly grassLight = uniform(1);
  readonly grassHue = uniform(0);
  readonly grassChroma = uniform(1);
  readonly shadeChroma = uniform(1);
  readonly bareLight = uniform(1);
  readonly objectShare = uniform(0);
  readonly objectFloor = uniform(1);
  readonly snowLift = uniform(1);
  readonly detail = uniform(1);
  readonly channelFloor = uniform(0);
  readonly goldTint = uniform(new Vector3(1, 1, 1));
  readonly wetTint = uniform(new Vector3(1, 1, 1));
  readonly objectShoulder = uniform(0);
  readonly objectHighlight = uniform(1);
  readonly bounce = uniform(0);
  readonly bounceHeight = uniform(1);
  readonly greenHue = uniform(new Vector4());
  readonly greenSat = uniform(new Vector2());
  readonly contextHue = uniform(new Vector4());
  readonly strength = uniform(0);
  readonly bareDepth = uniform(1);
  readonly cloudTintShare = uniform(1);
  readonly snowWhite = uniform(0);
  readonly neutralGradeLuma = uniform(1);
  readonly neutralGradeTint = uniform(new Vector3(1, 1, 1));
  readonly snowGate = uniform(new Vector4());
  readonly shadeHue = uniform(0);
  readonly gold = uniform(0);
  readonly contact = uniform(0);
  readonly wet = uniform(0);
  readonly snowBlue = uniform(0);
  readonly snow = uniform(0);
  readonly nightKeep = uniform(1);
  readonly sunGain = uniform(1);
  readonly shadeTint = uniform(new Vector3(1, 1, 1));
  readonly moonTint = uniform(new Vector3(1, 1, 1));
  readonly grassSunTint = uniform(new Vector3(1, 1, 1));
  readonly snowShadeTint = uniform(new Vector3(1, 1, 1));
  readonly wetDepth = uniform(0);
  readonly wetChroma = uniform(1);
  readonly wetSand = uniform(0);
  readonly wetSandTint = uniform(new Vector3(1, 1, 1));
  readonly wetSandDepth = uniform(0);
  readonly wetSandFloor = uniform(0);
  readonly wetSheen = uniform(0);
  readonly wetSheenGain = uniform(0);
  readonly wetSheenStart = uniform(0);
  readonly wetSandStart = uniform(0);
  readonly wetShallow = uniform(0);
  readonly wetShallowTint = uniform(new Vector3(1, 1, 1));
  readonly wetShallowLight = uniform(1);
  readonly wetPaintQ = uniform(new Vector2(0, 1));
  readonly wetLip = uniform(0);
  readonly wetLipLift = uniform(0);
  readonly wetFilm = uniform(0);
  readonly wetFilmStart = uniform(0);
  readonly treeBlend = uniform(0);
  readonly treeReference = uniform(new Vector3(1, 1, 1));
  readonly treeContact = uniform(0);
  readonly treeContactHeight = uniform(1);
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

/**
 * Snow theater objects (trees, rocks, buildings) sit in the same light as the lifted white snow ground: every object
 * pixel gets part of the snow lift (half of objectShare), and only bright, low-saturation pixels with a cool cast (the
 * snow on trees and rocks) are neutralised and lifted like the ground. Warm or dark paint keeps its colour.
 */
export function cloudObjectSnow(
  rgb: readonly [number, number, number],
  p: CloudToneParams,
  night = 0,
): [number, number, number] {
  if (p.snow <= 0) return [...rgb];
  const d = displayRgb(rgb);
  const y = Math.max(luma(rgb), 1e-6);
  const hi = Math.max(...d),
    lo = Math.min(...d);
  const w =
    p.snow *
    smooth(0, 0.03, d[2]! - Math.min(d[0]!, d[1]!)) *
    smooth(0.15, 0.35, luma(d)) *
    (1 - smooth(p.snowGate[2], p.snowGate[3], (hi - lo) / Math.max(hi, 1e-6))) *
    (1 - night);
  const neutral = rgb.map((v) => blend(v, y, w * p.snowWhite));
  const lifted = y * blend(1 + (p.snowLift - 1) * p.objectShare * 0.5, p.snowLift, w);
  const excess = Math.max(0, lifted - p.neutralGradeLuma);
  const scale = (lifted - excess ** 2 / (1 - p.neutralGradeLuma + excess)) / y;
  return neutral.map((v) => v * blend(1, scale, 1 - night)) as [number, number, number];
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
  const amount = p.strength * blend(1, p.nightKeep, night);
  if (amount === 0) return [...rgb];
  const y = luma(rgb),
    my = Math.max(luma(mean), 1e-6),
    lit = 1 - shade;
  if (object) {
    const src = cloudObjectSnow(rgb, p, night);
    const tinted = src.map(
      (v, i) =>
        v *
        blend(1, p.shadeTint[i]!, shade) *
        blend(1, p.moonTint[i]!, night) *
        blend(1, p.grassSunTint[i]! ** 0.25, p.bounce * (1 - smooth(0, p.bounceHeight, height)) * (1 - night)),
    );
    const sy = luma(src);
    const shoulder = 1 - (p.objectShoulder * Math.max(0, sy - p.objectHighlight)) / Math.max(sy, 1e-6);
    return tinted.map((v, i) =>
      Math.max(
        0,
        blend(
          rgb[i]!,
          ((v * sy) / Math.max(luma(tinted), 1e-6)) *
            Math.max(p.objectFloor, blend(1, p.grassLight, p.objectShare)) *
            shoulder,
          amount,
        ),
      ),
    ) as [number, number, number];
  }
  if (p.snow > 0) {
    const gain = cloudToneGain(mean, context, shade, p, night);
    return rgb.map((v, i) => Math.max(0, v * blend(1, gain[i]!, amount))) as [number, number, number];
  }
  const gain = cloudToneGain(mean, context, shade, p, night);
  const detailed = rgb.map((v, i) => v * gain[i]!);
  const materialGreen = cloudGreenness(mean, p);
  if (p.detail !== 1) {
    const contrast = (Math.max(y, 1e-6) / my) ** ((p.detail - 1) * (1 - night));
    detailed.forEach((v, i) => {
      detailed[i] = v * contrast;
    });
  }
  const wetGate = smooth(0, 0.3, wet);
  const wetShare = p.wet * wet * (1 - night);
  const tintY = Math.max(luma(p.wetTint), 1e-6);
  const bandSource = detailed.map((v, i) => blend(v, rgb[i]!, wetGate));
  detailed.forEach((_, i) => {
    detailed[i] = bandSource[i]! * blend(1, (p.wetTint[i]! * (1 - p.wetDepth)) / tintY, wetShare);
  });
  const wetY = luma(detailed);
  detailed.forEach((v, i) => {
    detailed[i] = wetY + (v - wetY) * blend(1, p.wetChroma, wetShare);
  });
  if (p.wetSand > 0 && wet > 0) {
    const ramp = smooth(p.wetSandStart, 1, wet);
    const share = ramp * Math.min(1, p.wet) * (1 - night);
    // Contract distance from a luminance floor: dim rim art cannot be multiplied into an olive-black strip.
    const targetY = p.wetSandFloor + Math.max(0, y - p.wetSandFloor) * (1 - p.wetSandDepth);
    const sandY = blend(y, targetY, ramp);
    const sandTintY = Math.max(luma(p.wetSandTint), 1e-6);
    // Shore tiles paint translucent shallows over the sand: those pixels keep reading as water (turquoise, from the
    // source luminance) and the crossing becomes a pale swash line, so tan and turquoise never mix into olive.
    const band = wetGate * Math.min(1, p.wet) * (1 - night);
    const water = smooth(p.wetPaintQ[0], p.wetPaintQ[1], (rgb[2] - 0.5 * rgb[0]) / Math.max(y, 1e-6));
    const shallow = water * band * p.wetShallow;
    const shallowTintY = Math.max(luma(p.wetShallowTint), 1e-6);
    const out = bandSource.map((b, i) =>
      // The new path starts before the r3 chroma expansion: partial replacement never boosts source green.
      blend(
        blend(b, (sandY * p.wetSandTint[i]!) / sandTintY, share),
        (y * p.wetShallowLight * p.wetShallowTint[i]!) / shallowTintY,
        shallow,
      ),
    );
    const lip = 4 * water * (1 - water) * band * p.wetLip * lit;
    const film = smooth(p.wetFilmStart, 1, wet) * band * p.wetLip * p.wetFilm;
    const outY = luma(out) * (1 + p.wetLipLift);
    detailed.forEach((v, i) => {
      const lipped = blend(out[i]!, outY, lip);
      const filmed = blend(lipped, outY * blend(1, p.wetShallowTint[i]! / shallowTintY, 0.4), film);
      detailed[i] = blend(v, filmed, p.wetSand);
    });
  }
  if (p.wetSheen > 0 && wet > 0) {
    const sheen =
      1 + p.wetSheenGain * p.wetSheen * smooth(p.wetSheenStart, 1, wet) * Math.min(1, p.wet) * (1 - night) * lit;
    detailed.forEach((v, i) => {
      detailed[i] = v * sheen;
    });
  }
  const gold =
    p.gold *
    materialGreen *
    (1 - wetGate) *
    lit ** 2 *
    (1 - night) *
    smooth(p.highlightGate[2], p.highlightGate[3], y / my);
  const warm = detailed.map((v, i) => v * p.goldTint[i]!);
  const warmY = Math.max(luma(warm), 1e-6),
    outY = luma(detailed);
  return detailed.map((v, i) =>
    Math.max(0, blend(rgb[i]!, blend(v, ((warm[i]! * outY) / warmY) * p.sunGain, gold), amount)),
  ) as [number, number, number];
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
    const amount = p.strength.mul(mix(float(1), p.nightKeep, night)).toVar();
    If(amount.greaterThan(0), () => {
      const y = dot(source, vec3(...W)).toVar();
      if (object) {
        const src = source.toVar();
        If(p.snow.greaterThan(0), () => {
          // GPU twin of cloudObjectSnow.
          const d: ShaderNode = (sRGBTransferOETF(source.max(0)) as ShaderNode).toVar();
          const hi = d.r.max(d.g).max(d.b),
            lo = d.r.min(d.g).min(d.b);
          const w = p.snow
            .mul(smoothstep(0, 0.03, d.b.sub(d.r.min(d.g))))
            .mul(smoothstep(0.15, 0.35, dot(d, vec3(...W))))
            .mul(float(1).sub(smoothstep(p.snowGate.z, p.snowGate.w, hi.sub(lo).div(hi.max(1e-6)))))
            .mul(float(1).sub(night))
            .toVar();
          const sy = y.max(1e-6);
          const lifted = sy
            .mul(mix(float(1).add(p.snowLift.sub(1).mul(p.objectShare).mul(0.5)), p.snowLift, w))
            .toVar();
          const excess = lifted.sub(p.neutralGradeLuma).max(0);
          const scale = lifted.sub(excess.pow(2).div(float(1).sub(p.neutralGradeLuma).add(excess))).div(sy);
          src.assign(mix(source, vec3(y), w.mul(p.snowWhite)).mul(mix(float(1), scale, float(1).sub(night))));
        });
        const oy = dot(src, vec3(...W)).toVar();
        const bounce = p.bounce.mul(float(1).sub(smoothstep(0, p.bounceHeight, height))).mul(float(1).sub(night));
        const tinted = src
          .mul(mix(vec3(1), p.shadeTint, shade))
          .mul(mix(vec3(1), p.moonTint, night))
          .mul(mix(vec3(1), p.grassSunTint.pow(0.25), bounce))
          .toVar();
        const shoulder = float(1).sub(p.objectShoulder.mul(oy.sub(p.objectHighlight).max(0)).div(oy.max(1e-6)));
        result.assign(
          mix(
            source,
            tinted
              .mul(oy.div(dot(tinted, vec3(...W)).max(1e-6)))
              .mul(p.objectFloor.max(mix(float(1), p.grassLight, p.objectShare)))
              .mul(shoulder),
            amount,
          ).max(0),
        );
      } else {
        If(p.snow.greaterThan(0), () => {
          result.assign(mix(source, source.mul(cloudToneGainNode(mean, context, shade, p, night)), amount).max(0));
        }).Else(() => {
          const gain = cloudToneGainNode(mean, context, shade, p, night);
          result.assign(
            cloudToneMappedNode(source, gain, dot(mean, vec3(...W)), cloudGreennessNode(mean, p), shade, p, night, wet),
          );
        });
      }
    });
    return result;
  })();
}

/** Atlas-only fallback: without a spatial mean, use the source as its own mean/context. */
export function cloudSnowToneNode(
  source: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
): ShaderNode {
  return mix(
    source,
    source.mul(cloudToneGainNode(source, source, shade, p, night)),
    p.strength.mul(mix(float(1), p.nightKeep, night)),
  ).max(0);
}

/** Mean-only snow recipe. Relative luminance admits dim mission snow without classifying individual strokes. */
function snowEndpointGain(
  mean: readonly [number, number, number],
  context: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night: number,
): [number, number, number] {
  const y = Math.max(luma(mean), 1e-6);
  const gate =
    p.snow *
    (1 - night) *
    smooth(p.snowGate[0], p.snowGate[1], luma(displayRgb(mean)) / Math.max(luma(displayRgb(context)), 0.05)) *
    (1 - smooth(p.snowGate[2], p.snowGate[3], displayHsv(mean).sat));
  const neutral = mean.map((v, i) => blend(v / y, 1, p.snowWhite) * p.neutralGradeTint[i]!);
  const chroma = neutral.map((v, i) => v / Math.max(mean[i]!, 1e-6));
  const cy = Math.max(luma(chroma), 1e-6);
  const display = displayRgb(mean);
  // Warm and neutral blocks receive only scalar lift; only a cool cast needs neutralisation.
  const cool = smooth(0, 0.03, display[2]! - Math.min(display[0]!, display[1]!));
  const lifted = y * p.snowLift;
  const excess = Math.max(0, lifted - p.neutralGradeLuma);
  const shoulder = lifted - excess ** 2 / (1 - p.neutralGradeLuma + excess);
  return mean.map(
    (_, i) =>
      blend(
        1,
        (shoulder / y) *
          p.bareDepth *
          blend(1, chroma[i]! / cy, cool) *
          blend(1, p.snowShadeTint[i]!, p.snowBlue * shade),
        gate,
      ) * blend(1, p.moonTint[i]!, night),
  ) as [number, number, number];
}

function snowEndpointGainNode(
  mean: ShaderNode,
  context: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
): ShaderNode {
  const y = dot(mean, vec3(...W))
    .max(1e-6)
    .toVar();
  const display: ShaderNode = (sRGBTransferOETF(mean.max(0)) as ShaderNode).toVar();
  const high = display.r.max(display.g).max(display.b),
    low = display.r.min(display.g).min(display.b);
  const scene: ShaderNode = sRGBTransferOETF(context.max(0));
  const gate = p.snow
    .mul(float(1).sub(night))
    .mul(smoothstep(p.snowGate.x, p.snowGate.y, dot(display, vec3(...W)).div(dot(scene, vec3(...W)).max(0.05))))
    .mul(float(1).sub(smoothstep(p.snowGate.z, p.snowGate.w, high.sub(low).div(high.max(1e-6)))));
  const neutral = mix(mean.div(y), vec3(1), p.snowWhite).mul(p.neutralGradeTint).toVar();
  const chroma = neutral.div(mean.max(1e-6)).toVar();
  const cool = smoothstep(0, 0.03, display.b.sub(display.r.min(display.g)));
  const lifted = y.mul(p.snowLift),
    excess = lifted.sub(p.neutralGradeLuma).max(0);
  const shoulder = lifted.sub(excess.pow(2).div(float(1).sub(p.neutralGradeLuma).add(excess)));
  return mix(
    vec3(1),
    mix(vec3(1), chroma.div(dot(chroma, vec3(...W)).max(1e-6)), cool)
      .mul(shoulder.div(y))
      .mul(p.bareDepth)
      .mul(mix(vec3(1), p.snowShadeTint, p.snowBlue.mul(shade))),
    gate,
  ).mul(mix(vec3(1), p.moonTint, night));
}

/** OKLab transforms operate in linear light. No game-specific palette values enter this module. */
const LMS = [
  [0.4122214708, 0.5363325363, 0.0514459929],
  [0.2119034982, 0.6806995451, 0.1073969566],
  [0.0883024619, 0.2817188376, 0.6299787005],
] as const;
const LAB = [
  [0.2104542553, 0.793617785, -0.0040720468],
  [1.9779984951, -2.428592205, 0.4505937099],
  [0.0259040371, 0.7827717662, -0.808675766],
] as const;
const INVLAB = [
  [1, 0.3963377774, 0.2158037573],
  [1, -0.1055613458, -0.0638541728],
  [1, -0.0894841775, -1.291485548],
] as const;
const INVLMS = [
  [4.0767416621, -3.3077115913, 0.2309699292],
  [-1.2684380046, 2.6097574011, -0.3413193965],
  [-0.0041960863, -0.7034186147, 1.707614701],
] as const;
const transform = (rgb: readonly number[], rows: readonly (readonly number[])[]) =>
  rows.map((row) => row.reduce((sum, v, i) => sum + v * rgb[i]!, 0));
const transformNode = (rgb: ShaderNode, rows: readonly (readonly [number, number, number])[]): ShaderNode =>
  vec3(dot(rgb, vec3(...rows[0]!)), dot(rgb, vec3(...rows[1]!)), dot(rgb, vec3(...rows[2]!)));

/** Smooth mean classification, also stored in the shade target's alpha. */
export function cloudGreennessNode(mean: ShaderNode, p: CloudToneUniforms, context = false): ShaderNode {
  const d: ShaderNode = (sRGBTransferOETF(mean.max(0)) as ShaderNode).toVar();
  const hi = d.r.max(d.g).max(d.b),
    lo = d.r.min(d.g).min(d.b),
    delta = hi.sub(lo),
    divisor = delta.max(1e-6);
  const h = hi
    .equal(d.r)
    .select(
      d.g.sub(d.b).div(divisor),
      hi.equal(d.g).select(d.b.sub(d.r).div(divisor).add(2), d.r.sub(d.g).div(divisor).add(4)),
    );
  const hue = h.mul(60).add(360).mod(360).toVar(),
    gate = context ? p.contextHue : p.greenHue;
  return smoothstep(gate.x, gate.y, hue)
    .mul(float(1).sub(smoothstep(gate.z, gate.w, hue)))
    .mul(smoothstep(p.greenSat.x, p.greenSat.y, delta.div(hi.max(1e-6))));
}

/** Mean-only gain, before strength, source detail, wet shoreline and highlight gold. */
function cloudToneEndpointGain(
  mean: readonly [number, number, number],
  context: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
): [number, number, number] {
  if (p.snow > 0) return snowEndpointGain(mean, context, shade, p, night);
  const green = cloudGreenness(mean, p),
    bare = cloudGreenness(context, { ...p, greenHue: p.contextHue }) * (1 - green);
  const lab = transform(
    transform(mean, LMS).map((v) => Math.cbrt(Math.max(0, v))),
    LAB,
  );
  const angle = ((p.grassHue + p.shadeHue * shade) * Math.PI) / 180;
  const c = p.grassChroma * blend(1, p.shadeChroma, shade);
  const a = lab[1]!,
    b = lab[2]!;
  const grass = transform(
    transform(
      [
        lab[0]! * p.grassLight,
        (a * Math.cos(angle) - b * Math.sin(angle)) * c,
        (a * Math.sin(angle) + b * Math.cos(angle)) * c,
      ],
      INVLAB,
    ).map((v) => v ** 3),
    INVLMS,
  );
  // Soft gamut compression around linear luma preserves neutral energy and never clips a channel.
  const gy = Math.max(luma(grass), 1e-6),
    floor = linearRgb([p.channelFloor * displayRgb([gy])[0]!])[0]!;
  const bound = (gy - floor) / Math.max(gy - Math.min(...grass), 1e-6);
  const safe = Math.max(0, softMin(1, bound, 0.001));
  const graded = mean.map((v, i) => blend(v, gy + (grass[i]! - gy) * safe, green) + v * (p.bareLight ** 3 - 1) * bare);
  return graded.map(
    (v, i) => (blend(v, mean[i]!, night) * blend(1, p.moonTint[i]!, night)) / Math.max(mean[i]!, 1e-6),
  ) as [number, number, number];
}

/** Shade is interpolated between the same two endpoint recipes stored by CloudToneMap. */
export function cloudToneGain(
  mean: readonly [number, number, number],
  context: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
): [number, number, number] {
  const lit = cloudToneEndpointGain(mean, context, 0, p, night),
    shadow = cloudToneEndpointGain(mean, context, 1, p, night);
  return lit.map((v, i) => blend(v, shadow[i]!, shade)) as [number, number, number];
}

/** GPU twin of cloudToneGain. Expensive classification/OKLab work belongs in the small cached targets. */
export function cloudToneEndpointGainNode(
  mean: ShaderNode,
  context: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
): ShaderNode {
  return Fn(() => {
    // Both uniform recipe branches and the alpha writer must see initialized low-pass values.
    mean.toStack();
    context.toStack();
    const result = vec3(1).toVar();
    If(p.snow.greaterThan(0), () => {
      result.assign(snowEndpointGainNode(mean, context, shade, p, night));
    }).Else(() => {
      const green = cloudGreennessNode(mean, p).toVar();
      const bare = cloudGreennessNode(context, p, true).mul(float(1).sub(green));
      const lab = transformNode(
        transformNode(mean, LMS)
          .max(0)
          .pow(1 / 3),
        LAB,
      ).toVar();
      const angle = p.grassHue.add(p.shadeHue.mul(shade)).mul(Math.PI / 180);
      const c = p.grassChroma.mul(mix(float(1), p.shadeChroma, shade));
      const cone = transformNode(
        vec3(
          lab.x.mul(p.grassLight),
          lab.y
            .mul(cos(angle))
            .sub(lab.z.mul(sin(angle)))
            .mul(c),
          lab.y
            .mul(sin(angle))
            .add(lab.z.mul(cos(angle)))
            .mul(c),
        ),
        INVLAB,
      ).toVar();
      const grass = transformNode(cone.mul(cone).mul(cone), INVLMS).toVar();
      const gy = dot(grass, vec3(...W))
        .max(1e-6)
        .toVar();
      const displayY = gy.lessThanEqual(0.0031308).select(
        gy.mul(12.92),
        gy
          .pow(1 / 2.4)
          .mul(1.055)
          .sub(0.055),
      );
      const df = displayY.mul(p.channelFloor).toVar();
      const floor = df.lessThanEqual(0.04045).select(df.div(12.92), df.add(0.055).div(1.055).pow(2.4));
      const bound = gy.sub(floor).div(gy.sub(grass.r.min(grass.g).min(grass.b)).max(1e-6));
      const safe = float(1).add(bound).sub(float(1).sub(bound).pow(2).add(0.000001).sqrt()).mul(0.5).max(0);
      const graded = mix(mean, vec3(gy).add(grass.sub(gy).mul(safe)), green).add(
        mean.mul(p.bareLight.pow(3).sub(1)).mul(bare),
      );
      result.assign(
        mix(graded, mean, night)
          .mul(mix(vec3(1), p.moonTint, night))
          .div(mean.max(1e-6)),
      );
    });
    return result;
  })();
}

/** Cheap full-resolution finish shared by terrain, walls, the fallback and the CPU reference. */
export function cloudToneMappedNode(
  source: ShaderNode,
  gain: ShaderNode,
  meanY: ShaderNode,
  green: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
  wet: ShaderNode = float(0),
): ShaderNode {
  return Fn(() => {
    const y = dot(source, vec3(...W)),
      my = meanY.max(1e-6);
    const detailed = source.mul(gain).toVar();
    If(p.snow.lessThanEqual(0), () => {
      If(p.detail.notEqual(1), () => {
        detailed.mulAssign(
          y
            .max(1e-6)
            .div(my)
            .pow(p.detail.sub(1).mul(float(1).sub(night))),
        );
      });
      const wetGate = smoothstep(0, 0.3, wet),
        wetShare = p.wet.mul(wet).mul(float(1).sub(night));
      If(wet.greaterThan(0), () => {
        const bandSource = mix(detailed, source, wetGate).toVar();
        detailed.assign(
          bandSource.mul(
            mix(vec3(1), p.wetTint.mul(float(1).sub(p.wetDepth)).div(dot(p.wetTint, vec3(...W)).max(1e-6)), wetShare),
          ),
        );
        const wy = dot(detailed, vec3(...W)).toVar();
        detailed.assign(vec3(wy).add(detailed.sub(wy).mul(mix(float(1), p.wetChroma, wetShare))));
        If(p.wetSand.greaterThan(0), () => {
          const ramp = smoothstep(p.wetSandStart, 1, wet);
          const share = ramp.mul(p.wet.min(1)).mul(float(1).sub(night));
          const targetY = p.wetSandFloor.add(y.sub(p.wetSandFloor).max(0).mul(float(1).sub(p.wetSandDepth)));
          const sandY = mix(y, targetY, ramp);
          const sand = p.wetSandTint.mul(sandY.div(dot(p.wetSandTint, vec3(...W)).max(1e-6)));
          // Painted shallows over sand stay water (turquoise); the crossing becomes a pale swash line (CPU twin).
          const band = wetGate.mul(p.wet.min(1)).mul(float(1).sub(night));
          const water = smoothstep(
            p.wetPaintQ.x,
            p.wetPaintQ.y,
            source.b.sub(source.r.mul(0.5)).div(y.max(1e-6)),
          ).toVar();
          const shallow = p.wetShallowTint.mul(
            y.mul(p.wetShallowLight).div(dot(p.wetShallowTint, vec3(...W)).max(1e-6)),
          );
          const out = mix(mix(bandSource, sand, share), shallow, water.mul(band).mul(p.wetShallow)).toVar();
          const lip = water.mul(float(1).sub(water)).mul(4).mul(band).mul(p.wetLip).mul(float(1).sub(shade));
          const film = smoothstep(p.wetFilmStart, 1, wet).mul(band).mul(p.wetLip).mul(p.wetFilm);
          const outY = dot(out, vec3(...W))
            .mul(float(1).add(p.wetLipLift))
            .toVar();
          const pale = mix(vec3(1), p.wetShallowTint.div(dot(p.wetShallowTint, vec3(...W)).max(1e-6)), 0.4).mul(outY);
          out.assign(mix(mix(out, vec3(outY), lip), pale, film));
          detailed.assign(mix(detailed, out, p.wetSand));
        });
        If(p.wetSheen.greaterThan(0), () => {
          const sheen = float(1).add(
            p.wetSheen
              .mul(p.wetSheenGain)
              .mul(smoothstep(p.wetSheenStart, 1, wet))
              .mul(p.wet.min(1))
              .mul(float(1).sub(night))
              .mul(float(1).sub(shade)),
          );
          detailed.mulAssign(sheen);
        });
      });
      If(p.gold.greaterThan(0), () => {
        const gold = p.gold
          .mul(green)
          .mul(float(1).sub(wetGate))
          .mul(float(1).sub(shade).pow(2))
          .mul(float(1).sub(night))
          .mul(smoothstep(p.highlightGate.z, p.highlightGate.w, y.div(my)));
        const warm = detailed.mul(p.goldTint).toVar();
        detailed.assign(
          mix(detailed, warm.mul(dot(detailed, vec3(...W)).div(dot(warm, vec3(...W)).max(1e-6))).mul(p.sunGain), gold),
        );
      });
    });
    return mix(source, detailed, p.strength.mul(mix(float(1), p.nightKeep, night))).max(0);
  })();
}

/** The atlas reference interpolates the same endpoint gains as mapped terrain. */
export function cloudToneGainNode(
  mean: ShaderNode,
  context: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
): ShaderNode {
  return mix(
    cloudToneEndpointGainNode(mean, context, float(0), p, night),
    cloudToneEndpointGainNode(mean, context, float(1), p, night),
    shade,
  );
}

/** CPU foliage twin: reference-grass gain, neutral on non-green pixels; contact is separate from the grade. */
export function cloudTreeToneRgb(
  rgb: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
  height = 0,
  eligible = true,
): [number, number, number] {
  const object = cloudToneRgb(rgb, shade, p, night, rgb, true, 0, height);
  if (!eligible || p.treeBlend === 0 || p.strength === 0) return object;
  const ref = p.treeReference;
  const gain = cloudToneGain(ref, ref, shade, p, night);
  const green = cloudGreenness(rgb, p);
  const lum = luma(ref.map((v, i) => v * gain[i]!)) / Math.max(luma(ref), 1e-6);
  const amount = p.strength * blend(1, p.nightKeep, night);
  const contact = 1 - p.treeContact * (1 - smooth(0, p.treeContactHeight, height)) * (1 - 0.5 * night);
  return object.map((v, i) =>
    blend(v, Math.max(0, blend(rgb[i]!, rgb[i]! * blend(lum, gain[i]!, green), amount)) * contact, p.treeBlend),
  ) as [number, number, number];
}

/** Caller owns the tree-only instance branch; this endpoint introduces no texture reads. */
export function cloudTreeToneNode(
  source: ShaderNode,
  object: ShaderNode,
  shade: ShaderNode,
  p: CloudToneUniforms,
  night: ShaderNode,
  height: ShaderNode,
): ShaderNode {
  // Reference grading depends only on uniforms and the instance's foot shade, never the source pixel.
  const ref = p.treeReference;
  const gain = cloudToneGainNode(ref, ref, shade, p, night).toVar();
  const green = cloudGreennessNode(source, p);
  const lum = dot(ref.mul(gain), vec3(...W)).div(dot(ref, vec3(...W)).max(1e-6));
  const treeGain = mix(vec3(lum), gain, green);
  const ground = mix(source, source.mul(treeGain), p.strength.mul(mix(float(1), p.nightKeep, night))).max(0);
  const contact = float(1).sub(
    p.treeContact.mul(float(1).sub(smoothstep(0, p.treeContactHeight, height))).mul(float(1).sub(night.mul(0.5))),
  );
  return mix(object, ground.mul(contact), p.treeBlend);
}
