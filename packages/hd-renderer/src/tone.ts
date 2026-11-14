import { Vector3, Vector4 } from 'three/webgpu';
import {
  atan,
  cos,
  dot,
  float,
  max,
  mix,
  screenSize,
  screenUV,
  sin,
  smoothstep,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { cinematicGrade, cinematicGradeRgb, neutralGrade, type CinematicGradeParams } from './post';
import type { ShaderNode } from './tsl';

/**
 * A hue window in tone space (hue of the opponent pair r - g, (r + g) / 2 - b of encoded colour: about 60 for sand,
 * 95 for sunlit grass, 205 for snow and shallow water, 222 for deep water). Weight 1 within width / 2 of the centre,
 * 0 beyond width, ramped in by chroma from minChroma to twice that, and optionally out by encoded luma towards
 * maxLuma (so bright near-white colour, such as snow, can be told from saturated water of the same hue).
 */
export interface ToneHueWindow {
  centre: number;
  width: number;
  minChroma: number;
  maxLuma?: number;
}
export interface ToneHueShift extends ToneHueWindow {
  /** Degrees, positive counter-clockwise in opponent-chroma space. */
  shift: number;
}
export interface FinalToneParams {
  strength: number;
  grade: CinematicGradeParams | null;
  /**
   * Depth on encoded luma: (1 - density) y + density y^6, an exposure-like scale (linear-light ratios, and so local
   * texture and patch contrast, are kept) with a shoulder that pins white. A gamma crushed dark scenes, and fading the
   * change out on dark pixels flattened the light/dark grass patches of darker maps (jungle patch contrast -28%).
   */
  density: number;
  /**
   * Sun/shade separation on encoded luma after the density curve: y + contrast * (y - pivot) * 4y(1 - y). Black and
   * white stay fixed; values below pivot darken, above it brighten. Monotonic while contrast < 1 / (4 * max(pivot,
   * 1 - pivot)).
   */
  contrast: number;
  pivot: number;
  chroma: [number, number, number];
  /** At most two windows; unused uniform slots have zero shift. */
  hueShifts: ToneHueShift[];
  protect: ToneHueWindow | null;
  /** Share of the colour change (chroma, hue, grade tints) on object pixels (silhouette code != 0). */
  objectShare: number;
  /** Share of the density/contrast change on object pixels: low keeps units and buildings bright and readable. */
  objectDensity: number;
  foliage: ToneHueWindow | null;
  grain: number;
  night: { colour: number; grain: number };
}
export type Rgb = readonly [number, number, number];
const WEIGHTS = [0.2126, 0.7152, 0.0722] as const;
const RAD = Math.PI / 180;
const clamp = (x: number): number => Math.max(0, Math.min(1, x));
const smooth = (lo: number, hi: number, x: number): number => {
  const t = clamp((x - lo) / (hi - lo));
  return t * t * (3 - 2 * t);
};
const blend = (x: number, y: number, t: number): number => x * (1 - t) + y * t;
const space = (rgb: Rgb) => {
  const [r, g, b] = rgb.map((x) => clamp(x) ** (1 / 2.2)) as [number, number, number];
  const a = r - g;
  const o = (r + g) * 0.5 - b;
  return {
    y: r * WEIGHTS[0] + g * WEIGHTS[1] + b * WEIGHTS[2],
    a,
    b: o,
    c: Math.hypot(a, o),
    h: Math.atan2(o, a) / RAD,
  };
};
const windowWeight = (h: number, c: number, y: number, w: ToneHueWindow | null): number => {
  if (!w) return 0;
  const d = ((((h - w.centre) % 360) + 540) % 360) - 180;
  const luma = w.maxLuma === undefined ? 1 : 1 - smooth(w.maxLuma - 0.1, w.maxLuma, y);
  return (1 - smooth(w.width * 0.5, w.width, Math.abs(d))) * smooth(w.minChroma, 2 * w.minChroma, c) * luma;
};
/** The density curve followed by the contrast S around the pivot, on encoded luma. */
export function toneLuma(y: number, p: Pick<FinalToneParams, 'density' | 'contrast' | 'pivot'>): number {
  const yd = (1 - p.density) * y + p.density * y ** 6;
  return yd + p.contrast * (yd - p.pivot) * 4 * yd * (1 - yd);
}

/** JavaScript twin: gradeShare is zero when the look has no cinematic grade. */
export function finalToneRgb(
  look: Rgb,
  lin: Rgb,
  p: FinalToneParams,
  opts: { object: boolean; gradeShare: number; night: number },
): [number, number, number] {
  const l = space(look);
  const foliage = windowWeight(l.h, l.c, l.y, p.foliage);
  const objectFactor = opts.object ? Math.max(p.objectShare, foliage) : 1;
  // Share of the luma change that survives the final mix: objectDensity on objects (foliage: all of it).
  const lumaKeep = opts.object ? Math.min(1, Math.max(p.objectDensity, foliage) / Math.max(objectFactor, 1e-3)) : 1;
  const m =
    p.strength * blend(1, p.night.colour, opts.night) * (1 - windowWeight(l.h, l.c, l.y, p.protect)) * objectFactor;
  if (m === 0) return [...look];
  const graded = p.grade ? cinematicGradeRgb(lin, p.grade) : null;
  const base: Rgb = graded
    ? (graded.map((v, i) => blend(lin[i]!, v, opts.gradeShare)) as [number, number, number])
    : look;
  const t = space(base);
  const ws = 1 - smooth(0.25, 0.45, t.y);
  const wh = smooth(0.62, 0.82, t.y);
  const gain = ws * p.chroma[0] + Math.max(0, 1 - ws - wh) * p.chroma[1] + wh * p.chroma[2];
  let a = t.a * gain;
  let b = t.b * gain;
  for (const shift of p.hueShifts.slice(0, 2)) {
    const angle = shift.shift * windowWeight(t.h, t.c, t.y, shift) * RAD;
    [a, b] = [a * Math.cos(angle) - b * Math.sin(angle), a * Math.sin(angle) + b * Math.cos(angle)];
  }
  const y = blend(t.y, toneLuma(t.y, p), lumaKeep);
  // Analytic inverse of [Rec709 luma, r-g, (r+g)/2-b].
  const g = y - (WEIGHTS[0] + WEIGHTS[2] * 0.5) * a + WEIGHTS[2] * b;
  return [g + a, g, g + a * 0.5 - b].map((v, i) => blend(look[i]!, clamp(v) ** 2.2, m)) as [number, number, number];
}

function toneSpace(rgb: ShaderNode) {
  const e: ShaderNode = vec3(rgb)
    .clamp(0, 1)
    .pow(1 / 2.2)
    .toVar();
  const a: ShaderNode = e.r.sub(e.g);
  const b: ShaderNode = e.r.add(e.g).mul(0.5).sub(e.b);
  const c: ShaderNode = a.mul(a).add(b.mul(b)).sqrt().toVar();
  // atan2(0,0) is undefined on some backends; achromatic pixels have no window weight anyway.
  const h: ShaderNode = atan(b, c.greaterThan(1e-8).select(a, float(1)))
    .div(RAD)
    .toVar();
  return { y: dot(e, vec3(...WEIGHTS)), a, b, c, h };
}
function hueWeight(t: ReturnType<typeof toneSpace>, w: ShaderNode, maxLuma: ShaderNode): ShaderNode {
  const d: ShaderNode = t.h.sub(w.x).abs().mod(360);
  const distance: ShaderNode = d.min(float(360).sub(d));
  return float(1)
    .sub(smoothstep(w.y.mul(0.5), w.y, distance))
    .mul(smoothstep(w.z, w.z.mul(2), t.c))
    .mul(float(1).sub(smoothstep(maxLuma.sub(0.1), maxLuma, t.y)));
}

/** Uniform-only final colour transform. The owner decides whether to include it in a pipeline. */
export class FinalTone {
  readonly strength = uniform(0);
  readonly density = uniform(0);
  readonly contrast = uniform(0);
  readonly pivot = uniform(0.5);
  readonly objectDensity = uniform(1);
  /** maxLuma of shift 0, shift 1, protect and foliage windows (2 = no luma limit). */
  readonly maxLuma = [uniform(2), uniform(2), uniform(2), uniform(2)] as const;
  readonly chroma = uniform(new Vector3(1, 1, 1));
  readonly shifts = [uniform(new Vector4(0, 1, 1, 0)), uniform(new Vector4(0, 1, 1, 0))] as const;
  readonly protect = uniform(new Vector4(0, 1, 1, 0));
  readonly foliage = uniform(new Vector4(0, 1, 1, 0));
  readonly objectShare = uniform(1);
  readonly grain = uniform(0);
  readonly nightColour = uniform(1);
  readonly nightGrain = uniform(1);
  readonly useGrade = uniform(0);
  readonly gradeUniforms = {
    gradeBalance: uniform(new Vector3(1, 1, 1)),
    gradeContrast: uniform(0),
    gradePivotLift: uniform(0),
    gradeSaturation: uniform(1),
    gradeShadowTint: uniform(new Vector3(1, 1, 1)),
    gradeHighlightTint: uniform(new Vector3(1, 1, 1)),
  };

  set(p: FinalToneParams, airless = false): void {
    this.strength.value = p.strength;
    this.density.value = p.density;
    this.contrast.value = p.contrast;
    this.pivot.value = p.pivot;
    this.chroma.value.fromArray(p.chroma);
    const window = (u: typeof this.protect, w: ToneHueWindow | null, amount: number) =>
      u.value.set(w?.centre ?? 0, Math.max(1e-6, w?.width ?? 1), Math.max(1e-6, w?.minChroma ?? 1), w ? amount : 0);
    this.shifts.forEach((u, i) => window(u, p.hueShifts[i] ?? null, p.hueShifts[i]?.shift ?? 0));
    window(this.protect, p.protect, 1);
    window(this.foliage, p.foliage, 1);
    [p.hueShifts[0], p.hueShifts[1], p.protect, p.foliage].forEach((w, i) => {
      this.maxLuma[i]!.value = w?.maxLuma ?? 2;
    });
    this.objectShare.value = p.objectShare;
    this.objectDensity.value = p.objectDensity;
    this.grain.value = p.grain;
    this.nightColour.value = p.night.colour;
    this.nightGrain.value = p.night.grain;
    this.useGrade.value = p.grade ? 1 : 0;
    const grade = p.grade ?? neutralGrade();
    for (const key of ['gradeBalance', 'gradeShadowTint', 'gradeHighlightTint'] as const)
      this.gradeUniforms[key].value.fromArray(airless ? [1, 1, 1] : grade[key]);
    this.gradeUniforms.gradeContrast.value = grade.gradeContrast;
    this.gradeUniforms.gradePivotLift.value = grade.gradePivotLift;
    this.gradeUniforms.gradeSaturation.value = grade.gradeSaturation;
  }

  node(
    lin: ShaderNode,
    look: ShaderNode,
    gradeShare: ShaderNode,
    night: ShaderNode,
    objectCode: ShaderNode | null,
  ): ShaderNode {
    // Classification reads the look's colour (before the tone): water protection and foliage.
    const l = toneSpace(look);
    const foliage: ShaderNode = hueWeight(l, this.foliage, this.maxLuma[3]).mul(this.foliage.w).toVar();
    const isObject: ShaderNode | null = objectCode ? dot(objectCode.abs(), vec2(1)).greaterThan(0) : null;
    const factor: ShaderNode = isObject ? isObject.select(max(this.objectShare, foliage), float(1)).toVar() : float(1);
    const lumaKeep: ShaderNode = isObject
      ? isObject.select(max(this.objectDensity, foliage).div(factor.max(1e-3)).min(1), float(1))
      : float(1);
    const base: ShaderNode = mix(look, mix(lin, cinematicGrade(lin, this.gradeUniforms), gradeShare), this.useGrade);
    const t = toneSpace(base);
    const ws: ShaderNode = float(1).sub(smoothstep(0.25, 0.45, t.y));
    const wh: ShaderNode = smoothstep(0.62, 0.82, t.y);
    const gain: ShaderNode = dot(vec3(ws, float(1).sub(ws).sub(wh).max(0), wh), this.chroma);
    let a: ShaderNode = t.a.mul(gain);
    let b: ShaderNode = t.b.mul(gain);
    this.shifts.forEach((shift, i) => {
      const angle: ShaderNode = hueWeight(t, shift, this.maxLuma[i]!).mul(shift.w).mul(RAD).toVar();
      const next: ShaderNode = a
        .mul(cos(angle))
        .sub(b.mul(sin(angle)))
        .toVar();
      b = a
        .mul(sin(angle))
        .add(b.mul(cos(angle)))
        .toVar();
      a = next;
    });
    // Depth, then the sun/shade S around the pivot (see FinalToneParams.density / contrast).
    const y0: ShaderNode = t.y.toVar();
    const yd: ShaderNode = mix(y0, y0.max(0).pow(6), this.density).toVar();
    const curved: ShaderNode = yd.add(this.contrast.mul(yd.sub(this.pivot)).mul(yd.mul(float(1).sub(yd)).mul(4)));
    const y: ShaderNode = mix(y0, curved, lumaKeep);
    const g: ShaderNode = y.sub(a.mul(WEIGHTS[0] + WEIGHTS[2] * 0.5)).add(b.mul(WEIGHTS[2]));
    const toned: ShaderNode = vec3(g.add(a), g, g.add(a.mul(0.5)).sub(b))
      .clamp(0, 1)
      .pow(2.2);
    const m: ShaderNode = this.strength
      .mul(mix(1, this.nightColour, night))
      .mul(float(1).sub(hueWeight(l, this.protect, this.maxLuma[2]).mul(this.protect.w)))
      .mul(factor);
    return mix(look, toned, m);
  }

  /** Store this mask in the shared picture's scratch alpha, avoiding a second evaluation of the scene/grade. */
  grainMask(look: ShaderNode, objectCode: ShaderNode): ShaderNode {
    return dot(objectCode.abs(), vec2(1))
      .greaterThan(0)
      .select(hueWeight(toneSpace(look), this.foliage, this.maxLuma[3]).mul(this.foliage.w), float(1));
  }

  /** Five display-picture taps, with a luminance-only tent residual and a guard for strong edges. */
  grainCut(picture: ShaderNode, night: ShaderNode): ShaderNode {
    const centre: ShaderNode = picture.sample(screenUV).toVar();
    // The copy is drawn at the output's size, so half a texel is half a screen pixel. textureSize(picture) here left a
    // texture-size uniform without a type in some builds (all-black frames, "Uniform null not implemented").
    const px: ShaderNode = vec2(0.5).div(screenSize);
    let blur: ShaderNode = vec3(0);
    for (const x of [-1, 1])
      for (const y of [-1, 1]) blur = blur.add(picture.sample(screenUV.add(px.mul(vec2(x, y)))).rgb.mul(0.25));
    const l: ShaderNode = dot(centre.rgb, vec3(...WEIGHTS));
    const r: ShaderNode = l.sub(dot(blur, vec3(...WEIGHTS)));
    const guard: ShaderNode = float(1).sub(smoothstep(0.06, 0.14, r.abs()));
    const gm: ShaderNode = this.grain.mul(mix(1, this.nightGrain, night)).mul(centre.a);
    const level: ShaderNode = l.sub(r.mul(gm).mul(guard));
    return vec4(centre.rgb.mul(level.div(l.max(1e-4))), 1);
  }
}
