import type { HdSettings } from '../HdRenderer';
import type { Theater } from '../contract';
import { hasCinematicGrade, type CinematicGradeParams } from '../post';
import type { FinalToneParams, ToneHueShift } from '../tone';

export const TONE_STYLES = ['off', 'moist1', 'moist2', 'moist3'] as const;
export type ToneStyle = (typeof TONE_STYLES)[number];
export function parseToneStyle(v: string | null): ToneStyle {
  return TONE_STYLES.find((style) => style === v) ?? 'off';
}
/**
 * Strength of each level relative to the theatre data (moist2). Direction: the owner's chosen reference (2026-11-14,
 * reference A of the moist diagnosis boards): warm sunlit light, strong sun/shade contrast, deep saturated greens in
 * shade; snow cold and crisp. Only colour and tone change, never shapes or patterns. Review limits per level: units at
 * least 16 L* apart from the ground around them, mid-scale contrast (grass clumps, cracks, panels) not below the
 * current look, clear water unchanged. moist1 stays within about 7 L* of the current look; moist2 and moist3 go deeper.
 */
export const MOIST_LEVEL_SCALE = { moist1: 0.6, moist2: 1, moist3: 1.4 } as const;
export interface MoistTheatre {
  balance?: [number, number, number];
  shadowTint?: [number, number, number];
  highlightTint?: [number, number, number];
  density: number;
  contrast: number;
  pivot: number;
  chroma: [number, number, number];
  hueShifts: ToneHueShift[];
  grain: number;
}
// Tone-space hues and encoded luma measured on Build 13 sandbox frames (quartiles of the blurred ground L*):
// XMP27T8 sunlit grass hue 89 / luma 0.69 / chroma 0.39, its shade grass 96 / 0.52 / 0.34; AMAZON01 jungle shade
// 128 / 0.31 / 0.18 and its lighter patches 97 / 0.49 / 0.16; sand 62-70; the wet band beside clear water 89 / 0.58 /
// 0.19; snow whites 185-218, shallow water 208, deep water 222.
/**
 * Reference A's look, classified by patch (FinalToneParams.keyRadius), not by pixel:
 * - sun: vivid lit yellow-green ground a little towards gold and a little deeper (Lab hue of base sunlit grass
 *   104 -> 99; a highlight gain below 1 or a larger shift turned it pale straw, which reads dry);
 * - shade: vivid mid-dark green (chroma gate 0.18: not the muted jungle, nor the wet band beside the water) towards
 *   deep green and deeper. Sun/shade patch L* difference ends near A's 20; more (exposure -0.16) outlined the map's
 *   tile-variant patches as hard camouflage blots;
 * - teal: the look's blue-green jungle shade back towards the original's olive (Lab hue 128 -> 123; original 122);
 * - muted: low-chroma lit ground (the wet band beside clear water, the jungle's lighter patches) warmer and a little
 *   lighter, so the band beside the water is not muddied and the jungle reads sunlit rather than dusk.
 */
const temperateShifts: ToneHueShift[] = [
  { centre: 88, width: 40, minChroma: 0.15, minLuma: 0.45, shift: -3, exposure: -0.06 },
  { centre: 105, width: 40, minChroma: 0.18, maxLuma: 0.64, shift: 10, exposure: -0.08 },
  { centre: 135, width: 40, minChroma: 0.06, shift: -8 },
  { centre: 88, width: 44, minChroma: 0.06, maxChroma: 0.22, minLuma: 0.45, maxLuma: 0.75, shift: -5, exposure: 0.1 },
];
const urban: MoistTheatre = {
  // Neutral pavement stays near the original's warmth (Lab b* about 5); lawns follow the temperate shifts.
  balance: [1, 1, 1],
  shadowTint: [1.02, 1, 0.97],
  highlightTint: [1.08, 1, 0.92],
  // City ground is mostly dark: a low pivot lifts lit facades and deepens shade instead of flattening the blocks.
  density: 0.08,
  contrast: 0.25,
  pivot: 0.35,
  chroma: [1.3, 1.2, 1.05],
  hueShifts: temperateShifts,
  grain: 0.12,
};
export const MOIST_THEATRES: Record<Theater, MoistTheatre> = {
  temperate: {
    // Deep green shade and warm golden sunlight instead of the look's grey-cool shade and bleached highlights. Part of
    // the depth comes from the sun/shade exposures (vivid grass only); the global density stays low so dark maps
    // (jungle) stay sunlit.
    balance: [0.99, 1, 1],
    shadowTint: [0.85, 1.1, 0.85],
    highlightTint: [1.25, 1.02, 0.7],
    density: 0.1,
    contrast: 0.2,
    pivot: 0.42,
    // Golden light needs its chroma: a highlight gain below 1 turned the warmed sunlit grass pale straw.
    chroma: [1.35, 1.18, 1.12],
    hueShifts: temperateShifts,
    grain: 0.25,
  },
  snow: {
    // Cold and crisp: the original's cyan ice (Lab hue about 217), not the look's cream and not violet. Blue gain with a
    // little less red and green keeps the hue cyan; there is no hue rotation and no darkening of the snow.
    balance: [0.97, 0.99, 1.06],
    shadowTint: [0.88, 0.98, 1.2],
    highlightTint: [0.97, 0.99, 1.05],
    density: 0,
    contrast: 0.12,
    pivot: 0.55,
    chroma: [1.15, 1.1, 1.12],
    hueShifts: [],
    grain: 0.2,
  },
  urban,
  newurban: urban,
  desert: {
    // Warm and rich but not dyed: golden light, slightly cool shade, sand keeps most of its brightness.
    balance: [1, 1, 1],
    shadowTint: [0.92, 1, 1.08],
    highlightTint: [1.1, 1, 0.88],
    density: 0.08,
    contrast: 0.2,
    pivot: 0.55,
    chroma: [1.2, 1.08, 1],
    hueShifts: [],
    grain: 0.2,
  },
  lunar: {
    // Airless terrain keeps its illuminant (no tints) and its brightness: crater contrast and a little chroma only.
    density: 0,
    contrast: 0.25,
    pivot: 0.35,
    chroma: [1.1, 1.08, 1.05],
    hueShifts: [],
    grain: 0.2,
  },
};

/** Contrast limit for a monotonic curve: contrast * 4 * max(pivot, 1 - pivot) < 1, with a margin. */
const contrastLimit = (pivot: number): number => 0.9 / (4 * Math.max(pivot, 1 - pivot));

export function moistTone(
  theater: string | undefined,
  style: ToneStyle,
  lookGrade: CinematicGradeParams,
): FinalToneParams | null {
  if (style === 'off') return null;
  const t = Object.hasOwn(MOIST_THEATRES, theater ?? '')
    ? MOIST_THEATRES[theater as Theater]
    : MOIST_THEATRES.temperate;
  const k = MOIST_LEVEL_SCALE[style];
  const tint = (look: [number, number, number], target = look): [number, number, number] =>
    look.map((v, i) => v + (target[i]! - v) * Math.min(1, k)) as [number, number, number];
  return {
    strength: 1,
    grade: hasCinematicGrade(lookGrade)
      ? {
          gradeContrast: lookGrade.gradeContrast,
          gradePivotLift: lookGrade.gradePivotLift,
          gradeSaturation: lookGrade.gradeSaturation,
          gradeBalance: tint(lookGrade.gradeBalance, t.balance),
          gradeShadowTint: tint(lookGrade.gradeShadowTint, t.shadowTint),
          gradeHighlightTint: tint(lookGrade.gradeHighlightTint, t.highlightTint),
        }
      : null,
    density: Math.min(0.5, t.density * k),
    contrast: Math.min(contrastLimit(t.pivot), t.contrast * k),
    pivot: t.pivot,
    chroma: t.chroma.map((g) => 1 + (g - 1) * k) as [number, number, number],
    hueShifts: t.hueShifts.map((shift) => ({
      ...shift,
      shift: shift.shift * k,
      ...(shift.exposure === undefined ? {} : { exposure: shift.exposure * k }),
    })),
    // About two engine pixels at the owner's 2.5 device pixels per engine pixel: grass texture is averaged out, light
    // and shade patches (tens of pixels) are not.
    keyRadius: 4,
    // The approved clear water (and shallows): saturated blue below near-white luma, left as the look draws it.
    protect: { centre: 215, width: 40, minChroma: 0.1, maxLuma: 0.75 },
    // Units and buildings take most of the colour and all of the depth, so they keep their separation from the deeper
    // ground (a low share kept them bright while the grass darkened, and they sank into it).
    objectShare: 0.6,
    objectDensity: 1,
    // Tree canopies are objects but belong to the landscape: green foliage takes the full tone and the grain cut.
    foliage: { centre: 105, width: 40, minChroma: 0.05 },
    // Only the 1-px band; capped so grass never turns velvety.
    grain: Math.min(0.4, t.grain * k),
    night: { colour: 0.35, grain: 0.6 },
  };
}

export function applyMoistTone(settings: HdSettings, theater: string | undefined, style: ToneStyle): void {
  settings.tone = moistTone(theater, style, settings);
}
