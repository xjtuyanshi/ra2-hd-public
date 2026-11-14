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
// Tone-space hues measured on Build 13 sandbox frames (FinalToneParams): sunlit grass 75-102 (median 93), jungle 127,
// sand 62, snow whites 185-218 (median 203), shallow water 208, deep water 222.
/** Sunlit grass a little towards green: the dry look is yellow-olive grass with bleached flecks. Sand (62) stays out. */
const green: ToneHueShift = { centre: 95, width: 30, minChroma: 0.04, shift: 6 };
const urban: MoistTheatre = {
  // Warm brick and lawn: neutral shadows keep their colour (the look's cool shadows turned them grey).
  balance: [1, 1, 0.99],
  shadowTint: [1.06, 1, 0.88],
  highlightTint: [1.15, 1, 0.82],
  // City ground is mostly dark: a low pivot lifts lit facades and deepens shade instead of flattening the blocks.
  density: 0.12,
  contrast: 0.25,
  pivot: 0.35,
  chroma: [1.3, 1.2, 1.05],
  hueShifts: [green],
  grain: 0.12,
};
export const MOIST_THEATRES: Record<Theater, MoistTheatre> = {
  temperate: {
    // Deep green shade and warm golden sunlight instead of the look's grey-cool shade and bleached highlights.
    balance: [0.99, 1, 1],
    shadowTint: [0.85, 1.1, 0.85],
    highlightTint: [1.2, 1, 0.75],
    density: 0.2,
    contrast: 0.25,
    pivot: 0.45,
    chroma: [1.3, 1.08, 0.95],
    hueShifts: [green],
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
    hueShifts: t.hueShifts.map((shift) => ({ ...shift, shift: shift.shift * k })),
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
