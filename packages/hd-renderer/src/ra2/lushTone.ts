import type { Theater } from '../contract';
import type { GroundToneParams } from '../groundTone';
import type { FinalToneParams } from '../tone';

export interface LushTheatre extends GroundToneParams {
  grain: number;
  groundClarity: number;
}
// Owner's reference: deep olive ground, golden light, broad soft light/shade patches and smoother grass.
// Only terrain materials change colour: units and buildings keep their brightness.
const temperate: LushTheatre = {
  strength: 1,
  // Dappled light: patches two to three cells across (roughly 100-400 device px at view 2.5), a little elongated,
  // with a strong second octave so they break up rather than read as broad cloud shadow.
  scale: 2.6,
  detailScale: 0.8,
  detailMix: 0.5,
  stretch: 1.4,
  angle: 30,
  coverage: 0.38,
  softness: 0.35,
  // Grass much deeper (the final frame about 20 L* darker), bare sand/rock less so, so it stays warm and light.
  depth: 0.56,
  bareDepth: 0.92,
  bareChroma: 1,
  // Jungle floors are already about as deep as the reference: they keep most of their light and take the colour.
  darkLo: 0.1,
  darkHi: 0.3,
  darkKeep: 0.25,
  darkPatch: 0.6,
  sunGain: 1.06,
  shadeGain: 0.37,
  bareShade: 0.82,
  // Golden-orange light in the lit patches (lit-patch R/G about 1.2, as in the reference), deep olive in the shade.
  // Tints are luminance-normalised; a blue cut below about 0.6 with more chroma turned the light lemon yellow, a red
  // push with chroma at or below 1 turned it beige. Shade chroma stays moderate: stronger gains amplified the art's
  // alternating yellow/green tile stripes into visible diagonal bands.
  sunTint: [1.32, 1, 0.62],
  // Cool shade (sky light): bare sand in shade stays light and bluish instead of turning muddy brown; grass
  // takes it back to deep olive through grassShadeTint.
  shadeTint: [0.86, 0.97, 1.16],
  grassShadeTint: [0.92, 1.03, 0.57],
  sunChroma: 1.12,
  shadeChroma: 1.15,
  // A light touch: soften 0.5 with a grain cut of 0.5 turned grass velvety at 1:1.
  soften: 0.12,
  grain: 0.3,
  groundClarity: 0.7,
  // At night half the dressing stays, with cool moonlit patches instead of golden ones: deeper, never brighter.
  nightKeep: 0.5,
  moonTint: [0.94, 1, 1.1],
};
const urban: LushTheatre = {
  ...temperate,
  // Deeper, cool, slightly more saturated pavement (reads damp slate rather than dry grey); lawns as temperate. The
  // light/shade split stays small on paving: a strong one drew tan and blue stains on the concrete.
  bareDepth: 0.68,
  bareShade: 0.86,
  bareChroma: 1.15,
  sunGain: 1.02,
  sunTint: [1.14, 1, 0.82],
  shadeTint: [0.9, 0.97, 1.12],
  grassShadeTint: [0.9, 1.03, 0.58],
  soften: 0.1,
};
export const LUSH_THEATRES: Record<Theater, LushTheatre | null> = {
  temperate,
  urban,
  newurban: urban,
  snow: {
    ...temperate,
    // Cold blue-white light, deep blue shade patches; the snow itself only a little deeper.
    depth: 0.9,
    bareDepth: 0.9,
    sunGain: 1.02,
    shadeGain: 0.7,
    bareShade: 0.74,
    sunTint: [0.98, 1, 1.04],
    shadeTint: [0.84, 0.95, 1.22],
    grassShadeTint: [1, 1, 1],
    sunChroma: 1,
    shadeChroma: 1.3,
    soften: 0.1,
  },
  desert: {
    ...temperate,
    // Warm orange light, cool (sky-lit) shade so shaded sand does not turn muddy brown; broader, calmer patches.
    scale: 3.4,
    detailMix: 0.35,
    softness: 0.4,
    coverage: 0.4,
    depth: 0.8,
    bareDepth: 0.85,
    bareShade: 0.66,
    sunGain: 1.1,
    shadeGain: 0.62,
    sunTint: [1.2, 1, 0.7],
    shadeTint: [0.8, 0.95, 1.28],
    grassShadeTint: [0.9, 1.03, 0.62],
    sunChroma: 1.05,
    shadeChroma: 1.1,
    bareChroma: 1.05,
    soften: 0.1,
  },
  lunar: null,
};
export function lushTheatre(theater: string | undefined): LushTheatre | null {
  const p = Object.hasOwn(LUSH_THEATRES, theater ?? '') ? LUSH_THEATRES[theater as Theater] : temperate;
  // Each renderer owns its mutable tuning data, including tint arrays.
  return p ? structuredClone(p) : null;
}
/** No final colour change: keep only the existing ground/foliage grain cut and its clarity weight. */
export function lushFinalTone(p: LushTheatre): FinalToneParams {
  return {
    strength: 1,
    grade: null,
    density: 0,
    contrast: 0,
    pivot: 0.5,
    chroma: [1, 1, 1],
    hueShifts: [],
    keyRadius: 0,
    protect: { centre: 215, width: 40, minChroma: 0.1, maxLuma: 0.75 },
    objectShare: 0,
    objectDensity: 0,
    foliage: { centre: 105, width: 40, minChroma: 0.05 },
    grain: p.grain,
    night: { colour: 0.35, grain: 0.6 },
    groundClarity: p.groundClarity,
  };
}
