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
  // Broad soft patches, a few cells across and elongated along one iso diagonal, like dappled sun and cloud shade.
  scale: 4.5,
  detailScale: 1.3,
  detailMix: 0.3,
  stretch: 2.2,
  angle: 30,
  coverage: 0.45,
  softness: 0.35,
  // Grass much deeper (the final frame about 20 L* darker), bare sand/rock less so, so it stays warm and light.
  depth: 0.6,
  bareDepth: 0.85,
  bareChroma: 1,
  // Jungle floors are already about as deep as the reference: they keep most of their light and take the colour.
  darkLo: 0.1,
  darkHi: 0.3,
  darkKeep: 0.25,
  sunGain: 1.05,
  shadeGain: 0.38,
  // Golden light in the lit patches, deep green in the shade. Chroma gains stay near 1: stronger gains amplified the
  // art's alternating yellow/green tile stripes into visible diagonal bands.
  sunTint: [1.1, 1, 0.78],
  shadeTint: [0.92, 1, 0.95],
  grassShadeTint: [0.8, 1, 0.75],
  sunChroma: 1,
  shadeChroma: 1.15,
  // A light touch: soften 0.5 with a grain cut of 0.5 turned grass velvety at 1:1.
  soften: 0.12,
  grain: 0.3,
  groundClarity: 0.7,
};
const urban: LushTheatre = {
  ...temperate,
  // Deeper, slightly cool pavement (reads damp), lawns follow the grass shade tint.
  depth: 0.72,
  bareDepth: 0.8,
  sunGain: 1,
  shadeGain: 0.6,
  sunTint: [1.04, 1, 0.9],
  shadeTint: [0.94, 1, 1.06],
  shadeChroma: 1.1,
  soften: 0.1,
};
export const LUSH_THEATRES: Record<Theater, LushTheatre | null> = {
  temperate,
  urban,
  newurban: urban,
  snow: {
    ...temperate,
    // Cold blue-white light, deeper blue shade; the snow itself only a little deeper.
    depth: 0.9,
    bareDepth: 0.9,
    sunGain: 1.02,
    shadeGain: 0.7,
    sunTint: [0.98, 1, 1.04],
    shadeTint: [0.88, 0.96, 1.15],
    grassShadeTint: [1, 1, 1],
    shadeChroma: 1.2,
    soften: 0.1,
  },
  desert: {
    ...temperate,
    // Warm light, cooler shade, sand keeps most of its brightness.
    depth: 0.8,
    bareDepth: 0.8,
    shadeGain: 0.62,
    sunTint: [1.08, 1, 0.85],
    shadeTint: [0.9, 0.97, 1.1],
    grassShadeTint: [0.85, 1, 0.8],
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
