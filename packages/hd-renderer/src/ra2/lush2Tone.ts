import type { HdSettings } from '../HdRenderer';
import type { Theater } from '../contract';
import type { CloudToneParams } from '../cloudTone';
import { ALIVE, ALIVE_CLOUD_SHAPES, type AliveOptions } from '../livingWorld';
import type { TuningField } from '../ui/tuningPanel';

export interface Lush2Values {
  strength: number;
  depth: number;
  chroma: number;
  shadeHue: number;
  gold: number;
  patchScale: number;
  softness: number;
  drift: number;
  unitMatch: number;
  contact: number;
  wet: number;
  snowBlue: number;
  shadeDepth: number;
}
export const LUSH2_SCHEMA: readonly TuningField<keyof Lush2Values>[] = [
  { key: 'strength', label: '总强度 [strength]', min: 0, max: 1.5, step: 0.01 },
  { key: 'depth', label: '地面深度 [depth]', min: 0, max: 1, step: 0.01 },
  { key: 'chroma', label: '饱和度 [chroma]', min: 0.6, max: 2, step: 0.01 },
  { key: 'shadeHue', label: '阴影色相 [shadeHue]', min: 0, max: 1, step: 0.01 },
  { key: 'gold', label: '金色量 [gold]', min: 0, max: 1, step: 0.01 },
  { key: 'patchScale', label: '片区大小 [patchScale]', min: 6, max: 24, step: 0.1 },
  { key: 'softness', label: '片区柔和 [softness]', min: 0.2, max: 1, step: 0.01 },
  { key: 'drift', label: '漂移速度 [drift]', min: 0, max: 1, step: 0.005 },
  { key: 'unitMatch', label: '单位跟随 [unitMatch]', min: 0, max: 1, step: 0.01 },
  { key: 'contact', label: '接触阴影 [contact]', min: 0, max: 1, step: 0.01 },
  { key: 'wet', label: '岸边湿润 [wet]', min: 0, max: 1, step: 0.01 },
  { key: 'snowBlue', label: '雪地蓝 [snowBlue]', min: 0, max: 1, step: 0.01 },
  { key: 'shadeDepth', label: '阴影深浅 [shadeDepth]', min: 0, max: 0.5, step: 0.005 },
];
const temperate: Readonly<Lush2Values> = Object.freeze({
  strength: 1,
  depth: 0.48,
  chroma: 1.3,
  shadeHue: 0.8,
  gold: 0.22,
  patchScale: 12,
  softness: 0.7,
  drift: 0.25,
  unitMatch: 0.9,
  contact: 0.8,
  wet: 0.8,
  snowBlue: 1,
  shadeDepth: 0.3,
});
export const LUSH2_DEFAULTS: Readonly<Record<Theater, Readonly<Lush2Values>>> = Object.freeze({
  temperate,
  urban: Object.freeze({ ...temperate, depth: 0.3, shadeDepth: 0.26 }),
  newurban: Object.freeze({ ...temperate, depth: 0.3, shadeDepth: 0.26 }),
  snow: Object.freeze({ ...temperate, depth: 0.02, chroma: 1, gold: 0, shadeDepth: 0.18 }),
  desert: Object.freeze({ ...temperate, depth: 0.12, chroma: 1.2, gold: 0.1, shadeDepth: 0.24 }),
  lunar: Object.freeze({ ...temperate, strength: 0, depth: 0, gold: 0, contact: 0, wet: 0, shadeDepth: 0 }),
});
/** RA2 hull policy; generic grading and UI modules contain no game-specific values. */
export const LUSH2_CONTACT = Object.freeze({ radiusScale: 1.3, falloff: 2, opacity: 0.7, opacityCap: 0.7 });
export const LUSH2_SHORE = Object.freeze({ width: 1 });
// smoothstep^-1(0.9) - smoothstep^-1(0.1), with a representative unit-gradient projection.
// Irregular noise has varying local slopes; this is a nominal normal-to-edge width, not a universal width.
const penumbraSpan = (softness: number) => ((softness / 0.7) * 2.5 * Math.SQRT1_2) / 0.6083997886818166;
export function lush2Defaults(theater: string | undefined): Lush2Values {
  return { ...LUSH2_DEFAULTS[theater && Object.hasOwn(LUSH2_DEFAULTS, theater) ? (theater as Theater) : 'temperate'] };
}
export function lush2Mapping(
  values: Lush2Values,
  theater: string | undefined,
): { tone: CloudToneParams; alive: AliveOptions } {
  const snow = theater === 'snow';
  return {
    tone: {
      strength: values.strength,
      depth: 1 - values.depth,
      chroma: values.chroma,
      bareChroma: 1,
      cloudTintShare: snow ? 0 : 0.1,
      snowWhite: 0.85,
      neutralGradeLuma: 0.8,
      neutralGradeTint: [1, 1, 1],
      snowGate: [0.42, 0.45, 0.3, 0.35],
      bareDepth: 1 - values.depth * (snow ? 1 : 0.35),
      shadeHue: values.shadeHue,
      gold: values.gold,
      unitMatch: values.unitMatch,
      contact: values.contact,
      // The recommended slider value owns the full wet-sand recipe; higher values extrapolate it.
      wet: values.wet / 0.8,
      snowBlue: values.snowBlue,
      snow: snow ? 1 : 0,
      nightKeep: 0.65,
      sunGain: 1,
      shadeTint: [0.995, 1, 1.005],
      moonTint: [0.96, 1, 1.04],
      grassShadeTint: [0.95 - 0.25 * values.shadeHue, 1, 0.55 + 0.07 * values.shadeHue],
      grassSunTint: [0.88, 1, 0.6],
      goldTint: [1.25, 1, 0.45],
      snowShadeTint: [0.93, 0.97, 1.08],
      wetTint: [1, 0.9, 0.74],
      wetDepth: 0.32,
      wetChroma: 1.15,
      highlightGate: [0.64, 0.76, 1.25, 1.5],
    },
    alive: {
      clouds: values.strength > 0,
      wind: false,
      preset: 'lush',
      cloud: values.strength,
      sway: 0,
      speed: values.drift / ALIVE.drift,
      debugCloud: false,
      shape: {
        ...ALIVE_CLOUD_SHAPES.lush,
        scale: values.patchScale,
        largeScale: values.patchScale * 2,
        edgeLow: -0.07,
        edgeHigh: -0.07 + penumbraSpan(values.softness) / values.patchScale,
        darkening: values.shadeDepth,
      },
    },
  };
}
export function lush2FromMapping(tone: CloudToneParams, alive: AliveOptions): Lush2Values {
  return {
    strength: tone.strength,
    depth: 1 - tone.depth,
    chroma: tone.chroma,
    shadeHue: tone.shadeHue,
    gold: tone.gold,
    patchScale: alive.shape!.scale,
    softness: ((alive.shape!.edgeHigh - alive.shape!.edgeLow) * alive.shape!.scale) / penumbraSpan(1),
    drift: alive.speed * ALIVE.drift,
    unitMatch: tone.unitMatch,
    contact: tone.contact,
    wet: tone.wet * 0.8,
    snowBlue: tone.snowBlue,
    shadeDepth: alive.shape!.darkening,
  };
}
/** Only renderer uniforms change. The post graph remains the exact no-tone graph on both sides of A/B. */
export function applyLush2(
  settings: HdSettings,
  theater: string | undefined,
  values = lush2Defaults(theater),
): AliveOptions {
  const mapped = lush2Mapping(values, theater);
  settings.groundTone = null;
  settings.tone = null;
  settings.cloudTone = mapped.tone;
  return mapped.alive;
}
