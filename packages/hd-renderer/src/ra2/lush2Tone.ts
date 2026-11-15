import type { HdSettings } from '../HdRenderer';
import type { Theater } from '../contract';
import type { CloudToneParams } from '../cloudTone';
import { ALIVE, ALIVE_CLOUD_SHAPES, cloudCoverThreshold, cloudThresholdCover, type AliveOptions } from '../livingWorld';
import type { TuningField } from '../ui/tuningPanel';

export interface Lush2Values {
  strength: number;
  grassLight: number;
  grassHue: number;
  shadeChroma: number;
  snowLift: number;
  objectFloor: number;
  grassChroma: number;
  detail: number;
  cover: number;
  shadeHue: number;
  gold: number;
  goldReach: number;
  patchScale: number;
  softness: number;
  drift: number;
  objectShare: number;
  contact: number;
  wet: number;
  snowBlue: number;
  shadeDepth: number;
}
export const LUSH2_SCHEMA: readonly TuningField<keyof Lush2Values>[] = [
  { key: 'strength', label: '总强度 [strength]', min: 0, max: 1.5, step: 0.01 },
  { key: 'grassLight', label: '草地明度 [grassLight]', min: 0.7, max: 1.15, step: 0.01 },
  { key: 'grassHue', label: '绿色色相 [grassHue]', min: 0, max: 30, step: 1 },
  { key: 'shadeHue', label: '阴影色相 [shadeHue]', min: 0, max: 40, step: 1 },
  { key: 'grassChroma', label: '饱和度 [grassChroma]', min: 0.8, max: 1.6, step: 0.01 },
  { key: 'shadeChroma', label: '阴影饱和 [shadeChroma]', min: 0.8, max: 1.6, step: 0.01 },
  { key: 'detail', label: '草纹对比 [detail]', min: 0.4, max: 1.2, step: 0.01 },
  { key: 'cover', label: '云影覆盖 [cover]', min: 0, max: 1, step: 0.01 },
  { key: 'shadeDepth', label: '阴影深浅 [shadeDepth]', min: 0, max: 0.5, step: 0.005 },
  { key: 'gold', label: '金色量 [gold]', min: 0, max: 1, step: 0.01 },
  { key: 'goldReach', label: '金色范围 [goldReach]', min: 0, max: 1, step: 0.01 },
  { key: 'patchScale', label: '片区大小 [patchScale]', min: 6, max: 24, step: 0.1 },
  { key: 'softness', label: '片区柔和 [softness]', min: 0.2, max: 1, step: 0.01 },
  { key: 'drift', label: '漂移速度 [drift]', min: 0, max: 1, step: 0.005 },
  { key: 'objectShare', label: '单位跟随 [objectShare]', min: 0, max: 1, step: 0.01 },
  { key: 'objectFloor', label: '单位最暗 [objectFloor]', min: 0.5, max: 1, step: 0.01 },
  { key: 'contact', label: '接触阴影 [contact]', min: 0, max: 1, step: 0.01 },
  { key: 'wet', label: '岸边湿润 [wet]', min: 0, max: 1, step: 0.01 },
  { key: 'snowBlue', label: '雪地蓝 [snowBlue]', min: 0, max: 1, step: 0.01 },
  { key: 'snowLift', label: '雪地提亮 [snowLift]', min: 1, max: 2, step: 0.01 },
];
const temperate: Readonly<Lush2Values> = Object.freeze({
  strength: 1,
  grassLight: 0.9,
  grassHue: 10,
  grassChroma: 1.2,
  shadeChroma: 1.25,
  snowLift: 1.05,
  objectFloor: 0.8,
  detail: 1,
  cover: 0.5,
  shadeHue: 12,
  gold: 0.75,
  goldReach: 0.35,
  patchScale: 12,
  softness: 0.7,
  drift: 0.6,
  objectShare: 0.5,
  contact: 1,
  wet: 0.8,
  snowBlue: 1,
  shadeDepth: 0.3,
});
export const LUSH2_DEFAULTS: Readonly<Record<Theater, Readonly<Lush2Values>>> = Object.freeze({
  temperate,
  urban: Object.freeze({
    ...temperate,
    grassLight: 0.98,
    grassHue: 5,
    grassChroma: 1.04,
    cover: 0.4,
    shadeDepth: 0.26,
  }),
  newurban: Object.freeze({
    ...temperate,
    grassLight: 0.98,
    grassHue: 5,
    grassChroma: 1.04,
    cover: 0.4,
    shadeDepth: 0.26,
  }),
  snow: Object.freeze({
    ...temperate,
    snowLift: 1.8,
    snowBlue: 0.5,
    grassLight: 1,
    grassHue: 0,
    grassChroma: 1,
    shadeHue: 0,
    shadeChroma: 1,
    gold: 0,
    cover: 0.28,
    shadeDepth: 0.12,
  }),
  desert: Object.freeze({
    ...temperate,
    grassLight: 1,
    grassHue: 3,
    grassChroma: 1.02,
    gold: 0.1,
    cover: 0.35,
    shadeDepth: 0.24,
  }),
  lunar: Object.freeze({ ...temperate, strength: 0, gold: 0, contact: 0, wet: 0, shadeDepth: 0 }),
});
/** RA2 hull policy; generic grading and UI modules contain no game-specific values. */
export const LUSH2_CONTACT = Object.freeze({ radiusScale: 1.6, falloff: 1.5, opacity: 0.9, opacityCap: 0.9 });
export const LUSH2_MEAN = Object.freeze({ blockWidth: 20, contextWidth: 96 });
export const LUSH2_SHORE = Object.freeze({ width: 1.6, noiseScale: 0.8, noiseAmplitude: 0.45 });
// smoothstep^-1(0.9) - smoothstep^-1(0.1), with a representative unit-gradient projection.
// Irregular noise has varying local slopes; this is a nominal normal-to-edge width, not a universal width.
const penumbraSpan = (softness: number) => ((softness / 0.7) * 2.25 * Math.SQRT1_2) / 0.6083997886818166;
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
      grassLight: values.grassLight,
      grassHue: values.grassHue,
      grassChroma: values.grassChroma,
      shadeChroma: values.shadeChroma,
      // Bare dirt inside a lawn darkens with the lawn (no pale halo); sand with a sandy context is untouched.
      bareLight: Math.min(0.97, values.grassLight),
      snowLift: values.snowLift,
      objectShare: values.objectShare,
      objectFloor: values.objectFloor,
      detail: values.detail,
      greenHue: [48, 62, 175, 195],
      greenSat: [0.08, 0.22],
      contextHue: [50, 52, 175, 195],
      channelFloor: 0.15,
      wetTint: [1, 0.88, 0.72],
      goldTint: [1.06, 1, 0.86],
      objectShoulder: 0.12,
      objectHighlight: 0.55,
      bounce: 0.08,
      bounceHeight: 0.75,
      cloudTintShare: snow ? 0 : 0.1,
      snowWhite: snow ? 1 : 0.75,
      neutralGradeLuma: 0.8,
      neutralGradeTint: [1, 1, 1],
      snowGate: snow ? [0.4, 0.7, 0.38, 0.5] : [0.42, 0.45, 0.3, 0.35],
      bareDepth: 1,
      shadeHue: values.shadeHue,
      gold: values.gold,
      contact: values.contact,
      // The recommended slider value owns the full wet-strip recipe; higher values extrapolate it.
      wet: values.wet / 0.8,
      snowBlue: values.snowBlue,
      snow: snow ? 1 : 0,
      nightKeep: 0.65,
      sunGain: 1.04,
      shadeTint: [1, 1, 1],
      moonTint: [0.96, 1, 1.04],
      grassSunTint: [0.96, 1, 0.7],
      snowShadeTint: [0.93, 0.97, 1.08],
      wetDepth: 0.4,
      wetChroma: 1.3,
      highlightGate: [0, 0, 1.12 - 0.14 * values.goldReach, 1.32 - 0.12 * values.goldReach],
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
        edgeLow: cloudCoverThreshold(values.cover) - penumbraSpan(values.softness) / values.patchScale / 2,
        edgeHigh: cloudCoverThreshold(values.cover) + penumbraSpan(values.softness) / values.patchScale / 2,
        darkening: values.shadeDepth,
      },
    },
  };
}
export function lush2FromMapping(tone: CloudToneParams, alive: AliveOptions): Lush2Values {
  return {
    strength: tone.strength,
    grassLight: tone.grassLight,
    grassHue: tone.grassHue,
    grassChroma: tone.grassChroma,
    shadeChroma: tone.shadeChroma,
    snowLift: tone.snowLift,
    objectShare: tone.objectShare,
    objectFloor: tone.objectFloor,
    detail: tone.detail,
    cover: cloudThresholdCover((alive.shape!.edgeLow + alive.shape!.edgeHigh) / 2),
    shadeHue: tone.shadeHue,
    gold: tone.gold,
    goldReach: (1.12 - tone.highlightGate[2]) / 0.14,
    patchScale: alive.shape!.scale,
    softness: ((alive.shape!.edgeHigh - alive.shape!.edgeLow) * alive.shape!.scale) / penumbraSpan(1),
    drift: alive.speed * ALIVE.drift,
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
