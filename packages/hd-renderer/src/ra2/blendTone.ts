import { parseAliveParams, type AliveOptions } from '../livingWorld';
import type { HdSettings } from '../HdRenderer';
import { applyLush2, lush2Defaults, type Lush2Values } from './lush2Tone';

export type BlendPreset = 'contact' | 'ground' | 'tuned';
/** Keep existing dramatic URLs as an explicit alias for the C2 ground recipe. */
export const parseBlendPreset = (value: string | null): BlendPreset =>
  value === 'ground' || value === 'dramatic' ? 'ground' : value === 'tuned' ? 'tuned' : 'contact';

/** The softer C recipe remains available; the C2 ground recipe is an explicit opt-in. */
export function blendDefaults(theater: string | undefined, preset: BlendPreset = 'contact'): Lush2Values {
  const values = { ...lush2Defaults(theater), objectShare: 0.35, objectFloor: 0.9, detail: 1 };
  if (theater === 'temperate' || theater === 'urban' || theater === 'newurban') {
    Object.assign(values, {
      grassLight: 1.01,
      grassHue: 5,
      grassChroma: 1.08,
      shadeChroma: 1.08,
      gold: 0.18,
      shadeDepth: 0.14,
    });
    if (preset === 'ground')
      Object.assign(values, {
        grassLight: 0.84,
        grassHue: 0,
        grassChroma: 1.32,
        shadeChroma: 1.2,
        shadeHue: 8,
        gold: 0.65,
        goldReach: 0.4,
        shadeDepth: 0.36,
        objectShare: 0.6,
        objectFloor: 0.76,
        detail: 0.95,
      });
  }
  return values;
}

export function applyBlendTone(
  settings: HdSettings,
  theater: string | undefined,
  preset: BlendPreset = 'contact',
  values = blendDefaults(theater, preset),
  currentAlive: AliveOptions = parseAliveParams(new URLSearchParams()),
) {
  // Contact never installs a terrain recipe. Reuse the selected tone's exact ground and cloud state.
  // A standalone contact entry stays AO-only; object warmth is available when a tone already owns lighting.
  if (preset === 'contact') {
    if (settings.cloudTone)
      settings.cloudTone = {
        ...settings.cloudTone,
        // Preserve object readability instead of inheriting the ground recipe's multiplicative dimming.
        objectFloor: 1,
        objectSunTint: [1 + 0.08 * values.gold, 1, 1 - 0.08 * values.gold],
      };
    return currentAlive;
  }
  const alive = applyLush2(settings, theater, values);
  const share =
    theater === 'temperate' || theater === 'urban' || theater === 'newurban'
      ? Math.max(0, Math.min(1, (1.01 - values.grassLight) / 0.17))
      : 0;
  const tint = (soft: [number, number, number], strong: [number, number, number]): [number, number, number] =>
    soft.map((v, i) => v + (strong[i]! - v) * share) as [number, number, number];
  settings.cloudTone!.grassSunTint = tint([1.01, 1, 0.93], [1.08, 1, 0.8]);
  settings.cloudTone!.shadeTint = tint([0.97, 1, 1.02], [0.82, 1, 1.1]);
  settings.cloudTone!.objectSunTint = tint([1, 1, 1], [1.08, 1, 0.88]);
  return alive;
}
