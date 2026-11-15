import type { HdSettings } from '../HdRenderer';
import { applyLush2, lush2Defaults } from './lush2Tone';

/** Balanced owner reference: retain moving shade, source clumps and existing water dressing. */
export function applyBlendTone(settings: HdSettings, theater: string | undefined) {
  const values = { ...lush2Defaults(theater), objectShare: 0.35, objectFloor: 0.9, detail: 1 };
  if (theater === 'temperate' || theater === 'urban' || theater === 'newurban')
    Object.assign(values, {
      grassLight: 1.01,
      grassHue: 5,
      grassChroma: 1.08,
      shadeChroma: 1.08,
      gold: 0.18,
      shadeDepth: 0.14,
    });
  const alive = applyLush2(settings, theater, values);
  settings.cloudTone!.grassSunTint = [1.01, 1, 0.93];
  settings.cloudTone!.shadeTint = [0.97, 1, 1.02];
  return alive;
}
