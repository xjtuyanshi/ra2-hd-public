import type { GroundToneParams } from '../src/groundTone';

const TRIPLES = ['sunTint', 'shadeTint', 'grassShadeTint', 'moonTint'] as const;
const NUMBERS = [
  'strength',
  'scale',
  'detailScale',
  'detailMix',
  'stretch',
  'angle',
  'coverage',
  'softness',
  'depth',
  'bareDepth',
  'bareChroma',
  'darkLo',
  'darkHi',
  'darkKeep',
  'darkPatch',
  'sunGain',
  'shadeGain',
  'bareShade',
  'sunChroma',
  'shadeChroma',
  'soften',
  'nightKeep',
] as const;
/** Sandbox-only ground overrides, after look/tone selection. Tints are linear multiplier triples. */
export function applyGroundOverrides(ground: GroundToneParams | null, overrides: string | null): void {
  if (!ground) return;
  for (const item of (overrides ?? '').split(',').filter(Boolean)) {
    const parts = item.split(':');
    const [key, raw] = parts;
    const triple = TRIPLES.find((k) => k === key);
    if (parts.length !== 2 || !raw?.trim()) throw new Error(`Invalid ground override: ${item}`);
    if (triple) {
      const values = raw.split('/').map((v) => (v.trim() ? Number(v) : NaN));
      if (values.length !== 3 || !values.every(Number.isFinite))
        throw new Error(`Invalid ground override: ${item}; expected ${triple}:a/b/c`);
      ground[triple] = values as [number, number, number];
    } else {
      const number = NUMBERS.find((k) => k === key);
      const value = Number(raw);
      if (!number || !Number.isFinite(value))
        throw new Error(`Invalid ground override: ${item}; expected numericKey:value or tint:a/b/c`);
      ground[number] = value;
    }
  }
}
