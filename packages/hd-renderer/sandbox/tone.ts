import type { FinalToneParams } from '../src/tone';

const TRIPLES = ['chroma', 'gradeBalance', 'gradeShadowTint', 'gradeHighlightTint'] as const;

/**
 * Sandbox-only numeric tuning, applied after the look and its setting overrides: numericKey:value, a triple a/b/c for
 * chroma or the grade's balance and tints, shift0-3 for the hue-shift angles and exposure0-3 for their exposures.
 */
export function applyToneOverrides(tone: FinalToneParams | null, overrides: string | null): void {
  if (!tone) return;
  const target = tone as unknown as Record<string, unknown>;
  for (const item of (overrides ?? '').split(',').filter(Boolean)) {
    const [key, raw] = item.split(':');
    const triple = TRIPLES.find((k) => k === key);
    if (triple && raw?.includes('/')) {
      const values = raw.split('/').map(Number);
      if (values.length !== 3 || !values.every(Number.isFinite))
        throw new Error(`Invalid tone override: ${item}; expected ${triple}:a/b/c`);
      if (triple === 'chroma') tone.chroma = values as [number, number, number];
      else if (tone.grade) tone.grade[triple] = values as [number, number, number];
      continue;
    }
    const value = Number(raw);
    const shift = /^(shift|exposure)([0-3])$/.exec(key ?? '');
    if (shift && raw !== undefined && Number.isFinite(value)) {
      const window = tone.hueShifts[Number(shift[2])];
      if (!window) throw new Error(`Invalid tone override: ${item}; this theatre has no hue shift ${shift[2]}`);
      if (shift[1] === 'shift') window.shift = value;
      else window.exposure = value;
      continue;
    }
    if (!key || typeof target[key] !== 'number' || raw === undefined || !Number.isFinite(value))
      throw new Error(
        `Invalid tone override: ${item}; expected numericKey:value, ${TRIPLES.join('|')}:a/b/c or shiftN`,
      );
    target[key] = value;
  }
}
