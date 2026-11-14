import { describe, expect, it } from 'vitest';
import { applyLookPreset, LOOK_PRESETS, type HdSettings } from '../src/HdRenderer';
import { neutralGrade } from '../src/post';
import {
  applyMoistTone,
  moistTone,
  MOIST_LEVEL_SCALE,
  MOIST_THEATRES,
  parseToneStyle,
  TONE_STYLES,
} from '../src/ra2/moistTone';
import { applyToneOverrides } from '../sandbox/tone';

const look = LOOK_PRESETS.cinematic.settings;
describe('RA2 moist tone data', () => {
  it('covers every theatre, falls back safely, and leaves off/unknown disabled', () => {
    expect(Object.keys(MOIST_THEATRES).sort()).toEqual(
      ['temperate', 'snow', 'urban', 'desert', 'lunar', 'newurban'].sort(),
    );
    for (const style of TONE_STYLES) expect(parseToneStyle(style)).toBe(style);
    for (const value of [null, '', 'moist4', 'Moist2']) expect(parseToneStyle(value)).toBe('off');
    expect(moistTone('snow', 'off', look)).toBeNull();
    expect(moistTone('unknown', 'moist2', look)).toEqual(moistTone('temperate', 'moist2', look));
    expect(moistTone(undefined, 'moist2', look)).toEqual(moistTone('temperate', 'moist2', look));
  });
  it.each(Object.keys(MOIST_THEATRES))('scales %s monotonically and preserves non-tint look grading', (theater) => {
    const [low, med, high] = (['moist1', 'moist2', 'moist3'] as const).map((s) => moistTone(theater, s, look)!);
    if (MOIST_THEATRES[theater as keyof typeof MOIST_THEATRES].density > 0) {
      expect(low!.density).toBeLessThan(med!.density);
      expect(med!.density).toBeLessThan(high!.density);
    }
    expect(low!.grain).toBeLessThan(med!.grain);
    expect(med!.grain).toBeLessThan(high!.grain);
    expect(high!.grain).toBeLessThanOrEqual(0.4);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(low!.chroma[i]! - 1)).toBeLessThanOrEqual(Math.abs(med!.chroma[i]! - 1));
      expect(Math.abs(med!.chroma[i]! - 1)).toBeLessThanOrEqual(Math.abs(high!.chroma[i]! - 1));
    }
    for (let i = 0; i < med!.hueShifts.length; i++) {
      expect(Math.abs(low!.hueShifts[i]!.shift)).toBeLessThan(Math.abs(med!.hueShifts[i]!.shift));
      expect(Math.abs(med!.hueShifts[i]!.shift)).toBeLessThan(Math.abs(high!.hueShifts[i]!.shift));
    }
    for (const p of [low!, med!, high!])
      for (const key of ['gradeContrast', 'gradePivotLift', 'gradeSaturation'] as const)
        expect(p.grade![key]).toBe(look[key]);
    expect(high!.grade).toEqual(med!.grade);
    expect(moistTone(theater, 'moist2', neutralGrade())!.grade).toBeNull();
    expect(moistTone(theater, 'moist2', LOOK_PRESETS.faithful.settings)!.grade).toBeNull();
  });
  it('keeps water protected but not snow, and every level monotonic', () => {
    for (const theater of Object.keys(MOIST_THEATRES))
      for (const style of ['moist1', 'moist2', 'moist3'] as const) {
        const p = moistTone(theater, style, look)!;
        // (1 - d) y + d y^6 is monotonic for d <= 1; the S-curve for contrast * 4 * max(pivot, 1 - pivot) < 1.
        expect(p.density).toBeGreaterThanOrEqual(0);
        expect(p.density).toBeLessThanOrEqual(0.5);
        // Units deepen with the ground so they keep their separation from it.
        expect(p.objectDensity).toBe(1);
        expect(p.contrast * 4 * Math.max(p.pivot, 1 - p.pivot)).toBeLessThan(1);
        // Snow whites (encoded luma > 0.85) are outside the water protection.
        expect(p.protect!.maxLuma).toBeLessThan(0.85);
      }
  });
  it('keeps snow cold and clones preset/grade arrays for live tuning', () => {
    const p = moistTone('snow', 'moist2', look)!;
    expect(p.grade!.gradeHighlightTint[2]).toBeGreaterThan(p.grade!.gradeHighlightTint[0]);
    expect(p.grade!.gradeShadowTint[2]).toBeGreaterThan(p.grade!.gradeShadowTint[1]);
    expect(p.density).toBe(0);
    expect(p.hueShifts).toEqual([]);
    p.grade!.gradeBalance[0] = 9;
    p.chroma[0] = 9;
    expect(moistTone('snow', 'moist2', look)!.grade!.gradeBalance[0]).toBe(0.97);
    for (const key of Object.keys(neutralGrade()) as (keyof ReturnType<typeof neutralGrade>)[])
      expect(moistTone('lunar', 'moist2', look)!.grade![key]).toEqual(look[key]);
  });
  it('applies tone after look overrides and accepts only numeric/chroma sandbox overrides', () => {
    const settings = {} as HdSettings;
    applyLookPreset(settings, 'cinematic');
    settings.gradeContrast = 0.2;
    settings.gradeBalance = [0.9, 1, 1.2];
    applyMoistTone(settings, 'temperate', 'moist1');
    expect(settings.tone!.grade!.gradeContrast).toBe(0.2);
    expect(settings.tone!.grade!.gradeBalance[0]).toBeCloseTo(0.9 + (0.99 - 0.9) * MOIST_LEVEL_SCALE.moist1, 12);
    applyToneOverrides(settings.tone, 'density:0.3,grain:0.6,objectShare:0.7,chroma:1.3/1.2/1');
    expect(settings.tone).toMatchObject({ density: 0.3, grain: 0.6, objectShare: 0.7, chroma: [1.3, 1.2, 1] });
    for (const s of ['grade:1', 'night:1', 'unknown:2', 'grain:NaN', 'density', 'chroma:1/2', 'chroma:1/2/NaN'])
      expect(() => applyToneOverrides(settings.tone, s)).toThrow();
    applyMoistTone(settings, 'snow', 'off');
    expect(settings.tone).toBeNull();
    expect(() => applyToneOverrides(null, 'density:1')).not.toThrow();
  });
});
