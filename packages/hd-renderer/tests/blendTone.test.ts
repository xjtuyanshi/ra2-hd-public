import { describe, it, expect } from 'vitest';
import type { HdSettings } from '../src/HdRenderer';
import { applyBlendTone, blendDefaults, parseBlendPreset } from '../src/ra2/blendTone';
import { applyLush2 } from '../src/ra2/lush2Tone';
import { cloudToneRgb } from '../src/cloudTone';

describe('opt-in blend recipes', () => {
  it('defaults to contact and retains explicit ground and legacy dramatic URLs', () => {
    expect(parseBlendPreset(null)).toBe('contact');
    expect(parseBlendPreset('invalid')).toBe('contact');
    expect(parseBlendPreset('dramatic')).toBe('ground');
    expect(parseBlendPreset('ground')).toBe('ground');
    expect(parseBlendPreset('tuned')).toBe('tuned');
    expect(blendDefaults('temperate', 'ground').grassLight).toBe(0.84);
  });
  it('composes with lush2 without changing terrain or clouds and keeps object luminance at least as bright', () => {
    const settings = {} as HdSettings;
    const alive = applyLush2(settings, 'temperate');
    const baseline = structuredClone(settings);
    expect(applyBlendTone(settings, 'temperate', 'contact', undefined, alive)).toBe(alive);
    const { objectSunTint: before, objectFloor: beforeFloor, ...groundBefore } = baseline.cloudTone!;
    const { objectSunTint: after, objectFloor: afterFloor, ...groundAfter } = settings.cloudTone!;
    expect(groundAfter).toEqual(groundBefore);
    expect(after).not.toEqual(before);
    expect(afterFloor).toBe(1);
    expect(afterFloor).toBeGreaterThanOrEqual(beforeFloor);
    const y = (rgb: readonly number[]) => rgb[0]! * 0.2126 + rgb[1]! * 0.7152 + rgb[2]! * 0.0722;
    for (const night of [0, 0.0666, 0.5, 1])
      for (const shade of [0, 0.5, 1]) {
        const rgb = [0.12, 0.24, 0.08] as const;
        expect(y(cloudToneRgb(rgb, shade, settings.cloudTone!, night, rgb, true))).toBeGreaterThanOrEqual(
          y(cloudToneRgb(rgb, shade, baseline.cloudTone!, night, rgb, true)),
        );
        expect(cloudToneRgb(rgb, shade, settings.cloudTone!, night)).toEqual(
          cloudToneRgb(rgb, shade, baseline.cloudTone!, night),
        );
      }
  });
  it('does not install any grade for standalone contact', () => {
    const settings = { cloudTone: null, groundTone: null, tone: null } as HdSettings;
    const before = structuredClone(settings);
    applyBlendTone(settings, 'temperate');
    expect(settings).toEqual(before);
  });
  it('retains C tuned values and produces the same tint after restoring saved slider values', () => {
    const settings = {} as HdSettings;
    applyBlendTone(settings, 'temperate', 'tuned');
    expect(settings.cloudTone).toMatchObject({
      grassLight: 1.01,
      objectFloor: 0.9,
      grassSunTint: [1.01, 1, 0.93],
      shadeTint: [0.97, 1, 1.02],
      objectSunTint: [1, 1, 1],
    });
    applyBlendTone(settings, 'temperate', 'tuned', blendDefaults('temperate', 'ground'));
    expect(settings.cloudTone!.objectSunTint).toEqual([1.08, 1, 0.88]);
  });
  it('warms lit objects without applying a green material classification to house paint', () => {
    const settings = {} as HdSettings;
    applyBlendTone(settings, 'temperate', 'ground');
    const neutral = [0.3, 0.3, 0.3] as const;
    const lit = cloudToneRgb(neutral, 0, settings.cloudTone!, 0, neutral, true);
    expect(lit[0]).toBeGreaterThan(lit[2]);
    const old = {} as HdSettings;
    applyLush2(old, 'temperate');
    expect(old.cloudTone!.objectSunTint).toEqual([1, 1, 1]);
  });
});
