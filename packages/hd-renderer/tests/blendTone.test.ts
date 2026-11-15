import { describe, it, expect } from 'vitest';
import type { HdSettings } from '../src/HdRenderer';
import { applyBlendTone, blendDefaults, parseBlendPreset } from '../src/ra2/blendTone';
import { applyLush2 } from '../src/ra2/lush2Tone';
import { cloudToneRgb } from '../src/cloudTone';

describe('opt-in blend recipes', () => {
  it('defaults to dramatic, preserves tuned and leaves non-green theaters at the softer recipe', () => {
    expect(parseBlendPreset(null)).toBe('dramatic');
    expect(parseBlendPreset('invalid')).toBe('dramatic');
    expect(parseBlendPreset('tuned')).toBe('tuned');
    expect(blendDefaults('temperate').grassLight).toBeLessThan(blendDefaults('temperate', 'tuned').grassLight - 0.1);
    for (const theater of ['snow', 'desert', 'lunar'])
      expect(blendDefaults(theater)).toEqual(blendDefaults(theater, 'tuned'));
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
    applyBlendTone(settings, 'temperate', 'tuned', blendDefaults('temperate'));
    expect(settings.cloudTone!.objectSunTint).toEqual([1.08, 1, 0.88]);
  });
  it('warms lit objects without applying a green material classification to house paint', () => {
    const settings = {} as HdSettings;
    applyBlendTone(settings, 'temperate');
    const neutral = [0.3, 0.3, 0.3] as const;
    const lit = cloudToneRgb(neutral, 0, settings.cloudTone!, 0, neutral, true);
    expect(lit[0]).toBeGreaterThan(lit[2]);
    const old = {} as HdSettings;
    applyLush2(old, 'temperate');
    expect(old.cloudTone!.objectSunTint).toEqual([1, 1, 1]);
  });
});
