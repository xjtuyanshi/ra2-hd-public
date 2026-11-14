import { describe, expect, it } from 'vitest';
import { float, vec3, vec4 } from 'three/tsl';
import {
  Mesh,
  BoxGeometry,
  NodeMaterial,
  OrthographicCamera,
  QuadMesh,
  Scene,
  WebGPURenderer,
  Group,
} from 'three/webgpu';
import {
  CloudToneUniforms,
  cloudToneNode,
  cloudToneRgb,
  cloudGreenness,
  cloudNeutralGradeTint,
} from '../src/cloudTone';
import {
  lush2Defaults,
  lush2Mapping,
  lush2FromMapping,
  LUSH2_DEFAULTS,
  LUSH2_SCHEMA,
  applyLush2,
} from '../src/ra2/lush2Tone';
import { AtmosphereNodes } from '../src/atmosphere';
import { ALIVE_CLOUD_SHAPES, ALIVE_DEFAULTS, resolveAlive, cloudCoverage, aliveDrift } from '../src/livingWorld';
import { VoxelSoftContactMesh, setShadowGround } from '../src/ra2/voxelShadow';
import { objectGroundAnchor, setObjectGroundAnchor } from '../src/objectGroundAnchor';
import { HdRenderer, LOOK_PRESETS, type HdSettings } from '../src/HdRenderer';
import { cinematicGradeRgb } from '../src/post';
const luma = (rgb: readonly number[]) => rgb.reduce((s, x, i) => s + x * [0.2126, 0.7152, 0.0722][i]!, 0);
const hue = (rgb: readonly number[]) => {
  const [r, g, b] = rgb as number[];
  const max = Math.max(...rgb),
    min = Math.min(...rgb),
    c = max - min;
  return ((max === r ? (g! - b!) / c : max === g ? (b! - r!) / c + 2 : (r! - g!) / c + 4) * 60 + 360) % 360;
};
const saturation = (rgb: readonly number[]) => 1 - Math.min(...rgb) / Math.max(...rgb);
const decode = (rgb: readonly number[]) =>
  rgb.map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
const encode = (rgb: readonly number[]) =>
  rgb.map((v) => 255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055)) as [number, number, number];
const recommended = (theater = 'temperate') => lush2Mapping(lush2Defaults(theater), theater).tone;

describe('lush2 colour reference', () => {
  it('uses the round-three grass defaults with rich lit green and greener shade in sRGB', () => {
    const rgb = decode([150, 160, 60]);
    const p = recommended();
    const lit = encode(cloudToneRgb(rgb, 0, p));
    const shade = encode(cloudToneRgb(rgb, 1, p));
    for (const colour of [lit, shade]) {
      expect(saturation(colour)).toBeGreaterThanOrEqual(0.75);
      expect(colour[0] / colour[1]).toBeGreaterThanOrEqual(0.7);
      expect(colour[0] / colour[1]).toBeLessThanOrEqual(0.95);
      expect(colour[2] / colour[1]).toBeLessThanOrEqual(0.35);
    }
    expect(lush2Defaults('temperate')).toMatchObject({ shadeHue: 0.8, chroma: 1.3 });
    expect(p.grassSunTint).toEqual([0.88, 1, 0.6]);
    expect(lit[0] / lit[1]).toBeGreaterThanOrEqual(0.8);
    expect(shade[0] / shade[1]).toBeLessThan(lit[0] / lit[1]);
    expect(Math.max(...lit) / 255).toBeGreaterThanOrEqual(0.45);
    expect(Math.max(...lit) / 255).toBeLessThanOrEqual(0.48);
    for (const shadeHue of [0, 0.5, 1]) {
      const { grassShadeTint } = lush2Mapping({ ...lush2Defaults('temperate'), shadeHue }, 'temperate').tone;
      expect(grassShadeTint[0]).toBeLessThan(1);
      expect(grassShadeTint[1]).toBe(1);
      expect(grassShadeTint[2]).toBeLessThan(1);
    }
  });
  it('changes ground chroma after tinting while preserving depth luma; objects ignore the slider', () => {
    const rgb = decode([150, 160, 60]);
    const a = lush2Mapping({ ...lush2Defaults('temperate'), chroma: 0.6 }, 'temperate').tone;
    const b = lush2Mapping({ ...lush2Defaults('temperate'), chroma: 2 }, 'temperate').tone;
    for (const shade of [0, 0.5, 1]) {
      const low = cloudToneRgb(rgb, shade, a),
        high = cloudToneRgb(rgb, shade, b);
      expect(saturation(encode(high))).toBeGreaterThan(saturation(encode(low)));
      expect(luma(high)).toBeCloseTo(luma(low), 12);
      expect(cloudToneRgb(rgb, shade, a, 0, rgb, true)).toEqual(cloudToneRgb(rgb, shade, b, 0, rgb, true));
    }
  });
  it('preserves bare sand/dirt hue, saturation and bare-depth luma independently of all grass controls', () => {
    for (const theater of ['temperate', 'desert']) {
      const p = recommended(theater);
      expect(p.bareChroma).toBeLessThanOrEqual(theater === 'desert' ? 1.08 : 1.05);
      for (const pixel of [
        [230, 190, 120],
        [190, 120, 80],
      ] as const) {
        const rgb = decode(pixel);
        expect(cloudGreenness(rgb)).toBe(0);
        // The grass-only daylight fix must retain the pre-existing moon tint on bare ground.
        expect(cloudToneRgb(rgb, 1, p, 1)).not.toEqual(cloudToneRgb(rgb, 1, { ...p, moonTint: [1, 1, 1] }, 1));
        for (const mean of [rgb, decode([150, 160, 60])])
          for (const shade of [0, 0.5, 1]) {
            const graded = cloudToneRgb(rgb, shade, p, 0, mean);
            const out = encode(graded);
            expect(Math.abs(hue(out) - hue(pixel))).toBeLessThan(3);
            expect(saturation(out) - saturation(pixel)).toBeLessThan(0.06);
            expect(luma(graded) / luma(rgb)).toBeCloseTo(p.bareDepth, 12);
            for (const chroma of [0.6, 2]) {
              const extreme = lush2Mapping({ ...lush2Defaults(theater), chroma, gold: 1, shadeHue: 1 }, theater).tone;
              expect(cloudToneRgb(rgb, shade, extreme, 0, mean)).toEqual(graded);
            }
          }
      }
    }
    expect(cloudGreenness(decode([150, 160, 60]))).toBeGreaterThan(0.9);
  });
  it('classifies pale yellow-green stripes with their lawn and preserves their luminance contrast', () => {
    const grass = decode([150, 160, 60]),
      stripe = decode([190, 190, 110]);
    const sharedMean = grass.map((v, i) => (v + stripe[i]!) / 2) as [number, number, number];
    for (const mean of [grass, stripe, sharedMean]) {
      for (const rgb of [grass, stripe]) expect(cloudGreenness(mean, rgb)).toBeGreaterThan(0.8);
      for (const shade of [0, 0.5, 1]) {
        const p = recommended();
        const a = cloudToneRgb(grass, shade, p, 0, mean),
          b = cloudToneRgb(stripe, shade, p, 0, mean);
        const contrastChange = luma(b) / luma(a) / (luma(stripe) / luma(grass));
        expect(contrastChange).toBeGreaterThanOrEqual(0.75);
        expect(contrastChange).toBeLessThanOrEqual(1.25);
        expect(luma(b) / luma(stripe) / (luma(a) / luma(grass))).toBeCloseTo(1, 12);
        expect(saturation(encode(b))).toBeGreaterThan(saturation([190, 190, 110]));
        expect(Math.min(...b)).toBeGreaterThan(0);
        expect(Math.min(...cloudToneRgb(stripe, shade, { ...p, chroma: 2 }, 0, mean))).toBeGreaterThan(0);
      }
      for (const pixel of [
        [230, 190, 120],
        [190, 120, 80],
      ] as const)
        expect(cloudGreenness(mean, decode(pixel))).toBeLessThan(0.1);
    }
    // Even separately classified broad patches differ in their depth gain by less than 10%.
    const p = recommended();
    const grassGain = luma(cloudToneRgb(grass, 0, p)) / luma(grass);
    const stripeGain = luma(cloudToneRgb(stripe, 0, p)) / luma(stripe);
    expect(Math.abs(stripeGain / grassGain - 1)).toBeLessThan(0.1);
  });
  it('colours low-saturation jungle rings green with a soft chroma limit instead of cyan halos', () => {
    const pixel = [120, 140, 110] as const,
      rgb = decode(pixel);
    const mean = decode([115, 135, 105]);
    expect(cloudGreenness(rgb)).toBeGreaterThan(0.6);
    expect(cloudGreenness(mean, rgb)).toBeGreaterThan(0.6);
    for (const shade of [0, 0.5, 1]) {
      const outputs = [1, 1.3, 2].map((chroma) => cloudToneRgb(rgb, shade, { ...recommended(), chroma }, 0, mean));
      for (const out of outputs) {
        const display = encode(out);
        expect(hue(display)).toBeGreaterThanOrEqual(60);
        expect(hue(display)).toBeLessThanOrEqual(hue(pixel));
        expect(display[1]).toBeGreaterThan(display[0]);
        expect(display[0]).toBeGreaterThan(display[2]);
        expect(Math.min(...out)).toBeGreaterThan(0);
        expect(luma(out) / luma(rgb)).toBeCloseTo(recommended().depth, 12);
      }
      expect(saturation(encode(outputs[1]!))).toBeGreaterThan(saturation(encode(outputs[0]!)));
      expect(saturation(encode(outputs[2]!))).toBeGreaterThan(saturation(encode(outputs[1]!)));
    }
  });
  it('uses the requested smooth HSV gates on the mean and has no source-grey hue discontinuity', () => {
    const hsv = (h: number, s: number) => {
      const x = (h / 60) % 2,
        c = 180 * s,
        low = 180 - c;
      const sectors = [
        [c, c * (1 - Math.abs(x - 1)), 0],
        [c * (1 - Math.abs(x - 1)), c, 0],
        [0, c, c * (1 - Math.abs(x - 1))],
        [0, c * (1 - Math.abs(x - 1)), c],
        [c * (1 - Math.abs(x - 1)), 0, c],
        [c, 0, c * (1 - Math.abs(x - 1))],
      ];
      return decode(sectors[Math.floor(h / 60)]!.map((v) => v + low));
    };
    const smooth = (a: number, b: number, x: number) => {
      const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
      return t * t * (3 - 2 * t);
    };
    for (const h of [20, 40, 48, 54, 58, 60, 62, 75, 100, 175, 185, 195, 220])
      for (const s of [0.04, 0.08, 0.15, 0.2, 0.22, 0.5])
        expect(cloudGreenness(hsv(h, s))).toBeCloseTo(
          smooth(48, 62, h) * (1 - smooth(175, 195, h)) * smooth(0.08, 0.22, s),
          10,
        );
    const mean = decode([150, 160, 60]);
    const neutral = cloudToneRgb(decode([140, 140, 140]), 0, recommended(), 0, mean);
    for (const pixel of [
      [140.00001, 140, 140],
      [140, 140.00001, 140],
      [140, 140, 140.00001],
    ]) {
      const out = cloudToneRgb(decode(pixel), 0, recommended(), 0, mean);
      expect(cloudGreenness(mean, decode(pixel))).toBe(1);
      out.forEach((v, i) => expect(v).toBeCloseTo(neutral[i]!, 6));
    }
  });
  it('gives wet sand the original-colour brown recipe, 0.68 gain and 1.15 chroma at the default waterline', () => {
    const rgb = decode([180, 150, 100]);
    for (const theater of ['temperate', 'desert']) {
      const p = recommended(theater);
      const dry = cloudToneRgb(rgb, 0, p);
      const wet = cloudToneRgb(rgb, 0, p, 0, rgb, false, 1);
      const tinted = rgb.map((v, i) => v * [1, 0.9, 0.74][i]!);
      const y = luma(tinted);
      const saturated = tinted.map((v) => Math.max(0, y + (v - y) * 1.15));
      const expected = saturated.map((v) => ((v * luma(rgb)) / luma(saturated)) * p.bareDepth * 0.68);
      wet.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 12));
      expect(luma(wet) / luma(dry)).toBeCloseTo(0.68, 12);
      expect(saturation(encode(wet))).toBeGreaterThan(saturation(encode(dry)));
      expect(wet[0] / wet[1]).toBeGreaterThan(dry[0] / dry[1]);
      const gains = [0, 0.25, 0.5, 0.75, 1].map((band) => luma(cloudToneRgb(rgb, 0, p, 0, rgb, false, band)));
      expect(gains).toEqual([...gains].sort((a, b) => b - a));
      expect(cloudToneRgb(rgb, 0, p, 0, rgb, true, 1)).toEqual(cloudToneRgb(rgb, 0, p, 0, rgb, true, 0));
    }
  });
  it('sets grass eligibility to zero throughout the wet band, even when taps see grass or the wet slider is zero', () => {
    const rgb = [0.65, 0.85, 0.3] as const;
    const mean = [0.3, 0.48, 0.15] as const;
    for (const wet of [0, 0.8, 1]) {
      const p = lush2Mapping({ ...lush2Defaults('temperate'), wet }, 'temperate').tone;
      const changedGrass = {
        ...p,
        chroma: 2,
        gold: 1,
        grassSunTint: [0.1, 1, 0.1] as [number, number, number],
        grassShadeTint: [0.1, 1, 0.1] as [number, number, number],
      };
      for (const band of [0.00001, 0.25, 1])
        for (const shade of [0, 1]) {
          const out = cloudToneRgb(rgb, shade, p, 0, mean, false, band);
          expect(cloudToneRgb(rgb, shade, changedGrass, 0, mean, false, band)).toEqual(out);
          expect(luma(out) / luma(rgb)).toBeCloseTo(p.bareDepth * (1 - p.wetDepth * p.wet * band), 12);
        }
    }
  });
  it('gates gold by lit share, absolute high luma and the local mean', () => {
    const p = { ...recommended(), gold: 1 };
    const noGold = { ...p, gold: 0 };
    for (const rgb of [
      [0.2, 0.4, 0.1],
      [0.5, 0.7, 0.3],
    ] as const) {
      expect(cloudToneRgb(rgb, 1, p, 0, [0.25, 0.4, 0.12])).toEqual(cloudToneRgb(rgb, 1, noGold, 0, [0.25, 0.4, 0.12]));
      expect(cloudToneRgb(rgb, 0, p)).toEqual(cloudToneRgb(rgb, 0, noGold));
    }
    const bright = [0.65, 0.85, 0.3] as const,
      mean = [0.3, 0.48, 0.15] as const;
    expect(cloudToneRgb(bright, 0, p, 0, mean)).not.toEqual(cloudToneRgb(bright, 0, noGold, 0, mean));
    expect(cloudToneRgb([0.2, 0.4, 0.1], 0, p, 0, mean)).toEqual(cloudToneRgb([0.2, 0.4, 0.1], 0, noGold, 0, mean));
    // At most the high tail of an evenly-spaced lawn ramp can be gilded.
    const ramp = Array.from({ length: 100 }, (_, i) => [0.1 + i * 0.005, 0.2 + i * 0.006, 0.05 + i * 0.002] as const);
    const gilded = ramp.filter(
      (rgb) =>
        JSON.stringify(cloudToneRgb(rgb, 0, p, 0, mean)) !== JSON.stringify(cloudToneRgb(rgb, 0, noGold, 0, mean)),
    );
    expect(gilded.length).toBeLessThanOrEqual(15);
    expect(gilded.length).toBeGreaterThanOrEqual(10);
  });
  it('neutralises cyan lit snow, adds blue only in shade and preserves grass/rock patches', () => {
    const p = recommended('snow');
    for (const pixel of [
      [195, 222, 217],
      [230, 240, 242],
      [255, 255, 255],
    ] as const) {
      const rgb = decode(pixel);
      const lit = encode(cloudToneRgb(rgb, 0, p));
      expect(saturation(lit)).toBeLessThan(0.05);
      expect(luma(cloudToneRgb(rgb, 0, p)) / luma(rgb)).toBeCloseTo(p.bareDepth, 12);
      const shade = cloudToneRgb(rgb, 1, p);
      expect(shade[2]).toBeGreaterThan(shade[0]);
      expect(shade[2] / shade[0]).toBeLessThan(1.23);
      expect(cloudToneRgb(rgb, 0, { ...p, snowBlue: 0 })).toEqual(cloudToneRgb(rgb, 0, p));
    }
    // Every pixel above sRGB luma .45 and below HSV S .3 receives the full .85 neutralisation.
    for (const pixel of [
      [118, 118, 118],
      [135, 175, 170],
      [200, 160, 145],
    ] as const) {
      const rgb = decode(pixel);
      const out = cloudToneRgb(rgb, 0, p);
      const y = luma(rgb);
      const expected = rgb.map((v) => (y + (v - y) * 0.15) * p.bareDepth);
      out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 12));
    }
    for (const rgb of [decode([90, 110, 45]), decode([90, 90, 90]), decode([225, 190, 100])]) {
      for (const shade of [0, 1]) {
        expect(cloudToneRgb(rgb, shade, p)).toEqual(cloudToneRgb(rgb, shade, { ...p, snowWhite: 0, snowBlue: 0 }));
      }
    }
  });
  it('keeps lit snow below S .05 through the real cinematic grade, including its brightest whites', () => {
    const nodes = new AtmosphereNodes(),
      mirror = new AtmosphereNodes();
    const settings = { ...LOOK_PRESETS.cinematic.settings, debugView: 'none', cloudTone: recommended('snow') };
    nodes.cloudTone.set(settings.cloudTone);
    mirror.cloudTone.set(settings.cloudTone);
    const receiver = { nodes, layerNodes: mirror, settings, gradeAirless: false, gradeShare: { value: 1 } };
    const update = (HdRenderer.prototype as unknown as { updateCloudNeutralGrade(): void }).updateCloudNeutralGrade;
    try {
      update.call(receiver);
      const p = {
        ...settings.cloudTone,
        neutralGradeTint: nodes.cloudTone.neutralGradeTint.value.toArray() as [number, number, number],
      };
      expect(p.neutralGradeTint).toEqual(
        cloudNeutralGradeTint(p.neutralGradeLuma, cinematicGradeRgb([0.8, 0.8, 0.8], settings), 1),
      );
      expect(mirror.cloudTone.neutralGradeTint.value.toArray()).toEqual(p.neutralGradeTint);
      for (const rgb of [decode([195, 222, 217]), decode([230, 240, 242]), decode([255, 255, 255])]) {
        const out = encode(cinematicGradeRgb(cloudToneRgb(rgb, 0, p), settings));
        expect(saturation(out)).toBeLessThan(0.05);
        expect(Math.max(...out)).toBeGreaterThan(210);
        if (luma(rgb) <= p.neutralGradeLuma)
          expect(luma(cloudToneRgb(rgb, 0, p)) / luma(rgb)).toBeCloseTo(p.bareDepth, 12);
      }
      // Highlight headroom remains monotonic: adjacent painted whites do not collapse to a flat clipped plateau.
      const ramp = [0.7, 0.8, 0.9, 1].map((y) => luma(cloudToneRgb([y, y, y], 0, p)));
      for (let i = 1; i < ramp.length; i++) expect(ramp[i]).toBeGreaterThan(ramp[i - 1]!);
      receiver.gradeShare.value = 0;
      update.call(receiver);
      expect(nodes.cloudTone.neutralGradeTint.value.toArray()).toEqual([1, 1, 1]);
      receiver.gradeShare.value = 1;
      nodes.cloudTone.set(null);
      update.call(receiver);
      expect(nodes.cloudTone.neutralGradeTint.value.toArray()).toEqual([1, 1, 1]);
    } finally {
      nodes.dispose();
      mirror.dispose();
    }
  });
  it('is exact identity at strength 0, including HDR, objects and wet banks', () => {
    for (const rgb of [
      [0, 0, 0],
      [1.7, 0.2, 0.8],
      [0.2, 0.4, 0.1],
    ] as const)
      for (const shade of [0, 0.4, 1])
        for (const night of [0, 0.5, 1])
          for (const object of [false, true])
            expect(
              cloudToneRgb(rgb, shade, { ...recommended(), strength: 0 }, night, [0.4, 0.5, 0.2], object, 1),
            ).toEqual(rgb);
  });
  it('preserves source detail and dark-pixel contrast rather than importing bright tap pixels', () => {
    const p = recommended();
    const mean = [0.4, 0.5, 0.2] as const;
    const rgb = [0.1, 0.2, 0.05] as const;
    for (const shade of [0, 0.5, 1]) {
      const a = cloudToneRgb(rgb, shade, p, 0, mean),
        b = cloudToneRgb([0.2, 0.4, 0.1], shade, p, 0, mean);
      // The pale-chroma shoulder is intentionally colour-dependent; painted luminance contrast stays exact.
      expect(luma(b)).toBeCloseTo(luma(a) * 2, 12);
      expect(luma(a)).toBeLessThan(luma(rgb));
    }
    expect(cloudGreenness([0.3, 0.3, 0.3])).toBe(0);
  });
  it('never lifts source luminance at slider extremes or by importing bright classification taps', () => {
    for (const strength of [0.5, 1, 1.5])
      for (const chroma of [0.6, 1.45, 2])
        for (const depth of [0, 1])
          for (const shade of [0, 0.5, 1]) {
            const p = lush2Mapping(
              { ...lush2Defaults('temperate'), strength, depth, chroma, gold: 1, shadeHue: 1 },
              'temperate',
            ).tone;
            for (const rgb of [
              [0.001, 0.002, 0.001],
              [0.5, 0.2, 0.8],
              [0.2, 0.4, 0.1],
              [0.1, 0.1, 0.1],
            ] as const)
              expect(luma(cloudToneRgb(rgb, shade, p, 0, [0.5, 0.8, 0.1]))).toBeLessThanOrEqual(luma(rgb) + 1e-12);
          }
  });
  it('matches the entire depth gain at unitMatch 1, with no grass-only tint; night never brightens', () => {
    const p = { ...recommended(), unitMatch: 1 };
    const rgb = [0.3, 0.4, 0.2] as const;
    for (const shade of [0, 0.5, 1]) {
      const object = cloudToneRgb(rgb, shade, p, 0, rgb, true);
      const ground = cloudToneRgb(rgb, shade, p, 0, [0.2, 0.5, 0.1]);
      expect(luma(object)).toBeCloseTo(luma(ground), 12);
      expect(object).not.toEqual(ground);
      expect(luma(cloudToneRgb(rgb, shade, p, 1, rgb, true))).toBeLessThanOrEqual(luma(rgb));
    }
  });
});

describe('lush2 runtime policy', () => {
  it('round-trips every slider for every theatre and does not mutate defaults', () => {
    const snapshot = structuredClone(LUSH2_DEFAULTS);
    for (const theater of Object.keys(LUSH2_DEFAULTS)) {
      for (const extreme of ['default', 'min', 'max'] as const) {
        const values = lush2Defaults(theater);
        if (extreme !== 'default') for (const field of LUSH2_SCHEMA) values[field.key] = field[extreme];
        const mapped = lush2Mapping(values, theater);
        const roundtrip = lush2FromMapping(mapped.tone, mapped.alive);
        for (const { key } of LUSH2_SCHEMA) expect(roundtrip[key]).toBeCloseTo(values[key], 12);
        expect(mapped.alive.wind).toBe(false);
        expect(mapped.alive.shape!.fineWeight).toBe(1);
        const settings = { groundTone: {}, tone: {} } as HdSettings;
        applyLush2(settings, theater, values);
        expect(settings.groundTone).toBeNull();
        expect(settings.tone).toBeNull();
      }
    }
    expect(LUSH2_DEFAULTS).toEqual(snapshot);
  });
  it('uses the requested ranges and theatre cloud depths; softer edges widen the same field', () => {
    const field = (key: string) => LUSH2_SCHEMA.find((f) => f.key === key)!;
    expect([field('drift').min, field('drift').max]).toEqual([0, 1]);
    expect([field('chroma').min, field('chroma').max]).toEqual([0.6, 2]);
    expect([field('softness').min, field('softness').max]).toEqual([0.2, 1]);
    expect([field('patchScale').min, field('patchScale').max]).toEqual([6, 24]);
    expect([field('shadeDepth').min, field('shadeDepth').max]).toEqual([0, 0.5]);
    for (const [theater, depth] of [
      ['temperate', 0.3],
      ['desert', 0.24],
      ['snow', 0.18],
      ['urban', 0.26],
    ] as const) {
      expect(lush2Defaults(theater).shadeDepth).toBe(depth);
      expect(lush2Defaults(theater).patchScale).toBe(12);
      expect(lush2Defaults(theater).contact).toBe(0.8);
      expect(lush2Defaults(theater).drift).toBe(0.25);
      const mapped = lush2Mapping(lush2Defaults(theater), theater);
      const [dx, dz] = aliveDrift(48, 0.6, mapped.alive.speed);
      expect(Math.hypot(dx, dz)).toBeCloseTo(12, 12);
    }
    // Deterministic transects across the actual JS cloud field, not a linear-ramp substitute.
    const widths = (softness: number) => {
      const shape = lush2Mapping({ ...lush2Defaults('temperate'), softness }, 'temperate').alive.shape!;
      const result: number[] = [];
      for (let z = 0; z < 64; z += 4) {
        let start: number | undefined;
        for (let i = 0; i < 12800; i++) {
          const x = -64 + i / 100,
            cover = cloudCoverage(x, z, 0, 0, 0, shape);
          if (cover < 0.1) start = undefined;
          if (cover >= 0.1 && cover < 0.9 && start === undefined) start = x;
          if (cover >= 0.9 && start !== undefined) {
            result.push(x - start);
            start = undefined;
          }
        }
      }
      result.sort((a, b) => a - b);
      return result[Math.floor(result.length / 2)]!;
    };
    const nominal = widths(0.7);
    expect(nominal).toBeGreaterThan(2.2);
    expect(nominal).toBeLessThan(2.9);
    expect(widths(0.2)).toBeLessThan(nominal);
    expect(widths(1)).toBeGreaterThan(nominal);
  });
  it('switches all uniforms in place, including mirrors; leaves soft/deep untouched and restores alive options', () => {
    const a = new AtmosphereNodes(),
      b = new AtmosphereNodes();
    const soft = structuredClone(ALIVE_CLOUD_SHAPES.soft),
      deep = structuredClone(ALIVE_CLOUD_SHAPES.deep);
    try {
      const identities = Object.values(a.cloudTone);
      const grass = a.cloudTone.grassShadeTint.value;
      const mapped = lush2Mapping({ ...lush2Defaults('snow'), patchScale: 9, shadeDepth: 0, drift: 0.1 }, 'snow');
      a.cloudTone.set(mapped.tone);
      a.alive.set(mapped.alive);
      a.alive.update(20, 0.6, 0, 0, false);
      b.copyValuesFrom(a);
      expect(a.alive.cloudLattice.value.x).toBe(1 / 9);
      expect(a.alive.cloudForm.value.x).toBe(1);
      expect(a.alive.cloudStrength.value).toBe(0);
      expect(a.alive.cloudAmount.value).toBe(1);
      expect(a.cloudTone.grassShadeTint.value).toBe(grass);
      expect(Object.values(a.cloudTone)).toEqual(identities);
      expect(b.cloudTone.strength.value).toBe(a.cloudTone.strength.value);
      expect(b.alive.cloudAmount.value).toBe(1);
      expect(resolveAlive(ALIVE_DEFAULTS, mapped.alive)).toEqual(ALIVE_DEFAULTS);
      expect(ALIVE_CLOUD_SHAPES.soft).toEqual(soft);
      expect(ALIVE_CLOUD_SHAPES.deep).toEqual(deep);
      expect(cloudCoverage(4, 9, 17, 0.6, 0.03, ALIVE_CLOUD_SHAPES.lush)).toBe(
        cloudCoverage(4, 9, 17, 0.6, 0.03, ALIVE_CLOUD_SHAPES.lush),
      );
    } finally {
      a.dispose();
      b.dispose();
    }
  });
  it('uses a shared unit root for hull, turret and barrel foot samples', () => {
    const root = new Group(),
      turret = new Group(),
      hull = new Mesh(),
      barrel = new Mesh();
    root.position.set(8, 0, 13);
    turret.position.set(0.3, 2, 0.1);
    root.add(hull, turret);
    turret.add(barrel);
    root.updateMatrixWorld(true);
    setObjectGroundAnchor(hull, root);
    setObjectGroundAnchor(barrel, root);
    // Exercise the actual per-draw uniform updater.
    const update = objectGroundAnchor.update as (frame: unknown) => void;
    update.call(objectGroundAnchor, { object: hull });
    expect(objectGroundAnchor.value.toArray()).toEqual([8, 0, 13]);
    update.call(objectGroundAnchor, { object: barrel });
    expect(objectGroundAnchor.value.toArray()).toEqual([8, 0, 13]);
  });
  it('keeps the ellipse under the hull, excludes water/flight, vanishes at zero and has an owned shared material', () => {
    const nodes = new AtmosphereNodes(),
      scene = new Scene(),
      root = new Group(),
      source = new Mesh(new BoxGeometry(0.8, 0.4, 0.5));
    scene.add(root);
    root.add(source);
    const contact = new VoxelSoftContactMesh(source, nodes);
    root.add(contact);
    setShadowGround(scene, () => 0);
    try {
      nodes.cloudTone.set(recommended());
      scene.updateMatrixWorld(true);
      expect(contact.visible).toBe(true);
      expect(contact.matrixWorld.elements[13]).toBeCloseTo(0.025);
      expect(contact.matrixWorld.elements[0]).toBeCloseTo((0.8 * 1.3) / 2);
      expect(contact.matrixWorld.elements[10]).toBeCloseTo((0.5 * 1.3) / 2);
      expect((contact.material as NodeMaterial).depthWrite).toBe(false);
      nodes.cloudTone.contact.value = 0;
      scene.updateMatrixWorld(true);
      expect(contact.visible).toBe(false);
      nodes.cloudTone.contact.value = 1;
      root.position.y = 5;
      scene.updateMatrixWorld(true);
      expect(contact.visible).toBe(false);
      root.position.y = 0;
      setShadowGround(scene, { height: () => 0, waterHeight: () => 0 });
      scene.updateMatrixWorld(true);
      expect(contact.visible).toBe(false);
    } finally {
      source.geometry.dispose();
      nodes.dispose();
    }
  });
});
it('builds the uniform-only cloud grade shader without noise or texture replacement', () => {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const material = new NodeMaterial();
  const uniforms = new CloudToneUniforms();
  uniforms.set(recommended());
  material.fragmentNode = vec4(
    cloudToneNode(vec3(0.2, 0.4, 0.1), vec3(0.2, 0.4, 0.1), float(1), uniforms, float(0)),
    1,
  );
  const builder = renderer.backend.createNodeBuilder(new QuadMesh(material), renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  try {
    builder.build();
    expect(builder.fragmentShader).toContain('@fragment');
    expect(builder.fragmentShader).not.toContain('textureSample');
  } finally {
    material.dispose();
  }
});
