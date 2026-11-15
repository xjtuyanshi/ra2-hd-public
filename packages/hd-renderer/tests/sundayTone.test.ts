import { describe, expect, it } from 'vitest';
import { cloudGreenness, cloudToneGain, cloudToneRgb, cloudTreeToneRgb, type CloudToneParams } from '../src/cloudTone';
import { LUSH2_R3_TEMPERATE, LUSH2_SCHEMA, lush2Defaults, lush2Mapping } from '../src/ra2/lush2Tone';
import { r3CloudToneRgb } from './fixtures/r3CloudTone';

const tone = (theater = 'temperate') => lush2Mapping(lush2Defaults(theater), theater).tone;
const y = (rgb: readonly number[]) => rgb.reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i]!, 0);
const linear = (rgb: readonly number[]) =>
  rgb.map((v) => ((v / 255 + 0.055) / 1.055) ** 2.4) as [number, number, number];
const lightness = (rgb: readonly number[]) => 116 * Math.cbrt(y(rgb)) - 16;
function lch(rgb: readonly number[]) {
  const [r, g, b] = rgb as [number, number, number];
  const f = (v: number) => (v > (6 / 29) ** 3 ? Math.cbrt(v) : v / (3 * (6 / 29) ** 2) + 4 / 29);
  const x = f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
  const yy = f(y(rgb));
  const z = f((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
  const a = 500 * (x - yy),
    bb = 200 * (yy - z);
  return [116 * yy - 16, Math.hypot(a, bb), ((Math.atan2(bb, a) * 180) / Math.PI + 360) % 360];
}

describe('Sunday same-build r3 gates', () => {
  it('exports the exact frozen temperate override and preserves all inherited theatre chroma values', () => {
    expect(Object.isFrozen(LUSH2_R3_TEMPERATE)).toBe(true);
    expect(LUSH2_R3_TEMPERATE).toEqual({ grassChroma: 1.2, shadeChroma: 1.25, wetSand: 0, wetSheen: 0, treeBlend: 0 });
    for (const [theater, grass, shade, enabled] of [
      ['temperate', 1, 1.2, 1],
      ['urban', 1.04, 1.25, 1],
      ['newurban', 1.04, 1.25, 1],
      ['desert', 1.02, 1.25, 1],
      ['snow', 1, 1, 0],
      ['lunar', 1.2, 1.25, 0],
    ] as const) {
      const p = tone(theater);
      expect([p.grassChroma, p.shadeChroma, p.wetSand, p.treeBlend]).toEqual([grass, shade, enabled, enabled]);
      expect(p.treeReference).toEqual([0.356, 0.372, 0.0884]);
      expect(p.wetSandTint).toEqual([0.5161, 0.267, 0.0458]);
      expect([p.wetSandDepth, p.wetSheen]).toEqual([0.7, enabled * 0.5]);
    }
    for (const key of ['wetSand', 'wetSheen', 'wetSandDepth', 'treeBlend']) {
      expect(LUSH2_SCHEMA.find((f) => f.key === key)).toMatchObject({ min: 0, max: 1, step: 0.01 });
    }
  });

  it('matches the frozen f2542b5 CPU pixel recipe exactly at r3 overrides, including objects, wet ramps and night', () => {
    const p = { ...tone(), ...LUSH2_R3_TEMPERATE };
    for (const rgb of [
      [0, 0, 0],
      [0.2, 0.3, 0.05],
      [0.6, 0.4, 0.1],
      [0.8, 0.8, 0.8],
    ] as const)
      for (const shade of [0, 0.35, 1])
        for (const night of [0, 0.5, 1])
          for (const wet of [0, 1e-6, 0.1, 0.3, 0.75, 1])
            for (const object of [false, true])
              for (const detail of [0.4, 1, 1.2]) {
                const params = { ...p, detail };
                const args = [rgb, shade, params, night, [0.3, 0.4, 0.1], object, wet, 0.2, [0.4, 0.5, 0.2]] as const;
                expect(cloudToneRgb(...args)).toEqual(r3CloudToneRgb(...args));
                expect(cloudTreeToneRgb(rgb, shade, params, night, 0.2)).toEqual(
                  r3CloudToneRgb(rgb, shade, params, night, rgb, true, 0, 0.2),
                );
              }
  });

  it('keeps the snow defaults exactly on the old CPU recipe for ground and objects', () => {
    const p = tone('snow');
    for (const rgb of [
      [0.2, 0.3, 0.05],
      [0.6, 0.55, 0.7],
      [1, 1, 1],
    ] as const)
      for (const shade of [0, 0.5, 1])
        for (const night of [0, 0.5, 1])
          for (const object of [false, true]) {
            expect(cloudToneRgb(rgb, shade, p, night, rgb, object, 1)).toEqual(
              r3CloudToneRgb(rgb, shade, p, night, rgb, object, 1),
            );
            expect(cloudTreeToneRgb(rgb, shade, p, night)).toEqual(r3CloudToneRgb(rgb, shade, p, night, rgb, true));
          }
  });
});

it('reduces default grass chroma while preserving OKLab lightness and hue', () => {
  const oklab = (rgb: readonly number[]) => {
    const [r, g, b] = rgb as [number, number, number];
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [
      0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    ];
  };
  const rgb = linear([150, 160, 60]);
  for (const shade of [0, 0.5, 1]) {
    const p = tone();
    const old = oklab(cloudToneRgb(rgb, shade, { ...p, ...LUSH2_R3_TEMPERATE }));
    const next = oklab(cloudToneRgb(rgb, shade, p));
    const ratio = Math.hypot(next[1]!, next[2]!) / Math.hypot(old[1]!, old[2]!);
    expect(ratio).toBeGreaterThan(0.69);
    expect(ratio).toBeLessThan(0.9);
    // The unchanged display-channel safety compressor can move OKLab L slightly.
    expect(Math.abs(next[0]! - old[0]!)).toBeLessThan(0.002);
    expect((Math.abs(Math.atan2(next[2]!, next[1]!) - Math.atan2(old[2]!, old[1]!)) * 180) / Math.PI).toBeLessThan(1);
  }
});

describe('Sunday warm land-side sand', () => {
  it('uses a pre-light/display L*62/C*50/h78 tint and retains brighter source texture', () => {
    const p = tone();
    lch(p.wetSandTint).forEach((v, i) => expect(v).toBeCloseTo([62, 50, 78][i]!, 1));
    const dryY = ((77 + 16) / 116) ** 3;
    const rgb = [dryY, dryY, dryY] as const;
    const out = cloudToneRgb(rgb, 0, p, 0, rgb, false, 1);
    const [l, c, h] = lch(out);
    expect(l).toBeGreaterThanOrEqual(60);
    expect(l).toBeLessThanOrEqual(64);
    expect(c).toBeGreaterThanOrEqual(49);
    expect(c).toBeLessThanOrEqual(51);
    expect(h).toBeGreaterThanOrEqual(77);
    expect(h).toBeLessThanOrEqual(79);
    const values = [0.3, 0.4, 0.5, 0.6].map((v) => y(cloudToneRgb([v, v, v], 0, p, 0, [v, v, v], false, 1)));
    expect(values.every((v, i) => i === 0 || v > values[i - 1]!)).toBe(true);
  });

  it('replaces green source chroma, floors dark rim luminance, and has a continuous ordered shore ramp', () => {
    const p = tone();
    for (const rgb of [linear([150, 160, 60]), linear([105, 115, 50]), linear([215, 185, 130])]) {
      const out = cloudToneRgb(rgb, 0, p, 0, rgb, false, 1);
      expect(lightness(out)).toBeGreaterThanOrEqual(52);
      expect(lch(out)[2]).toBeGreaterThanOrEqual(77);
      expect(lch(out)[2]).toBeLessThanOrEqual(79);
      const dry = cloudToneRgb(rgb, 0, p);
      cloudToneRgb(rgb, 0, p, 0, rgb, false, 1e-6).forEach((v, i) => expect(v).toBeCloseTo(dry[i]!, 6));
      const before = cloudToneRgb(rgb, 0, p, 0, rgb, false, 1 - 1e-6);
      before.forEach((v, i) => expect(v).toBeCloseTo(out[i]!, 6));
    }
    const sand = linear([215, 185, 130]);
    const ramp = Array.from({ length: 101 }, (_, i) =>
      y(cloudToneRgb(sand, 0, { ...p, wetSheen: 0 }, 0, sand, false, i / 100)),
    );
    expect(ramp.every((v, i) => i === 0 || v <= ramp[i - 1]! + 1e-12)).toBe(true);
  });

  it('never applies the legacy chroma expansion to the new sand path, including partial wet ramps', () => {
    const p = tone(),
      rgb = linear([105, 115, 50]);
    for (const wet of [0.1, 0.3, 0.5, 0.85, 1]) {
      const base = cloudToneRgb(rgb, 0, { ...p, wetChroma: 1 }, 0, rgb, false, wet);
      cloudToneRgb(rgb, 0, { ...p, wetChroma: 3 }, 0, rgb, false, wet).forEach((v, i) =>
        expect(v).toBeCloseTo(base[i]!, 12),
      );
    }
  });

  it('limits sheen to outer, lit, daytime band and keeps objects/dry land untouched', () => {
    const rgb = linear([215, 185, 130]);
    const p = tone();
    for (const wet of [0, 0.5, 0.7, 0.85, 1])
      for (const night of [0, 0.5, 1])
        for (const shade of [0, 0.5, 1]) {
          const off = cloudToneRgb(rgb, shade, { ...p, wetSheen: 0 }, night, rgb, false, wet);
          const on = cloudToneRgb(rgb, shade, { ...p, wetSheen: 1 }, night, rgb, false, wet);
          expect(y(on) / y(off)).toBeLessThanOrEqual(1.04 + 1e-12);
          if (wet <= 0.7 || night === 1 || shade === 1) expect(on).toEqual(off);
        }
    expect(cloudToneRgb(rgb, 0, p, 0, rgb, true, 1)).toEqual(r3CloudToneRgb(rgb, 0, p, 0, rgb, true, 1));
    expect(cloudToneRgb(rgb, 0, p, 0, rgb, false, 0)).toEqual(r3CloudToneRgb(rgb, 0, p, 0, rgb, false, 0));
  });
});

describe('Sunday reference-grass foliage grade', () => {
  it('gives green leaves exactly the reference grass gain, regardless of their own albedo', () => {
    const p = tone();
    for (const rgb of [linear([150, 160, 60]), linear([60, 95, 30]), linear([35, 50, 20])])
      for (const shade of [0, 0.5, 1])
        for (const night of [0, 0.5, 1]) {
          expect(cloudGreenness(rgb, p)).toBe(1);
          const gain = cloudToneGain(p.treeReference, p.treeReference, shade, p, night);
          const amount = p.strength * (1 + (p.nightKeep - 1) * night);
          const ground = rgb.map((v, i) => v + (v * gain[i]! - v) * amount);
          const object = cloudToneRgb(rgb, shade, p, night, rgb, true, 0, p.treeContactHeight);
          expect(cloudTreeToneRgb(rgb, shade, p, night, p.treeContactHeight)).toEqual(
            object.map((v, i) => v + (ground[i]! - v) * p.treeBlend),
          );
        }
  });

  it('gives trunks, red/brown leaves and neutral crown shadows only the reference luminance gain', () => {
    const p = tone();
    for (const rgb of [linear([140, 95, 60]), linear([150, 45, 35]), linear([30, 30, 30])])
      for (const shade of [0, 0.5, 1])
        for (const night of [0, 0.5, 1]) {
          expect(cloudGreenness(rgb, p)).toBe(0);
          const gain = cloudToneGain(p.treeReference, p.treeReference, shade, p, night);
          const lum = y(p.treeReference.map((v, i) => v * gain[i]!)) / y(p.treeReference);
          const amount = p.strength * (1 + (p.nightKeep - 1) * night);
          const expectedGain = 1 + (lum - 1) * amount;
          const out = cloudTreeToneRgb(rgb, shade, p, night, p.treeContactHeight);
          out.forEach((v, i) => expect(v / rgb[i]!).toBeCloseTo(expectedGain, 12));
        }
  });

  it('fades contact softly over .25 world cells, scales it with blend and halves it at night', () => {
    const p = tone(),
      rgb = linear([150, 160, 60]);
    expect(p.treeContactHeight).toBe(0.25);
    for (const night of [0, 1]) {
      const top = cloudTreeToneRgb(rgb, 0, p, night, p.treeContactHeight);
      const bottom = cloudTreeToneRgb(rgb, 0, p, night, 0);
      expect(y(bottom) / y(top)).toBeCloseTo(1 - 0.25 * (1 - 0.5 * night), 12);
      expect(y(cloudTreeToneRgb(rgb, 0, p, night, 0.125)) / y(top)).toBeCloseTo(1 - 0.125 * (1 - 0.5 * night), 12);
    }
    const old = cloudTreeToneRgb(rgb, 0, { ...p, treeBlend: 0 }, 0, 0);
    const full = cloudTreeToneRgb(rgb, 0, p, 0, 0);
    cloudTreeToneRgb(rgb, 0, { ...p, treeBlend: 0.5 }, 0, 0).forEach((v, i) =>
      expect(v).toBeCloseTo((old[i]! + full[i]!) / 2, 12),
    );
  });

  it('leaves ineligible trees and non-tree objects exactly on r3, independent of the foliage knobs', () => {
    const rgb = linear([150, 160, 60]);
    for (const theater of ['temperate', 'snow', 'lunar'])
      for (const treeBlend of [0, 0.5, 1]) {
        const p: CloudToneParams = { ...tone(theater), treeBlend };
        const old = r3CloudToneRgb(rgb, 0.4, p, 0.3, rgb, true);
        // The caller supplies false for non-trees, flat cards, lighting-none, grade-exempt and house paint.
        expect(cloudTreeToneRgb(rgb, 0.4, p, 0.3, 0, false)).toEqual(old);
        expect(cloudToneRgb(rgb, 0.4, p, 0.3, rgb, true)).toEqual(old);
      }
    const black = [0, 0, 0] as const;
    expect(cloudTreeToneRgb(black, 0, tone(), 0, 1)).toEqual([0, 0, 0]);
    expect(cloudTreeToneRgb(rgb, 0, { ...tone(), strength: 0 })).toEqual([...rgb]);
    expect(cloudToneGain(rgb, rgb, 0, tone())).toEqual(cloudToneGain(rgb, rgb, 0, { ...tone(), treeBlend: 0 }));
  });
});
