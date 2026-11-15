import { linearCloudFactor, shadeCloudColour } from '../src/livingWorldNodes';
import { describe, expect, it } from 'vitest';
import { float, vec2, vec3, vec4 } from 'three/tsl';
import {
  Mesh,
  BoxGeometry,
  NodeMaterial,
  OrthographicCamera,
  QuadMesh,
  Scene,
  WebGPURenderer,
  Group,
  DataTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
} from 'three/webgpu';
import {
  CloudToneUniforms,
  cloudToneNode,
  cloudToneRgb,
  cloudGreenness,
  cloudNeutralGradeTint,
  cloudToneMeanNode,
  cloudToneGain,
} from '../src/cloudTone';
import {
  lush2Defaults,
  lush2Mapping,
  lush2FromMapping,
  LUSH2_DEFAULTS,
  LUSH2_SCHEMA,
  applyLush2,
  LUSH2_CONTACT,
  LUSH2_MEAN,
} from '../src/ra2/lush2Tone';
import { AtmosphereNodes } from '../src/atmosphere';
import {
  ALIVE_CLOUD_SHAPES,
  ALIVE_DEFAULTS,
  resolveAlive,
  cloudCoverage,
  aliveDrift,
  cloudCoverThreshold,
} from '../src/livingWorld';
import { VoxelSoftContactMesh, setShadowGround } from '../src/ra2/voxelShadow';
import { objectGroundAnchor, setObjectGroundAnchor } from '../src/objectGroundAnchor';
import { HdRenderer, LOOK_PRESETS, type HdSettings } from '../src/HdRenderer';
import { objectCloudTone } from '../src/objectCloudTone';
import { vi } from 'vitest';
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
const snowGain = (y: number, p: ReturnType<typeof lush2Mapping>['tone']) => {
  const lift = p.snowLift;
  const lifted = y * lift,
    excess = Math.max(0, lifted - p.neutralGradeLuma);
  return p.bareDepth * lift * (1 - excess ** 2 / ((1 - p.neutralGradeLuma + excess) * Math.max(lifted, 1e-6)));
};
const recommended = (theater = 'temperate') => lush2Mapping(lush2Defaults(theater), theater).tone;

describe('lush2 colour reference', () => {
  it('keeps lit grass bright with the new OKLab hue and adds a green-cyan shade shift', () => {
    const rgb = decode([157, 161, 68]),
      p = recommended();
    const lit = encode(cloudToneRgb(rgb, 0, p)),
      shade = encode(cloudToneRgb(rgb, 1, p));
    expect(hue(lit)).toBeGreaterThanOrEqual(70);
    expect(hue(lit)).toBeLessThanOrEqual(80);
    expect(Math.max(...lit) / 255).toBeGreaterThanOrEqual(0.56);
    expect(Math.max(...lit) / 255).toBeLessThanOrEqual(0.66);
    expect(saturation(lit)).toBeGreaterThanOrEqual(0.6);
    expect(saturation(lit)).toBeLessThanOrEqual(0.85);
    expect(hue(shade)).toBeGreaterThan(hue(lit));
    expect(saturation(shade)).toBeGreaterThanOrEqual(0.5);
    // Physical attenuation uses this same coverage, outside the albedo grade.
    expect(luma(cloudToneRgb(rgb, 1, p)) * (1 - lush2Defaults('temperate').shadeDepth)).toBeLessThan(
      luma(cloudToneRgb(rgb, 0, p)),
    );
  });
  it('uses only the shared mean for grade and bounds luminance contrast by the requested detail exponent', () => {
    const grass = decode([150, 160, 60]),
      stripe = decode([190, 190, 110]);
    const mean = grass.map((v, i) => (v + stripe[i]!) / 2) as [number, number, number];
    for (const detail of [0.4, 0.75, 1, 1.2])
      for (const shade of [0, 0.5, 1]) {
        const p = { ...recommended(), detail, gold: 0 };
        const a = cloudToneRgb(grass, shade, p, 0, mean),
          b = cloudToneRgb(stripe, shade, p, 0, mean);
        const ratio = luma(b) / luma(a),
          original = (luma(stripe) / luma(grass)) ** detail;
        expect(ratio).toBeLessThanOrEqual(original);
        expect(ratio).toBeGreaterThanOrEqual(original / 1.05);
        expect(Math.abs(saturation(encode(a)) - saturation(encode(b)))).toBeLessThanOrEqual(
          Math.abs(saturation(encode(grass)) - saturation(encode(stripe))),
        );
        // At detail=1 every channel multiplier is shared, preserving the original source ratios.
        if (detail === 1)
          expect(a[0] / a[1] / (grass[0] / grass[1])).toBeCloseTo(b[0] / b[1] / (stripe[0] / stripe[1]), 5);
      }
  });
  it('does not amplify diagonal-stripe display-luma RMS and supports optional detail reduction', () => {
    const grass = decode([150, 160, 60]),
      stripe = decode([190, 190, 110]);
    const source = Array.from({ length: 64 * 64 }, (_, i) =>
      ((i % 64) + Math.floor(i / 64) * 2) % 8 < 2 ? stripe : grass,
    );
    // Exact box mip pyramid, bilinear + trilinear sampling, matching the two rotated GPU taps.
    const levels = [source];
    for (let size = 64; size > 1; size /= 2)
      levels.push(
        Array.from({ length: (size / 2) ** 2 }, (_, i) => {
          const x = (i % (size / 2)) * 2,
            z = Math.floor(i / (size / 2)) * 2;
          return [0, 1, 2].map(
            (c) =>
              [0, 1].reduce(
                (a, dz) => a + [0, 1].reduce((b, dx) => b + levels.at(-1)![(z + dz) * size + x + dx]![c]!, 0),
                0,
              ) / 4,
          ) as [number, number, number];
        }),
      );
    const bilinear = (level: number, x: number, z: number) => {
      const size = 64 / 2 ** level,
        px = x / 2 ** level - 0.5,
        pz = z / 2 ** level - 0.5,
        ix = Math.floor(px),
        iz = Math.floor(pz);
      return [0, 1, 2].map((c) =>
        [0, 1].reduce(
          (a, dz) =>
            a +
            [0, 1].reduce(
              (b, dx) =>
                b +
                levels[level]![((iz + dz + size) % size) * size + ((ix + dx + size) % size)]![c]! *
                  (dx ? px - ix : 1 - px + ix) *
                  (dz ? pz - iz : 1 - pz + iz),
              0,
            ),
          0,
        ),
      );
    };
    const lod = Math.log2(20),
      low = Math.floor(lod),
      blend = lod - low;
    const means = source.map(
      (_, i) =>
        [0, 1, 2].map((c) =>
          [-1, 1].reduce((sum, sign) => {
            const x = (i % 64) + 0.5 + sign * 20 * 0.46,
              z = Math.floor(i / 64) + 0.5 + sign * 20 * 0.19;
            return sum + (bilinear(low, x, z)[c]! * (1 - blend) + bilinear(low + 1, x, z)[c]! * blend) / 2;
          }, 0),
        ) as [number, number, number],
    );
    const rms = (pixels: readonly (readonly number[])[]) => {
      const ys = pixels.map((pixel) => luma(encode(pixel))),
        avg = ys.reduce((a, b) => a + b, 0) / ys.length;
      return Math.sqrt(ys.reduce((a, b) => a + (b - avg) ** 2, 0) / ys.length);
    };
    const input = rms(source);
    for (const detail of [1, 0.75]) {
      const out = source.map((pixel, i) => cloudToneRgb(pixel, 0, { ...recommended(), detail }, 0, means[i]!));
      expect(rms(out) / input).toBeLessThanOrEqual(detail === 1 ? 1 : 0.8);
      if (detail === 0.75) {
        const average = (pixels: readonly (readonly number[])[]) =>
          pixels.reduce((sum, rgb) => sum + luma(encode(rgb)), 0) / pixels.length;
        expect(rms(out) / average(out) / (input / average(source))).toBeLessThanOrEqual(0.8);
      }
    }
  });
  it('keeps the graded grass mean above the display-space channel floor, including chroma extremes', () => {
    // Check the graded mean before source multiplication; a zero source channel still stays zero.
    for (const r of [0, 80, 120, 150, 180])
      for (const g of [r, r + 10, r + 30, 160])
        for (const b of [0, 20, 60, 100]) {
          const rgb = decode([r, g, b]);
          if (cloudGreenness(rgb, recommended()) < 0.99) continue;
          for (const grassChroma of [0.8, 1.08, 1.6])
            for (const shade of [0, 0.5, 1]) {
              const gain = cloudToneGain(rgb, rgb, shade, { ...recommended(), grassChroma });
              const out = encode(gain.map((v, i) => v * Math.max(rgb[i]!, 1e-6)));
              const sourceOut = cloudToneRgb(rgb, shade, { ...recommended(), grassChroma });
              rgb.forEach((v, i) => {
                if (v === 0) expect(sourceOut[i]).toBe(0);
              });
              expect(Math.min(...out)).toBeGreaterThanOrEqual(0.15 * luma(out));
            }
        }
  });
  it('retains monotonic grass chroma without changing object colours', () => {
    const rgb = decode([150, 160, 60]),
      a = { ...recommended(), grassChroma: 0.8 },
      b = { ...recommended(), grassChroma: 1.6 };
    for (const shade of [0, 0.5, 1]) {
      expect(saturation(encode(cloudToneRgb(rgb, shade, b)))).toBeGreaterThan(
        saturation(encode(cloudToneRgb(rgb, shade, a))),
      );
      // OKLab lightness is fixed; linear luminance may change with hue/chroma.
      expect(luma(cloudToneRgb(rgb, shade, b))).toBeGreaterThan(0.7 * luma(rgb));
      expect(cloudToneRgb(rgb, shade, a, 0, rgb, true)).toEqual(cloudToneRgb(rgb, shade, b, 0, rgb, true));
    }
  });
  it('preserves dry warm sand/dirt hue, saturation and >=.95 luminance across theatres', () => {
    for (const theater of ['temperate', 'desert', 'urban'])
      for (const pixel of [
        [215, 185, 130],
        [190, 140, 90],
        [230, 190, 120],
        [190, 120, 80],
      ]) {
        const rgb = decode(pixel),
          p = recommended(theater);
        expect(hue(pixel)).toBeGreaterThanOrEqual(20);
        expect(hue(pixel)).toBeLessThanOrEqual(45);
        for (const shade of [0, 0.5, 1]) {
          const out = cloudToneRgb(rgb, shade, p),
            display = encode(out);
          expect(Math.abs(hue(display) - hue(pixel))).toBeLessThanOrEqual(2);
          expect(saturation(display) + 1e-12).toBeGreaterThanOrEqual(saturation(pixel));
          expect(luma(out)).toBeGreaterThanOrEqual(0.95 * luma(rgb));
          expect(cloudToneRgb(rgb, shade, { ...p, grassChroma: 1.6, gold: 1, grassSunTint: [0.1, 1, 0.1] })).toEqual(
            out,
          );
        }
        expect(cloudToneRgb(rgb, 1, p, 1)).not.toEqual(cloudToneRgb(rgb, 1, { ...p, moonTint: [1, 1, 1] }, 1));
      }
  });
  it('avoids a new halo on a moving-mean grass/ring/dirt profile and stays continuous at the wet boundary', () => {
    const anchors = [
      [150, 160, 60],
      [175, 170, 100],
      [190, 140, 90],
    ].map(decode);
    const profile = Array.from({ length: 65 }, (_, i) => {
      const t = i / 32,
        j = Math.min(1, Math.floor(t)),
        f = t - j;
      return anchors[j]!.map((v, c) => v + (anchors[j + 1]![c]! - v) * f) as [number, number, number];
    });
    const means = profile.map(
      (_, i) =>
        [0, 1, 2].map(
          (c) =>
            Array.from({ length: 17 }, (_, j) => profile[Math.min(64, Math.max(0, i + j - 8))]![c]!).reduce(
              (a, b) => a + b,
              0,
            ) / 17,
        ) as [number, number, number],
    );
    const input = profile.map(luma),
      output = profile.map((rgb, i) => luma(cloudToneRgb(rgb, 0, recommended(), 0, means[i]!)));
    const ringRatio = (ys: number[]) => ys[32]! / Math.max(ys[0]!, ys[64]!);
    expect(ringRatio(output) / ringRatio(input)).toBeLessThanOrEqual(1.1);
    // Every new local peak must also have existed in the original profile within the 10% tolerance.
    for (let i = 1; i < 64; i++)
      if (output[i]! > output[i - 1]! && output[i]! > output[i + 1]!)
        expect(output[i]! / Math.max(output[i - 1]!, output[i + 1]!)).toBeLessThanOrEqual(
          (1.1 * input[i]!) / Math.max(input[i - 1]!, input[i + 1]!),
        );
    const rgb = decode([150, 160, 60]),
      p = recommended();
    const dry = cloudToneRgb(rgb, 0, p),
      edge = cloudToneRgb(rgb, 0, p, 0, rgb, false, 1e-6);
    dry.forEach((v, i) => expect(edge[i]).toBeCloseTo(v, 6));
  });
  it('uses one warm wet multiplier and own-luma chroma for sand, surf and grass', () => {
    const contaminated = decode([120, 170, 160]);
    for (const theater of ['temperate', 'desert']) {
      const p = recommended(theater);
      for (const pixel of [
        [215, 185, 130],
        [170, 165, 140],
        [150, 170, 150],
        [150, 160, 60],
      ]) {
        const rgb = decode(pixel);
        for (const band of [0.3, 0.5, 0.75, 1]) {
          const share = p.wet * band;
          const tinted = rgb.map(
            (v, i) => v * (1 + ((p.wetTint[i]! * (1 - p.wetDepth)) / luma(p.wetTint) - 1) * share),
          );
          const y = luma(tinted);
          const expected = tinted.map((v) => y + (v - y) * (1 + (p.wetChroma - 1) * share));
          const wet = cloudToneRgb(rgb, 0, p, 0, contaminated, false, band, 0, contaminated);
          wet.forEach((v, i) => expect(v).toBeCloseTo(Math.max(0, expected[i]!), 12));
        }
        expect(cloudToneRgb(rgb, 0, p, 0, rgb, true, 1)).toEqual(cloudToneRgb(rgb, 0, p, 0, rgb, true, 0));
      }
    }
  });
  it('has no source hue classification, including adjacent grey-channel permutations', () => {
    const mean = decode([150, 160, 60]),
      neutral = cloudToneRgb(decode([140, 140, 140]), 0, recommended(), 0, mean);
    for (const rgb of [
      [140.00001, 140, 140],
      [140, 140.00001, 140],
      [140, 140, 140.00001],
    ])
      cloudToneRgb(decode(rgb), 0, recommended(), 0, mean).forEach((v, i) => expect(v).toBeCloseTo(neutral[i]!, 6));
    expect(cloudGreenness([0.3, 0.3, 0.3], recommended())).toBe(0);
  });
  it('gates luma-normalised warm gold by relative source highlights and clear-lit mean grass', () => {
    const p = { ...recommended(), gold: 1 },
      noGold = { ...p, gold: 0 },
      mean = [0.3, 0.48, 0.15] as const;
    const bright = [0.65, 0.85, 0.3] as const,
      dark = [0.2, 0.4, 0.1] as const;
    expect(cloudToneRgb(dark, 0, p, 0, mean)).toEqual(cloudToneRgb(dark, 0, noGold, 0, mean));
    expect(cloudToneRgb(bright, 1, p, 0, mean)).toEqual(cloudToneRgb(bright, 1, noGold, 0, mean));
    const a = cloudToneRgb(bright, 0, p, 0, mean),
      b = cloudToneRgb(bright, 0, noGold, 0, mean);
    expect(luma(a) / luma(b)).toBeGreaterThan(1);
    expect(luma(a) / luma(b)).toBeLessThanOrEqual(1.04 + 1e-12);
    expect(a[0] / a[2]).toBeGreaterThan(b[0] / b[2]);
  });
  it('grades dim lavender snow and its darker strokes with the same mean-only gain', () => {
    const p = recommended('snow');
    expect(p.snowLift).toBe(1.8);
    expect(p.snowWhite).toBe(1);
    const mean = [0.18, 0.2, 0.27] as const;
    const pixels = [mean, [0.08, 0.09, 0.15], [0.28, 0.3, 0.38]] as const;
    for (const shade of [0, 0.4, 1]) {
      const gain = cloudToneGain(mean, mean, shade, p);
      for (const rgb of pixels) {
        const out = cloudToneRgb(rgb, shade, p, 0, mean);
        out.forEach((v, i) => expect(v / rgb[i]!).toBeCloseTo(gain[i]!, 12));
        expect(out[2] / out[0]).toBeLessThan(rgb[2] / rgb[0]);
      }
    }
    const lit = cloudToneRgb(mean, 0, p),
      shadow = cloudToneRgb(mean, 1, p);
    expect(lit[2]).toBeGreaterThan(lit[0]); // Retain a hint of mission colour.
    expect(shadow[2] / shadow[0]).toBeGreaterThan(lit[2] / lit[0]);
    expect(cloudToneRgb(mean, 0, { ...p, snowBlue: 0 })).toEqual(lit);
    for (const rgb of [decode([90, 110, 45]), decode([225, 190, 100])])
      expect(cloudToneRgb(rgb, 0, p)).toEqual([...rgb]);
    // A dark neutral patch relative to its surroundings is not snow, even though its saturation is zero.
    expect(cloudToneRgb([0.01, 0.01, 0.01], 0, p, 0, [0.01, 0.01, 0.01], false, 0, 0, mean)).toEqual([
      0.01, 0.01, 0.01,
    ]);
    const ramp = [0.7, 0.8, 0.9, 1].map((y) => cloudToneRgb([y, y, y], 0, { ...p, snowLift: 1.3 }));
    ramp.forEach((rgb) => expect(Math.max(...rgb)).toBeLessThan(1));
    for (let i = 1; i < ramp.length; i++) expect(luma(ramp[i]!)).toBeGreaterThan(luma(ramp[i - 1]!));
    // Snow objects follow the white ground: never bluer than without the snow step, and cool bright snow neutralised.
    for (const rgb of pixels) {
      const on = cloudToneRgb(rgb, 0.5, p, 0, mean, true),
        off = cloudToneRgb(rgb, 0.5, { ...p, snow: 0 }, 0, mean, true);
      expect(on[2] / on[0]).toBeLessThanOrEqual(off[2] / off[0] + 1e-9);
      expect(luma(on)).toBeGreaterThan(luma(off));
    }
  });
  it('keeps a mild cool hint through the real cinematic grade, including its brightest whites', () => {
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
        expect(saturation(out)).toBeLessThan(0.08);
        expect(Math.max(...out)).toBeGreaterThan(210);
        // Chromatic correction is gain-luma-one; source-weighted output luma may change on cool means.
        if (luma(rgb) <= p.neutralGradeLuma)
          expect(luma(cloudToneGain(rgb, rgb, 0, p))).toBeCloseTo(snowGain(luma(rgb), p), 12);
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
  it('keeps black exactly black and stays finite/nonnegative at slider extremes', () => {
    for (const strength of [0.5, 1, 1.5])
      for (const grassLight of [0.7, 1.15])
        for (const grassChroma of [0.8, 1.6]) {
          const p = lush2Mapping(
            { ...lush2Defaults('temperate'), strength, grassLight, grassChroma },
            'temperate',
          ).tone;
          expect(cloudToneRgb([0, 0, 0], 0, p, 0, [0.5, 0.8, 0.1])).toEqual([0, 0, 0]);
          for (const rgb of [
            [0.001, 0.002, 0.001],
            [0.1, 0.3, 0.05],
            [1, 1, 1],
          ] as const)
            cloudToneRgb(rgb, 1, p, 0, [0.5, 0.8, 0.1]).forEach((v) => {
              expect(Number.isFinite(v)).toBe(true);
              expect(v).toBeGreaterThanOrEqual(0);
            });
        }
  });
  it('gives objects their independent floor/share, hue-preserving shoulder and subtle low bounce', () => {
    const p = recommended(),
      rgb = decode([150, 160, 60]);
    const multiplier = Math.max(p.objectFloor, 1 + (p.grassLight - 1) * p.objectShare);
    expect(luma(cloudToneRgb(rgb, 0, p, 0, rgb, true)) / luma(rgb)).toBeCloseTo(multiplier, 12);
    expect(multiplier).toBe(0.95);
    const white = [0.8, 0.75, 0.7] as const;
    const high = cloudToneRgb(white, 0, { ...p, bounce: 0 }, 0, white, true),
      plain = cloudToneRgb(white, 0, { ...p, bounce: 0, objectShoulder: 0 }, 0, white, true);
    expect(luma(high)).toBeCloseTo((luma(white) - p.objectShoulder * (luma(white) - 0.55)) * multiplier, 12);
    expect(high[0] / high[1]).toBeCloseTo(plain[0] / plain[1], 12);
    const lower = cloudToneRgb(white, 0, p, 0, white, true, 0, 0),
      upper = cloudToneRgb(white, 0, p, 0, white, true, 0, p.bounceHeight);
    expect(lower[1] / lower[2]).toBeGreaterThan(upper[1] / upper[2]);
    expect(Math.max(...lower.map((v, i) => Math.abs(v / upper[i]! - 1)))).toBeLessThan(0.02);
    const night = { ...p, objectShoulder: 0 },
      ground = cloudToneRgb(rgb, 0, night, 1),
      object = cloudToneRgb(rgb, 0, night, 1, rgb, true);
    expect(luma(object)).toBeGreaterThan(p.objectFloor * luma(rgb));
    expect(ground[2] / ground[0]).toBeGreaterThan(rgb[2] / rgb[0]);
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
  it('keeps the original gold gate by default and extends its reach monotonically', () => {
    for (const theater of Object.keys(LUSH2_DEFAULTS)) {
      const reach = lush2Defaults(theater).goldReach;
      expect(reach).toBe(0.35);
      expect(recommended(theater).highlightGate).toEqual([0, 0, 1.12 - 0.14 * reach, 1.32 - 0.12 * reach]);
      for (const goldReach of [0, 0.25, 0.5, 1]) {
        const mapped = lush2Mapping({ ...lush2Defaults(theater), goldReach }, theater);
        expect(mapped.tone.highlightGate).toEqual([0, 0, 1.12 - 0.14 * goldReach, 1.32 - 0.12 * goldReach]);
        expect(lush2FromMapping(mapped.tone, mapped.alive).goldReach).toBeCloseTo(goldReach, 12);
      }
    }
    const mean = decode([150, 160, 60]);
    const narrow = recommended();
    const broad = lush2Mapping({ ...lush2Defaults('temperate'), goldReach: 1 }, 'temperate').tone;
    const a = cloudToneRgb(mean, 0, narrow),
      b = cloudToneRgb(mean, 0, broad);
    expect(b[0] / b[2]).toBeGreaterThan(a[0] / a[2]);
  });
  it('uses the requested ranges and theatre cloud depths; softer edges widen the same field', () => {
    const field = (key: string) => LUSH2_SCHEMA.find((f) => f.key === key)!;
    expect([field('snowLift').min, field('snowLift').max, field('snowLift').step]).toEqual([1, 2, 0.01]);
    expect(field('goldReach')).toEqual({ key: 'goldReach', label: '金色范围 [goldReach]', min: 0, max: 1, step: 0.01 });
    expect([field('drift').min, field('drift').max]).toEqual([0, 1]);
    expect([field('grassChroma').min, field('grassChroma').max]).toEqual([0.8, 1.6]);
    expect([field('softness').min, field('softness').max]).toEqual([0.2, 1]);
    expect([field('patchScale').min, field('patchScale').max]).toEqual([6, 24]);
    expect([field('shadeDepth').min, field('shadeDepth').max]).toEqual([0, 0.5]);
    for (const [theater, depth] of [
      ['temperate', 0.3],
      ['desert', 0.24],
      ['snow', 0.12],
      ['urban', 0.26],
    ] as const) {
      expect(lush2Defaults(theater).shadeDepth).toBe(depth);
      expect(lush2Defaults(theater).patchScale).toBe(12);
      expect(lush2Defaults(theater).contact).toBe(1);
      expect(lush2Defaults(theater).drift).toBe(0.6);
      const mapped = lush2Mapping(lush2Defaults(theater), theater);
      const [dx, dz] = aliveDrift(48, 0.6, mapped.alive.speed);
      expect(Math.hypot(dx, dz)).toBeCloseTo(28.8, 12);
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
      const grass = a.cloudTone.goldTint.value;
      const mapped = lush2Mapping({ ...lush2Defaults('snow'), patchScale: 9, shadeDepth: 0, drift: 0.1 }, 'snow');
      a.cloudTone.set(mapped.tone);
      a.alive.set(mapped.alive);
      a.alive.update(20, 0.6, 0, 0, false);
      b.copyValuesFrom(a);
      expect(a.alive.cloudLattice.value.x).toBe(1 / 9);
      expect(a.alive.cloudForm.value.x).toBe(1);
      expect(a.alive.cloudStrength.value).toBe(0);
      expect(a.alive.cloudAmount.value).toBe(1);
      expect(a.cloudTone.goldTint.value).toBe(grass);
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
  it('maps cover monotonically to the real shared field, with the requested theatre fractions', () => {
    const points = Array.from({ length: 128 * 128 }, (_, i) => [
      (i % 128) * 1.71 - 91,
      Math.floor(i / 128) * 1.93 - 100,
    ]);
    let previous = 0;
    for (const cover of [0, 0.1, 0.28, 0.35, 0.4, 0.45, 0.6, 0.8, 1]) {
      const { alive } = lush2Mapping({ ...lush2Defaults('temperate'), cover }, 'temperate');
      const shaded =
        points.filter(([x, z]) => cloudCoverage(x!, z!, 17, 0.6, alive.speed, alive.shape) >= 0.5).length /
        points.length;
      expect(shaded).toBeGreaterThanOrEqual(previous);
      expect(Math.abs(shaded - cover)).toBeLessThan(0.045);
      previous = shaded;
    }
    expect(cloudCoverThreshold(0)).toBeGreaterThan(cloudCoverThreshold(1));
    for (const [theater, cover] of [
      ['temperate', 0.5],
      ['snow', 0.28],
      ['desert', 0.35],
      ['urban', 0.4],
    ] as const)
      expect(lush2Defaults(theater).cover).toBe(cover);
    expect(LUSH2_SCHEMA.find((f) => f.key === 'detail')).toMatchObject({ min: 0.4, max: 1.2 });
    expect(LUSH2_SCHEMA.find((f) => f.key === 'cover')).toMatchObject({ min: 0, max: 1 });
  });
  it('shares one cache texture and world bounds between terrain and object feet', () => {
    const nodes = new AtmosphereNodes();
    try {
      const sample = vi.spyOn(nodes.alive, 'coverageAt');
      const object = objectCloudTone(nodes);
      expect(sample).toHaveBeenCalledTimes(1);
      const terrainShader = fragmentShader(vec4(vec3(nodes.alive.coverage), 1));
      const objectShader = fragmentShader(vec4(vec3(object.coverage), 1));
      for (const code of [terrainShader, objectShader]) {
        expect(code.match(/textureSample\(/g)).toHaveLength(1);
        expect(code).not.toContain('aliveCloudShadow');
        expect(code).toContain('aliveCoverageEnabled');
      }
    } finally {
      nodes.dispose();
    }
  });
  it('sets a strong soft contact core with 1.6x hull diameter and keeps its slider', () => {
    const p = recommended();
    const alpha = (r: number) =>
      Math.min(
        LUSH2_CONTACT.opacityCap,
        Math.max(
          0,
          (Math.exp(-r * r * LUSH2_CONTACT.falloff) - Math.exp(-LUSH2_CONTACT.falloff)) /
            (1 - Math.exp(-LUSH2_CONTACT.falloff)),
        ) *
          LUSH2_CONTACT.opacity *
          p.contact *
          p.strength,
      );
    expect(alpha(0)).toBe(0.9);
    expect(alpha(0.5)).toBeGreaterThan(0.35);
    expect(alpha(1)).toBe(0);
    expect(LUSH2_CONTACT.radiusScale).toBe(1.6);
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
      expect(contact.matrixWorld.elements[0]).toBeCloseTo((0.8 * 1.6) / 2);
      expect(contact.matrixWorld.elements[10]).toBeCloseTo((0.5 * 1.6) / 2);
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
function fragmentShader(fragment: ReturnType<typeof vec4>): string {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const material = new NodeMaterial();
  material.fragmentNode = fragment;
  const builder = renderer.backend.createNodeBuilder(new QuadMesh(material), renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  try {
    builder.build();
    return builder.fragmentShader;
  } finally {
    material.dispose();
  }
}
it('builds uniform-gated ground, snow and object shaders without texture replacement', () => {
  const uniforms = new CloudToneUniforms();
  uniforms.set(recommended());
  for (const object of [false, true]) {
    const shader = fragmentShader(
      vec4(cloudToneNode(vec3(0.2, 0.4, 0.1), vec3(0.2, 0.4, 0.1), float(1), uniforms, float(0), object), 1),
    );
    expect(shader).toContain('@fragment');
    expect(shader).not.toContain('textureSample');
    if (!object) expect(shader).toContain('else');
  }
});
it('builds two local and two context explicit-LOD samples, independent of screen derivatives', () => {
  const map = new DataTexture(new Uint8Array(64 * 64 * 4), 64, 64);
  map.magFilter = LinearFilter;
  map.minFilter = LinearMipmapLinearFilter;
  map.generateMipmaps = true;
  try {
    const local = cloudToneMeanNode(map, vec2(0.5), vec2(1 / 64), 20, vec4(0, 0, 1, 1));
    const context = cloudToneMeanNode(map, vec2(0.5), vec2(1 / 64), 96, vec4(0, 0, 1, 1));
    const shader = fragmentShader(vec4(local.add(context), 1));
    expect(shader.match(/textureSampleLevel\(/g)).toHaveLength(4);
    expect(shader).toContain(String(Math.log2(96)));
    expect(shader).toContain(String(Math.log2(20)));
    expect(shader).not.toMatch(/textureSample\(/);
  } finally {
    map.dispose();
  }
});

it('keeps gain within 12% over a six-mean-radius dirt patch in lawn and preserves a sandy context', () => {
  const grass = decode([150, 160, 60]),
    dirt = decode([190, 140, 90]),
    sand = decode([215, 185, 130]);
  const radius = LUSH2_MEAN.blockWidth / 2;
  const profile = (background: typeof grass) => (x: number) => {
    const t = Math.max(0, Math.min(1, (3 * radius - Math.abs(x)) / 3));
    return background.map((v, c) => v + (dirt[c]! - v) * t) as typeof grass;
  };
  // Moving linear-light box means, with the shader's antithetic rotated offset projected onto X.
  const mean = (source: ReturnType<typeof profile>, x: number, width: number) =>
    [0, 1, 2].map((c) =>
      [-1, 1].reduce(
        (total, sign) =>
          total +
          Array.from(
            { length: 128 },
            (_, i) => source(x + sign * 0.46 * width + ((i + 0.5) / 128 - 0.5) * width)[c]!,
          ).reduce((a, b) => a + b, 0) /
            256,
        0,
      ),
    ) as typeof grass;
  for (const background of [grass, sand]) {
    const source = profile(background);
    const gains = Array.from({ length: 241 }, (_, i) => {
      const x = i - 120,
        rgb = source(x);
      return (
        luma(
          cloudToneRgb(
            rgb,
            0,
            recommended(),
            0,
            mean(source, x, LUSH2_MEAN.blockWidth),
            false,
            0,
            0,
            mean(source, x, LUSH2_MEAN.contextWidth),
          ),
        ) / luma(rgb)
      );
    });
    if (background === grass) expect(Math.max(...gains) / Math.min(...gains) - 1).toBeLessThan(0.12);
    else expect(Math.min(...gains)).toBeGreaterThanOrEqual(0.95);
  }
});
it('gives neighbouring sand and surf identical pre-chroma gains, independent of mean and source hue', () => {
  const pixels = [
    [215, 185, 130],
    [150, 170, 150],
  ].map(decode);
  for (const slider of [0, 0.4, 0.8, 1])
    for (const wet of [0.3, 0.5, 1]) {
      const { tone: p } = lush2Mapping({ ...lush2Defaults('temperate'), wet: slider }, 'temperate');
      const gains = pixels.map((rgb) => {
        const out = cloudToneRgb(rgb, 0, { ...p, wetChroma: 1 }, 0, decode([120, 170, 160]), false, wet);
        return out.map((v, i) => v / rgb[i]!);
      });
      gains[0]!.forEach((v, i) => expect(gains[1]![i]).toBeCloseTo(v, 12));
    }
  const p = recommended();
  for (const mean of [decode([120, 170, 160]), decode([150, 160, 60]), decode([215, 185, 130])]) {
    const sand = cloudToneRgb(pixels[0]!, 0, p, 0, mean, false, 1, 0, mean);
    expect(hue(encode(sand))).toBeGreaterThanOrEqual(25);
    expect(hue(encode(sand))).toBeLessThanOrEqual(40);
    // No hue/saturation selector: the full-strip response is a linear map of source RGB.
    const surf = pixels[1]!;
    const surfOut = cloudToneRgb(surf, 0, p, 0, mean, false, 1, 0, mean);
    for (let step = 0; step <= 64; step++) {
      const t = step / 64;
      const pixel = pixels[0]!.map((v, i) => v + (surf[i]! - v) * t) as [number, number, number];
      cloudToneRgb(pixel, 0, p, 0, mean, false, 1, 0, mean).forEach((v, i) =>
        expect(v).toBeCloseTo(sand[i]! + (surfOut[i]! - sand[i]!) * t, 12),
      );
    }
  }
});
it('builds cached receiver coverage with one bilinear fetch and no analytic noise', () => {
  const nodes = new AtmosphereNodes();
  try {
    const receiver = fragmentShader(vec4(vec3(nodes.alive.coverage), 1));
    expect(receiver).not.toContain('aliveCloudShadow');
    expect(receiver.match(/textureSample\(/g)).toHaveLength(1);
  } finally {
    nodes.dispose();
  }
});

it('gives context dirt only the bare-light OKLab lightness change, preserving hue', () => {
  const dirt = decode([190, 140, 90]),
    grass = decode([150, 160, 60]),
    p = recommended();
  const out = cloudToneRgb(dirt, 0, p, 0, dirt, false, 0, 0, grass),
    display = encode(out);
  expect(Math.abs(luma(out) / luma(dirt) / p.bareLight ** 3 - 1)).toBeLessThan(0.015);
  expect(hue(display)).toBeGreaterThan(20);
  expect(hue(display)).toBeLessThan(40);
  expect(saturation(display)).toBeCloseTo(saturation(encode(dirt)), 1);
});

it('shares one linear conversion between object art and its physical lighting finish', () => {
  const nodes = new AtmosphereNodes();
  try {
    const cloud = objectCloudTone(nodes);
    const code = fragmentShader(
      vec4(cloud.shade(vec3(0.2)).add(shadeCloudColour(vec3(0.4), linearCloudFactor(cloud.shadow))), 1),
    );
    expect(code.match(/pow\( [^\n]+2\.2/g)).toHaveLength(1);
    expect(code).not.toContain('sRGBTransferEOTF');
    expect(code).not.toContain('sRGBTransferOETF');
  } finally {
    nodes.dispose();
  }
});
