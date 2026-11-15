import { describe, expect, it } from 'vitest';
import {
  colourFilterWeight,
  quotientGradient,
  coverageAt as evaluateCoverage,
  integratedAlpha,
  softRange,
  highlightRoll,
  SpriteIntegration,
  nonGreyFraction,
} from '../src/spriteIntegration';
import { RECOMMENDED } from '../src/ra2/spriteIntegrationTuning';

const coverageAt = (x: number, y: number, occupied: (x: number, y: number) => number, sigma = RECOMMENDED.edgeSigma) =>
  evaluateCoverage(x, y, occupied, sigma);
const alphaAt = (coverage: Parameters<typeof integratedAlpha>[0]) =>
  integratedAlpha(coverage, 2.5, RECOMMENDED.edgeSoft, RECOMMENDED.thinKeep);
const paintRatiosAt = (
  x: number,
  y: number,
  opaque: (x: number, y: number) => number,
  paint: (x: number, y: number) => number,
  sigma = RECOMMENDED.edgeSigma,
) => {
  const a = coverageAt(x, y, opaque, sigma);
  const h = coverageAt(x, y, (x, y) => opaque(x, y) * paint(x, y), sigma);
  return {
    wide: h.wide / Math.max(a.wide, 1e-8),
    narrow: h.narrow / Math.max(a.narrow, 1e-8),
    gradWide: quotientGradient(h.wide, h.gradWide, a.wide, a.gradWide),
    gradNarrow: quotientGradient(h.narrow, h.gradNarrow, a.narrow, a.gradNarrow),
  };
};

describe('sprite reconstruction CPU twins', () => {
  it.each([2, 4])('bounds the wide 0.5 iso-line ripple on a %i:1 stair', (slope) => {
    const residuals: number[] = [];
    const mask = (x: number, y: number) => (y <= Math.floor(x / slope) ? 1 : 0);
    for (let i = 0; i < 400; i++) {
      const x = 8 + (i * slope) / 400;
      let lo = x / slope - 2,
        hi = x / slope + 3;
      for (let n = 0; n < 40; n++) {
        const mid = (lo + hi) / 2;
        if (coverageAt(x, mid, mask).wide > 0.5) lo = mid;
        else hi = mid;
      }
      residuals.push(((lo + hi) / 2 - x / slope) / Math.hypot(1, 1 / slope));
    }
    const ripple = Math.max(...residuals) - Math.min(...residuals);
    expect(ripple).toBeLessThan(slope === 2 ? 0.1 : 0.3);
  });
  it('keeps a one-texel line and a 2x2 block opaque at their centres', () => {
    const line = coverageAt(0.5, 0.5, (x) => (x === 0 ? 1 : 0));
    expect(line.wide).toBeLessThan(0.5);
    expect(line.narrow).toBeCloseTo(0.75);
    expect(alphaAt(line)).toBe(1);
    expect(alphaAt(coverageAt(1, 1, (x, y) => (x >= 0 && x < 2 && y >= 0 && y < 2 ? 1 : 0)))).toBe(1);
  });
  it('keeps a one-texel hole in a solid block transparent at its centre', () => {
    const block = (x: number, y: number) => (Math.abs(x) <= 3 && Math.abs(y) <= 3 ? 1 : 0);
    const hole = coverageAt(0.5, 0.5, (x, y) => (x === 0 && y === 0 ? 0 : block(x, y)));
    expect(hole.wide).toBeGreaterThan(0.5);
    expect(hole.narrow).toBeCloseTo(0.4375);
    expect(alphaAt(hole)).toBe(0);
    const gap = coverageAt(0.5, 0.5, (x, y) => (y === 0 ? 0 : block(x, y)));
    expect(gap.wide).toBeGreaterThan(0.5);
    expect(alphaAt(gap)).toBe(0);
  });
  it('keeps a one-texel non-paint slat between paint rows, and keeps a thin paint row', () => {
    const opaque = (x: number, y: number) => (Math.abs(x) <= 3 && Math.abs(y) <= 3 ? 1 : 0);
    const slat = paintRatiosAt(0.5, 0.5, opaque, (_x, y) => (y === 0 ? 0 : 1));
    expect(slat.wide).toBeGreaterThan(0.5);
    expect(alphaAt(slat)).toBe(0);
    const row = paintRatiosAt(0.5, 0.5, opaque, (_x, y) => (y === 0 ? 1 : 0));
    expect(row.wide).toBeLessThan(0.5);
    expect(alphaAt(row)).toBe(1);
  });
  it('can skip outer house sums for homogeneous opaque inner taps without changing paint coverage', () => {
    const inner = (x: number, y: number) => Math.abs(x) <= 1 && Math.abs(y) <= 1;
    for (const sigma of [0.5, 0.95, 1.4])
      for (const paint of [0, 1])
        for (const x of [0.1, 0.5, 0.9]) {
          const opaque = (x: number, y: number) => (inner(x, y) || (x + y) % 2 === 0 ? 1 : 0);
          const full = paintRatiosAt(x, 0.3, opaque, (x, y) => (inner(x, y) ? paint : 1 - paint), sigma);
          const skipped = paintRatiosAt(x, 0.3, opaque, (x, y) => (inner(x, y) ? paint : 0), sigma);
          for (const px of [0.75, 2.5, 6]) {
            expect(integratedAlpha(full, px, 1, 0.56)).toBe(paint);
            expect(integratedAlpha(skipped, px, 1, 0.56)).toBe(paint);
          }
        }
  });
  it('uses the weighted share of individually non-grey paint taps, ignoring sRGB code noise', () => {
    const linear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const grey = [0.4, 0.4, 0.4];
    const nearGrey = [linear(0.5), linear(0.5 + 0.9 / 255), linear(0.5)];
    expect(
      nonGreyFraction([
        { rgb: grey, weight: 1 },
        { rgb: nearGrey, weight: 2 },
      ]),
    ).toBe(0);
    expect(
      nonGreyFraction([
        { rgb: grey, weight: 7 },
        { rgb: [0.9, 0.1, 0.1], weight: 2 },
        { rgb: [0.1, 0.9, 0.9], weight: 1 },
      ]),
    ).toBeCloseTo(0.3);
    // Two coloured taps can average to grey; they must still invalidate a baked-colour page.
    expect(
      nonGreyFraction([
        { rgb: [0.9, 0.1, 0.1], weight: 1 },
        { rgb: [0.1, 0.9, 0.9], weight: 1 },
      ]),
    ).toBe(1);
    // A tiny coloured contribution must not reject otherwise genuinely encoded grey shading.
    expect(
      nonGreyFraction([
        { rgb: grey, weight: 999 },
        { rgb: [1, 0, 0], weight: 1 },
      ]),
    ).toBe(0.001);
    expect(nonGreyFraction([{ rgb: [1, 0, 0], weight: 0 }])).toBe(0);
    expect(nonGreyFraction([])).toBe(0);
    for (const base of [0.001, 0.05, 0.5, 0.9]) {
      expect(nonGreyFraction([{ rgb: [linear(base), linear(base + 1.1 / 255), linear(base)], weight: 1 }])).toBe(1);
    }
  });
  it('matches analytic wide and narrow gradients to finite differences, including the Gaussian normaliser', () => {
    const mask = (x: number, y: number) => (x + 2 * y > 0 && x !== 2 ? 1 : 0);
    const e = 1e-5;
    for (const sigma of [0.5, 0.95, 1.4])
      for (const [x, y] of [
        [0.13, 0.81],
        [1.7, -0.2],
        [-2.36, 2.23],
      ]) {
        const c = coverageAt(x!, y!, mask, sigma);
        for (const [axis, dx, dy] of [
          [0, e, 0],
          [1, 0, e],
        ]) {
          const a = coverageAt(x! + dx!, y! + dy!, mask, sigma),
            b = coverageAt(x! - dx!, y! - dy!, mask, sigma);
          expect(c.gradWide[axis!]).toBeCloseTo((a.wide - b.wide) / (2 * e), 7);
          expect(c.gradNarrow[axis!]).toBeCloseTo((a.narrow - b.narrow) / (2 * e), 7);
        }
      }
  });
  it('soft range is identity inside the range and continuous at both boundaries; roll preserves its knee', () => {
    for (const x of [0.2, 0.3, 0.8]) expect(softRange(x, 0.2, 0.8)).toBe(x);
    for (const x of [0.2, 0.8]) {
      expect(Math.abs(softRange(x - 1e-8, 0.2, 0.8) - softRange(x + 1e-8, 0.2, 0.8))).toBeLessThan(3e-8);
    }
    expect(softRange(2, 0.2, 0.8)).toBeGreaterThan(0.8);
    expect(softRange(2, 0.2, 0.8)).toBeLessThan(2);
    expect(highlightRoll(0.3, 0.6, 1)).toBe(0.3);
    expect(highlightRoll(1, 0.6, 0)).toBe(1);
    expect(highlightRoll(1, 0.6, 0.6)).toBeLessThan(1);
  });
  it('HD filter weights stay positive and normalised, and opacity-share gradients use the quotient rule', () => {
    for (const t of [-0.5, -0.17, 0, 0.23, 0.5])
      for (const soft of [0, 0.5, 1]) {
        const weights = [-1, 0, 1].map((o) => colourFilterWeight(t, o, soft));
        expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
        expect(Math.min(...weights)).toBeGreaterThanOrEqual(0);
      }
    expect(quotientGradient(2, [1, 3], 4, [2, 1])).toEqual([0, 0.625]);
  });

  it('edits uniforms in place and restores A/B without graph changes', () => {
    const group = new SpriteIntegration();
    group.set(RECOMMENDED);
    const node = group.uniforms.edgeSigma;
    group.set({ edgeSigma: 1.2 });
    group.ab(false);
    expect(group.get().on).toBe(0);
    group.ab(true);
    expect(group.get()).toEqual({ ...RECOMMENDED, edgeSigma: 1.2 });
    expect(group.uniforms.edgeSigma).toBe(node);
  });
});
