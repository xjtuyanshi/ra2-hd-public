import { describe, expect, it } from 'vitest';
import {
  GROUND_PULL,
  SHADOW_EXCLUSIVE_PULL,
  SHADOW_PULL_LEVELS,
  overPassSample,
  shadowGroundPull,
  shadowPullRank,
} from '../src/ra2/sprites';

describe('enhanced shadow pulls', () => {
  it('spans the ground pull range in equal steps, clamps coverage, and stays behind infantry cores and reflections', () => {
    expect(SHADOW_PULL_LEVELS).toBe(32);
    expect(shadowGroundPull(0)).toBe(GROUND_PULL.shadow[0]);
    expect(shadowGroundPull(1)).toBeCloseTo(GROUND_PULL.shadow[1], 12);
    expect(shadowPullRank(-1)).toBe(0);
    expect(shadowPullRank(2)).toBe(SHADOW_PULL_LEVELS);
    const ranks = new Set<number>();
    let previous = -1;
    for (let i = 0; i <= 3200; i++) {
      const rank = shadowPullRank(i / 3200);
      expect(rank).toBeGreaterThanOrEqual(previous);
      ranks.add(rank);
      previous = rank;
    }
    expect(ranks.size).toBe(SHADOW_PULL_LEVELS + 1);
    expect(shadowGroundPull(1, true)).toBeLessThan(GROUND_PULL.infantryCore);
    expect(shadowGroundPull(1, true)).toBeLessThan(GROUND_PULL.reflection);
    // The margin separates neighbouring steps both ways: half a step.
    const step = (GROUND_PULL.shadow[1] - GROUND_PULL.shadow[0]) / SHADOW_PULL_LEVELS;
    expect(SHADOW_EXCLUSIVE_PULL).toBeCloseTo(step / 2, 15);
    // More than four float32 depth steps anywhere below depth 1 (spacing 2^-24) of the 400-unit view range.
    expect(SHADOW_EXCLUSIVE_PULL / 400).toBeGreaterThan(4 * 2 ** -24);
  });

  it('darkens an overlap once, by the densest shadow (to within one step), whatever the draw order', () => {
    const opacity = 0.7;
    const values = [0, 0.03, 0.1, 0.3, 0.49, 0.5, 0.51, 0.6, 0.62, 0.9, 0.99, 1];
    for (const a of values)
      for (const b of values)
        for (const order of [
          [a, b],
          [b, a],
        ]) {
          const light = overPassSample(order.map((coverage) => ({ coverage, opacity })));
          const densest = Math.max(a, b);
          const expected = densest * opacity >= 0.04 ? 1 - densest * opacity : 1;
          // Never two darkenings; at most one coverage step lighter than the densest.
          expect(light).toBeGreaterThanOrEqual(expected - 1e-12);
          expect(light).toBeLessThanOrEqual(expected + opacity / SHADOW_PULL_LEVELS + 1e-12);
        }
    expect(overPassSample([0.4, 1, 0.7, 1].map((coverage) => ({ coverage, opacity })))).toBeCloseTo(1 - opacity, 12);
  });

  it('lets a later solid shadow replace faint softened leaves instead of leaving a light hole', () => {
    // L (thin leaves, drawn first) draws only 0.1 after the penumbra; P (solid, later) draws 0.5.
    const light = overPassSample([
      { coverage: 0.1, opacity: 0.6 },
      { coverage: 0.5, opacity: 0.6 },
    ]);
    expect(light).toBeCloseTo(1 - 0.5 * 0.6, 12);
  });

  it('draws nothing where every shadow is below the alpha test', () => {
    expect(overPassSample([{ coverage: 0.05, opacity: 0.6 }])).toBe(1);
  });
});
