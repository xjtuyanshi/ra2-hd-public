import { describe, expect, it } from 'vitest';
import { GROUND_PULL, SHADOW_PULL_LEVELS, shadowGroundPull, shadowPullRank } from '../src/ra2/sprites';

describe('enhanced shadow coverage ranks', () => {
  it('spans the ground pull range in sixteen equal steps and clamps coverage', () => {
    expect(SHADOW_PULL_LEVELS).toBe(16);
    expect(shadowGroundPull(shadowPullRank(0))).toBe(GROUND_PULL.shadow[0]);
    expect(shadowGroundPull(shadowPullRank(1))).toBe(GROUND_PULL.shadow[1]);
    expect(shadowPullRank(-1)).toBe(0);
    expect(shadowPullRank(2)).toBe(16);
    const ranks = new Set<number>();
    let previous = -1;
    for (let i = 0; i <= 1600; i++) {
      const rank = shadowPullRank(i / 1600);
      expect(rank).toBeGreaterThanOrEqual(previous);
      ranks.add(rank);
      previous = rank;
    }
    expect([...ranks]).toEqual(Array.from({ length: 17 }, (_, i) => i));
    for (let level = 0; level <= 16; level++) {
      expect(shadowPullRank(level / 16)).toBe(level);
      expect(shadowGroundPull(level)).toBeCloseTo(GROUND_PULL.shadow[0] + level * 0.0005, 12);
    }
  });

  it('lets a later solid shadow pass faint softened leaves under LessDepth', () => {
    // L may have denser indexed coverage, but draws only 0.1 after the penumbra. P draws 0.5: it must be nearer.
    const leaves = shadowPullRank(0.1);
    const solid = shadowPullRank(0.5);
    expect(leaves).toBe(2);
    expect(solid).toBe(8);
    expect(shadowGroundPull(solid)).toBeGreaterThan(shadowGroundPull(leaves));
  });
});
