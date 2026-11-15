import { describe, expect, it } from 'vitest';
import { buildingPaintCoverage, parseBuildingPilot } from '../src/buildingPilot';
import { attachedLayerDepth, GROUND_DEPTH, type NativeSpriteDepth } from '../src/ra2/nativeDepth';

describe('building pilot gate', () => {
  it('accepts only the explicit pilot value; absent and unknown preserve defaults', () => {
    for (const value of [undefined, null, '', '1', 'true', 'Pilot']) expect(parseBuildingPilot(value)).toBe(false);
    expect(parseBuildingPilot('pilot')).toBe(true);
  });
  it('keeps fractional building coverage while units and the default path retain indexed fallback', () => {
    expect(buildingPaintCoverage(0.25, 1, true, true)).toBe(0.25);
    expect(buildingPaintCoverage(0.25, 1, true)).toBe(1);
    expect(buildingPaintCoverage(0.25, 1, false, true)).toBe(1);
  });
  it('preserves default layer depth and uses signed per-layer art depth only when supplied', () => {
    const body: NativeSpriteDepth = { mode: 'card', bottom: 50, bonus: 0 };
    expect(attachedLayerDepth(body, [0, 3], 3 * GROUND_DEPTH).anchorPush).toBeCloseTo(-GROUND_DEPTH, 12);
    expect(attachedLayerDepth(body, [0, 3], 3 * GROUND_DEPTH, -5).anchorPush).toBeCloseTo(-2 * GROUND_DEPTH, 12);
    expect(attachedLayerDepth(body, [0, 3], 3 * GROUND_DEPTH, 2).anchorPush).toBeCloseTo(5 * GROUND_DEPTH, 12);
    expect(attachedLayerDepth(body, [0, 3], 3 * GROUND_DEPTH, 0).anchorPush).toBeCloseTo(3 * GROUND_DEPTH, 12);
  });
});
