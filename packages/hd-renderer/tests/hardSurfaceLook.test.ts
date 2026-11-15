import { describe, expect, it } from 'vitest';
import {
  combParallelSeams,
  isStaircaseSeam,
  panelLineProfile,
  classifyPaintRegion,
  levelledPaint,
  edgeWear,
  wallOcclusion,
  hoopProfile,
  hatchRingProfile,
  DEFAULT_PAINTED_STEEL_MATERIALS,
  type SeamSegment,
} from '../src/hardSurfaceLook';
import { RA2_HARD_SURFACE_LOOK, ra2HardSurfaceLook } from '../src/ra2/hardSurfaceLook';

const seam = (start: SeamSegment['start'], end: SeamSegment['end'], strength = 0): SeamSegment => ({
  start,
  end,
  strength,
});

describe('hard surface look', () => {
  it('combs by length plus strength, handles reversed directions and preserves input', () => {
    const long = seam([0, 0, 0], [10, 0, 0]),
      strong = seam([9, 1, 0], [0, 1, 0], 2),
      crossing = seam([5, -4, 0], [5, 4, 0]),
      far = seam([0, 4, 0], [9, 4, 0]),
      short = seam([0, 8, 0], [2, 8, 0]);
    const segments = [long, short, far, crossing, strong],
      before = structuredClone(segments);
    expect(combParallelSeams(segments, { minLength: 3 })).toEqual([strong, far, crossing]);
    expect(combParallelSeams(segments, { minLength: 3 })).toEqual([strong, far, crossing]);
    expect(segments).toEqual(before);
  });
  it('uses strict gap/overlap limits, the length tolerance and stable score ties', () => {
    const a = seam([0, 0, 0], [10, 0, 0]),
      gap = seam([0, 1.6, 0], [9, 1.6, 0]),
      overlap = seam([9.5, 1, 0], [12.5, 1, 0]),
      justLong = seam([20, 0, 0], [23 - 0.5e-6, 0, 0]),
      justShort = seam([30, 0, 0], [33 - 2e-6, 0, 0]);
    expect(combParallelSeams([a, gap], { minLength: 3 })).toEqual([a, gap]);
    expect(combParallelSeams([a, gap], { minLength: 3, maxGap: 2 })).toEqual([a]);
    expect(combParallelSeams([a, overlap], { minLength: 3 })).toEqual([a, overlap]);
    expect(combParallelSeams([a, overlap], { minLength: 3, minOverlap: 0.4 })).toEqual([a]);
    expect(combParallelSeams([justLong, justShort], { minLength: 3 })).toEqual([justLong]);
    const tie = seam([0, 0.5, 0], [10, 0.5, 0]);
    expect(combParallelSeams([tie, a], { minLength: 3 })).toEqual([tie]);
    expect(combParallelSeams([seam([0, 0, 0], [0, 0, 0])], { minLength: 0 })).toEqual([]);
  });
  it('recognises only axis-aligned staircase seams on slopes with an untilted axis', () => {
    const n = [1 / Math.sqrt(10), 0, 3 / Math.sqrt(10)] as const;
    expect(isStaircaseSeam([0, -1, 0], n)).toBe(true);
    expect(isStaircaseSeam([1, 0, 0], n)).toBe(false);
    expect(isStaircaseSeam([0, 1, 0], [0, 0, 1])).toBe(false);
    expect(isStaircaseSeam([0.6, 0.8, 0], n)).toBe(false);
    expect(isStaircaseSeam([0, 1, 0], [Math.sqrt(1 - 0.985 ** 2), 0, 0.985])).toBe(false);
    expect(isStaircaseSeam([0, 0.99, Math.sqrt(1 - 0.99 ** 2)], n)).toBe(false);
    expect(isStaircaseSeam([0, 1, 0], [Math.sqrt(1 - 0.9 ** 2 - 0.05 ** 2), 0.05, 0.9])).toBe(false);
  });
  it('returns a narrow groove and a broader quadratic halo with clamped strength', () => {
    const options = { width: 0.4, aoWidth: 1 };
    expect(panelLineProfile(0, -3, options)).toEqual({ dark: 0.6, halo: 0.6 });
    expect(panelLineProfile(0, 3, options)).toEqual({ dark: 1, halo: 1 });
    const profile = panelLineProfile(0.2, 0.4, options);
    expect(profile.dark).toBeCloseTo(0.75 ** 0.7 * 0.8, 12);
    expect(profile.halo).toBeCloseTo(0.64 * 0.8, 12);
    expect(panelLineProfile(0.5, 0.6, options)).toEqual({ dark: 0, halo: 0.25 });
    expect(panelLineProfile(1, 0.6, options)).toEqual({ dark: 0, halo: 0 });
    expect(panelLineProfile(2, 0.6, options)).toEqual({ dark: 0, halo: 0 });
    expect(panelLineProfile(0.5, 1, { width: 1, aoWidth: 0.5 })).toEqual({ dark: 0, halo: 0 });
  });
  it('classifies glints before neutral steel and excludes metal and house regions', () => {
    const flags = { metal: false, house: false };
    expect(classifyPaintRegion([0.9, 0.9, 0.9], flags)).toBe('glint');
    expect(classifyPaintRegion([0.75, 0.75, 0.75], flags)).toBe('bareSteel');
    expect(classifyPaintRegion([0.5, 0.3, 0.48], flags)).toBe('bareSteel');
    expect(classifyPaintRegion([0.6, 0.55, 0.4], flags)).toBe('paint');
    expect(classifyPaintRegion([0.03, 0, 0], flags)).toBe('paint');
    expect(classifyPaintRegion([0.9, 0.9, 0.9], { ...flags, metal: true })).toBe('other');
    expect(classifyPaintRegion([0.6, 0.55, 0.4], { ...flags, house: true })).toBe('other');
  });
  it('levels sRGB paint by weighted luma and clamps its relative level', () => {
    const cream = [0.6, 0.5, 0.4] as const,
      base = [0.5, 0.5, 0.5] as const;
    expect(levelledPaint(base, base, cream)).toEqual(cream);
    expect(levelledPaint([0, 0, 0], base, cream)).toEqual([0.6 * 0.45, 0.5 * 0.45, 0.4 * 0.45]);
    expect(levelledPaint([1, 1, 1], base, cream)).toEqual([0.6 * 1.15, 0.5 * 1.15, 0.4 * 1.15]);
    const ratio = (0.4 * 0.3 + 0.5 * 0.59 + 0.2 * 0.11) / 0.5;
    levelledPaint([0.4, 0.5, 0.2], base, cream).forEach((v, i) => expect(v).toBeCloseTo(cream[i] * ratio, 12));
  });
  it('limits wear to bevels, up-facing normals and sparse noise patches', () => {
    expect(edgeWear(0, 1, 1, 0.3)).toBe(0);
    expect(edgeWear(1, -1, 1, 0.3)).toBe(0);
    expect(edgeWear(1, 1, 0.45, 0.3)).toBe(0);
    expect(edgeWear(1, 1, 1, 0.3)).toBe(0.3);
    expect(edgeWear(1, 0, 0.55, 0.3)).toBeCloseTo(0.03, 12);
  });
  it('returns multiplicative wall occlusion and grime with height and normal clamps', () => {
    expect(wallOcclusion(0, 0, 0.45, 1.5)).toBe(0.55);
    expect(wallOcclusion(1, 0, 0.45, 1.5)).toBe(1);
    expect(wallOcclusion(-1, 0, 0.45, 1.5)).toBe(1);
    expect(wallOcclusion(0, 2, 0.45, 1.5)).toBe(1);
    expect(wallOcclusion(0, -1, 0.45, 1.5)).toBe(0.55);
    expect(wallOcclusion(0.2, 0.5, 0.45, 1.5)).toBeCloseTo(1 - 0.45 * 0.7 * 0.5 ** 1.5, 12);
    expect(wallOcclusion(0, 0.5, 0.32, 1.2)).toBeCloseTo(1 - 0.32 * 0.5 ** 1.2, 12);
  });
  it('profiles hoops symmetrically from either axial end and rings around their radius', () => {
    expect(hoopProfile(-0.75, [-2, 8], 1.25, 0.2)).toBe(1);
    expect(hoopProfile(6.75, [-2, 8], 1.25, 0.2)).toBe(1);
    expect(hoopProfile(-0.65, [-2, 8], 1.25, 0.2)).toBeCloseTo(0.5, 12);
    expect(hoopProfile(3, [-2, 8], 1.25, 0.2)).toBe(0);
    expect(hoopProfile(-3, [-2, 8], 1.25, 0.2)).toBe(0);
    expect(hatchRingProfile(2.7, 2.7, 0.16)).toBe(1);
    expect(hatchRingProfile(2.78, 2.7, 0.16)).toBeCloseTo(0.5, 12);
    expect(hatchRingProfile(2.62, 2.7, 0.16)).toBeCloseTo(0.5, 12);
    expect(hatchRingProfile(0, 2.7, 0.16)).toBe(0);
  });
  it('preserves the reference material table and keeps named paint policy separate', () => {
    expect(DEFAULT_PAINTED_STEEL_MATERIALS).toEqual({
      paint: { roughness: 0.45, metalness: 0 },
      darkMetal: { roughness: 0.42, metalness: 0.55 },
      track: { roughness: 0.7, metalness: 0.15 },
      bareSteel: { roughness: 0.45, metalness: 0.45 },
      barrelSteel: { roughness: 0.3, metalness: 0.85 },
      drumRoughness: 0.3,
    });
    expect(RA2_HARD_SURFACE_LOOK).toEqual({
      cream: [0.64, 0.615, 0.565],
      houseTone: [0.6, 0.6, 0.6],
      hatchColour: [0.36, 0.38, 0.43],
      barrelSteel: [0.42, 0.44, 0.47],
      wear: 0.3,
      wallOcclusion: 0.45,
      wallGrime: 0.32,
      lines: { width: 0.42, aoWidth: 1 },
      seams: { minLength: 2, maxGap: 1.6, minOverlap: 0.5 },
      paintCodes: { auto: 0, house: 1, steel: 2, flat: 3, wheel: 4, drum: 5, hatch: 6 },
      materials: DEFAULT_PAINTED_STEEL_MATERIALS,
    });
  });
  it('takes house colour explicitly and keeps the neutral policy isolated between players', () => {
    const blue = ra2HardSurfaceLook([0.1, 0.3, 0.8]);
    expect(blue.houseTone).toEqual([0.1, 0.3, 0.8]);
    blue.paintCodes.house = 9;
    expect(RA2_HARD_SURFACE_LOOK.paintCodes.house).toBe(1);
    expect(RA2_HARD_SURFACE_LOOK.houseTone).toEqual([0.6, 0.6, 0.6]);
    expect(() => ra2HardSurfaceLook([NaN, 0, 0])).toThrow('house colour');
  });
});
