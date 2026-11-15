import { afterEach, expect, it, vi } from 'vitest';
import { coverageRaster } from '../../src/graphics/coverageRaster';
import { refineCommandCursor } from '../../src/games/yr/commandCursor';

afterEach(() => vi.unstubAllGlobals());
it('interpolates premultiplied coverage without a black rim or exterior glow', () => {
  const result = coverageRaster(new Uint8ClampedArray([0, 200, 0, 255, 0, 0, 0, 0]), 2, 1, 2, (r, g, b) => [r, g, b]);
  for (let i = 0; i < result.length; i += 4) if (result[i + 3]) expect(result[i + 1]).toBe(200);
  expect(result[15]).toBe(0);
});
it('retains non-command art and keeps the exact logical extent of reconstructed commands', () => {
  vi.stubGlobal(
    'ImageData',
    class {
      constructor(
        public data: Uint8ClampedArray,
        public width: number,
        public height: number,
      ) {}
    },
  );
  const make = (rgb: number[]) => new ImageData(new Uint8ClampedArray([...rgb, 255]), 1, 1);
  expect(refineCommandCursor(make([255, 20, 0]), 53)).toBeNull();
  expect(refineCommandCursor(make([255, 255, 255]), 0)).toBeNull();
  expect(refineCommandCursor(make([0, 0, 0]), 41)).toBeNull();
  expect(refineCommandCursor(make([0, 255, 0]), 2)).toBeNull();
  const refined = refineCommandCursor(make([0, 255, 0]), 43)!;
  expect([refined.width, refined.height]).toEqual([2, 2]);
  expect(refined.data[1]).toBeLessThan(255);
  expect(refined.data[1]).toBeGreaterThan(refined.data[0]! * 3);
});
