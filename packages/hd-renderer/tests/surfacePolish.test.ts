import { describe, it, expect } from 'vitest';
import { polishSurfaceCPU } from '../src/surfacePolish';
const policy = { category: 1, sigma: 1.5, range: 0.06 };
describe('category-scoped residual polish', () => {
  it('suppresses flat speckle while preserving a high-contrast seam and excluding vehicles/resources', () => {
    const width = 41,
      pixels = Array.from({ length: width * width }, (_, k) => {
        const x = k % width,
          y = Math.floor(k / width),
          v = x === 20 ? 0.05 : 0.45 + ((x + y) % 2 ? 0.018 : -0.018);
        return [v, v, v];
      }),
      codes = Array(pixels.length).fill(9);
    const value = polishSurfaceCPU(pixels, codes, width, 10, 10, policy)[0]!;
    expect(Math.abs(value - 0.45)).toBeLessThan(0.018 * 0.25);
    expect(polishSurfaceCPU(pixels, codes, width, 20, 20, policy)[0]).toBeLessThan(0.051);
    for (const code of [0, 2, 3, 4, 5]) {
      codes.fill(code);
      expect(polishSurfaceCPU(pixels, codes, width, 10, 10, policy)).toEqual(pixels[10 * width + 10]);
    }
  });
  it('never borrows a neighbour object even when its category matches', () => {
    const pixels = Array.from({ length: 25 }, () => [0.55, 0.55, 0.55]);
    pixels[12] = [0.5, 0.5, 0.5];
    const codes = Array(25).fill(17);
    codes[12] = 9;
    expect(polishSurfaceCPU(pixels, codes, 5, 2, 2, policy)).toEqual([0.5, 0.5, 0.5]);
  });
});
