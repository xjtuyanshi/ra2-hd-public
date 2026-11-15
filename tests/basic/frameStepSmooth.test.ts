import { describe, expect, it } from 'vitest';
import { STEP_SMOOTH_GLSL, stepSmooth2x, stepSmooth4x } from '../../src/graphics/frameStepSmooth';

const K = 0xf81f; // key / background
const W = 0xffff;

function picture(rows: string[], colours: Record<string, number>): { pixels: number[]; width: number; height: number } {
  return {
    pixels: rows.flatMap((row) => [...row].map((c) => colours[c] ?? K)),
    width: rows[0].length,
    height: rows.length,
  };
}

/** Each source pixel as its 4x4 block: what nearest sampling at 4x shows. */
function blocks(pixels: number[], width: number, height: number): Uint32Array {
  const out = new Uint32Array(width * height * 16);
  for (let y = 0; y < height * 4; y++)
    for (let x = 0; x < width * 4; x++) out[y * width * 4 + x] = pixels[(y >> 2) * width + (x >> 2)];
  return out;
}

describe('step smoothing', () => {
  it('keeps axis-aligned rectangles, outlines and solid fills exactly', () => {
    const { pixels, width, height } = picture(
      ['........', '.bbbbbb.', '.bggggb.', '.bbbbbb.', '........', '...ww...', '...ww...', '........'],
      { b: 0x001f, g: 0x07e0, w: W },
    );
    expect([...stepSmooth4x(pixels, width, height, K)]).toEqual([...blocks(pixels, width, height)]);
  });

  it('keeps the inside of checkerboards (dithering) exactly', () => {
    const { pixels, width, height } = picture(
      ['.w.w.w.w', 'w.w.w.w.', '.w.w.w.w', 'w.w.w.w.', '.w.w.w.w', 'w.w.w.w.'],
      {
        w: W,
      },
    );
    const full = stepSmooth4x(pixels, width, height, K),
      plain = blocks(pixels, width, height);
    // Pixels away from the edge see only the pattern (the edge itself meets the outside value).
    for (let y = 8; y < (height - 2) * 4; y++)
      for (let x = 8; x < (width - 2) * 4; x++) expect(full[y * width * 4 + x]).toBe(plain[y * width * 4 + x]);
    // Without the second step's gate the 2x blocks would be filled like diagonal steps.
    const once = stepSmooth2x(pixels, width, height, K);
    const ungated = stepSmooth2x(once, width * 2, height * 2, K);
    expect(ungated.some((value, index) => value !== plain[index] && index > 8 * width * 4)).toBe(true);
  });

  it('fills the concave corners of a one-pixel 2:1 line on both sides, without new colours', () => {
    // A 2:1 bracket diagonal: pairs of pixels stepping down one row.
    const { pixels, width, height } = picture(['ww......', '..ww....', '....ww..', '......ww'], { w: W });
    const half = stepSmooth2x(pixels, width, height, K);
    const at = (x: number, y: number) => half[y * width * 2 + x];
    // Source pixel (1,1) is background below the end of the first run: its top-right quadrant joins the step...
    expect(at(3, 2)).toBe(W);
    // ...and source pixel (2,0) above the start of the second run fills its bottom-left quadrant.
    expect(at(4, 1)).toBe(W);
    // The line pixels themselves are unchanged.
    for (const [x, y] of [
      [0, 0],
      [1, 0],
      [2, 1],
      [3, 1],
    ])
      for (let q = 0; q < 4; q++) expect(at(2 * x + (q & 1), 2 * y + (q >> 1))).toBe(W);
    const full = stepSmooth4x(pixels, width, height, K);
    expect(new Set(full)).toEqual(new Set([K, W]));
    expect([...full]).not.toEqual([...blocks(pixels, width, height)]);
  });

  it('treats outside pixels as the given value so marks at the picture edge stay put', () => {
    const { pixels, width, height } = picture(['w.', '.w'], { w: W });
    const inside = stepSmooth2x(pixels, width, height, K);
    // The 45-degree pair fills the shared corners of the two background pixels.
    expect(inside[0 * 4 + 3]).toBe(K);
    expect(inside[1 * 4 + 2]).toBe(W);
    expect(inside[2 * 4 + 1]).toBe(W);
  });

  it('exposes the shader functions the renderer includes', () => {
    for (const name of ['uint stepSmoothAt(vec2 s)', 'bool stepSmoothFlat(ivec2 lo, ivec2 hi)', 'stepSmoothFetch('])
      expect(STEP_SMOOTH_GLSL).toContain(name);
  });
});
