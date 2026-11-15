import { describe, expect, it } from 'vitest';
import {
  FrameMarkReplacer,
  clearFrameMarks,
  findFrameMarks,
  frameMarkVertices,
  indexFrameMarks,
  parseFrameMarkManifest,
} from '../../src/graphics/frameMarks';

const KEY = 0xf81f;
const A = 0xffe0,
  B = 0xfc00;

/** A 3x2 "chevron" and a 3x4 double chevron whose top half repeats it. */
function manifest() {
  const single = [-1, A, -1, B, -1, B];
  return {
    version: 1,
    scale: 4,
    file: 'marks.png',
    marks: [
      { id: 'single', width: 3, height: 2, pixels: single, atlas: { x: 0, y: 0, width: 12, height: 8 } },
      {
        id: 'double',
        width: 3,
        height: 4,
        pixels: [...single, ...single],
        atlas: { x: 16, y: 0, width: 12, height: 16 },
      },
    ],
  };
}

function picture(width: number, height: number): Uint16Array {
  return new Uint16Array(width * height).fill(KEY);
}

function stamp(pixels: Uint16Array, width: number, x: number, y: number, pattern: number[], w: number) {
  pattern.forEach((value, index) => {
    if (value >= 0) pixels[(y + Math.floor(index / w)) * width + x + (index % w)] = value;
  });
}

describe('native mark replacement', () => {
  const set = parseFrameMarkManifest(manifest(), 32, 16)!;
  const index = indexFrameMarks(set);
  const single = [-1, A, -1, B, -1, B];

  it('validates the manifest against the atlas and rejects malformed entries', () => {
    expect(set.templates.map((t) => t.id)).toEqual(['single', 'double']);
    expect(parseFrameMarkManifest(manifest(), 20, 16)).toBeNull();
    const bad = manifest();
    bad.marks[0]!.pixels = [-1, -1, -1, -1, -1, -1];
    expect(parseFrameMarkManifest(bad, 32, 16)).toBeNull();
    const wrongScale = manifest();
    wrongScale.marks[0]!.atlas.width = 11;
    expect(parseFrameMarkManifest(wrongScale, 32, 16)).toBeNull();
    expect(parseFrameMarkManifest({ ...manifest(), version: 2 }, 32, 16)).toBeNull();
  });

  it('finds exact occurrences, prefers the larger mark and never shares a pixel', () => {
    const pixels = picture(12, 8);
    stamp(pixels, 12, 1, 1, single, 3);
    stamp(pixels, 12, 6, 2, [...single, ...single], 3);
    const matches = findFrameMarks(pixels, 12, 8, { x: 0, y: 0, width: 12, height: 8 }, index, KEY);
    expect(matches).toEqual([
      { template: 1, x: 6, y: 2 },
      { template: 0, x: 1, y: 1 },
    ]);
  });

  it('keeps native pixels for overdrawn, clipped or out-of-rectangle marks', () => {
    const pixels = picture(12, 8);
    stamp(pixels, 12, 1, 1, single, 3);
    pixels[2 * 12 + 1] = 0x1234; // a later native draw over one opaque pixel
    stamp(pixels, 12, 9, 6, single, 3); // partly outside the key rectangle below
    expect(findFrameMarks(pixels, 12, 8, { x: 0, y: 0, width: 12, height: 7 }, index, KEY)).toEqual([]);
    // Transparent template pixels may hold anything.
    const other = picture(12, 8);
    stamp(other, 12, 1, 1, single, 3);
    other[1 * 12 + 1] = 0x1234;
    expect(findFrameMarks(other, 12, 8, { x: 0, y: 0, width: 12, height: 8 }, index, KEY)).toHaveLength(1);
  });

  it('clears only matched opaque pixels in a copy and maps quads to the native box', () => {
    const pixels = picture(12, 8);
    stamp(pixels, 12, 1, 1, single, 3);
    pixels[1 * 12 + 1] = 0x1234;
    const replacer = new FrameMarkReplacer(set);
    const clean = replacer.prepare(pixels, 12, 8, { x: 0, y: 0, width: 12, height: 8 }, KEY)!;
    expect(clean).not.toBe(pixels);
    expect(pixels[1 * 12 + 2]).toBe(A);
    expect(clean[1 * 12 + 2]).toBe(KEY);
    expect(clean[2 * 12 + 1]).toBe(KEY);
    expect(clean[1 * 12 + 1]).toBe(0x1234);
    expect(replacer.prepare(pixels, 12, 8, { x: 0, y: 0, width: 12, height: 8 }, KEY)).toBe(clean);
    const quad = frameMarkVertices(replacer.matches, set, 12, 8);
    expect(quad.length).toBe(30);
    [1 / 6 - 1, 1 - 0.25, 0, 0, 1].forEach((value, i) => expect(quad[i]).toBeCloseTo(value, 6));
    [4 / 6 - 1, 1 - 0.75, 12 / 32, 8 / 16, 1].forEach((value, i) => expect(quad[25 + i]).toBeCloseTo(value, 6));
    // Light is sampled at the native box corners (picture pixels) and clamped; the mark here spans x 1..4, y 1..3.
    const seen: [number, number][] = [];
    const lit = frameMarkVertices(replacer.matches, set, 12, 8, (x, y) => {
      seen.push([x, y]);
      return x === 1 ? (y === 1 ? -2 : 0.25) : y === 1 ? Number.NaN : 3;
    });
    expect(seen).toEqual([
      [1, 1],
      [4, 1],
      [1, 3],
      [4, 3],
    ]);
    expect([0, 1, 2, 3, 4, 5].map((v) => lit[v * 5 + 4])).toEqual([0, 1, 0.25, 0.25, 1, 1]);
    const none = picture(12, 8);
    expect(replacer.prepare(none, 12, 8, { x: 0, y: 0, width: 12, height: 8 }, KEY)).toBeNull();
    const copy = picture(12, 8);
    clearFrameMarks(copy, 12, [], set, KEY);
    expect(copy.every((v) => v === KEY)).toBe(true);
  });
});
