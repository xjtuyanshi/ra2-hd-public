import { describe, expect, it } from 'vitest';
import {
  analyseVoxelPaint,
  regionLocalVariance,
  srgbToLab,
  type PaintMaterial,
  type VoxelPaintInput,
  type VoxelPaintPolicy,
} from '../src/voxelPaint';
import { ra2PaintMaterial, ra2VoxelPaintPolicy } from '../src/ra2/voxelPaintPolicy';

// Synthetic palette: 0..63 a grey ramp from white to black (~4 sRGB steps), 64..79 a masked red ramp.
const palette = new Uint8Array(768);
for (let i = 0; i < 64; i++) palette.fill(Math.round(252 - i * 4), i * 3, i * 3 + 3);
for (let i = 0; i < 16; i++) palette.set([252 - i * 14, 0, 0], (64 + i) * 3);
const policy: VoxelPaintPolicy = {
  palette,
  families: [{ name: 'grey' }, { name: 'house', masked: true }],
  familyOf: (c) => (c < 64 ? 0 : c < 80 ? 1 : -1),
  materialOf: (c): PaintMaterial => (c >= 56 && c < 64 ? 'rubber' : 'painted'),
};

/** A one-voxel-thick plate facing +z, coloured by fn(x, y). */
function plate(w: number, h: number, fn: (x: number, y: number) => number | null): VoxelPaintInput {
  const xs: number[] = [];
  const ys: number[] = [];
  const cs: number[] = [];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const c = fn(x, y);
      if (c === null) continue;
      xs.push(x);
      ys.push(y);
      cs.push(c);
    }
  const normal = new Float32Array(xs.length * 3);
  for (let i = 0; i < xs.length; i++) normal[i * 3 + 2] = 1;
  return { size: [w, h, 1], x: xs, y: ys, z: xs.map(() => 0), colour: cs, normal };
}

const L = (c: number) => srgbToLab(palette[c * 3], palette[c * 3 + 1], palette[c * 3 + 2])[0];
const paletteRgb = (input: VoxelPaintInput) => {
  const out = new Uint8Array(input.colour.length * 3);
  for (let i = 0; i < input.colour.length; i++)
    out.set(palette.subarray(input.colour[i] * 3, input.colour[i] * 3 + 3), i * 3);
  return out;
};

describe('voxel paint analysis', () => {
  it('collapses a two-entry dither into one flat region between the two shades', () => {
    const input = plate(12, 12, (x, y) => ((x + y) % 2 ? 20 : 22));
    const r = analyseVoxelPaint(input, policy);
    expect(r.regions).toHaveLength(1);
    expect(r.seams).toHaveLength(0);
    const l = r.regions[0].lightness;
    expect(l).toBeGreaterThan(L(22));
    expect(l).toBeLessThan(L(20));
    expect(regionLocalVariance(input, r, paletteRgb(input)).meanDeltaE).toBeGreaterThan(1);
    expect(regionLocalVariance(input, r, r.albedo).meanDeltaE).toBe(0);
  });

  it('keeps a drawn dark line as a seam and leaves it out of the host albedo', () => {
    const input = plate(16, 9, (_x, y) => (y === 4 ? 50 : 20));
    const r = analyseVoxelPaint(input, policy);
    const lines = r.seams.filter((s) => s.kind === 'line');
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ voxels: 16, length: 16, axis: 'x' });
    expect(lines[0].darkness).toBeCloseTo(L(20) - L(50), 5);
    // Two panels either side of the line share one albedo equal to the panel paint.
    const panelIds = new Set(Array.from(r.region).filter((_, i) => r.seamFeature[i] < 0));
    for (const id of panelIds) expect(r.regions[id].albedo).toEqual([palette[60], palette[61], palette[62]]);
    for (let i = 0; i < input.colour.length; i++) expect(r.seam[i] > 0).toBe(input.colour[i] === 50);
  });

  it('removes isolated speckle but keeps a strongly dark dot as detail', () => {
    const input = plate(10, 10, (x, y) => (x === 2 && y === 2 ? 26 : x === 7 && y === 7 ? 60 : 20));
    const r = analyseVoxelPaint(input, policy);
    expect(r.regions).toHaveLength(1);
    expect(r.seams.map((s) => s.kind)).toEqual(['dot']);
    expect(r.seamFeature[input.colour.length - 1 - 2 * 10 - 2]).toBeGreaterThanOrEqual(0);
  });

  it('keeps a filled dark patch as its own part, not a seam', () => {
    const input = plate(14, 14, (x, y) => (x >= 4 && x < 10 && y >= 4 && y < 10 ? 60 : 20));
    const r = analyseVoxelPaint(input, policy);
    expect(r.seams).toHaveLength(0);
    expect(r.regions.map((g) => g.material).sort()).toEqual(['painted', 'rubber']);
  });

  it('does not chain a long shading ramp into one region', () => {
    const input = plate(32, 4, (x) => 8 + x);
    const r = analyseVoxelPaint(input, policy);
    expect(r.regions.length).toBeGreaterThan(1);
    for (const reg of r.regions) {
      const ls = Object.keys(reg.indices).map(Number).map(L);
      expect(Math.max(...ls) - Math.min(...ls)).toBeLessThanOrEqual(20);
    }
  });

  it('keeps masked voxels exact, even a single one', () => {
    const input = plate(10, 10, (x, y) => (x === 5 && y === 5 ? 70 : x < 3 ? 66 : 20));
    const r = analyseVoxelPaint(input, policy);
    for (let i = 0; i < input.colour.length; i++) expect(r.mask[i]).toBe(input.colour[i] >= 64 ? 1 : 0);
    for (const reg of r.regions) {
      const masked = Object.keys(reg.indices)
        .map(Number)
        .map((c) => c >= 64);
      expect(new Set(masked).size).toBe(1);
      expect(reg.masked).toBe(masked[0]);
    }
  });

  it('splits facets whose normals disagree when their paint differs', () => {
    // An L-shaped bend: floor facing +z in light paint, wall facing +x in a darker shade of the same ramp.
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    const cs: number[] = [];
    const ns: number[] = [];
    for (let y = 0; y < 6; y++) {
      for (let x = 0; x < 6; x++) (xs.push(x), ys.push(y), zs.push(0), cs.push(20), ns.push(0, 0, 1));
      for (let z = 1; z < 6; z++) (xs.push(6), ys.push(y), zs.push(z), cs.push(28), ns.push(1, 0, 0));
    }
    const r = analyseVoxelPaint({ size: [7, 6, 6], x: xs, y: ys, z: zs, colour: cs, normal: ns }, policy);
    expect(r.regions).toHaveLength(2);
  });

  it('is deterministic', () => {
    const input = plate(20, 20, (x, y) => ((x * 7 + y * 3) % 5 === 0 ? 23 : y === 10 ? 52 : 21));
    const a = analyseVoxelPaint(input, policy);
    const b = analyseVoxelPaint(input, policy);
    expect(Array.from(a.region)).toEqual(Array.from(b.region));
    expect(Array.from(a.albedo)).toEqual(Array.from(b.albedo));
    expect(a.seams).toEqual(b.seams);
  });
});

describe('RA2 unit-palette paint policy', () => {
  const pal = new Uint8Array(768).fill(128);
  const ra2 = ra2VoxelPaintPolicy(pal);

  it('masks exactly the house remap ramp', () => {
    for (let c = 0; c < 256; c++) {
      const f = ra2.familyOf(c);
      expect(f >= 0 && !!ra2.families[f].masked).toBe(c >= 16 && c <= 31);
    }
  });

  it('keeps the grey ramp and the low greys together and leaves lights and unused entries alone', () => {
    expect(ra2.familyOf(13)).toBe(ra2.familyOf(47));
    expect(ra2.familyOf(32)).toBe(ra2.familyOf(63));
    expect(ra2.familyOf(70)).not.toBe(ra2.familyOf(47));
    for (const c of [0, 5, 204, 239, 240, 255]) expect(ra2.familyOf(c)).toBe(-1);
  });

  it('classifies materials from the shared voxel class table', () => {
    expect(ra2PaintMaterial(20)).toBe('painted');
    expect(ra2PaintMaterial(40)).toBe('painted');
    expect(ra2PaintMaterial(52)).toBe('bare-metal');
    expect(ra2PaintMaterial(60)).toBe('rubber');
    expect(ra2PaintMaterial(165)).toBe('rubber');
    expect(ra2PaintMaterial(245)).toBe('light');
  });

  it('rejects a short palette', () => {
    expect(() => ra2VoxelPaintPolicy(new Uint8Array(10))).toThrow();
  });
});
