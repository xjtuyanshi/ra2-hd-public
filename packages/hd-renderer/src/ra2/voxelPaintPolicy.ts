import type { PaintFamily, PaintMaterial, VoxelPaintPolicy } from '../voxelPaint';
import { ra2VoxelClass } from './voxelClasses';

/**
 * RA2/YR unit-palette policy for the generic voxel paint analysis (voxelPaint.ts).
 *
 * The stock unit palettes (unittem.pal and its theatre siblings) are laid out as 16-entry shading ramps, with the
 * grey ramp spanning 32..63 and the house (player) remap ramp at 16..31. A family is one ramp: voxel artists shade
 * and dither within a ramp, so neighbours of one ramp may collapse into one flat albedo, while a different ramp is a
 * different paint. 12..15 are the fixed greys of the low block and belong with the grey ramp. 204..239 are unused
 * magenta in the stock palette and 240..255 are the engine's light/special entries: each of those stays a family of
 * its own (-1) so no averaging ever mixes them.
 */
const RA2_UNIT_FAMILIES: readonly (PaintFamily & { readonly ranges: readonly (readonly [number, number])[] })[] = [
  { name: 'house', masked: true, ranges: [[16, 31]] },
  {
    name: 'grey',
    ranges: [
      [12, 15],
      [32, 63],
    ],
  },
  { name: 'khaki', ranges: [[64, 79]] },
  { name: 'blue-grey', ranges: [[80, 95]] },
  { name: 'rust', ranges: [[96, 111]] },
  { name: 'olive-brown', ranges: [[112, 127]] },
  { name: 'tan', ranges: [[128, 143]] },
  { name: 'brown', ranges: [[144, 167]] },
  { name: 'green', ranges: [[168, 175]] },
  { name: 'yellow-orange', ranges: [[176, 191]] },
  { name: 'blue', ranges: [[192, 199]] },
  { name: 'red', ranges: [[200, 203]] },
];

const FAMILY_OF = (() => {
  const out = new Int16Array(256).fill(-1);
  RA2_UNIT_FAMILIES.forEach((f, k) => {
    for (const [a, b] of f.ranges) for (let i = a; i <= b; i++) out[i] = k;
  });
  return out;
})();

/** First and last palette index of the house remap ramp; the mask is exactly these entries. */
export const RA2_HOUSE_RAMP: readonly [number, number] = [16, 31];

/** Material per palette entry, from the shared RA2 voxel class table (grey armour and house colour are paint). */
export function ra2PaintMaterial(index: number): PaintMaterial {
  switch (ra2VoxelClass(index, 0, 0, 0).kind) {
    case 'dark':
      return 'bare-metal';
    case 'rubber':
      return 'rubber';
    case 'glass':
      return 'glass';
    case 'light':
      return 'light';
    default:
      return 'painted';
  }
}

/** Policy for one 256-colour unit palette (8-bit sRGB triples, e.g. a decoded unittem.pal). */
export function ra2VoxelPaintPolicy(palette: ArrayLike<number>): VoxelPaintPolicy {
  if (palette.length < 768) throw new Error(`unit palette needs 768 bytes, got ${palette.length}`);
  return {
    palette,
    families: RA2_UNIT_FAMILIES.map(({ name, masked }) => ({ name, masked })),
    familyOf: (index) => FAMILY_OF[index],
    materialOf: ra2PaintMaterial,
  };
}
