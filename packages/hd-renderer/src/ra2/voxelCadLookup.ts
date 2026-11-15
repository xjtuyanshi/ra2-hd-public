import { classifyPaintRegion, levelledPaint, type LookVector } from '../hardSurfaceLook';
import { RA2_HARD_SURFACE_LOOK as policy, RA2_CAD_HOUSE_LIMITS as houseLimits } from './hardSurfaceLook';
import { ra2VoxelClass } from './voxelClasses';

export const CAD_CLASS = { paint: 0, house: 1, darkMetal: 2, track: 3, bareSteel: 4, barrelSteel: 5 } as const;
export type CadClass = keyof typeof CAD_CLASS;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
/** Hue is kept; a neutral default player colour must not collapse every live player hue to grey. */
export function cadHouseTone(colour: LookVector): LookVector {
  const peak = Math.max(...colour, 1e-6),
    rgb = colour.map((v) => v / peak),
    low = Math.min(...rgb);
  const saturation = Math.min(1, (1 - houseLimits.trough / houseLimits.peak) / Math.max(1e-6, 1 - low));
  return rgb.map((v) => (1 + (v - 1) * saturation) * houseLimits.peak) as [number, number, number];
}
export function cadPaletteEntry(palette: Uint8Array, index: number, steel = false) {
  const rgb = Array.from(palette.subarray(index * 3, index * 3 + 3), (v) => v / 255) as [number, number, number];
  let colour: LookVector = rgb,
    kind: CadClass = 'paint';
  const original = ra2VoxelClass(index, ...rgb).kind;
  if (steel) {
    colour = policy.barrelSteel;
    kind = 'barrelSteel';
  } else if (original === 'house') {
    kind = 'house';
    colour = cadHouseTone(rgb).map((v) => v * clamp(Math.max(...rgb) * 1.25, 0.35, 0.9)) as [number, number, number];
  } else if (original === 'rubber') {
    kind = 'track';
    colour = rgb.map((v, a) => Math.max(v, [0.12, 0.12, 0.115][a])) as [number, number, number];
  } else if (original === 'dark') {
    kind = 'darkMetal';
    colour = rgb.map((v, a) => Math.max(v, [0.3, 0.31, 0.34][a])) as [number, number, number];
  } else {
    const region = classifyPaintRegion(rgb, { metal: false, house: false });
    if (region === 'glint') colour = policy.cream;
    else if (region === 'bareSteel') {
      kind = 'bareSteel';
      colour = rgb.map((v) => clamp(v * 0.95, 0.3, 0.55)) as [number, number, number];
    } else colour = levelledPaint(rgb, [138 / 255, 138 / 255, 113 / 255], policy.cream);
  }
  const material = policy.materials[kind === 'house' ? 'paint' : kind];
  return { colour, kind, ...material };
}
/** Three independent rows: sRGB albedo, linear roughness/metalness/class, sRGB house hue at unit level. */
export function cadLookupBytes(palette: Uint8Array): Uint8Array {
  if (palette.length !== 768) throw new Error('CAD palette must contain 768 RGB bytes');
  const bytes = new Uint8Array(256 * 3 * 4);
  const fallback = cadHouseTone(
    Array.from(palette.subarray(24 * 3, 24 * 3 + 3), (v) => v / 255) as [number, number, number],
  );
  for (let i = 0; i < 256; i++) {
    const p = cadPaletteEntry(palette, i);
    bytes.set([...p.colour.map((v) => Math.round(clamp(v, 0, 1) * 255)), 255], i * 4);
    bytes.set([Math.round(p.roughness * 255), Math.round(p.metalness * 255), CAD_CLASS[p.kind], 0], (i + 256) * 4);
    const hue =
      i >= 16 && i <= 31
        ? cadHouseTone(Array.from(palette.subarray(i * 3, i * 3 + 3), (v) => v / 255) as [number, number, number])
        : fallback;
    bytes.set([...hue.map((v) => Math.round(v * 255)), 255], (i + 512) * 4);
  }
  return bytes;
}
