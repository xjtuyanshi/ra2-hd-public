import { coverageRaster } from '../../graphics/coverageRaster';

/** Recognise green order art as a whole; arrows, red attack and sidebar cursors retain their original pixels. */
export function refineCommandCursor(data: ImageData, frame: number): ImageData | null {
  // Pinned YR 1.001 mouseControls entries 17/18: 18+13, 31+10 and the small 42+10 sequence. Menu arrows, attack, harvest and scroll art are separate shapes.
  if (!((frame >= 18 && frame <= 40) || (frame >= 42 && frame <= 51))) return null;
  const pixels = coverageRaster(data.data, data.width, data.height, 2, (r, g, b) => {
    // Keep green identity and internal highlights; reduce the neon separation without adding an exterior halo.
    if (g <= r * 1.4 || g <= b * 1.4) return [r, g, b];
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return [r, g, b].map((c) => (c * 0.88 + l * 0.12) * 0.9);
  });
  return new ImageData(pixels, data.width * 2, data.height * 2);
}
