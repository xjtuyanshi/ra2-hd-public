/** Premultiplied bilinear coverage reconstruction. No dilation, external pixels, or guest memory writes. */
export function coverageRaster(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  factor: number,
  colour: (r: number, g: number, b: number) => readonly number[],
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(width * height * factor * factor * 4);
  const targetWidth = width * factor;
  for (let y = 0; y < height * factor; y++)
    for (let x = 0; x < targetWidth; x++) {
      const sx = (x + 0.5) / factor - 0.5,
        sy = (y + 0.5) / factor - 0.5;
      const ix = Math.floor(sx),
        iy = Math.floor(sy),
        fx = sx - ix,
        fy = sy - iy;
      let alpha = 0;
      const rgb = [0, 0, 0];
      for (let dy = 0; dy < 2; dy++)
        for (let dx = 0; dx < 2; dx++) {
          const px = ix + dx,
            py = iy + dy;
          if (px < 0 || px >= width || py < 0 || py >= height) continue;
          const k = (py * width + px) * 4;
          const a = source[k + 3]! * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
          alpha += a;
          const c = colour(source[k]!, source[k + 1]!, source[k + 2]!);
          for (let i = 0; i < 3; i++) rgb[i]! += c[i]! * a;
        }
      const k = (y * targetWidth + x) * 4;
      if (alpha > 0) for (let i = 0; i < 3; i++) out[k + i] = rgb[i]! / alpha;
      out[k + 3] = alpha;
    }
  return out;
}
