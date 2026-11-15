import { BufferGeometry, DataTexture, FloatType, NearestFilter, RGBAFormat } from 'three/webgpu';

/** Rasterize an existing mesh's world positions into its art UVs, for small offscreen material caches.
 * The caller supplies the visibility axis; overlapping UV triangles keep the nearest surface on that axis.
 * No art is copied. Uncovered texels remain zero with alpha zero. */
export function uvPositionTexture(
  geometry: BufferGeometry,
  uvAttribute: string,
  width: number,
  height: number,
  visibilityAxis: readonly [number, number, number],
): DataTexture {
  const positions = geometry.getAttribute('position'),
    uv = geometry.getAttribute(uvAttribute),
    index = geometry.index;
  const data = new Float32Array(width * height * 4),
    depths = new Float32Array(width * height).fill(-Infinity);
  const count = index?.count ?? positions.count;
  for (let face = 0; face < count; face += 3) {
    const ids = [0, 1, 2].map((k) => (index ? index.getX(face + k) : face + k));
    const px = ids.map((i) => uv.getX(i) * width),
      py = ids.map((i) => uv.getY(i) * height);
    const area = (py[1]! - py[2]!) * (px[0]! - px[2]!) + (px[2]! - px[1]!) * (py[0]! - py[2]!);
    if (Math.abs(area) < 1e-10) continue;
    const minX = Math.max(0, Math.ceil(Math.min(...px) - 0.5)),
      maxX = Math.min(width - 1, Math.floor(Math.max(...px) - 0.5));
    const minY = Math.max(0, Math.ceil(Math.min(...py) - 0.5)),
      maxY = Math.min(height - 1, Math.floor(Math.max(...py) - 0.5));
    for (let y = minY; y <= maxY; y++)
      for (let x = minX; x <= maxX; x++) {
        const a = ((py[1]! - py[2]!) * (x + 0.5 - px[2]!) + (px[2]! - px[1]!) * (y + 0.5 - py[2]!)) / area;
        const b = ((py[2]! - py[0]!) * (x + 0.5 - px[2]!) + (px[0]! - px[2]!) * (y + 0.5 - py[2]!)) / area;
        const c = 1 - a - b;
        if (Math.min(a, b, c) < -1e-7) continue;
        const weights = [a, b, c];
        const point = [0, 1, 2].map((axis) =>
          ids.reduce((v, i, k) => v + positions.getComponent(i, axis) * weights[k]!, 0),
        );
        const depth = point.reduce((v, p, k) => v + p * visibilityAxis[k]!, 0),
          at = y * width + x;
        if (depth < depths[at]!) continue;
        depths[at] = depth;
        data.set([...point, 1], at * 4);
      }
  }
  const result = new DataTexture(data, width, height, RGBAFormat, FloatType);
  result.minFilter = result.magFilter = NearestFilter;
  result.generateMipmaps = false;
  result.needsUpdate = true;
  return result;
}
