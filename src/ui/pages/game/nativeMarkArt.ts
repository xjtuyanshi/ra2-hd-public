import { parseFrameMarkManifest } from '../../../graphics/frameMarks';
import type { FrameMarkArt } from './vmFrameRenderer';

/**
 * Loads the farm's sharp art for fixed native battlefield marks (manifest and atlas written by the asset tools; see
 * docs/remaster/HD_OVERLAYS.md). A missing or malformed farm entry returns null and every native mark stays.
 * The caller owns the result and calls dispose after removing it from the renderer.
 */
export async function loadNativeMarkArt(
  base = '/hd-cache/_ui/marks',
  signal?: AbortSignal,
): Promise<(FrameMarkArt & { dispose(): void }) | null> {
  const response = await fetch(`${base}/marks.json`, { signal });
  if (!response.ok) return null;
  const manifest = (await response.json()) as { file?: unknown };
  const file = manifest.file ?? 'marks.png';
  if (typeof file !== 'string' || !/^[a-z0-9_-]+\.png$/i.test(file)) return null;
  const imageResponse = await fetch(`${base}/${file}`, { signal });
  if (!imageResponse.ok) return null;
  const bitmap = await createImageBitmap(await imageResponse.blob(), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  });
  const set = parseFrameMarkManifest(manifest, bitmap.width, bitmap.height);
  if (!set || signal?.aborted) {
    bitmap.close();
    return null;
  }
  return { set, atlas: bitmap, dispose: () => bitmap.close() };
}
