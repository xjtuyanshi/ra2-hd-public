import type { Palette, ShpFile } from 'ra2-formats';

export interface UiRaster {
  width: number;
  height: number;
  channels: 4;
  data: Uint8Array;
}

/** Keep crop offsets, optionally clipping/padding to a caller-owned slot without resampling pixels. */
export function rasterizeUiFrame(
  shape: ShpFile,
  palette: Palette,
  index: number,
  size?: { width: number; height: number },
): UiRaster {
  const width = size?.width ?? shape.cx,
    height = size?.height ?? shape.cy;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 4096 * 4096)
    throw new Error('Invalid UI shape bounds');
  const frame = shape.frame(index),
    data = new Uint8Array(width * height * 4);
  for (let y = 0; y < frame.cy; y++) {
    for (let x = 0; x < frame.cx; x++) {
      const color = frame.pixels[y * frame.cx + x];
      const dx = frame.x + x,
        dy = frame.y + y;
      if (!color || dx < 0 || dy < 0 || dx >= width || dy >= height) continue;
      data.set(palette.rgba(color), (dy * width + dx) * 4);
    }
  }
  return { width, height, channels: 4, data };
}

export interface CursorIndex {
  version: 1;
  frames: { x: number; y: number; width: number; height: number }[];
  /** Optional caller-supplied cursor type metadata; runtime hotspots need not be baked into the atlas. */
  types?: readonly unknown[];
}

/** Export all canvases side by side, retaining cropped offsets and empty frames. */
export function rasterizeUiStrip(shape: ShpFile, palette: Palette): UiRaster {
  const width = shape.cx * shape.count,
    height = shape.cy;
  if (!shape.count || width < 1 || height < 1 || width > 16384 || height > 16384 || width * height > 64 * 1024 * 1024)
    throw new Error('UI strip exceeds bounds');
  const image: UiRaster = { width, height, channels: 4, data: new Uint8Array(width * height * 4) };
  for (let i = 0; i < shape.count; i++) {
    const frame = rasterizeUiFrame(shape, palette, i);
    for (let y = 0; y < height; y++) {
      image.data.set(frame.data.subarray(y * shape.cx * 4, (y + 1) * shape.cx * 4), (y * width + i * shape.cx) * 4);
    }
  }
  return image;
}

/** Pack every frame, including empty placeholders, without assuming body/shadow pairs. */
export function buildCursorAtlas(
  shape: ShpFile,
  palette: Palette,
  types?: readonly unknown[],
): {
  image: UiRaster;
  index: CursorIndex;
} {
  if (!shape.count) throw new Error('Cursor shape has no frames');
  const columns = Math.ceil(Math.sqrt(shape.count)),
    rows = Math.ceil(shape.count / columns);
  const width = columns * shape.cx,
    height = rows * shape.cy;
  if (width < 1 || height < 1 || width > 16384 || height > 16384 || width * height > 64 * 1024 * 1024)
    throw new Error('Cursor atlas exceeds bounds');
  const image: UiRaster = { width, height, channels: 4, data: new Uint8Array(width * height * 4) };
  const frames: CursorIndex['frames'] = [];
  for (let i = 0; i < shape.count; i++) {
    const raster = rasterizeUiFrame(shape, palette, i);
    const x = (i % columns) * shape.cx,
      y = Math.floor(i / columns) * shape.cy;
    for (let row = 0; row < raster.height; row++) {
      image.data.set(
        raster.data.subarray(row * raster.width * 4, (row + 1) * raster.width * 4),
        ((y + row) * width + x) * 4,
      );
    }
    frames.push({ x, y, width: shape.cx, height: shape.cy });
  }
  return { image, index: { version: 1, frames, ...(types ? { types } : {}) } };
}

/**
 * Scale2x (EPX): each pixel becomes 2x2 copies, and a corner takes a neighbour's colour only where two orthogonal
 * neighbours agree across a diagonal. No new colours are made; diagonal stair steps become finer steps and straight
 * runs stay straight. Edges clamp.
 */
export function scale2xRgba(raster: UiRaster): UiRaster {
  const { width, height, data } = raster;
  const out = new Uint8Array(width * height * 16);
  const view = new Uint32Array(data.buffer, data.byteOffset, width * height);
  const target = new Uint32Array(out.buffer);
  const at = (x: number, y: number) =>
    view[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))]!;
  const outWidth = width * 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = at(x, y),
        a = at(x, y - 1),
        b = at(x + 1, y),
        c = at(x - 1, y),
        d = at(x, y + 1);
      const row = y * 2 * outWidth + x * 2;
      target[row] = c === a && c !== d && a !== b ? a : p;
      target[row + 1] = a === b && a !== c && b !== d ? b : p;
      target[row + outWidth] = d === c && d !== b && c !== a ? c : p;
      target[row + outWidth + 1] = b === d && b !== a && d !== c ? d : p;
    }
  }
  return { width: outWidth, height: height * 2, channels: 4, data: out };
}

export interface MarkDefinition {
  id: string;
  frame: number;
}

/** Version 1 manifest read by src/graphics/frameMarks.ts: native pixel patterns plus art rectangles. */
export interface MarkManifest {
  version: 1;
  scale: number;
  file: string;
  marks: {
    id: string;
    frame: number;
    width: number;
    height: number;
    /** Row-major RGB565 the 16-bit game writes for each crop pixel; -1 transparent. */
    pixels: number[];
    atlas: { x: number; y: number; width: number; height: number };
  }[];
}

/** RGB565 of an 8-bit colour as a 16-bit DirectDraw surface stores it (top bits of each channel). */
export function rgb565Of(r: number, g: number, b: number): number {
  return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

const MARK_CELL_ALIGN = 16;

/**
 * Sharp mark art: each listed frame's own crop, enlarged by repeated Scale2x (`scale` must be 2, 4 or 8) and placed in
 * one row on a 16-texel grid with transparent gaps, so mipmaps do not bleed between marks.
 */
export function buildMarkAtlas(
  shape: ShpFile,
  palette: Palette,
  marks: readonly MarkDefinition[],
  scale = 4,
  file = 'marks.png',
): { image: UiRaster; manifest: MarkManifest } {
  if (![2, 4, 8].includes(scale)) throw new Error('Mark scale must be 2, 4 or 8');
  if (!marks.length) throw new Error('No marks listed');
  const cells = marks.map((mark) => {
    if (!Number.isInteger(mark.frame) || mark.frame < 0 || mark.frame >= shape.count)
      throw new Error(`Mark ${mark.id}: frame ${mark.frame} is not in the shape`);
    const frame = shape.frame(mark.frame);
    if (frame.cx < 1 || frame.cy < 1) throw new Error(`Mark ${mark.id}: frame ${mark.frame} is empty`);
    // One transparent pixel of margin so Scale2x sees the true outline at the crop edges.
    const padded: UiRaster = {
      width: frame.cx + 2,
      height: frame.cy + 2,
      channels: 4,
      data: new Uint8Array((frame.cx + 2) * (frame.cy + 2) * 4),
    };
    const pixels: number[] = [];
    for (let y = 0; y < frame.cy; y++)
      for (let x = 0; x < frame.cx; x++) {
        const index = frame.pixels[y * frame.cx + x]!;
        if (!index) {
          pixels.push(-1);
          continue;
        }
        const [r, g, b] = palette.rgba(index);
        pixels.push(rgb565Of(r, g, b));
        padded.data.set([r, g, b, 255], ((y + 1) * padded.width + x + 1) * 4);
      }
    if (!pixels.some((pixel) => pixel >= 0)) throw new Error(`Mark ${mark.id}: frame ${mark.frame} is empty`);
    let art = padded;
    for (let factor = 1; factor < scale; factor *= 2) art = scale2xRgba(art);
    return { mark, width: frame.cx, height: frame.cy, pixels, art };
  });
  const slot = (size: number) => Math.ceil(size / MARK_CELL_ALIGN) * MARK_CELL_ALIGN + MARK_CELL_ALIGN;
  const width = MARK_CELL_ALIGN + cells.reduce((sum, cell) => sum + slot(cell.width * scale), 0);
  const height = MARK_CELL_ALIGN + Math.max(...cells.map((cell) => slot(cell.height * scale)));
  const image: UiRaster = { width, height, channels: 4, data: new Uint8Array(width * height * 4) };
  const manifest: MarkManifest = { version: 1, scale, file, marks: [] };
  let x = MARK_CELL_ALIGN;
  for (const cell of cells) {
    const w = cell.width * scale,
      h = cell.height * scale,
      y = MARK_CELL_ALIGN;
    for (let row = 0; row < h; row++) {
      const source = ((row + scale) * cell.art.width + scale) * 4;
      image.data.set(cell.art.data.subarray(source, source + w * 4), ((y + row) * width + x) * 4);
    }
    manifest.marks.push({
      id: cell.mark.id,
      frame: cell.mark.frame,
      width: cell.width,
      height: cell.height,
      pixels: cell.pixels,
      atlas: { x, y, width: w, height: h },
    });
    x += slot(w);
  }
  return { image, manifest };
}
