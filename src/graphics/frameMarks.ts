/**
 * Sharp replacements for small fixed native marks (for example a game's rank chevrons) in a keyed overlay picture.
 *
 * The native game still draws every mark itself. Each template is the exact 16-bit pixel pattern one source frame
 * writes (opaque pixels only) plus its higher-resolution art. A match requires every opaque template pixel to be
 * present unchanged inside the key rectangle. Matched pixels become the key colour in a copy of the picture and the
 * art is drawn over the same native box at output resolution. Anything drawn over a mark afterwards breaks the exact
 * match, so that mark keeps its native pixels; nothing is erased by guessed rectangle or colour. Where opaque pixels
 * of several candidates overlap, the template with more opaque pixels wins and the other is rejected.
 *
 * Game policy (which frames, palette, file names) lives with the game's asset tools; this module only consumes the
 * resulting manifest and art.
 */

export interface FrameMarkRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FrameMarkTemplate {
  id: string;
  /** Native box of the drawn frame (its opaque crop). */
  width: number;
  height: number;
  /** Packed RGB565 value of each box pixel; only meaningful where `opaque` is 1. */
  rgb565: Uint16Array;
  opaque: Uint8Array;
  /** Box pixel offsets (y * width + x) of the opaque pixels, in raster order. */
  opaqueOffsets: Int32Array;
  /** Art rectangle in atlas texels: the native box times the set's scale. */
  atlas: FrameMarkRect;
}

export interface FrameMarkSet {
  /** Art texels per native pixel. */
  scale: number;
  atlasWidth: number;
  atlasHeight: number;
  templates: FrameMarkTemplate[];
}

export interface FrameMarkMatch {
  template: number;
  x: number;
  y: number;
}

const MAX_TEMPLATES = 256;
const MAX_TEMPLATE_SIDE = 64;

/**
 * Manifest (version 1): `{ version, scale, file, marks: [{ id, width, height, pixels, atlas }] }`. `pixels` lists
 * width * height values: -1 for transparent, otherwise the packed RGB565 value the native game writes there.
 * Returns null for anything malformed, so a bad farm keeps every native mark.
 */
export function parseFrameMarkManifest(value: unknown, atlasWidth: number, atlasHeight: number): FrameMarkSet | null {
  if (!value || typeof value !== 'object') return null;
  const manifest = value as { version?: unknown; scale?: unknown; marks?: unknown };
  const scale = manifest.scale;
  if (
    manifest.version !== 1 ||
    typeof scale !== 'number' ||
    !Number.isInteger(scale) ||
    scale < 1 ||
    scale > 16 ||
    !Number.isInteger(atlasWidth) ||
    !Number.isInteger(atlasHeight) ||
    atlasWidth < 1 ||
    atlasHeight < 1 ||
    !Array.isArray(manifest.marks) ||
    !manifest.marks.length ||
    manifest.marks.length > MAX_TEMPLATES
  )
    return null;
  const templates: FrameMarkTemplate[] = [];
  for (const entry of manifest.marks as unknown[]) {
    if (!entry || typeof entry !== 'object') return null;
    const mark = entry as { id?: unknown; width?: unknown; height?: unknown; pixels?: unknown; atlas?: unknown };
    const { width, height, pixels, atlas } = mark;
    if (
      typeof mark.id !== 'string' ||
      !/^[a-z0-9_-]{1,64}$/i.test(mark.id) ||
      typeof width !== 'number' ||
      typeof height !== 'number' ||
      !Number.isInteger(width) ||
      !Number.isInteger(height) ||
      width < 1 ||
      height < 1 ||
      width > MAX_TEMPLATE_SIDE ||
      height > MAX_TEMPLATE_SIDE ||
      !Array.isArray(pixels) ||
      pixels.length !== width * height ||
      !atlas ||
      typeof atlas !== 'object'
    )
      return null;
    const rect = atlas as Partial<FrameMarkRect>;
    if (
      ![rect.x, rect.y, rect.width, rect.height].every((v) => typeof v === 'number' && Number.isInteger(v)) ||
      rect.x! < 0 ||
      rect.y! < 0 ||
      rect.width !== width * scale ||
      rect.height !== height * scale ||
      rect.x! + rect.width! > atlasWidth ||
      rect.y! + rect.height! > atlasHeight
    )
      return null;
    const rgb565 = new Uint16Array(width * height),
      opaque = new Uint8Array(width * height),
      offsets: number[] = [];
    for (let index = 0; index < pixels.length; index++) {
      const pixel = pixels[index] as unknown;
      if (pixel === -1) continue;
      if (typeof pixel !== 'number' || !Number.isInteger(pixel) || pixel < 0 || pixel > 0xffff) return null;
      rgb565[index] = pixel;
      opaque[index] = 1;
      offsets.push(index);
    }
    if (!offsets.length) return null;
    templates.push({
      id: mark.id,
      width,
      height,
      rgb565,
      opaque,
      opaqueOffsets: Int32Array.from(offsets),
      atlas: { x: rect.x!, y: rect.y!, width: rect.width!, height: rect.height! },
    });
  }
  return { scale, atlasWidth, atlasHeight, templates };
}

/** Lookup by the colour of each template's first opaque pixel, prepared once per set. */
export interface FrameMarkIndex {
  readonly set: FrameMarkSet;
  /** Templates whose first opaque pixel has this colour, larger (more opaque pixels) first. */
  readonly byColour: Map<number, number[]>;
  readonly seed: Uint8Array;
}

export function indexFrameMarks(set: FrameMarkSet): FrameMarkIndex {
  const byColour = new Map<number, number[]>();
  const seed = new Uint8Array(65536);
  set.templates.forEach((template, index) => {
    const colour = template.rgb565[template.opaqueOffsets[0]!]!;
    seed[colour] = 1;
    const list = byColour.get(colour) ?? [];
    list.push(index);
    byColour.set(colour, list);
  });
  for (const list of byColour.values())
    list.sort((a, b) => set.templates[b]!.opaqueOffsets.length - set.templates[a]!.opaqueOffsets.length);
  return { set, byColour, seed };
}

/**
 * Exact occurrences of every template entirely inside `rect` (clamped to the picture). A pixel equal to `key` is
 * never part of a mark. Accepted matches never share an opaque pixel; larger templates are preferred.
 */
export function findFrameMarks(
  pixels: Uint16Array,
  width: number,
  height: number,
  rect: FrameMarkRect,
  index: FrameMarkIndex,
  key: number,
): FrameMarkMatch[] {
  const x0 = Math.max(0, Math.trunc(rect.x)),
    y0 = Math.max(0, Math.trunc(rect.y));
  const x1 = Math.min(width, Math.trunc(rect.x + rect.width)),
    y1 = Math.min(height, Math.trunc(rect.y + rect.height));
  if (x1 <= x0 || y1 <= y0 || pixels.length < width * height) return [];
  const { seed, byColour, set } = index;
  const candidates: FrameMarkMatch[] = [];
  for (let y = y0; y < y1; y++) {
    const row = y * width;
    for (let x = x0; x < x1; x++) {
      const pixel = pixels[row + x]!;
      if (pixel === key || !seed[pixel]) continue;
      for (const templateIndex of byColour.get(pixel)!) {
        const template = set.templates[templateIndex]!;
        const first = template.opaqueOffsets[0]!;
        const ox = x - (first % template.width),
          oy = y - Math.floor(first / template.width);
        if (ox < x0 || oy < y0 || ox + template.width > x1 || oy + template.height > y1) continue;
        if (matchesAt(pixels, width, template, ox, oy)) candidates.push({ template: templateIndex, x: ox, y: oy });
      }
    }
  }
  if (candidates.length < 2) return candidates;
  // Larger marks first; a candidate sharing an opaque pixel with an accepted one is part of it (or overlapped).
  candidates.sort(
    (a, b) =>
      set.templates[b.template]!.opaqueOffsets.length - set.templates[a.template]!.opaqueOffsets.length ||
      a.y - b.y ||
      a.x - b.x,
  );
  const claimed = new Set<number>();
  const accepted: FrameMarkMatch[] = [];
  for (const match of candidates) {
    const template = set.templates[match.template]!;
    let free = true;
    for (const offset of template.opaqueOffsets) {
      const at = (match.y + Math.floor(offset / template.width)) * width + match.x + (offset % template.width);
      if (claimed.has(at)) {
        free = false;
        break;
      }
    }
    if (!free) continue;
    for (const offset of template.opaqueOffsets)
      claimed.add((match.y + Math.floor(offset / template.width)) * width + match.x + (offset % template.width));
    accepted.push(match);
  }
  return accepted;
}

function matchesAt(pixels: Uint16Array, width: number, template: FrameMarkTemplate, ox: number, oy: number): boolean {
  const { opaqueOffsets, rgb565 } = template;
  for (let i = 0; i < opaqueOffsets.length; i++) {
    const offset = opaqueOffsets[i]!;
    const tx = offset % template.width,
      ty = (offset - tx) / template.width;
    if (pixels[(oy + ty) * width + ox + tx] !== rgb565[offset]) return false;
  }
  return true;
}

/** Write `key` over every opaque pixel of each match (in place: callers pass their own copy). */
export function clearFrameMarks(
  pixels: Uint16Array,
  width: number,
  matches: readonly FrameMarkMatch[],
  set: FrameMarkSet,
  key: number,
): void {
  for (const match of matches) {
    const template = set.templates[match.template]!;
    for (const offset of template.opaqueOffsets) {
      const tx = offset % template.width,
        ty = (offset - tx) / template.width;
      pixels[(match.y + ty) * width + match.x + tx] = key;
    }
  }
}

/** Two triangles per match: clip-space position over the native box and atlas texture coordinates. */
export function frameMarkVertices(
  matches: readonly FrameMarkMatch[],
  set: FrameMarkSet,
  frameWidth: number,
  frameHeight: number,
): Float32Array {
  const values = new Float32Array(matches.length * 24);
  let at = 0;
  for (const match of matches) {
    const template = set.templates[match.template]!;
    const left = (match.x / frameWidth) * 2 - 1,
      right = ((match.x + template.width) / frameWidth) * 2 - 1;
    const top = 1 - (match.y / frameHeight) * 2,
      bottom = 1 - ((match.y + template.height) / frameHeight) * 2;
    const u0 = template.atlas.x / set.atlasWidth,
      u1 = (template.atlas.x + template.atlas.width) / set.atlasWidth;
    const v0 = template.atlas.y / set.atlasHeight,
      v1 = (template.atlas.y + template.atlas.height) / set.atlasHeight;
    for (const [x, y, u, v] of [
      [left, top, u0, v0],
      [right, top, u1, v0],
      [left, bottom, u0, v1],
      [left, bottom, u0, v1],
      [right, top, u1, v0],
      [right, bottom, u1, v1],
    ] as const) {
      values[at++] = x;
      values[at++] = y;
      values[at++] = u;
      values[at++] = v;
    }
  }
  return values;
}

/**
 * Per-picture replacement state with a reused clean buffer. `prepare` returns the clean picture (or null when no mark
 * matched) and keeps the matches for `vertices`. The same source picture is never rescanned.
 */
export class FrameMarkReplacer {
  private readonly index: FrameMarkIndex;
  private source: Uint16Array | null = null;
  private clean: Uint16Array | null = null;
  private result: Uint16Array | null = null;
  matches: FrameMarkMatch[] = [];

  constructor(readonly set: FrameMarkSet) {
    this.index = indexFrameMarks(set);
  }

  prepare(pixels: Uint16Array, width: number, height: number, rect: FrameMarkRect, key: number): Uint16Array | null {
    if (this.source === pixels) return this.result;
    this.source = pixels;
    this.matches = findFrameMarks(pixels, width, height, rect, this.index, key);
    if (!this.matches.length) return (this.result = null);
    if (!this.clean || this.clean.length !== pixels.length) this.clean = new Uint16Array(pixels.length);
    this.clean.set(pixels);
    clearFrameMarks(this.clean, width, this.matches, this.set, key);
    return (this.result = this.clean);
  }

  reset(): void {
    this.source = null;
    this.result = null;
    this.matches = [];
  }
}

/** WebGL2 pass drawing matched art with premultiplied alpha; owns its program, buffer and atlas texture. */
export class FrameMarkPass {
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private texture: WebGLTexture | null = null;
  private atlasLocation: WebGLUniformLocation | null = null;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    atlas: TexImageSource,
  ) {
    const shaders: WebGLShader[] = [];
    const previousActive = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
    gl.activeTexture(gl.TEXTURE4);
    const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null;
    // Drain errors left by earlier unrelated calls (bounded), so the allocation check below sees only its own.
    for (let i = 0; i < 8 && gl.getError() !== gl.NO_ERROR; i++);
    try {
      this.program = gl.createProgram();
      this.buffer = gl.createBuffer();
      this.vao = gl.createVertexArray();
      this.texture = gl.createTexture();
      if (!this.program || !this.buffer || !this.vao || !this.texture) throw new Error('Mark resources unavailable');
      for (const [kind, source] of [
        [
          gl.VERTEX_SHADER,
          `#version 300 es
          layout(location = 0) in vec2 position;
          layout(location = 1) in vec2 texel;
          out vec2 uv;
          void main() { gl_Position = vec4(position, 0.0, 1.0); uv = texel; }`,
        ],
        [
          gl.FRAGMENT_SHADER,
          // A small negative bias keeps edges crisp when the art is shown below its own resolution.
          `#version 300 es
          precision highp float;
          uniform sampler2D atlas;
          in vec2 uv;
          out vec4 colour;
          void main() { colour = texture(atlas, uv, -0.5); }`,
        ],
      ] as const) {
        const shader = gl.createShader(kind);
        if (!shader) throw new Error('Mark shader unavailable');
        shaders.push(shader);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Mark shader failed');
        gl.attachShader(this.program, shader);
      }
      gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('Mark program failed');
      this.atlasLocation = gl.getUniformLocation(this.program, 'atlas');
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlas);
      } finally {
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      }
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error('Mark device unavailable');
    } catch (error) {
      this.destroy();
      throw error;
    } finally {
      for (const shader of shaders) gl.deleteShader(shader);
      gl.bindTexture(gl.TEXTURE_2D, previousTexture);
      gl.activeTexture(previousActive);
    }
  }

  /** Draw onto the bound framebuffer/viewport; the caller's viewport covers the whole native picture. */
  draw(vertices: Float32Array): void {
    if (!vertices.length) return;
    const { gl } = this;
    const previousVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING) as WebGLVertexArrayObject | null;
    const previousProgram = gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null;
    const previousBuffer = gl.getParameter(gl.ARRAY_BUFFER_BINDING) as WebGLBuffer | null;
    const previousActive = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
    gl.activeTexture(gl.TEXTURE4);
    const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null;
    try {
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
      gl.useProgram(this.program);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.atlasLocation, 4);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 4);
      gl.disable(gl.BLEND);
      // Only context loss is checked per draw: gl.getError() synchronises with the GPU process and would also report
      // errors of unrelated earlier calls. The allocation was validated once in the constructor.
      if (gl.isContextLost()) throw new Error('Mark device lost');
    } finally {
      gl.bindTexture(gl.TEXTURE_2D, previousTexture);
      gl.activeTexture(previousActive);
      gl.bindVertexArray(previousVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, previousBuffer);
      gl.useProgram(previousProgram);
    }
  }

  destroy(): void {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.program);
    this.gl.deleteTexture(this.texture);
    this.buffer = null;
    this.vao = null;
    this.program = null;
    this.texture = null;
  }
}
