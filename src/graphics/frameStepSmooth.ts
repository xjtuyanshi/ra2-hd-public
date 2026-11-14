/**
 * Step smoothing for flat-colour overlay pictures drawn at a non-integer display scale (generic, no game data).
 *
 * Thin diagonal marks (lines, brackets, pixel-font strokes, chevrons) show their source-pixel stair steps when a
 * picture is drawn nearest-neighbour at, say, 2.5 device pixels per source pixel. This rule refines every source
 * pixel into 4x4 sub-pixels by applying a 2x step twice; the renderer evaluates it per output pixel on the GPU and
 * averages a few sub-pixel samples. No colour is invented: existing colours only fill the corners of steps.
 *
 * The 2x step is Scale2x (AdvMAME2x) with one extra condition: a quadrant of pixel E takes the colour C of its two
 * orthogonal neighbours only when the diagonal pixel between them is not C. Plain Scale2x rounds the corners of
 * rectangles and outlines (health bars, boxes, vertical bracket edges); with the condition, axis-aligned shapes,
 * solid fills and checkerboards stay exactly as they were and only one-pixel steps of thin strokes are filled. Thick
 * diagonal edges stay stepped: the rule targets marks drawn with one-pixel lines. The second step fills only next to
 * a first-step fill (the quadrant itself, its two neighbours or the diagonal between them); otherwise the 2x blocks of
 * a one-pixel checkerboard would look like diagonal steps to it.
 *
 * The CPU functions define the rule for tests and offline boards; STEP_SMOOTH_GLSL is the same rule for a fragment
 * shader. Pixel values are opaque identifiers (for example packed RGB565); equality is the only operation used.
 */

/** One 2x step over a row-major picture; pixels outside the picture have the value `outside`. */
export function stepSmooth2x(pixels: ArrayLike<number>, width: number, height: number, outside: number): Uint32Array {
  return step(pixels, width, height, outside).values;
}

function step(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  outside: number,
  gate?: Uint8Array,
): { values: Uint32Array; filled: Uint8Array } {
  const out = new Uint32Array(width * height * 4);
  const filled = new Uint8Array(width * height * 4);
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height;
  const at = (x: number, y: number) => (inside(x, y) ? pixels[y * width + x] : outside);
  const gated = (x: number, y: number) => !gate || (inside(x, y) && gate[y * width + x] === 1);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const e = at(x, y);
      for (let qy = 0; qy < 2; qy++)
        for (let qx = 0; qx < 2; qx++) {
          const dx = qx ? 1 : -1,
            dy = qy ? 1 : -1;
          const vertical = at(x, y + dy),
            horizontal = at(x + dx, y);
          const fill =
            vertical === horizontal &&
            vertical !== at(x - dx, y) &&
            horizontal !== at(x, y - dy) &&
            at(x + dx, y + dy) !== vertical &&
            (gated(x, y) || gated(x, y + dy) || gated(x + dx, y) || gated(x + dx, y + dy));
          const index = (2 * y + qy) * 2 * width + 2 * x + qx;
          out[index] = fill ? vertical : e;
          filled[index] = fill ? 1 : 0;
        }
    }
  return { values: out, filled };
}

/**
 * Two 2x steps: the 4x4 sub-pixel picture the shader samples. The second step treats pixels outside the 2x picture
 * as `outside` too, which is what the shader's fetch returns there.
 */
export function stepSmooth4x(pixels: ArrayLike<number>, width: number, height: number, outside: number): Uint32Array {
  const first = step(pixels, width, height, outside);
  return step(first.values, width * 2, height * 2, outside, first.filled).values;
}

/**
 * GLSL ES 3.00 implementation. The including shader defines, before this source,
 *   `uint stepSmoothFetch(ivec2 p)`: the packed source value at integer pixel p, `outside` beyond the picture.
 * It provides
 *   `uint stepSmoothAt(vec2 s)`: the 4x sub-pixel value at source position s (pixel units; pixel (x, y) spans
 *      [x, x+1) x [y, y+1)), equal to stepSmooth4x at every position inside the picture;
 *   `bool stepSmoothFlat(ivec2 lo, ivec2 hi)`: true when every pixel of [lo-1, hi+1] has one value, so no sub-pixel of
 *      pixels lo..hi can differ from its source pixel (early exit for empty areas).
 */
export const STEP_SMOOTH_GLSL = `
uint stepSmoothWindow[16];
ivec2 stepSmoothBase;
uint stepSmoothWin(ivec2 p) {
  ivec2 d = p - stepSmoothBase;
  return stepSmoothWindow[d.y * 4 + d.x];
}
// Value of 2x sub-pixel p2 (2x units) and whether the first step filled it; its source neighbourhood lies inside the
// loaded 4x4 window.
uint stepSmoothHalf(ivec2 p2, out bool filled) {
  ivec2 c = p2 >> 1;
  ivec2 d = ((p2 & 1) << 1) - 1;
  uint v = stepSmoothWin(c + ivec2(0, d.y));
  uint h = stepSmoothWin(c + ivec2(d.x, 0));
  filled = v == h && v != stepSmoothWin(c - ivec2(d.x, 0)) && h != stepSmoothWin(c - ivec2(0, d.y)) &&
    stepSmoothWin(c + d) != v;
  return filled ? v : stepSmoothWin(c);
}
uint stepSmoothAt(vec2 s) {
  ivec2 p4 = ivec2(floor(s * 4.0));
  ivec2 p2 = p4 >> 1;
  stepSmoothBase = (p2 >> 1) - 2 + (p2 & 1);
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++) stepSmoothWindow[y * 4 + x] = stepSmoothFetch(stepSmoothBase + ivec2(x, y));
  ivec2 d = ((p4 & 1) << 1) - 1;
  bool fe, fv, fh, fc, unused;
  uint e = stepSmoothHalf(p2, fe);
  uint v = stepSmoothHalf(p2 + ivec2(0, d.y), fv);
  uint h = stepSmoothHalf(p2 + ivec2(d.x, 0), fh);
  bool fill = v == h && v != stepSmoothHalf(p2 - ivec2(d.x, 0), unused) &&
    h != stepSmoothHalf(p2 - ivec2(0, d.y), unused) && stepSmoothHalf(p2 + d, fc) != v && (fe || fv || fh || fc);
  return fill ? v : e;
}
bool stepSmoothFlat(ivec2 lo, ivec2 hi) {
  uint first = stepSmoothFetch(lo - 1);
  for (int y = lo.y - 1; y <= hi.y + 1; y++)
    for (int x = lo.x - 1; x <= hi.x + 1; x++)
      if (stepSmoothFetch(ivec2(x, y)) != first) return false;
  return true;
}
`;

/**
 * Per-source-pixel "may change" flags (R8: 1 when the pixel's 3x3 neighbourhood is not one value), computed on the GPU
 * once per picture so the per-output-pixel shader can skip flat areas with a few flag reads instead of a 4x4 window
 * of source reads. A flat 3x3 neighbourhood implies every 4x sub-pixel of that pixel equals it (fills need a differing
 * neighbour, second-step fills a first-step fill next to them), so the skip is exact.
 *
 * The packed source must be bound as an unsigned integer texture on `sourceUnit`. update() leaves its own framebuffer
 * unbound and restores the caller's program, vertex array, viewport and active texture; the caller rebinds anything
 * else it needs. Throws when the device cannot provide the target (the caller then keeps nearest output).
 */
export class StepSmoothFlags {
  private program: WebGLProgram | null;
  private texture: WebGLTexture | null;
  private framebuffer: WebGLFramebuffer | null;
  private vao: WebGLVertexArrayObject | null;
  private width = 0;
  private height = 0;
  private readonly keyLocation: WebGLUniformLocation | null;
  private readonly rectLocation: WebGLUniformLocation | null;
  private readonly sourceLocation: WebGLUniformLocation | null;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const shaders: WebGLShader[] = [];
    try {
      this.program = gl.createProgram();
      this.texture = gl.createTexture();
      this.framebuffer = gl.createFramebuffer();
      this.vao = gl.createVertexArray();
      if (!this.program || !this.texture || !this.framebuffer || !this.vao) throw new Error('Step flags unavailable');
      for (const [kind, source] of [
        [
          gl.VERTEX_SHADER,
          `#version 300 es
          void main() { gl_Position = vec4(vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0, 0.0, 1.0); }`,
        ],
        [
          gl.FRAGMENT_SHADER,
          `#version 300 es
          precision highp float;
          precision highp int;
          uniform highp usampler2D packedFrame;
          uniform ivec4 keyTexels;
          uniform uint keyPacked;
          out vec4 flag;
          uint stepSmoothFetch(ivec2 p) {
            if (any(lessThan(p, keyTexels.xy)) || any(greaterThanEqual(p, keyTexels.zw))) return keyPacked;
            return texelFetch(packedFrame, p, 0).r;
          }
          ${STEP_SMOOTH_GLSL}
          void main() {
            ivec2 c = ivec2(gl_FragCoord.xy);
            flag = vec4(stepSmoothFlat(c, c) ? 0.0 : 1.0);
          }`,
        ],
      ] as const) {
        const shader = gl.createShader(kind);
        if (!shader) throw new Error('Step flag shader unavailable');
        shaders.push(shader);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Step flag shader failed');
        gl.attachShader(this.program, shader);
      }
      gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('Step flag program failed');
      this.keyLocation = gl.getUniformLocation(this.program, 'keyPacked');
      this.rectLocation = gl.getUniformLocation(this.program, 'keyTexels');
      this.sourceLocation = gl.getUniformLocation(this.program, 'packedFrame');
    } catch (error) {
      this.destroy();
      throw error;
    } finally {
      for (const shader of shaders) gl.deleteShader(shader);
    }
  }

  /** Recompute the flags for the bound source picture; key rectangle in texels [x0, y0, x1, y1). */
  update(sourceUnit: number, width: number, height: number, keyTexels: readonly number[], keyPacked: number): void {
    const { gl } = this;
    if (!this.program || !this.texture || !this.framebuffer) throw new Error('Step flags destroyed');
    const previousProgram = gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null;
    const previousVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING) as WebGLVertexArrayObject | null;
    const previousActive = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
    const previousViewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
    gl.activeTexture(gl.TEXTURE0 + sourceUnit);
    const previousTexture = gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null;
    try {
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      if (width !== this.width || height !== this.height) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
          throw new Error('Step flag target unavailable');
        this.width = width;
        this.height = height;
      }
      gl.bindTexture(gl.TEXTURE_2D, previousTexture);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
      gl.viewport(0, 0, width, height);
      gl.useProgram(this.program);
      gl.bindVertexArray(this.vao);
      gl.uniform1i(this.sourceLocation, sourceUnit);
      gl.uniform4i(this.rectLocation, keyTexels[0], keyTexels[1], keyTexels[2], keyTexels[3]);
      gl.uniform1ui(this.keyLocation, keyPacked & 0xffff);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (gl.isContextLost()) throw new Error('Step flag device lost');
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindVertexArray(previousVao);
      gl.useProgram(previousProgram);
      gl.bindTexture(gl.TEXTURE_2D, previousTexture);
      gl.activeTexture(previousActive);
      gl.viewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
    }
  }

  /** Bind the flags (R8, nearest) on a texture unit for the drawing pass. */
  bind(unit: number): void {
    const { gl } = this;
    const previousActive = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.activeTexture(previousActive);
  }

  destroy(): void {
    this.gl.deleteProgram(this.program);
    this.gl.deleteTexture(this.texture);
    this.gl.deleteFramebuffer(this.framebuffer);
    this.gl.deleteVertexArray(this.vao);
    this.program = null;
    this.texture = null;
    this.framebuffer = null;
    this.vao = null;
  }
}
