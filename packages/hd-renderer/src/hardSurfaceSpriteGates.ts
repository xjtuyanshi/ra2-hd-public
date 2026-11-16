/** Offline, camera-independent sprite measurements. All distances are native pixels. */
export interface SpriteRgb {
  width: number;
  height: number;
  /** Interleaved display-sRGB bytes; alpha is supplied separately. */
  rgb: ArrayLike<number>;
}
export type PixelMask = Uint8Array;
export interface SpriteEdgeOptions {
  deDitherMethod: 'median' | 'bilateral';
  medianRadius: number;
  medianIterations: number;
  bilateralSpatialSigma: number;
  bilateralDeltaE: number;
  mergeDeltaE: number;
  minRegionArea: number;
  boundaryDeltaE: number;
  boundaryLuminanceStep: number;
  structuralMinLength: number;
  normalRadius: number;
}
export const defaultSpriteEdgeOptions: Readonly<SpriteEdgeOptions> = {
  deDitherMethod: 'bilateral',
  medianRadius: 2,
  medianIterations: 3,
  bilateralSpatialSigma: 2,
  bilateralDeltaE: 10,
  mergeDeltaE: 8,
  minRegionArea: 6,
  boundaryDeltaE: 10,
  boundaryLuminanceStep: 12,
  structuralMinLength: 6,
  normalRadius: 3,
};
function dimensions(width: number, height: number, ...arrays: ArrayLike<number>[]): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1)
    throw new Error('Invalid raster dimensions');
  if (arrays.some((a) => a.length !== width * height)) throw new Error('Raster dimensions do not match');
}
export function median(values: readonly number[]): number {
  if (!values.length) throw new Error('Median requires samples');
  const a = [...values].sort((x, y) => x - y),
    m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
/** Channel-wise 3x3 median by default; transparent neighbours never contaminate a visible pixel. */
export function deDitherSprite(
  image: SpriteRgb,
  alpha: ArrayLike<number>,
  radius = 1,
  classes?: ArrayLike<number>,
): Float64Array {
  const { width: w, height: h, rgb } = image;
  dimensions(w, h, alpha);
  if (classes) dimensions(w, h, classes);
  if (rgb.length !== w * h * 3 || !Number.isInteger(radius) || radius < 0) throw new Error('Invalid median input');
  const out = new Float64Array(rgb.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!alpha[i]) continue;
      for (let c = 0; c < 3; c++) {
        const values: number[] = [];
        for (let yy = Math.max(0, y - radius); yy <= Math.min(h - 1, y + radius); yy++)
          for (let xx = Math.max(0, x - radius); xx <= Math.min(w - 1, x + radius); xx++)
            if (alpha[yy * w + xx] && (!classes || classes[yy * w + xx] === classes[i]))
              values.push(rgb[3 * (yy * w + xx) + c]);
        out[3 * i + c] = median(values);
      }
    }
  return out;
}
export function dilateMask(mask: ArrayLike<number>, w: number, h: number, radius: number): PixelMask {
  dimensions(w, h, mask);
  if (!Number.isInteger(radius) || radius < 0) throw new Error('Invalid pixel tolerance');
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let yy = Math.max(0, y - radius); yy <= Math.min(h - 1, y + radius); yy++)
        for (let xx = Math.max(0, x - radius); xx <= Math.min(w - 1, x + radius); xx++) out[yy * w + xx] = 1;
    }
  return out;
}
export function erodeMask(mask: ArrayLike<number>, w: number, h: number, radius = 1): PixelMask {
  dimensions(w, h, mask);
  if (!Number.isInteger(radius) || radius < 0) throw new Error('Invalid erosion radius');
  const out = new Uint8Array(w * h);
  for (let y = radius; y < h - radius; y++)
    for (let x = radius; x < w - radius; x++) {
      let ok = true;
      for (let yy = y - radius; yy <= y + radius; yy++)
        for (let xx = x - radius; xx <= x + radius; xx++) if (!mask[yy * w + xx]) ok = false;
      out[y * w + x] = +ok;
    }
  return out;
}
export function maskBoundary(mask: ArrayLike<number>, w: number, h: number): PixelMask {
  const eroded = erodeMask(mask, w, h);
  return Uint8Array.from(mask, (v, i) => +(!!v && !eroded[i]));
}
/** Eight-connected components; pixel lists make every unsupported segment reviewable. */
export function maskSegments(mask: ArrayLike<number>, w: number, h: number): number[][] {
  dimensions(w, h, mask);
  const seen = new Uint8Array(w * h),
    out: number[][] = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    const queue = [i];
    seen[i] = 1;
    for (let k = 0; k < queue.length; k++) {
      const p = queue[k],
        x = p % w,
        y = Math.floor(p / w);
      for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
        for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++) {
          const q = yy * w + xx;
          if (mask[q] && !seen[q]) {
            seen[q] = 1;
            queue.push(q);
          }
        }
    }
    out.push(queue);
  }
  return out;
}
/** Sobel L2, four-direction nonmaximum suppression, then eight-connected hysteresis. No hidden blur. */
export function spriteCanny(gray: ArrayLike<number>, w: number, h: number, low: number, high: number): PixelMask {
  dimensions(w, h, gray);
  if (!(low >= 0 && high >= low)) throw new Error('Invalid Canny thresholds');
  const mag = new Float64Array(w * h),
    direction = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] + gray[i - w + 1] - 2 * gray[i - 1] + 2 * gray[i + 1] - gray[i + w - 1] + gray[i + w + 1];
      const gy =
        -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
      mag[i] = Math.hypot(gx, gy);
      direction[i] = Math.round(((Math.atan2(gy, gx) * 180) / Math.PI + 180) / 45) % 4;
    }
  const thin = new Float64Array(w * h),
    offsets = [1, w + 1, w, w - 1];
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x,
        d = offsets[direction[i]];
      if (mag[i] >= mag[i - d] && mag[i] > mag[i + d]) thin[i] = mag[i];
    }
  const out = new Uint8Array(w * h),
    queue: number[] = [];
  for (let i = 0; i < out.length; i++)
    if (thin[i] > 0 && thin[i] >= high) {
      out[i] = 1;
      queue.push(i);
    }
  for (let k = 0; k < queue.length; k++) {
    const p = queue[k],
      x = p % w,
      y = Math.floor(p / w);
    for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
      for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++) {
        const q = yy * w + xx;
        if (!out[q] && thin[q] > 0 && thin[q] >= low) {
          out[q] = 1;
          queue.push(q);
        }
      }
  }
  return out;
}
export function labToSrgb(lab: readonly number[]): [number, number, number] {
  const fy = (lab[0] + 16) / 116,
    fx = fy + lab[1] / 500,
    fz = fy - lab[2] / 200,
    delta = 6 / 29;
  const inverse = (v: number) => (v > delta ? v * v * v : 3 * delta * delta * (v - 4 / 29));
  const x = inverse(fx) * 0.95047,
    y = inverse(fy),
    z = inverse(fz) * 1.08883;
  const values = [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
  return values.map((v) =>
    Math.max(0, Math.min(255, 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055))),
  ) as [number, number, number];
}
/** Lab bilateral filtering preserves coherent narrow high-contrast rails while averaging local palette dither. */
export function bilateralSprite(
  image: SpriteRgb,
  alpha: ArrayLike<number>,
  radius: number,
  spatialSigma: number,
  rangeDeltaE: number,
  classes?: ArrayLike<number>,
): Float64Array {
  const { width: w, height: h, rgb } = image;
  dimensions(w, h, alpha, ...(classes ? [classes] : []));
  if (rgb.length !== w * h * 3 || !Number.isInteger(radius) || radius < 1 || !(spatialSigma > 0) || !(rangeDeltaE > 0))
    throw new Error('Invalid bilateral input');
  const labs = Array.from({ length: w * h }, (_, i) => srgbToLab([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]])),
    out = new Float64Array(rgb.length);
  const spatial = new Map<string, number>();
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++)
      spatial.set(`${dx}:${dy}`, Math.exp(-(dx * dx + dy * dy) / (2 * spatialSigma * spatialSigma)));
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!alpha[i]) continue;
      const sum = [0, 0, 0];
      let weights = 0;
      for (let yy = Math.max(0, y - radius); yy <= Math.min(h - 1, y + radius); yy++)
        for (let xx = Math.max(0, x - radius); xx <= Math.min(w - 1, x + radius); xx++) {
          const j = yy * w + xx;
          if (!alpha[j] || (classes && classes[j] !== classes[i])) continue;
          const d = deltaE76(labs[i], labs[j]),
            weight = spatial.get(`${xx - x}:${yy - y}`)! * Math.exp((-d * d) / (2 * rangeDeltaE * rangeDeltaE));
          weights += weight;
          for (let c = 0; c < 3; c++) sum[c] += labs[j][c] * weight;
        }
      const colour = labToSrgb(sum.map((v) => v / weights));
      for (let c = 0; c < 3; c++) out[3 * i + c] = colour[c];
    }
  return out;
}
export interface SpriteRegion {
  id: number;
  area: number;
  medianRgb: number[];
  medianLab: number[];
  sourceIndices: number[];
  semanticClasses: number[];
}
/** Zhang-Suen thinning. Endpoints survive; no components are silently dropped. */
export function thinSpriteEdges(mask: ArrayLike<number>, w: number, h: number): PixelMask {
  dimensions(w, h, mask);
  const a = Uint8Array.from(mask, (v) => +!!v);
  let changed = true;
  while (changed) {
    changed = false;
    for (let pass = 0; pass < 2; pass++) {
      const remove: number[] = [];
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (!a[i]) continue;
          const get = (xx: number, yy: number) => (xx < 0 || yy < 0 || xx >= w || yy >= h ? 0 : a[yy * w + xx]);
          const p = [
            get(x, y - 1),
            get(x + 1, y - 1),
            get(x + 1, y),
            get(x + 1, y + 1),
            get(x, y + 1),
            get(x - 1, y + 1),
            get(x - 1, y),
            get(x - 1, y - 1),
          ];
          const count = p.reduce((s, v) => s + v, 0);
          let transitions = 0;
          for (let k = 0; k < 8; k++) if (p[k] === 0 && p[(k + 1) % 8] === 1) transitions++;
          if (count < 2 || count > 6 || transitions !== 1) continue;
          if (
            pass === 0
              ? p[0] * p[2] * p[4] === 0 && p[2] * p[4] * p[6] === 0
              : p[0] * p[2] * p[6] === 0 && p[0] * p[4] * p[6] === 0
          )
            remove.push(i);
        }
      for (const i of remove) a[i] = 0;
      if (remove.length) changed = true;
    }
  }
  return a;
}
/** Boundary normals in radians modulo pi, from the local geometric tangent. Unknown directions remain NaN. */
export function spriteEdgeNormals(mask: ArrayLike<number>, w: number, h: number, radius = 3): Float64Array {
  dimensions(w, h, mask);
  if (!Number.isInteger(radius) || radius < 1) throw new Error('Invalid edge normal radius');
  const normals = new Float64Array(w * h).fill(NaN);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const points: number[][] = [];
      for (let yy = Math.max(0, y - radius); yy <= Math.min(h - 1, y + radius); yy++)
        for (let xx = Math.max(0, x - radius); xx <= Math.min(w - 1, x + radius); xx++)
          if (mask[yy * w + xx]) points.push([xx - x, yy - y]);
      if (points.length < 3) continue;
      const mx = points.reduce((s, p) => s + p[0], 0) / points.length,
        my = points.reduce((s, p) => s + p[1], 0) / points.length;
      let xx = 0,
        yy = 0,
        xy = 0;
      for (const p of points) {
        const a = p[0] - mx,
          b = p[1] - my;
        xx += a * a;
        yy += b * b;
        xy += a * b;
      }
      if (xx + yy < 1e-9 || Math.hypot(xx - yy, 2 * xy) / (xx + yy) < 0.1) continue;
      const tangent = 0.5 * Math.atan2(2 * xy, xx - yy);
      normals[i] = (tangent + Math.PI * 1.5) % Math.PI;
    }
  return normals;
}
export function orientedEdgeAgreement(
  predicted: ArrayLike<number>,
  predictedNormals: ArrayLike<number>,
  reference: ArrayLike<number>,
  referenceNormals: ArrayLike<number>,
  w: number,
  h: number,
  k = 1,
  angleDegrees = 30,
) {
  dimensions(w, h, predicted, predictedNormals, reference, referenceNormals);
  if (!Number.isInteger(k) || k < 0 || !(angleDegrees >= 0 && angleDegrees <= 90))
    throw new Error('Invalid oriented matching tolerance');
  const failures = new Uint8Array(w * h);
  let total = 0,
    matched = 0,
    unknownNormals = 0;
  const limit = (angleDegrees * Math.PI) / 180;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!predicted[i]) continue;
      total++;
      const angle = predictedNormals[i];
      let ok = false;
      if (!Number.isFinite(angle)) unknownNormals++;
      else
        for (let yy = Math.max(0, y - k); yy <= Math.min(h - 1, y + k) && !ok; yy++)
          for (let xx = Math.max(0, x - k); xx <= Math.min(w - 1, x + k); xx++) {
            const q = yy * w + xx;
            if (!reference[q] || !Number.isFinite(referenceNormals[q])) continue;
            const raw = Math.abs(angle - referenceNormals[q]) % Math.PI,
              diff = Math.min(raw, Math.PI - raw);
            if (diff <= limit + 1e-12) {
              ok = true;
              break;
            }
          }
      if (ok) matched++;
      else failures[i] = 1;
    }
  return {
    fraction: total ? matched / total : null,
    total,
    matched,
    unknownNormals,
    failures,
    segments: maskSegments(failures, w, h),
  };
}
export function shiftSpriteEdges(
  mask: ArrayLike<number>,
  normals: ArrayLike<number>,
  w: number,
  h: number,
  dx: number,
  dy: number,
) {
  dimensions(w, h, mask, normals);
  if (!Number.isInteger(dx) || !Number.isInteger(dy)) throw new Error('Edge shifts must be integer pixels');
  const shifted = new Uint8Array(w * h),
    angles = new Float64Array(w * h).fill(NaN);
  let clipped = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (mask[y * w + x]) {
        const xx = x + dx,
          yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
          clipped++;
          continue;
        }
        shifted[yy * w + xx] = 1;
        angles[yy * w + xx] = normals[y * w + x];
      }
  return { mask: shifted, normals: angles, clipped };
}
/** Multichannel Sobel gradient of the region-median Lab image (Di Zenzo principal direction). */
export function spriteRegionGradientNormals(
  labels: ArrayLike<number>,
  regions: readonly SpriteRegion[],
  w: number,
  h: number,
): Float64Array {
  dimensions(w, h, labels);
  const out = new Float64Array(w * h).fill(NaN),
    sx = [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    sy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!labels[i]) continue;
      const centre = regions[labels[i] - 1].medianLab;
      let xx = 0,
        yy = 0,
        xy = 0;
      for (let c = 0; c < 3; c++) {
        let gx = 0,
          gy = 0,
          k = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++, k++) {
            const px = x + dx,
              py = y + dy,
              id = px >= 0 && py >= 0 && px < w && py < h ? labels[py * w + px] : 0;
            const value = id ? regions[id - 1].medianLab[c] : centre[c];
            gx += sx[k] * value;
            gy += sy[k] * value;
          }
        xx += gx * gx;
        yy += gy * gy;
        xy += gx * gy;
      }
      if (xx + yy > 1e-9) out[i] = (0.5 * Math.atan2(2 * xy, xx - yy) + Math.PI) % Math.PI;
    }
  return out;
}
export function spriteEdgeNullControls(input: {
  width: number;
  height: number;
  interior: ArrayLike<number>;
  interiorNormals: ArrayLike<number>;
  allEdges: ArrayLike<number>;
  allNormals: ArrayLike<number>;
  original: ArrayLike<number>;
  originalNormals: ArrayLike<number>;
}) {
  const { width: w, height: h, interior, interiorNormals, allEdges, allNormals, original, originalNormals } = input;
  const truePrecision = orientedEdgeAgreement(interior, interiorNormals, original, originalNormals, w, h).fraction;
  const trueRecall = orientedEdgeAgreement(original, originalNormals, allEdges, allNormals, w, h).fraction;
  const controls = [
    [-2, 0],
    [2, 0],
    [0, -2],
    [0, 2],
    [-3, -3],
    [-3, 3],
    [3, -3],
    [3, 3],
  ].map(([dx, dy]) => {
    const a = shiftSpriteEdges(interior, interiorNormals, w, h, dx, dy),
      b = shiftSpriteEdges(allEdges, allNormals, w, h, dx, dy);
    const precision = orientedEdgeAgreement(a.mask, a.normals, original, originalNormals, w, h),
      recall = orientedEdgeAgreement(original, originalNormals, b.mask, b.normals, w, h);
    return {
      shift: [dx, dy],
      precision: precision.fraction,
      recall: recall.fraction,
      predictedPixels: precision.total,
      matchedPrecisionPixels: precision.matched,
      matchedRecallPixels: recall.matched,
      clippedInteriorPixels: a.clipped,
      clippedAllEdgePixels: b.clipped,
    };
  });
  const values = controls.map((c) => c.precision).filter((v): v is number => v !== null);
  const worstShiftedPrecision = values.length ? Math.max(...values) : null;
  const trueMinusWorstShift =
    truePrecision !== null && worstShiftedPrecision !== null ? truePrecision - worstShiftedPrecision : null;
  return {
    truePrecision,
    trueRecall,
    controls,
    worstShiftedPrecision,
    trueMinusWorstShift,
    requiredMargin: 0.25,
    pass: values.length === controls.length && trueMinusWorstShift !== null && trueMinusWorstShift >= 0.25,
  };
}
/** The source set is defined by coherent regions, not by texture gradients or a raw Canny union. */
export function originalSpriteEdges(
  image: SpriteRgb,
  alpha: ArrayLike<number>,
  indices?: ArrayLike<number>,
  options: SpriteEdgeOptions = { ...defaultSpriteEdgeOptions },
  classes?: ArrayLike<number>,
) {
  const { width: w, height: h, rgb } = image,
    n = w * h;
  dimensions(w, h, alpha, ...(indices ? [indices] : []), ...(classes ? [classes] : []));
  for (const key of [
    'medianRadius',
    'medianIterations',
    'minRegionArea',
    'structuralMinLength',
    'normalRadius',
  ] as const)
    if (!Number.isInteger(options[key]) || options[key] < 1) throw new Error(`Invalid region option: ${key}`);
  for (const key of ['mergeDeltaE', 'boundaryDeltaE', 'boundaryLuminanceStep'] as const)
    if (!Number.isFinite(options[key]) || options[key] < 0) throw new Error(`Invalid region option: ${key}`);
  if (
    !['median', 'bilateral'].includes(options.deDitherMethod) ||
    !(options.bilateralSpatialSigma > 0) ||
    !(options.bilateralDeltaE > 0)
  )
    throw new Error('Invalid de-dither method');
  let clean: ReturnType<typeof deDitherSprite> = Float64Array.from(rgb);
  for (let pass = 0; pass < options.medianIterations; pass++)
    clean =
      options.deDitherMethod === 'median'
        ? deDitherSprite({ width: w, height: h, rgb: clean }, alpha, options.medianRadius, classes)
        : bilateralSprite(
            { width: w, height: h, rgb: clean },
            alpha,
            options.medianRadius,
            options.bilateralSpatialSigma,
            options.bilateralDeltaE,
            classes,
          );
  const labs = Array.from({ length: n }, (_, i) =>
    alpha[i] ? srgbToLab(Array.from(clean.slice(i * 3, i * 3 + 3))) : [0, 0, 0],
  );
  // Palette-aware seeds: quantize the filtered colour to observed original palette entries within the caller's class.
  const palette = new Map<string, { index: number; classId: number; rgb: number[]; lab: number[] }>();
  for (let i = 0; i < n; i++)
    if (alpha[i]) {
      const classId = classes?.[i] ?? 0,
        index = indices?.[i] ?? i,
        key = `${classId}:${index}`;
      if (!palette.has(key)) {
        const colour = [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]];
        palette.set(key, { index, classId, rgb: colour, lab: srgbToLab(colour) });
      }
    }
  const choices = [...palette.values()],
    quantized = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++)
    if (alpha[i]) {
      let best = Infinity,
        chosen = -1;
      const classId = classes?.[i] ?? 0;
      for (let j = 0; j < choices.length; j++)
        if (choices[j].classId === classId) {
          const d = deltaE76(labs[i], choices[j].lab);
          if (d < best) {
            best = d;
            chosen = j;
          }
        }
      quantized[i] = chosen;
    }
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const root = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const join = (a: number, b: number) => {
    a = root(a);
    b = root(b);
    if (a === b) return false;
    parent[Math.max(a, b)] = Math.min(a, b);
    return true;
  };
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!alpha[i]) continue;
      for (const j of [x + 1 < w ? i + 1 : -1, y + 1 < h ? i + w : -1])
        if (j >= 0 && alpha[j] && quantized[i] === quantized[j]) join(i, j);
    }
  const regionState = () => {
    const pixels = new Map<number, number[]>();
    for (let i = 0; i < n; i++)
      if (alpha[i]) {
        const id = root(i),
          list = pixels.get(id);
        if (list) list.push(i);
        else pixels.set(id, [i]);
      }
    const regions = new Map<number, { pixels: number[]; rgb: number[]; lab: number[]; classId: number }>();
    for (const [id, p] of pixels) {
      const colour = [0, 1, 2].map((c) => median(p.map((i) => clean[3 * i + c])));
      const counts = new Map<number, number>();
      for (const i of p) {
        const k = classes?.[i] ?? 0;
        counts.set(k, (counts.get(k) ?? 0) + 1);
      }
      const classId = [...counts].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
      regions.set(id, { pixels: p, rgb: colour, lab: srgbToLab(colour), classId });
    }
    const pairs = new Map<string, [number, number]>();
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!alpha[i]) continue;
        for (const j of [x + 1 < w ? i + 1 : -1, y + 1 < h ? i + w : -1])
          if (j >= 0 && alpha[j]) {
            const a = root(i),
              b = root(j);
            if (a !== b) {
              const pair = [Math.min(a, b), Math.max(a, b)] as [number, number];
              pairs.set(pair.join(':'), pair);
            }
          }
      }
    return { regions, pairs: [...pairs.values()] };
  };
  let mergePasses = 0;
  // Each iteration merges at least one pair or stops, so termination is bounded by the number of opaque pixels.
  while (mergePasses < n) {
    const state = regionState();
    let changed = false;
    const candidates = state.pairs
      .map(([a, b]) => ({ a, b, d: deltaE76(state.regions.get(a)!.lab, state.regions.get(b)!.lab) }))
      .sort((a, b) => a.d - b.d || a.a - b.a || a.b - b.b);
    // Recompute region medians after each merge; a chain of stale pair comparisons must not
    // erase a high-contrast face through a sequence of individually similar texture colours.
    for (const { a, b, d } of candidates) {
      const A = state.regions.get(a)!,
        B = state.regions.get(b)!;
      if (A.classId === B.classId && d <= options.mergeDeltaE) {
        changed = join(a, b);
        break;
      }
    }
    if (changed) {
      mergePasses++;
      continue;
    }
    // Merge one small region at a time, preferring its semantic class and nearest median colour.
    const small = [...state.regions]
      .filter(([, r]) => r.pixels.length < options.minRegionArea)
      .sort((a, b) => a[1].pixels.length - b[1].pixels.length || a[0] - b[0]);
    for (const [id, A] of small) {
      const neighbours = state.pairs.filter((p) => p.includes(id)).map((p) => (p[0] === id ? p[1] : p[0]));
      neighbours.sort(
        (a, b) =>
          +(state.regions.get(a)!.classId !== A.classId) - +(state.regions.get(b)!.classId !== A.classId) ||
          deltaE76(A.lab, state.regions.get(a)!.lab) - deltaE76(A.lab, state.regions.get(b)!.lab) ||
          a - b,
      );
      if (neighbours.length) {
        join(id, neighbours[0]);
        changed = true;
        break;
      }
    }
    if (!changed) break;
    mergePasses++;
  }
  const final = regionState(),
    labels = new Uint32Array(n),
    regions: SpriteRegion[] = [];
  for (const r of final.regions.values()) {
    const id = regions.length + 1;
    for (const i of r.pixels) labels[i] = id;
    regions.push({
      id,
      area: r.pixels.length,
      medianRgb: r.rgb,
      medianLab: r.lab,
      sourceIndices: [...new Set(r.pixels.map((i) => indices?.[i] ?? quantized[i]))].sort((a, b) => a - b),
      semanticClasses: [...new Set(r.pixels.map((i) => classes?.[i] ?? 0))].sort((a, b) => a - b),
    });
  }
  const pairMasks = new Map<string, PixelMask>(),
    inside = erodeMask(alpha, w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!alpha[i]) continue;
      for (const j of [x + 1 < w ? i + 1 : -1, y + 1 < h ? i + w : -1])
        if (j >= 0 && alpha[j] && labels[i] !== labels[j]) {
          const a = regions[labels[i] - 1],
            b = regions[labels[j] - 1];
          if (a.area < options.minRegionArea || b.area < options.minRegionArea) continue;
          const lum = (q: number[]) => q[0] * 0.2126 + q[1] * 0.7152 + q[2] * 0.0722;
          if (
            deltaE76(a.medianLab, b.medianLab) < options.boundaryDeltaE &&
            Math.abs(lum(a.medianRgb) - lum(b.medianRgb)) < options.boundaryLuminanceStep
          )
            continue;
          const key = [a.id, b.id].sort((a, b) => a - b).join(':');
          let mask = pairMasks.get(key);
          if (!mask) {
            mask = new Uint8Array(n);
            pairMasks.set(key, mask);
          }
          // Both incident source pixels are a boundary candidate; thinning below makes a single-pixel locus.
          if (inside[i]) mask[i] = 1;
          if (inside[j]) mask[j] = 1;
        }
    }
  const boundary = new Uint8Array(n);
  for (const mask of pairMasks.values()) {
    const thin = thinSpriteEdges(mask, w, h);
    for (const component of maskSegments(thin, w, h))
      if (component.length >= options.structuralMinLength) {
        for (const i of component) boundary[i] = 1;
      }
  }
  const structural = thinSpriteEdges(boundary, w, h);
  const finalSegments: { id: number; regions: number[]; pixels: number[] }[] = [];
  for (const pixels of maskSegments(structural, w, h)) {
    if (pixels.length < options.structuralMinLength) {
      for (const i of pixels) structural[i] = 0;
      continue;
    }
    const nearby = new Set<number>();
    for (const i of pixels) {
      const x = i % w,
        y = Math.floor(i / w);
      for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
        for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++)
          if (labels[yy * w + xx]) nearby.add(labels[yy * w + xx]);
    }
    finalSegments.push({ id: finalSegments.length + 1, regions: [...nearby].sort((a, b) => a - b), pixels });
  }
  const normals = spriteRegionGradientNormals(labels, regions, w, h);
  const near = dilateMask(structural, w, h, 1);
  let opaque = 0,
    covered = 0;
  for (let i = 0; i < n; i++)
    if (alpha[i]) {
      opaque++;
      covered += near[i];
    }
  return {
    deDithered: clean,
    regions,
    labels,
    structural,
    support: structural,
    normals,
    segments: finalSegments,
    coverage: opaque ? covered / opaque : null,
    opaquePixels: opaque,
    coveredOpaquePixels: covered,
    mergePasses,
  };
}

export function silhouetteIoU(reference: ArrayLike<number>, coverage: ArrayLike<number>, threshold = 0.5) {
  if (reference.length !== coverage.length || !(threshold > 0 && threshold <= 1))
    throw new Error('Invalid silhouette input');
  let intersection = 0,
    union = 0,
    missing = 0,
    extra = 0;
  for (let i = 0; i < reference.length; i++) {
    if (!Number.isFinite(coverage[i]) || coverage[i] < 0 || coverage[i] > 1)
      throw new Error('Coverage must be in [0,1]');
    const a = !!reference[i],
      b = coverage[i] >= threshold;
    intersection += +(a && b);
    union += +(a || b);
    missing += +(a && !b);
    extra += +(!a && b);
  }
  return { iou: union ? intersection / union : 1, intersection, union, missing, extra };
}
export function edgeAgreement(predicted: ArrayLike<number>, reference: ArrayLike<number>, w: number, h: number, k = 1) {
  dimensions(w, h, predicted, reference);
  const near = dilateMask(reference, w, h, k),
    failures = new Uint8Array(w * h);
  let total = 0,
    matched = 0;
  for (let i = 0; i < predicted.length; i++)
    if (predicted[i]) {
      total++;
      if (near[i]) matched++;
      else failures[i] = 1;
    }
  return { fraction: total ? matched / total : null, total, matched, failures, segments: maskSegments(failures, w, h) };
}
export function srgbToLab(rgb: readonly number[]): [number, number, number] {
  const [r, g, b] = rgb.map((v) => {
    const x = v / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  const xyz = [
    (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047,
    0.2126729 * r + 0.7151522 * g + 0.072175 * b,
    (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883,
  ].map((v) => (v > (6 / 29) ** 3 ? Math.cbrt(v) : v / (3 * (6 / 29) ** 2) + 4 / 29));
  return [116 * xyz[1] - 16, 500 * (xyz[0] - xyz[1]), 200 * (xyz[1] - xyz[2])];
}
export const deltaE76 = (a: readonly number[], b: readonly number[]) =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export interface FaceColourMeasurement {
  face: number;
  house: boolean;
  pixels: number;
  referencePixels: number;
  missingReferencePixels: number;
  originalMedian: number[] | null;
  renderedMean: number[];
  deltaE76: number | null;
}
/** Face IDs are geometric regions; an exact caller-supplied house mask splits paint without inventing a crease. */
export function faceColourMeasurements(
  original: SpriteRgb,
  originalAlpha: ArrayLike<number>,
  render: SpriteRgb,
  faceIds: ArrayLike<number>,
  coverage: ArrayLike<number>,
  house: ArrayLike<number>,
  minPixels = 6,
): FaceColourMeasurement[] {
  const { width: w, height: h } = original;
  dimensions(w, h, originalAlpha, faceIds, coverage, house);
  if (render.width !== w || render.height !== h || original.rgb.length !== w * h * 3 || render.rgb.length !== w * h * 3)
    throw new Error('Colour raster dimensions do not match');
  const groups = new Map<string, number[]>();
  for (let i = 0; i < w * h; i++)
    if (faceIds[i] > 0 && coverage[i] >= 0.5) {
      const key = `${faceIds[i]}:${+!!house[i]}`,
        list = groups.get(key);
      if (list) list.push(i);
      else groups.set(key, [i]);
    }
  const out: FaceColourMeasurement[] = [];
  for (const [key, ids] of groups) {
    if (ids.length < minPixels) continue;
    const [face, hm] = key.split(':').map(Number),
      ref = ids.filter((i) => !!originalAlpha[i]);
    const originalMedian = ref.length ? [0, 1, 2].map((c) => median(ref.map((i) => original.rgb[3 * i + c]))) : null;
    const renderedMean = [0, 1, 2].map((c) => ids.reduce((s, i) => s + render.rgb[3 * i + c], 0) / ids.length);
    out.push({
      face,
      house: !!hm,
      pixels: ids.length,
      referencePixels: ref.length,
      missingReferencePixels: ids.length - ref.length,
      originalMedian,
      renderedMean,
      deltaE76: originalMedian ? deltaE76(srgbToLab(originalMedian), srgbToLab(renderedMean)) : null,
    });
  }
  return out.sort((a, b) => a.face - b.face || +a.house - +b.house);
}
export function colourSummary(rows: readonly FaceColourMeasurement[]) {
  const values = rows
    .map((r) => r.deltaE76)
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);
  if (!values.length) return { faces: rows.length, measured: 0, mean: null, p90: null, worst: null };
  const p = (values.length - 1) * 0.9,
    lo = Math.floor(p),
    hi = Math.ceil(p);
  return {
    faces: rows.length,
    measured: values.length,
    mean: values.reduce((s, v) => s + v, 0) / values.length,
    p90: values[lo] + (values[hi] - values[lo]) * (p - lo),
    worst: values[values.length - 1],
  };
}
