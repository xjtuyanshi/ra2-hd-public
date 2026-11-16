import { describe, expect, it } from 'vitest';
import {
  deDitherSprite,
  originalSpriteEdges,
  silhouetteIoU,
  edgeAgreement,
  srgbToLab,
  deltaE76,
  faceColourMeasurements,
  maskSegments,
  thinSpriteEdges,
  spriteEdgeNormals,
  orientedEdgeAgreement,
  spriteEdgeNullControls,
  bilateralSprite,
  labToSrgb,
  spriteLocalEvidence,
  spriteLocalEvidenceNullControls,
  spriteDetailRoiMeasurements,
} from '../src/hardSurfaceSpriteGates';
import { projectSpritePoint, spriteGeometryEdges, type SpriteCamera } from '../src/hardSurfaceSpriteCamera';
const camera: SpriteCamera = { right: [1, 0, 0], up: [0, 1, 0], towards: [0, 0, 1], origin: [4, 5], pixelsPerWorld: 2 };
describe('native sprite gates on synthetic evidence', () => {
  it('requires local evidence to peak on the analytic fold, not beside it or on a uniform face', () => {
    const width = 40,
      height = 30,
      alpha = new Uint8Array(width * height).fill(1);
    const rgb = Uint8Array.from({ length: width * height * 3 }, (_, i) => (Math.floor(i / 3) % width < 20 ? 40 : 180));
    const runs = [{ id: 'wall', samples: Array.from({ length: 12 }, (_, k) => ({ x: 20, y: k + 8.5, normal: 0 })) }];
    const actual = spriteLocalEvidence({ width, height, rgb }, alpha, runs);
    expect(actual.segmentFraction).toBe(1);
    expect(actual.uniqueTotalPixels).toBe(12);
    expect(actual.uniquePixelFraction).toBe(1);
    expect(actual.segments[0].nearbyMaximumDeltaE76).toBeGreaterThan(8);
    expect(spriteLocalEvidence({ width, height, rgb }, alpha, runs, undefined, [3, 0]).segmentFraction).toBe(0);
    expect(
      spriteLocalEvidence({ width, height, rgb: new Uint8Array(rgb.length).fill(80) }, alpha, runs).segmentFraction,
    ).toBe(0);
    const missing = alpha.slice();
    missing.fill(0);
    expect(spriteLocalEvidence({ width, height, rgb }, missing, runs).segments[0].measurable).toBe(false);
  });
  it('keeps all eight local null controls and rejects a translation-invariant wall as meaningful', () => {
    const width = 40,
      height = 40,
      alpha = new Uint8Array(width * height).fill(1);
    const rgb = Uint8Array.from({ length: width * height * 3 }, (_, i) => (Math.floor(i / 3) % width < 20 ? 40 : 180));
    const runs = [{ id: 'wall', samples: Array.from({ length: 12 }, (_, k) => ({ x: 20, y: k + 12.5, normal: 0 })) }];
    const result = spriteLocalEvidenceNullControls({ width, height, rgb }, alpha, runs);
    expect(result.controls).toHaveLength(8);
    expect(result.actual.segmentFraction).toBe(1);
    expect(result.segmentLead).toBe(0);
    expect(result.meaningful).toBe(false);
    expect(result.pass).toBe(false);
  });
  it('measures ROI detail with the same distance and orientation rule and excludes other ROIs', () => {
    const width = 16,
      height = 16,
      source = new Uint8Array(256),
      rendered = new Uint8Array(256),
      normals = new Float64Array(256);
    for (let y = 3; y < 10; y++) {
      source[y * 16 + 5] = 1;
      rendered[y * 16 + 6] = 1;
    }
    rendered[14 * 16 + 14] = 1;
    const input = {
      width,
      height,
      source,
      sourceNormals: normals,
      rendered,
      renderedNormals: normals,
      rois: [{ id: 'wall', bounds: [2, 2, 11, 11] as const }],
    };
    expect(spriteDetailRoiMeasurements(input)[0]).toMatchObject({
      precision: 1,
      recall: 1,
      sourcePixels: 7,
      renderedPixels: 7,
    });
    expect(
      spriteDetailRoiMeasurements({ ...input, renderedNormals: new Float64Array(256).fill(Math.PI / 2) })[0].recall,
    ).toBe(0);
    expect(() => spriteDetailRoiMeasurements({ ...input, rois: [{ id: 'bad', bounds: [-1, 0, 5, 5] }] })).toThrow(
      'Invalid detail ROI',
    );
  });
  it('thresholds true area coverage at 0.5 and rejects mismatched or invalid buffers', () => {
    expect(silhouetteIoU([1, 1, 0], [0.5, 0.49, 0.51])).toEqual({
      iou: 1 / 3,
      intersection: 1,
      union: 3,
      missing: 1,
      extra: 1,
    });
    expect(() => silhouetteIoU([1], [NaN])).toThrow('Coverage');
    expect(() => silhouetteIoU([1], [])).toThrow();
  });
  it('uses Chebyshev tolerance and lists every failing connected segment', () => {
    const a = new Uint8Array(25),
      b = new Uint8Array(25);
    a[6] = a[23] = a[24] = 1;
    b[0] = 1;
    const q = edgeAgreement(a, b, 5, 5, 1);
    expect(q.fraction).toBe(1 / 3);
    expect(q.segments).toEqual([[23, 24]]);
    expect(maskSegments(a, 5, 5)).toHaveLength(2);
    expect(edgeAgreement(new Uint8Array(25), b, 5, 5).fraction).toBeNull();
  });
  it('removes single-pixel dither without using transparent RGB or altering source bytes', () => {
    const rgb = new Uint8Array(27).fill(100),
      a = new Uint8Array(9).fill(1);
    rgb[12] = 250;
    a[0] = 0;
    rgb[0] = 0;
    const clean = deDitherSprite({ width: 3, height: 3, rgb }, a);
    expect(clean[12]).toBe(100);
    expect(rgb[12]).toBe(250);
    expect(clean[0]).toBe(0);
  });
  it('keeps coherent palette-region boundaries and rejects sub-area texture fragments', () => {
    const w = 40,
      h = 30,
      rgb = new Uint8Array(w * h * 3),
      indices = new Uint8Array(w * h),
      alpha = new Uint8Array(w * h).fill(1);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x,
          right = x >= 20,
          d = (x + y) % 2 ? 8 : -8,
          base = right ? [180, 180, 185] : [95, 80, 55];
        for (let c = 0; c < 3; c++) rgb[3 * i + c] = base[c] + d;
        indices[i] = (right ? 4 : 0) + ((x + y) % 2);
      }
    const edges = originalSpriteEdges({ width: w, height: h, rgb }, alpha, indices);
    expect(edges.regions.every((r) => r.area >= 6)).toBe(true);
    expect(edges.coverage).toBeLessThan(0.4);
    expect(edges.segments.every((s) => s.pixels.length >= 6)).toBe(true);
    expect(edges.structural.some(Boolean)).toBe(true);
    expect(edges.structural[0]).toBe(0);
  });
  it('exposes an over-dense source rather than silently thinning its reference to a passing density', () => {
    const w = 40,
      h = 30,
      rgb = new Uint8Array(w * h * 3),
      indices = new Uint8Array(w * h),
      alpha = new Uint8Array(w * h).fill(1);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const index = Math.floor(x / 3) % 2;
        indices[y * w + x] = index;
        for (let c = 0; c < 3; c++) rgb[3 * (y * w + x) + c] = index ? 245 : 10;
      }
    expect(originalSpriteEdges({ width: w, height: h, rgb }, alpha, indices).coverage).toBeGreaterThan(0.5);
  });
  it('measures standard D65 Lab and separates exact house paint from a shared face', () => {
    expect(srgbToLab([255, 255, 255])[0]).toBeCloseTo(100, 4);
    expect(srgbToLab([255, 0, 0])[1]).toBeCloseTo(80.0925, 3);
    expect(deltaE76([0, 0, 0], [3, 4, 0])).toBe(5);
    const image = { width: 4, height: 3, rgb: new Uint8Array(36).fill(120) },
      mask = new Uint8Array(12).fill(1);
    const house = Uint8Array.from({ length: 12 }, (_, i) => +(i < 6));
    const rows = faceColourMeasurements(image, mask, image, mask, mask, house);
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => r.deltaE76 === 0 && r.pixels === 6)).toBe(true);
  });
});
describe('oriented sparse edges and negative controls', () => {
  it('keeps a coherent two-pixel rail through Lab bilateral de-dithering and round-trips Lab', () => {
    const w = 32,
      h = 28,
      rgb = new Uint8Array(w * h * 3),
      alpha = new Uint8Array(w * h).fill(1);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const rail = x >= 15 && x <= 16 && y >= 4 && y < 24,
          base = rail ? [145, 150, 175] : [100, 75, 45];
        for (let c = 0; c < 3; c++) rgb[3 * (y * w + x) + c] = base[c];
      }
    const out = bilateralSprite({ width: w, height: h, rgb }, alpha, 2, 2, 10);
    expect(out[3 * (12 * w + 15) + 2]).toBeGreaterThan(160);
    const colour = [45, 125, 210];
    expect(labToSrgb(srgbToLab(colour))[0]).toBeCloseTo(colour[0], 3);
  });
  it('thins a thick edge and rejects a nearby edge with an incompatible normal', () => {
    const w = 20,
      h = 20,
      thick = new Uint8Array(w * h);
    for (let y = 2; y < 18; y++) for (let x = 8; x < 11; x++) thick[y * w + x] = 1;
    const thin = thinSpriteEdges(thick, w, h);
    expect(thin.reduce((s, v) => s + v, 0)).toBeLessThan(20);
    const normal = spriteEdgeNormals(thin, w, h),
      wrong = Float64Array.from(normal, (a) => a + Math.PI / 2);
    expect(orientedEdgeAgreement(thin, normal, thin, normal, w, h).fraction).toBe(1);
    expect(orientedEdgeAgreement(thin, normal, thin, wrong, w, h).fraction).toBe(0);
    expect(
      orientedEdgeAgreement(
        thin,
        normal,
        thin,
        Float64Array.from(normal, (a) => a + Math.PI / 6),
        w,
        h,
      ).fraction,
    ).toBe(1);
    expect(
      orientedEdgeAgreement(
        thin,
        normal,
        thin,
        Float64Array.from(normal, (a) => a + Math.PI / 6 + 0.01),
        w,
        h,
      ).fraction,
    ).toBe(0);
  });
  it('reports all eight shifts and requires a 0.25 precision margin', () => {
    const w = 44,
      h = 40,
      mask = new Uint8Array(w * h);
    for (let y = 3; y < 21; y++) mask[y * w + 8] = 1;
    for (let x = 23; x < 41; x++) mask[30 * w + x] = 1;
    const normals = spriteEdgeNormals(mask, w, h);
    const result = spriteEdgeNullControls({
      width: w,
      height: h,
      interior: mask,
      interiorNormals: normals,
      allEdges: mask,
      allNormals: normals,
      original: mask,
      originalNormals: normals,
    });
    expect(result.controls).toHaveLength(8);
    expect(result.truePrecision).toBe(1);
    expect(result.trueRecall).toBe(1);
    expect(result.pass).toBe(true);
    expect(result.trueMinusWorstShift).toBeGreaterThanOrEqual(0.25);
    const dense = new Uint8Array(w * h).fill(1),
      angles = new Float64Array(w * h);
    expect(
      spriteEdgeNullControls({
        width: w,
        height: h,
        interior: dense,
        interiorNormals: angles,
        allEdges: dense,
        allNormals: angles,
        original: dense,
        originalNormals: angles,
      }).pass,
    ).toBe(false);
  });
});
describe('generic camera and geometry edges', () => {
  it('projects with caller axes and scale', () => expect(projectSpritePoint([1, 2, 3], camera)).toEqual([6, 1, 3]));
  it('excludes planar slope depth but detects a parallel-surface depth jump', () => {
    const w = 28,
      h = 28,
      ss = 4,
      coverage = new Float64Array(w * h).fill(1),
      depth = new Float64Array(w * h),
      normals = new Float64Array(w * h * 3);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        normals[3 * i] = -Math.SQRT1_2;
        normals[3 * i + 2] = Math.SQRT1_2;
        depth[i] = x / (2 * ss);
      }
    const b = { width: w, height: h, supersampling: ss, coverage, normals, depth };
    const options = { normalAngleDegrees: 20, depthJumpWorld: 0.1 };
    expect(spriteGeometryEdges(b, camera, options).interior.some(Boolean)).toBe(false);
    for (let y = 0; y < h; y++) for (let x = w / 2; x < w; x++) depth[y * w + x] += 1;
    const edge = spriteGeometryEdges(b, camera, options).interior;
    expect(edge[3 * 7 + 3]).toBe(1);
    expect(edge.slice(0, 7).some(Boolean)).toBe(false);
  });
  it('detects a normal crease at continuous depth, and rejects nonorthogonal cameras', () => {
    const coverage = new Float64Array(144).fill(1),
      depth = new Float64Array(144),
      normals = new Float64Array(432);
    for (let y = 0; y < 12; y++)
      for (let x = 0; x < 12; x++) {
        const i = y * 12 + x;
        normals[3 * i] = x < 6 ? 0 : Math.SQRT1_2;
        normals[3 * i + 2] = x < 6 ? 1 : Math.SQRT1_2;
      }
    const buffer = { width: 12, height: 12, supersampling: 4, coverage, normals, depth };
    expect(spriteGeometryEdges(buffer, camera, { normalAngleDegrees: 20, depthJumpWorld: 0.1 }).interior[4]).toBe(1);
    expect(() =>
      spriteGeometryEdges(buffer, { ...camera, up: [1, 0, 0] }, { normalAngleDegrees: 20, depthJumpWorld: 0.1 }),
    ).toThrow('orthonormal');
  });
  it('detects a small parallel depth gap while retaining a connected shallow crease', () => {
    const w = 28,
      h = 28,
      ss = 4,
      coverage = new Float64Array(w * h).fill(1),
      depth = new Float64Array(w * h),
      normals = new Float64Array(w * h * 3);
    const boundary = 14,
      slope = 0.2;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x,
          dx = (x + 0.5 - boundary) / (2 * ss);
        if (x < boundary) normals[3 * i + 2] = 1;
        else {
          normals[3 * i] = -slope / Math.hypot(1, slope);
          normals[3 * i + 2] = 1 / Math.hypot(1, slope);
          depth[i] = dx * slope;
        }
      }
    const buffer = { width: w, height: h, supersampling: ss, coverage, normals, depth },
      options = { normalAngleDegrees: 20, depthJumpWorld: 1e-5 };
    expect(spriteGeometryEdges(buffer, camera, options).interior.some(Boolean)).toBe(false);
    normals.fill(0);
    depth.fill(0);
    for (let i = 0; i < w * h; i++) normals[3 * i + 2] = 1;
    for (let y = 0; y < h; y++) for (let x = boundary; x < w; x++) depth[y * w + x] += 0.01;
    expect(spriteGeometryEdges(buffer, camera, options).interior.some(Boolean)).toBe(true);
  });
});
