/** Offline PNG gate adapter. Inputs and all diagnostic outputs are explicitly caller-owned. */
import { readFile, writeFile, realpath, lstat } from 'node:fs/promises';
import { resolve, dirname, basename, relative, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { decodePng, encodePng, type PngImage } from '../../ra2-assets/src/node/png';
import {
  defaultSpriteEdgeOptions,
  originalSpriteEdges,
  silhouetteIoU,
  orientedEdgeAgreement,
  thinSpriteEdges,
  spriteEdgeNormals,
  spriteEdgeNullControls,
  maskBoundary,
  faceColourMeasurements,
  colourSummary,
  maskSegments,
  defaultSpriteLocalEvidenceOptions,
  spriteLocalEvidenceNullControls,
  spriteDetailRoiMeasurements,
  type SpriteDetailRoi,
  type SpriteLocalEvidenceSegment,
  type SpriteEdgeOptions,
} from '../src/hardSurfaceSpriteGates';
import { spriteGeometryEdges, type SpriteCamera } from '../src/hardSurfaceSpriteCamera';
import { analyticSpriteEdges, type SpriteAnalyticMesh } from '../src/hardSurfaceSpriteAnalyticEdges';

interface Manifest {
  original: string;
  analyticMesh?: string;
  indices: string;
  coverage: string;
  normals: string;
  /** Optional lossless float32-LE component PNGs avoid normal quantization in geometric measurements. */
  normalX?: string;
  normalY?: string;
  normalZ?: string;
  depth: string;
  faceIds: string;
  colour: string;
  faceIdsSamples?: string;
  /** Additional diagnostics; neither changes the primary composite-colour measurement. */
  surfaceColour?: string;
  emissive?: string;
  /** Optional paired-state colour audit, supplementary to the unchanged primary reference. */
  comparisonOriginal?: string;
  comparisonColour?: string;
  comparisonState?: string;
  /** Optional independent edge raster: reject disagreement rather than silently trusting it. */
  edges?: string;
  supersampling: number;
  camera: SpriteCamera;
  normalAngleDegrees: number;
  depthJumpWorld: number;
  houseIndexRange: [number, number];
  edgeOptions?: SpriteEdgeOptions;
  frozenSource?: { edgeMaskSha256: string; optionsSha256: string; normalSha256?: string };
  faceLabels?: Record<
    string,
    { part: string; material?: string; primitive?: number; group?: number; category?: string }
  >;
  /** Independent local detail protocol, with source-only ROI/configuration JSON. */
  localEvidence?: {
    rois: string;
    frozen?: {
      deDitherSha256: string;
      detailMaskSha256: string;
      detailOptionsSha256: string;
      roiSha256?: string;
      profileOptionsSha256?: string;
    };
  };
}
const args = process.argv.slice(2);
if (args.length !== 2)
  throw new Error('Usage: hardSurfaceSpriteGates.ts <buffers.json> <outside-repository/gates.json>');
const manifestPath = await realpath(resolve(args[0])),
  base = dirname(manifestPath);
const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as Manifest;
const repo = await realpath(resolve(dirname(fileURLToPath(import.meta.url)), '../../..'));
const inputs = new Set([manifestPath]),
  hashes: Record<string, string> = {};
async function load(name: keyof Manifest): Promise<PngImage> {
  const value = manifest[name];
  if (typeof value !== 'string') throw new Error(`Missing raster: ${name}`);
  const path = await realpath(resolve(base, value)),
    bytes = await readFile(path);
  inputs.add(path);
  hashes[name] = createHash('sha256').update(bytes).digest('hex');
  return decodePng(bytes);
}
const names = ['original', 'indices', 'coverage', 'normals', 'depth', 'faceIds', 'colour'] as const;
const images = await Promise.all(names.map(load));
const [original, indices, cov, normals, depth, faceIds, colour] = images;
const w = original.width,
  h = original.height,
  ss = manifest.supersampling;
if (
  original.channels !== 4 ||
  indices.channels !== 1 ||
  cov.channels !== 1 ||
  normals.channels !== 3 ||
  depth.channels !== 4 ||
  faceIds.channels !== 3
)
  throw new Error('Expected RGBA original, L indices/coverage, RGB normals/face IDs, RGBA float32-LE depth');
if (!Number.isInteger(ss) || ss < 4) throw new Error('Gates require at least 4x4 supersampling');
for (const im of [indices, faceIds, colour])
  if (im.width !== w || im.height !== h) throw new Error('Native raster dimensions differ');
for (const im of [cov, normals, depth])
  if (im.width !== w * ss || im.height !== h * ss) throw new Error('Supersample dimensions differ');
if (colour.channels !== 3 && colour.channels !== 4) throw new Error('Colour must be RGB or RGBA');
const alpha = Uint8Array.from({ length: w * h }, (_, i) => +(original.data[4 * i + 3] >= 128));
if (Array.from({ length: w * h }, (_, i) => original.data[4 * i + 3]).some((v) => v !== 0 && v !== 255))
  throw new Error('Original alpha must be binary');
const rgb = (im: PngImage) => ({
  width: im.width,
  height: im.height,
  rgb: Uint8Array.from(
    { length: im.width * im.height * 3 },
    (_, i) => im.data[Math.floor(i / 3) * im.channels + (i % 3)],
  ),
});
const normalData = new Float64Array(w * h * ss * ss * 3),
  z = new Float64Array(w * h * ss * ss);
const depthView = new DataView(depth.data.buffer, depth.data.byteOffset, depth.data.byteLength);
for (let i = 0; i < z.length; i++) {
  z[i] = depthView.getFloat32(4 * i, true);
  const v = [0, 1, 2].map((c) => (normals.data[3 * i + c] / 255) * 2 - 1),
    length = Math.hypot(...v);
  for (let c = 0; c < 3; c++) normalData[3 * i + c] = v[c] / length;
}
if ([manifest.normalX, manifest.normalY, manifest.normalZ].some(Boolean)) {
  if (!manifest.normalX || !manifest.normalY || !manifest.normalZ)
    throw new Error('All three float normal components are required');
  const components = await Promise.all((['normalX', 'normalY', 'normalZ'] as const).map(load));
  for (const [c, component] of components.entries()) {
    if (component.width !== w * ss || component.height !== h * ss || component.channels !== 4)
      throw new Error('Invalid float normal component PNG');
    const view = new DataView(component.data.buffer, component.data.byteOffset, component.data.byteLength);
    for (let i = 0; i < z.length; i++) normalData[3 * i + c] = view.getFloat32(4 * i, true);
  }
}
const decodeIds = (image: PngImage) =>
  Uint32Array.from(
    { length: image.width * image.height },
    (_, i) => image.data[3 * i] + 256 * image.data[3 * i + 1] + 65536 * image.data[3 * i + 2],
  );
let sampleIds: Uint32Array | undefined;
if (manifest.faceIdsSamples) {
  const samples = await load('faceIdsSamples');
  if (samples.width !== w * ss || samples.height !== h * ss || samples.channels !== 3)
    throw new Error('Invalid sample-resolution face IDs');
  sampleIds = decodeIds(samples);
}
const geometry = spriteGeometryEdges(
  {
    width: w * ss,
    height: h * ss,
    supersampling: ss,
    coverage: Float64Array.from(cov.data, (v) => v / 255),
    normals: normalData,
    depth: z,
    faceIds: sampleIds,
  },
  manifest.camera,
  { normalAngleDegrees: manifest.normalAngleDegrees, depthJumpWorld: manifest.depthJumpWorld },
);
if (manifest.normalAngleDegrees !== 20) throw new Error('This gate contract requires a 20 degree crease threshold');
if (manifest.edges) {
  const edge = await load('edges');
  if (edge.width !== w || edge.height !== h || edge.channels !== 1) throw new Error('Invalid supplied edge raster');
  const supplied = thinSpriteEdges(
      Uint8Array.from(edge.data, (v) => +(v >= 128)),
      w,
      h,
    ),
    expected = thinSpriteEdges(geometry.interior, w, h);
  if (supplied.some((v, i) => v !== expected[i])) throw new Error('Supplied edges disagree with geometry buffers');
}
const range = manifest.houseIndexRange;
if (
  !Array.isArray(range) ||
  range.length !== 2 ||
  !range.every(Number.isInteger) ||
  range[0] < 0 ||
  range[1] > 255 ||
  range[0] > range[1]
)
  throw new Error('Invalid exact house index range');
const house = Uint8Array.from(indices.data, (v, i) => +(!!alpha[i] && v >= range[0] && v <= range[1]));
const edgeOptions = manifest.edgeOptions ?? { ...defaultSpriteEdgeOptions };
const source = originalSpriteEdges(rgb(original), alpha, indices.data, edgeOptions, house);
const edgeMaskSha256 = createHash('sha256').update(source.structural).digest('hex');
const optionsSha256 = createHash('sha256')
  .update(JSON.stringify(Object.fromEntries(Object.entries(edgeOptions).sort(([a], [b]) => a.localeCompare(b)))))
  .digest('hex');
const normalSha256 = createHash('sha256')
  .update(
    JSON.stringify(
      Array.from(source.structural, (v, i) =>
        v && Number.isFinite(source.normals[i]) ? Math.round(source.normals[i] * 1e9) : null,
      ),
    ),
  )
  .digest('hex');
const structuralSegmentIds = new Uint32Array(w * h);
for (const segment of source.segments) for (const i of segment.pixels) structuralSegmentIds[i] = segment.id;
if (
  manifest.frozenSource &&
  (manifest.frozenSource.edgeMaskSha256 !== edgeMaskSha256 ||
    manifest.frozenSource.optionsSha256 !== optionsSha256 ||
    (manifest.frozenSource.normalSha256 !== undefined && manifest.frozenSource.normalSha256 !== normalSha256))
)
  throw new Error('Frozen source structure changed; geometry comparisons must keep the same source edge set');
const silhouette = silhouetteIoU(alpha, geometry.coverage);
let analytic: ReturnType<typeof analyticSpriteEdges> | undefined;
if (manifest.analyticMesh) {
  const path = await realpath(resolve(base, manifest.analyticMesh)),
    bytes = await readFile(path);
  inputs.add(path);
  hashes.analyticMesh = createHash('sha256').update(bytes).digest('hex');
  analytic = analyticSpriteEdges(
    JSON.parse(bytes.toString('utf8')) as SpriteAnalyticMesh,
    manifest.camera,
    w,
    h,
    geometry.coverage,
  );
}
const interior = analytic?.interior ?? thinSpriteEdges(geometry.interior, w, h);
const interiorNormals = analytic?.normals ?? spriteEdgeNormals(interior, w, h, edgeOptions.normalRadius);
const boundary = thinSpriteEdges(
  maskBoundary(
    Uint8Array.from(geometry.coverage, (v) => +(v >= 0.5)),
    w,
    h,
  ),
  w,
  h,
);
const allEdges = thinSpriteEdges(
  Uint8Array.from(boundary, (v, i) => +(!!v || !!(analytic?.allEdges[i] ?? interior[i]))),
  w,
  h,
);
const allNormals = spriteEdgeNormals(allEdges, w, h, edgeOptions.normalRadius);
if (analytic)
  for (let i = 0; i < allNormals.length; i++)
    if (analytic.allEdges[i] && Number.isFinite(analytic.normals[i])) allNormals[i] = analytic.normals[i];
const precision = orientedEdgeAgreement(interior, interiorNormals, source.structural, source.normals, w, h);
const recall = orientedEdgeAgreement(source.structural, source.normals, allEdges, allNormals, w, h);
const nullControl = spriteEdgeNullControls({
  width: w,
  height: h,
  interior,
  interiorNormals,
  allEdges,
  allNormals,
  original: source.structural,
  originalNormals: source.normals,
});
const sourceCoveragePass = source.coverage !== null && source.coverage < 0.4;
let localEvidence:
  | {
      options: typeof defaultSpriteLocalEvidenceOptions;
      hashes: {
        deDitherSha256: string;
        detailMaskSha256: string;
        detailOptionsSha256: string;
        roiSha256: string;
        profileOptionsSha256: string;
      };
      runs: SpriteLocalEvidenceSegment[];
      G2L: ReturnType<typeof spriteLocalEvidenceNullControls>;
      G5: ReturnType<typeof spriteDetailRoiMeasurements>;
      detailSource: ReturnType<typeof originalSpriteEdges>;
      detailEdgeOptions: SpriteEdgeOptions;
    }
  | undefined;
if (manifest.localEvidence) {
  if (!analytic) throw new Error('Local segment evidence requires an analytic mesh');
  const path = await realpath(resolve(base, manifest.localEvidence.rois)),
    bytes = await readFile(path);
  inputs.add(path);
  hashes.localEvidenceRois = createHash('sha256').update(bytes).digest('hex');
  const config = JSON.parse(bytes.toString('utf8')) as {
    rois: SpriteDetailRoi[];
    detailEdgeOptions: SpriteEdgeOptions;
  };
  if (
    !config.detailEdgeOptions ||
    config.detailEdgeOptions.boundaryDeltaE !== 8 ||
    config.detailEdgeOptions.structuralMinLength !== 4
  )
    throw new Error('Local detail source requires dE76 8 and minimum length 4');
  const detailSource = originalSpriteEdges(rgb(original), alpha, indices.data, config.detailEdgeOptions, house);
  const localHashes = {
    roiSha256: hashes.localEvidenceRois,
    profileOptionsSha256: createHash('sha256').update(JSON.stringify(defaultSpriteLocalEvidenceOptions)).digest('hex'),
    deDitherSha256: createHash('sha256')
      .update(JSON.stringify(Array.from(source.deDithered, (v) => Math.round(v * 1e8))))
      .digest('hex'),
    detailMaskSha256: createHash('sha256').update(detailSource.structural).digest('hex'),
    detailOptionsSha256: createHash('sha256')
      .update(
        JSON.stringify(
          Object.fromEntries(Object.entries(config.detailEdgeOptions).sort(([a], [b]) => a.localeCompare(b))),
        ),
      )
      .digest('hex'),
  };
  if (
    manifest.localEvidence.frozen &&
    Object.entries(manifest.localEvidence.frozen).some(
      ([key, value]) => localHashes[key as keyof typeof localHashes] !== value,
    )
  )
    throw new Error('Frozen local source evidence changed');
  const runs: SpriteLocalEvidenceSegment[] = [];
  for (const [group, crease] of analytic.creases.entries()) {
    const mask = new Uint8Array(w * h);
    const runNormals = new Map(crease.pixels.map((i, k) => [i, crease.normals[k]]));
    for (const i of crease.pixels) mask[i] = 1;
    for (const [component, pixels] of maskSegments(mask, w, h).entries())
      if (pixels.length >= 3)
        runs.push({
          id: `L${group + 1}.${component + 1}`,
          faces: crease.faces,
          samples: pixels.map((i) => ({ x: (i % w) + 0.5, y: Math.floor(i / w) + 0.5, normal: runNormals.get(i)! })),
        });
  }
  localEvidence = {
    options: defaultSpriteLocalEvidenceOptions,
    hashes: localHashes,
    runs,
    detailSource,
    detailEdgeOptions: config.detailEdgeOptions,
    G2L: spriteLocalEvidenceNullControls({ width: w, height: h, rgb: source.deDithered }, alpha, runs),
    G5: spriteDetailRoiMeasurements({
      width: w,
      height: h,
      source: detailSource.structural,
      sourceNormals: detailSource.normals,
      rendered: allEdges,
      renderedNormals: allNormals,
      rois: config.rois,
    }),
  };
}
const ids = Uint32Array.from(
  { length: w * h },
  (_, i) => faceIds.data[3 * i] + 256 * faceIds.data[3 * i + 1] + 65536 * faceIds.data[3 * i + 2],
);
let emissive: PngImage | undefined;
if (manifest.emissive) {
  emissive = await load('emissive');
  if (emissive.width !== w || emissive.height !== h || emissive.channels !== 1)
    throw new Error('Invalid native emissive mask');
}
const labelRows = (measurements: ReturnType<typeof faceColourMeasurements>) =>
  measurements.map((row) => {
    const label = manifest.faceLabels?.[String(row.face)];
    let emissivePixels = 0;
    if (emissive)
      for (let i = 0; i < ids.length; i++)
        if (ids[i] === row.face && geometry.coverage[i] >= 0.5 && !!house[i] === row.house && emissive.data[i] > 0)
          emissivePixels++;
    // Labels cannot overwrite measured numbers, even if the input JSON contains extra properties.
    return {
      ...row,
      part: label?.part,
      material: label?.material,
      primitive: label?.primitive,
      group: label?.group,
      category: label?.category,
      emissivePixels,
    };
  });
const rows = labelRows(faceColourMeasurements(rgb(original), alpha, rgb(colour), ids, geometry.coverage, house));
const geometricFaces = labelRows(
  faceColourMeasurements(rgb(original), alpha, rgb(colour), ids, geometry.coverage, new Uint8Array(w * h)),
);
let comparison: Record<string, unknown> | undefined;
if (manifest.comparisonOriginal || manifest.comparisonColour) {
  if (!manifest.comparisonOriginal || !manifest.comparisonColour)
    throw new Error('Paired colour audit requires both rasters');
  const [reference, rendered] = await Promise.all((['comparisonOriginal', 'comparisonColour'] as const).map(load));
  if (
    reference.channels !== 4 ||
    (rendered.channels !== 3 && rendered.channels !== 4) ||
    [reference, rendered].some((im) => im.width !== w || im.height !== h)
  )
    throw new Error('Invalid paired colour audit rasters');
  const referenceAlpha = Uint8Array.from({ length: w * h }, (_, i) => +(reference.data[4 * i + 3] >= 128));
  const withoutEmission = (r: ReturnType<typeof labelRows>[number]) => {
    const { emissivePixels: _primaryEmissionCount, ...row } = r;
    return row;
  };
  const faces = labelRows(
    faceColourMeasurements(rgb(reference), referenceAlpha, rgb(rendered), ids, geometry.coverage, house),
  ).map(withoutEmission);
  const complete = labelRows(
    faceColourMeasurements(
      rgb(reference),
      referenceAlpha,
      rgb(rendered),
      ids,
      geometry.coverage,
      new Uint8Array(w * h),
    ),
  ).map(withoutEmission);
  comparison = {
    state: manifest.comparisonState ?? 'caller-supplied matched state',
    summary: colourSummary(faces),
    house: colourSummary(faces.filter((r) => r.house)),
    nonHouse: colourSummary(faces.filter((r) => !r.house)),
    geometricFaceSummary: colourSummary(complete),
    faces,
    geometricFaces: complete,
    byCategory: Object.fromEntries(
      [...new Set(complete.map((r) => r.category).filter((s): s is string => !!s))]
        .sort()
        .map((category) => [category, colourSummary(complete.filter((r) => r.category === category))]),
    ),
  };
}
let surfaceOnly:
  | {
      summary: ReturnType<typeof colourSummary>;
      house: ReturnType<typeof colourSummary>;
      nonHouse: ReturnType<typeof colourSummary>;
      faces: ReturnType<typeof labelRows>;
    }
  | undefined;
if (manifest.surfaceColour) {
  const surface = await load('surfaceColour');
  if (surface.width !== w || surface.height !== h || (surface.channels !== 3 && surface.channels !== 4))
    throw new Error('Invalid native surface colour');
  const faces = labelRows(faceColourMeasurements(rgb(original), alpha, rgb(surface), ids, geometry.coverage, house));
  surfaceOnly = {
    summary: colourSummary(faces),
    house: colourSummary(faces.filter((r) => r.house)),
    nonHouse: colourSummary(faces.filter((r) => !r.house)),
    faces,
  };
}
const segmentRows = (segments: number[][]) =>
  segments.map((segment, id) => ({
    id: id + 1,
    pixels: segment.length,
    coordinates: segment.map((i) => [i % w, Math.floor(i / w)]),
    faces: [...new Set(segment.map((i) => ids[i]))],
  }));
const report = {
  version: 3,
  protocol: analytic ? 'region-structural-analytic-union-v2' : 'region-structural-buffer-fallback-v2',
  nativeSize: [w, h],
  supersampling: ss,
  inputSha256: hashes,
  definitions: {
    coverageThreshold: 0.5,
    toleranceChebyshev: 1,
    normalAngleDegrees: manifest.normalAngleDegrees,
    depthJumpWorld: manifest.depthJumpWorld,
    edgeExtraction: analytic
      ? 'analytic visible union segments; coplanar surfaces merged; chamfer midlines'
      : 'sample-buffer fallback',
    normalPrecision: manifest.normalX ? 'float32-LE components' : 'quantized RGB8 unit vectors',
    edgeOptions,
    deDither:
      'iterated alpha- and caller-class-aware Lab bilateral or channel median, selected by edgeOptions; quantized to observed source palette entries',
    segmentation:
      '4-connected palette seeds; adjacent median-Lab merge; merge undersized regions into nearest adjacent median, preferring caller classes',
    structural:
      'region boundaries with median dE76 or luminance contrast; alpha eroded one; per-pair thinning and min length; final skeleton',
    sourceNormals: 'multichannel Sobel of region-median Lab, Di Zenzo principal gradient direction',
    renderedNormals: analytic
      ? 'perpendicular to the analytic projected segment; doubled-angle aggregation at junctions; ambiguous normals fail'
      : 'PCA tangent of the thinned geometry edge locus, rotated 90 degrees, radius from edgeOptions; ambiguous normals fail',
    orientationToleranceDegrees: 30,
    interior: analytic
      ? 'continuous visibility of union crease/intersection segments; coplanar surfaces suppressed; chamfer midlines; exclude silhouette band; native thinning'
      : 'supersample normal crease or tangent-corrected depth jump; OR reduction; exclude silhouette band; Zhang-Suen thinning',
    colours: 'native display-sRGB mean vs channel median source RGB, D65 Lab dE76; >=6 pixels per face/house partition',
    g4Threshold: null,
  },
  analytic: analytic
    ? {
        triangles: analytic.triangleCount,
        candidates: analytic.candidateCount,
        visibleSegments: analytic.segments.length,
      }
    : undefined,
  sourceStructure: {
    edgeMaskSha256,
    optionsSha256,
    normalSha256,
    coverage: source.coverage,
    coveredOpaquePixels: source.coveredOpaquePixels,
    opaquePixels: source.opaquePixels,
    maximumCoverage: 0.4,
    pass: sourceCoveragePass,
    regions: source.regions.length,
    segments: source.segments.length,
    mergePasses: source.mergePasses,
  },
  nullControl,
  G2L: localEvidence
    ? {
        definition:
          'analytic host-face-pair visible connected runs >=3 native pixels; per-offset median of sample-wise Lab step across two 2px bands; all seven offsets need >=50% opaque samples; peak at -1/0/+1 and dE76>=8',
        options: localEvidence.options,
        sourceHashes: localEvidence.hashes,
        runs: localEvidence.runs,
        ...localEvidence.G2L,
      }
    : undefined,
  G5: localEvidence
    ? {
        definition:
          'source-only ROIs; local source region boundaries dE76>=8, component length>=4; rendered interior+silhouette; distance 1 and normal difference 30 degrees',
        detailSourceCoverage: localEvidence.detailSource.coverage,
        sourceHashes: localEvidence.hashes,
        rows: localEvidence.G5,
      }
    : undefined,
  G1: { ...silhouette, target: 0.99, pass: silhouette.iou >= 0.99 },
  G2: {
    precision: precision.fraction,
    total: precision.total,
    matched: precision.matched,
    target: 1,
    precisionPass: precision.total > 0 && precision.fraction === 1,
    meaningful: sourceCoveragePass && nullControl.pass,
    pass: precision.total > 0 && precision.fraction === 1 && sourceCoveragePass && nullControl.pass,
    unknownNormalPixels: precision.unknownNormals,
    failingSegments: segmentRows(precision.segments),
    creaseEvidence: (analytic?.creases ?? geometry.creases)
      .filter((c) => c.pixels.some((i) => interior[i]))
      .map((crease) => ({
        faces: crease.faces,
        pixels: crease.pixels
          .filter((i) => !!interior[i])
          .map((i) => {
            const x = i % w,
              y = Math.floor(i / w),
              support: number[][] = [];
            for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
              for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++)
                if (
                  source.structural[yy * w + xx] &&
                  Number.isFinite(interiorNormals[i]) &&
                  Number.isFinite(source.normals[yy * w + xx])
                ) {
                  const a = Math.abs(interiorNormals[i] - source.normals[yy * w + xx]) % Math.PI;
                  if (Math.min(a, Math.PI - a) <= Math.PI / 6 + 1e-12) support.push([xx, yy]);
                }
            return {
              coordinate: [x, y],
              normalRadians: Number.isFinite(interiorNormals[i]) ? interiorNormals[i] : null,
              originalEdgePixels: support,
              sourceSegmentIds: [...new Set(support.map(([sx, sy]) => `S${structuralSegmentIds[sy * w + sx]}`))],
            };
          }),
      })),
  },
  G3: {
    recall: recall.fraction,
    total: recall.total,
    matched: recall.matched,
    target: 0.85,
    pass: sourceCoveragePass && recall.total > 0 && recall.fraction !== null && recall.fraction >= 0.85,
    unknownNormalPixels: recall.unknownNormals,
    missedSegments: segmentRows(recall.segments),
  },
  G4: {
    comparison,
    byCategory: Object.fromEntries(
      [...new Set(geometricFaces.map((r) => r.category).filter((s): s is string => !!s))]
        .sort()
        .map((category) => [category, colourSummary(geometricFaces.filter((r) => r.category === category))]),
    ),
    originalSelfControl: {
      notice:
        'Original mean versus its own median, excluding missing original pixels; not a correction to the primary metric',
      house: colourSummary(
        faceColourMeasurements(
          rgb(original),
          alpha,
          rgb(original),
          ids,
          Float64Array.from(geometry.coverage, (v, i) => (alpha[i] ? v : 0)),
          house,
        ).filter((r) => r.house),
      ),
      geometricFaceSummary: colourSummary(
        faceColourMeasurements(
          rgb(original),
          alpha,
          rgb(original),
          ids,
          Float64Array.from(geometry.coverage, (v, i) => (alpha[i] ? v : 0)),
          new Uint8Array(w * h),
        ),
      ),
      geometricFaces: labelRows(
        faceColourMeasurements(
          rgb(original),
          alpha,
          rgb(original),
          ids,
          Float64Array.from(geometry.coverage, (v, i) => (alpha[i] ? v : 0)),
          new Uint8Array(w * h),
        ),
      ),
    },
    summary: colourSummary(rows),
    geometricFaceSummary: colourSummary(geometricFaces),
    geometricFaces,
    house: colourSummary(rows.filter((r) => r.house)),
    nonHouse: colourSummary(rows.filter((r) => !r.house)),
    faces: rows,
    surfaceOnly,
    originalHousePixels: house.reduce((s, v) => s + v, 0),
    coveredHousePixels: house.reduce((s, v, i) => s + +(!!v && geometry.coverage[i] >= 0.5), 0),
  },
};
async function outputPath(path: string): Promise<string> {
  const candidate = resolve(await realpath(dirname(resolve(path))), basename(path));
  const existing = await lstat(candidate).catch((error: NodeJS.ErrnoException) => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  });
  const target = existing ? await realpath(candidate) : candidate,
    rel = relative(repo, target);
  if (rel === '' || (!(rel === '..' || rel.startsWith('..' + sep)) && !isAbsolute(rel)))
    throw new Error('Gate outputs must be outside the repository');
  if (inputs.has(target)) throw new Error('Gate output must not overwrite an input');
  return target;
}
const out = await outputPath(args[1]),
  dest = dirname(out);
const diagnostics: Record<string, Uint8Array> = {
  original_edges: source.structural,
  original_structural: source.structural,
  rendered_interior: interior,
  rendered_silhouette: boundary,
  failing_edges: precision.failures,
  missed_structural: recall.failures,
  ...(localEvidence ? { local_detail_source: localEvidence.detailSource.structural } : {}),
};
// Resolve every output before the first write, including symlink targets.
const targets = await Promise.all(Object.keys(diagnostics).map((name) => outputPath(resolve(dest, `${name}.png`))));
const structurePath = await outputPath(resolve(dest, 'structural_regions.json'));
const analyticPath = analytic ? await outputPath(resolve(dest, 'analytic_segments.json')) : undefined;
const localImagePath = localEvidence ? await outputPath(resolve(dest, 'local_de_dithered.png')) : undefined;
const localRegionsPath = localEvidence ? await outputPath(resolve(dest, 'local_detail_regions.json')) : undefined;
const originalSegments = source.segments.map((segment) => ({
  id: `S${segment.id}`,
  regions: segment.regions,
  pixels: segment.pixels.map((i) => [i % w, Math.floor(i / w)]),
  normalRadians: segment.pixels.map((i) => (Number.isFinite(source.normals[i]) ? source.normals[i] : null)),
}));
if (
  new Set([
    out,
    structurePath,
    ...(analyticPath ? [analyticPath] : []),
    ...(localImagePath ? [localImagePath] : []),
    ...(localRegionsPath ? [localRegionsPath] : []),
    ...targets,
  ]).size !==
  targets.length + 2 + +!!analyticPath + +!!localImagePath + +!!localRegionsPath
)
  throw new Error('Diagnostic outputs must be distinct');
for (const [i, mask] of Object.values(diagnostics).entries())
  await writeFile(
    targets[i],
    encodePng({ width: w, height: h, channels: 1, data: Uint8Array.from(mask, (v) => (v ? 255 : 0)) }),
  );
if (localEvidence && localImagePath && localRegionsPath) {
  await writeFile(
    localImagePath,
    encodePng({
      width: w,
      height: h,
      channels: 3,
      data: Uint8Array.from(source.deDithered, (v) => Math.round(Math.min(255, Math.max(0, v)))),
    }),
  );
  await writeFile(
    localRegionsPath,
    JSON.stringify(
      {
        edgeOptions: localEvidence.detailEdgeOptions,
        hashes: localEvidence.hashes,
        regions: localEvidence.detailSource.regions,
        labels: Array.from(localEvidence.detailSource.labels),
        segments: localEvidence.detailSource.segments.map((s) => ({
          id: `D${s.id}`,
          pixels: s.pixels.map((i) => [i % w, Math.floor(i / w)]),
          regions: s.regions,
        })),
      },
      null,
      2,
    ) + '\n',
  );
}
await writeFile(
  structurePath,
  JSON.stringify(
    {
      protocol: report.protocol,
      edgeOptions,
      coverage: source.coverage,
      regions: source.regions,
      labels: Array.from(source.labels),
      segments: originalSegments,
    },
    null,
    2,
  ) + '\n',
);
if (analytic && analyticPath)
  await writeFile(
    analyticPath,
    JSON.stringify({ camera: manifest.camera, segments: analytic.segments }, null, 2) + '\n',
  );
await writeFile(out, JSON.stringify(report, null, 2) + '\n');
console.log(
  JSON.stringify({
    output: out,
    G1: report.G1.iou,
    G2: report.G2.precision,
    G3: report.G3.recall,
    G4: report.G4.summary,
    sourceCoverage: source.coverage,
    nullMargin: nullControl.trueMinusWorstShift,
  }),
);
