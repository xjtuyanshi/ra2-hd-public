/** Pure surface decisions from the offline research renderer; inputs use its lattice units and sRGB colours. */
export type LookVector = readonly [number, number, number];
export interface SeamSegment {
  readonly start: LookVector;
  readonly end: LookVector;
  readonly strength: number;
}
export interface SeamCombOptions {
  minLength: number;
  maxGap?: number;
  minOverlap?: number;
}
export interface PanelLineOptions {
  width: number;
  aoWidth: number;
}
export interface PaintRegionFlags {
  metal: boolean;
  house: boolean;
}
export type PaintRegion = 'paint' | 'bareSteel' | 'glint' | 'other';
export interface SurfaceMaterial {
  readonly roughness: number;
  readonly metalness: number;
}
/** Material classes 0, 1, 2, 3 and 5 respectively; class 4 is outside the painted-steel look. */
export type PaintedSteelMaterial = Readonly<
  Record<'paint' | 'darkMetal' | 'track' | 'bareSteel' | 'barrelSteel', SurfaceMaterial> & { drumRoughness: number }
>;
export const DEFAULT_PAINTED_STEEL_MATERIALS: PaintedSteelMaterial = {
  paint: { roughness: 0.45, metalness: 0 },
  darkMetal: { roughness: 0.42, metalness: 0.55 },
  track: { roughness: 0.7, metalness: 0.15 },
  bareSteel: { roughness: 0.45, metalness: 0.45 },
  barrelSteel: { roughness: 0.3, metalness: 0.85 },
  drumRoughness: 0.3,
};
/** Caller policy for colours, named paint codes and surface-profile strengths; contains no geometry. */
export interface HardSurfaceLookParameters {
  cream: LookVector;
  houseTone: LookVector;
  hatchColour: LookVector;
  barrelSteel: LookVector;
  wear: number;
  wallOcclusion: number;
  wallGrime: number;
  lines: PanelLineOptions;
  seams: SeamCombOptions;
  paintCodes: Readonly<Record<string, number>>;
  materials: PaintedSteelMaterial;
}
const clamp = (v: number, lo = 0, hi = 1): number => Math.max(lo, Math.min(hi, v));
const dot = (a: LookVector, b: LookVector): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: LookVector, b: LookVector): LookVector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const length = (v: LookVector): number => Math.hypot(...v);
const direction = (v: LookVector, len: number): LookVector => [v[0] / len, v[1] / len, v[2] / len];

/** Keep the strongest/longest representative of nearby overlapping parallel seams. Does not mutate input order. */
export function combParallelSeams<T extends SeamSegment>(
  segments: readonly T[],
  { minLength, maxGap = 1.6, minOverlap = 0.5 }: SeamCombOptions,
): T[] {
  const sorted = segments
    .map((segment) => {
      const delta = sub(segment.end, segment.start);
      return { segment, delta, len: length(delta) };
    })
    .filter(({ len }) => len > 0 && len >= minLength - 1e-6)
    .sort((a, b) => b.len + b.segment.strength - (a.len + a.segment.strength));
  const keep: typeof sorted = [];
  for (const candidate of sorted) {
    const u = direction(candidate.delta, candidate.len);
    const clash = keep.some(({ segment, delta, len }) => {
      const u2 = direction(delta, len);
      if (Math.abs(dot(u, u2)) < 0.99) return false;
      const r = sub(candidate.segment.start, segment.start),
        t = dot(r, u2),
        perp = length([r[0] - t * u2[0], r[1] - t * u2[1], r[2] - t * u2[2]]),
        tEnd = dot(sub(candidate.segment.end, segment.start), u2),
        t0 = Math.min(t, tEnd),
        t1 = Math.max(t, tEnd);
      return perp < maxGap && Math.min(t1, len) - Math.max(t0, 0) > minOverlap;
    });
    if (!clash) keep.push(candidate);
  }
  return keep.map(({ segment }) => segment);
}

/** Both inputs are unit vectors in the same frame, as in the reference panel-line pass. */
export function isStaircaseSeam(direction: LookVector, faceNormal: LookVector): boolean {
  const n = faceNormal.map(Math.abs),
    u = direction.map(Math.abs),
    axis = u.indexOf(Math.max(...u));
  return Math.max(...n) < 0.985 && u[axis] > 0.99 && n[axis] < 0.05;
}

/** Tangent-plane distance to a seam; the caller handles face depth and staircase rejection. Widths must be positive. */
export function panelLineProfile(
  dist: number,
  strength: number,
  { width, aoWidth }: PanelLineOptions,
): { dark: number; halo: number } {
  if (dist >= aoWidth) return { dark: 0, halo: 0 };
  const st = clamp(0.4 + strength, 0.6, 1);
  return { dark: clamp(1 - (dist / width) ** 2) ** 0.7 * st, halo: clamp(1 - dist / aoWidth) ** 2 * st };
}

/** Glints are folded into their parent face before neutral steel and warm paint are classified. */
export function classifyPaintRegion(albedoSrgb: LookVector, { metal, house }: PaintRegionFlags): PaintRegion {
  if (metal || house) return 'other';
  if ((albedoSrgb[0] + albedoSrgb[1] + albedoSrgb[2]) / 3 > 0.8) return 'glint';
  return Math.abs(albedoSrgb[0] - albedoSrgb[2]) < 0.03 ? 'bareSteel' : 'paint';
}
export function levelledPaint(albedoSrgb: LookVector, baseSrgb: LookVector, cream: LookVector): LookVector {
  const luma = (v: LookVector) => dot(v, [0.3, 0.59, 0.11]),
    ratio = clamp(luma(albedoSrgb) / luma(baseSrgb), 0.45, 1.15);
  return [cream[0] * ratio, cream[1] * ratio, cream[2] * ratio];
}
/** Wear blend amount. The caller limits this to painted/house surfaces and supplies deterministic noise. */
export function edgeWear(chamferFlag: number, normalUp: number, noise01: number, strength: number): number {
  return chamferFlag * clamp(normalUp * 1.4 + 0.2) * clamp((noise01 - 0.45) / 0.2) * strength;
}
/** Multiplicative wall shading, shared by sky occlusion (exponent 1.5) and grime (1.2). */
export function wallOcclusion(normalUp: number, heightFrac: number, strength: number, exponent: number): number {
  const wall = 1 - clamp(Math.abs(normalUp) * 1.5);
  return 1 - strength * wall * (1 - clamp(heightFrac)) ** exponent;
}
/** Triangular hoop groove profile measured from the nearer cylinder end. Width must be positive. */
export function hoopProfile(axial: number, range: readonly [number, number], at: number, width: number): number {
  const endDistance = Math.min(axial - range[0], range[1] - axial);
  return clamp(1 - Math.abs(endDistance - at) / width);
}
/** Triangular roof-ring groove profile. The caller masks cylinder roofs; width must be positive. */
export function hatchRingProfile(radial: number, r: number, width: number): number {
  return clamp(1 - Math.abs(radial - r) / width);
}
