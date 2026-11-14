export type AlivePreset = 'soft' | 'deep';

/** Opt-in presentation only; all distances are cells and all times are renderer seconds. */
export interface AliveOptions {
  clouds: boolean;
  /** Cloud field shape and core depth; `soft` is the reviewed default, `deep` the stronger opt-in variant. */
  preset: AlivePreset;
  wind: boolean;
  cloud: number;
  sway: number;
  speed: number;
  debugCloud: boolean;
}

export type AliveChange = boolean | Partial<AliveOptions>;

export const ALIVE_DEFAULTS: Readonly<AliveOptions> = Object.freeze({
  clouds: false,
  preset: 'soft',
  wind: false,
  cloud: 1,
  sway: 1,
  speed: 1,
  debugCloud: false,
});

export const ALIVE = Object.freeze({
  cloudScale: 13,
  cloudLargeScale: 26,
  cloudEdgeLow: 0.07,
  cloudEdgeHigh: 0.23,
  cloudDarkening: 0.22,
  cloudFineWeight: 0.8,
  cloudStretch: 1.4,
  cloudMorph: 0.35,
  waterBodyWeight: 0.2,
  waterSkyWeight: 0.45,
  drift: 1.5,
  crownPixels: 3.8,
  gustPixels: 6.2,
  period: 3,
  gustLength: 8.5,
  heightPower: 1.4,
});

/** The noise lattice, threshold and core depth that define a cloud look; shared by the CPU reference and shader. */
export interface CloudShape {
  scale: number;
  largeScale: number;
  edgeLow: number;
  edgeHigh: number;
  darkening: number;
  fineWeight: number;
  stretch: number;
  /** Share of the land shadow (darkening and tint) taken by the water body/seabed and by sky reflection. */
  waterBodyWeight: number;
  waterSkyWeight: number;
}

/**
 * `soft` keeps the reviewed field (13-cell patches, ~2.7-cell edge, 22% core). `deep` is for small/phone views
 * (about ten cells across), where soft patches are larger than the frame and read as overall dimming: 6/14-cell
 * lattices give patches of about seven cells with lit gaps inside one view, a 0.04..0.16 threshold tightens the edge
 * to about one cell, and the core darkens by 33%; water takes most of that shadow. Drift, morph and tint are shared.
 */
export const ALIVE_CLOUD_SHAPES: Readonly<Record<AlivePreset, Readonly<CloudShape>>> = Object.freeze({
  soft: Object.freeze({
    scale: ALIVE.cloudScale,
    largeScale: ALIVE.cloudLargeScale,
    edgeLow: ALIVE.cloudEdgeLow,
    edgeHigh: ALIVE.cloudEdgeHigh,
    darkening: ALIVE.cloudDarkening,
    fineWeight: ALIVE.cloudFineWeight,
    stretch: ALIVE.cloudStretch,
    waterBodyWeight: ALIVE.waterBodyWeight,
    waterSkyWeight: ALIVE.waterSkyWeight,
  }),
  deep: Object.freeze({
    scale: 6,
    largeScale: 14,
    edgeLow: 0.04,
    edgeHigh: 0.16,
    darkening: 0.33,
    fineWeight: 0.75,
    stretch: 1.35,
    // Owner 2026-11-13: clouds must also show on the water. Body/seabed 0.85 (core 0.72) and sky reflection 0.95
    // (core 0.69) keep the shadow continuous across the coast with land (0.67); glints stay at 1.0. The attenuation is
    // a palette-domain multiply with the shared faint cool tint, so the turquoise hue is kept (no murky stain).
    waterBodyWeight: 0.85,
    waterSkyWeight: 0.95,
  }),
});

const cloudTintLuma = 0.98 * 0.2126 + 0.99 * 0.7152 + 1.01 * 0.0722;
/** Rec.709-normalized cool tint; neutral gaps and faded clouds must not inherit the full tint. */
export const ALIVE_CLOUD_TINT = Object.freeze([0.98, 0.99, 1.01].map((v) => v / cloudTintLuma));

/** CPU reference for the shader's palette-domain RGB multiplier (debug retains the untinted grey value). */
export function cloudShadowColour(
  coverage: number,
  scale = 1,
  dayWeight = 1,
  weight = 1,
  shape: Readonly<CloudShape> = ALIVE_CLOUD_SHAPES.soft,
): number[] {
  const grey = 1 - coverage * scale * dayWeight * shape.darkening;
  const tintWeight = Math.min(1, Math.max(0, (1 - grey) / shape.darkening));
  return ALIVE_CLOUD_TINT.map((tint) => mix(1, grey * (1 + (tint - 1) * tintWeight), weight));
}

const bounded = (value: number | undefined, fallback: number, max: number) =>
  value !== undefined && Number.isFinite(value) ? Math.min(max, Math.max(0, value)) : fallback;

export function resolveAlive(change: AliveChange, previous: Readonly<AliveOptions> = ALIVE_DEFAULTS): AliveOptions {
  if (typeof change === 'boolean')
    return { ...previous, clouds: change, wind: false, debugCloud: change && previous.debugCloud };
  const preset = change.preset;
  return {
    clouds: change.clouds ?? previous.clouds,
    preset: preset === 'soft' || preset === 'deep' ? preset : previous.preset,
    wind: change.wind ?? previous.wind,
    cloud: bounded(change.cloud, previous.cloud, 2),
    sway: bounded(change.sway, previous.sway, 3),
    speed: bounded(change.speed, previous.speed, 4),
    debugCloud: change.debugCloud ?? previous.debugCloud,
  };
}

/** Both benches use alive=1; the game uses hd-alive=1. No implicit enable from tunables. */
export function parseAliveParams(params: URLSearchParams, flag = 'alive'): AliveOptions {
  const number = (key: string): number | undefined => {
    const value = params.get(key)?.trim();
    return value ? Number(value) : undefined;
  };
  return resolveAlive({
    clouds: params.get(flag) === '1',
    preset: params.get('alive-preset')?.trim() as AlivePreset | undefined,
    wind: params.get(flag) === '1' && params.get('alive-wind') === '1',
    cloud: number('alive-cloud'),
    sway: number('alive-sway'),
    speed: number('alive-speed'),
    debugCloud: params.get('alive-debug') === 'cloud',
  });
}

const fract = (x: number) => x - Math.floor(x);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (x: number) => x * x * (3 - 2 * x);

/** Small polynomial hash, mirrored in livingWorldNodes; no sine hash or repeating texture tile. */
export function aliveHash(x: number, z: number): number {
  let a = fract(x * 0.1031);
  let b = fract(z * 0.1031);
  let c = a;
  const d = a * (b + 33.33) + b * (c + 33.33) + c * (a + 33.33);
  a += d;
  b += d;
  c += d;
  return fract((a + b) * c);
}

export function aliveNoise(x: number, z: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const u = smooth(fract(x));
  const v = smooth(fract(z));
  return mix(
    mix(aliveHash(ix, iz), aliveHash(ix + 1, iz), u),
    mix(aliveHash(ix, iz + 1), aliveHash(ix + 1, iz + 1), u),
    v,
  );
}

/** Zero-centred gradient noise keeps nearby lit gaps even in small, zoomed-in crops. */
export function aliveGradientNoise(x: number, z: number): number {
  const ix = Math.floor(x),
    iz = Math.floor(z);
  const fx = fract(x),
    fz = fract(z);
  const fade = (v: number) => v * v * v * (v * (v * 6 - 15) + 10);
  const gradient = (ox: number, oz: number) => {
    const h = aliveHash(ix + ox, iz + oz);
    return ((h < 0.5 ? -1 : 1) * (fx - ox) + (fract(h * 2) < 0.5 ? -1 : 1) * (fz - oz)) * Math.SQRT1_2;
  };
  return mix(mix(gradient(0, 0), gradient(1, 0), fade(fx)), mix(gradient(0, 1), gradient(1, 1), fade(fx)), fade(fz));
}

export function aliveDrift(seconds: number, direction: number, speed = 1): [number, number] {
  const distance = seconds * speed * ALIVE.drift;
  return [Math.cos(direction) * distance, Math.sin(direction) * distance];
}

/** Broad clear gaps and full patches, with a slow sub-cell evolution in addition to advection. */
export function cloudCoverage(
  x: number,
  z: number,
  seconds: number,
  direction: number,
  speed = 1,
  shape: Readonly<CloudShape> = ALIVE_CLOUD_SHAPES.soft,
): number {
  const [dx, dz] = aliveDrift(seconds, direction, speed);
  const t = seconds * speed;
  // Stretch along the wind, then blend two lattices: irregular edges without a third noise evaluation.
  const along = ((x - dx) * Math.cos(direction) + (z - dz) * Math.sin(direction)) / shape.stretch;
  const across = -(x - dx) * Math.sin(direction) + (z - dz) * Math.cos(direction);
  const mx = (Math.sin(t * 0.07) * ALIVE.cloudMorph) / shape.scale;
  const mz = (Math.cos(t * 0.09) * ALIVE.cloudMorph) / shape.scale;
  const n =
    aliveGradientNoise(along / shape.scale + mx, across / shape.scale + mz) * shape.fineWeight +
    aliveGradientNoise(along / shape.largeScale + 17.2 - mz, across / shape.largeScale + 9.4 + mx) *
      (1 - shape.fineWeight);
  return smooth(Math.min(1, Math.max(0, (n - shape.edgeLow) / (shape.edgeHigh - shape.edgeLow))));
}

export function aliveDayWeight(night: number, overcast: number, airless = false): number {
  return airless ? 0 : (1 - bounded(night, 0, 1)) * (1 - bounded(overcast, 0, 1));
}

export function aliveGust(x: number, z: number, seconds: number, direction: number, speed = 1): number {
  const wave =
    (x * Math.cos(direction) + z * Math.sin(direction) - seconds * speed * ALIVE.drift) *
    ((Math.PI * 2) / ALIVE.gustLength);
  return (0.5 + 0.5 * Math.sin(wave)) ** 4;
}

/** Crown displacement in native pixels; the bottom is exactly pinned, including during a gust. */
export function treeSway(
  x: number,
  z: number,
  heightFraction: number,
  seconds: number,
  direction: number,
  sway = 1,
  speed = 1,
): number {
  const seed = aliveHash(x, z);
  const t = seconds * speed;
  const period = ALIVE.period + (seed - 0.5) * 0.8;
  const oscillation = Math.sin((t * Math.PI * 2) / period + seed * Math.PI * 2);
  return (
    oscillation *
    sway *
    (ALIVE.crownPixels + (ALIVE.gustPixels - ALIVE.crownPixels) * aliveGust(x, z, seconds, direction, speed)) *
    Math.min(1, Math.max(0, heightFraction)) ** ALIVE.heightPower
  );
}

/** Signed grass ripple shares the trees' travelling gust phase; no additional noise sample. */
export function grassRipple(x: number, z: number, seconds: number, direction: number, speed = 1): number {
  return (2 * aliveGust(x, z, seconds, direction, speed) - 1) * 0.05;
}
