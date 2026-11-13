/** Opt-in presentation only; all distances are cells and all times are renderer seconds. */
export interface AliveOptions {
  clouds: boolean;
  wind: boolean;
  cloud: number;
  sway: number;
  speed: number;
  debugCloud: boolean;
}

export type AliveChange = boolean | Partial<AliveOptions>;

export const ALIVE_DEFAULTS: Readonly<AliveOptions> = Object.freeze({
  clouds: false,
  wind: false,
  cloud: 1,
  sway: 1,
  speed: 1,
  debugCloud: false,
});

export const ALIVE = Object.freeze({
  cloudScale: 5.75,
  cloudLargeScale: 12,
  cloudEdgeLow: -0.085,
  cloudEdgeHigh: 0.145,
  cloudDarkening: 0.36,
  drift: 0.85,
  crownPixels: 3.8,
  gustPixels: 6.2,
  period: 3,
  gustLength: 8.5,
  heightPower: 1.4,
});

const cloudTintLuma = 0.9 * 0.2126 + 0.95 * 0.7152 + 1.05 * 0.0722;
/** Rec.709-normalized cool tint; neutral gaps and faded clouds must not inherit the full tint. */
export const ALIVE_CLOUD_TINT = Object.freeze([0.9, 0.95, 1.05].map((v) => v / cloudTintLuma));

/** CPU reference for the shader's palette-domain RGB multiplier (debug retains the untinted grey value). */
export function cloudShadowColour(coverage: number, scale = 1, dayWeight = 1): number[] {
  const grey = 1 - coverage * scale * dayWeight * ALIVE.cloudDarkening;
  const tintWeight = Math.min(1, Math.max(0, (1 - grey) / ALIVE.cloudDarkening));
  return ALIVE_CLOUD_TINT.map((tint) => grey * (1 + (tint - 1) * tintWeight));
}

const bounded = (value: number | undefined, fallback: number, max: number) =>
  value !== undefined && Number.isFinite(value) ? Math.min(max, Math.max(0, value)) : fallback;

export function resolveAlive(change: AliveChange, previous: Readonly<AliveOptions> = ALIVE_DEFAULTS): AliveOptions {
  if (typeof change === 'boolean')
    return { ...previous, clouds: change, wind: change, debugCloud: change && previous.debugCloud };
  return {
    clouds: change.clouds ?? previous.clouds,
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
    wind: params.get(flag) === '1',
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
export function cloudCoverage(x: number, z: number, seconds: number, direction: number, speed = 1): number {
  const [dx, dz] = aliveDrift(seconds, direction, speed);
  const t = seconds * speed;
  const px = (x - dx) / ALIVE.cloudScale;
  const pz = (z - dz) / ALIVE.cloudScale;
  const mx = (Math.sin(t * 0.07) * 0.8) / ALIVE.cloudScale;
  const mz = (Math.cos(t * 0.09) * 0.8) / ALIVE.cloudScale;
  // Gradient noise crosses zero locally, avoiding the broad exposure drift of value-noise plateaus.
  const n =
    aliveGradientNoise(px + mx, pz + mz) * 0.85 +
    aliveGradientNoise((x - dx) / ALIVE.cloudLargeScale + 17.2 - mz, (z - dz) / ALIVE.cloudLargeScale + 9.4 + mx) *
      0.15;
  return smooth(Math.min(1, Math.max(0, (n - ALIVE.cloudEdgeLow) / (ALIVE.cloudEdgeHigh - ALIVE.cloudEdgeLow))));
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
