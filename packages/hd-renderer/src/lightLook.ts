import { uniform } from 'three/tsl';

/** Presentation only. Zero-valued master switches reproduce 1697702, including its chrono pool. */
export const LIGHT_LOOK_CURRENT = {
  smokeVolume: 0,
  smokeWisp: 0,
  smokeWarp: 0,
  smokeStretch: 0,
  smokeDetail: 1,
  beamLight: 0,
  beamCore: 1,
  beamGlow: 1,
  beamFlicker: 0,
  beamImpact: 0,
  chronoPool: 1,
  flashWarm: 0,
  flashNeutral: 0,
  wreckGlow: 0,
} as const;
export const LIGHT_LOOK_NEW = {
  smokeVolume: 1,
  smokeWisp: 0.75,
  smokeWarp: 0.8,
  smokeStretch: 0.35,
  smokeDetail: 2,
  beamLight: 1,
  beamCore: 0.35,
  beamGlow: 1.9,
  beamFlicker: 0.12,
  beamImpact: 0.18,
  chronoPool: 0,
  flashWarm: 1,
  flashNeutral: 0.9,
  wreckGlow: 1,
} as const;
export type LightLookValues = { [K in keyof typeof LIGHT_LOOK_NEW]: number };
const lookUniform = (v: number) => uniform(v);
export type LightLook = { [K in keyof LightLookValues]: ReturnType<typeof lookUniform> };
export type LightLookArgument = 'cur' | 'new' | Partial<LightLookValues>;
export function createLightLook(): LightLook {
  return Object.fromEntries(Object.entries(LIGHT_LOOK_NEW).map(([k, v]) => [k, uniform(v as number)])) as LightLook;
}
/** Mutate existing uniforms, never graphs. Readback is a detached numeric snapshot for the development bridge. */
export function lightLook(look: LightLook, arg?: LightLookArgument): LightLookValues {
  const values = arg === 'cur' ? LIGHT_LOOK_CURRENT : arg === 'new' ? LIGHT_LOOK_NEW : arg;
  if (values)
    for (const [key, value] of Object.entries(values)) {
      if (!Object.hasOwn(LIGHT_LOOK_NEW, key) || !Number.isFinite(value))
        throw new Error(`Invalid light look parameter: ${key}`);
    }
  if (values)
    for (const [key, value] of Object.entries(values)) {
      const k = key as keyof LightLookValues;
      look[k].value = Math.max(0, Math.min(k === 'smokeDetail' || k === 'beamGlow' ? 4 : 1, value!));
    }
  return Object.fromEntries(Object.entries(look).map(([k, v]) => [k, v.value])) as LightLookValues;
}
/** Smooth, calm 3 Hz wreck convection; bounded to +/-10%, independent of native simulation time. */
export function wreckFlicker(seed: number, seconds: number): number {
  return (
    1 + 0.1 * (0.7 * Math.sin(seconds * Math.PI * 6 + seed * 41) + 0.3 * Math.sin(seconds * Math.PI * 4 + seed * 17))
  );
}
