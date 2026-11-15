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
  beamSpread: 1,
  beamTail: 0,
  beamCoreGain: 0.82,
  chronoPool: 1,
  flashWarm: 0,
  flashNeutral: 0,
  flashSat: 0.32,
  flashSize: 1,
  smokeEdge: 0.4,
  smokeRidge: 0,
  wreckGlow: 0,
  beamOverbright: 1,
  beamKnee: 0,
  beamDepthLift: 0,
  beamFade: 0,
  beamOver: 0,
  beamAgeKnee: 0,
  beamAgeWhite: 0,
  beamAgeCore: 0,
  beamAgeClip: 0,
} as const;
export const LIGHT_LOOK_NEW = {
  smokeVolume: 1,
  smokeWisp: 0.75,
  smokeWarp: 0.8,
  smokeStretch: 0.35,
  smokeDetail: 2,
  beamLight: 1,
  beamCore: 0.2,
  beamGlow: 2.4,
  beamFlicker: 0.12,
  beamImpact: 0.7,
  beamSpread: 2,
  beamTail: 0.5,
  beamCoreGain: 1.15,
  chronoPool: 0,
  flashWarm: 1,
  flashNeutral: 0.9,
  flashSat: 0.85,
  flashSize: 0.55,
  smokeEdge: 0.22,
  smokeRidge: 0.6,
  wreckGlow: 1,
  beamOverbright: 1,
  beamKnee: 0,
  beamDepthLift: 0,
  beamFade: 0,
  beamOver: 0,
  beamAgeKnee: 0,
  beamAgeWhite: 0,
  beamAgeCore: 0,
  beamAgeClip: 0,
} as const;
/**
 * hd-light=beam (round 8, the default-on candidate): round 7's laser light, chrono and warm-flash fixes with the smoke
 * left exactly as Build 13 draws it (the smoke graphs are not even built in this preset; see AtmosphereNodes).
 * Three laser fixes on top of round 7:
 * - beamOverbright: the glow takes the hue the ORIGINAL shows at the beam centre. The original adds the house colour
 *   over the scene, so its two-channel colours overbright: a live SOV07TMD Prism beam with the inner colour
 *   (137, 0, 197) displayed (225, 43, 255), i.e. about 1.64x per display channel, clipped at white. Round 7 used the
 *   inner colour's own linear ratios, which showed ~272 deg instead of the original's ~291 deg (violet read blue).
 *   Single-channel colours (the red house) are unchanged by this.
 * - beamKnee: the soft saturation acts on the glow's amplitude, not on its radial profile, so a full-gain beam keeps
 *   the same falloff as a budgeted one instead of growing a flat saturated plateau (the "neon tube"). With a slightly
 *   smaller glow radius and tail (2 and 0.4; live SOV07TMD): the violet tower beam's bright band is ~19-25 device px
 *   instead of 34, while the budgeted red Prism tank beams keep round 7's look.
 * - beamDepthLift: a gain on the depth lift of glow fragments towards the camera, by the height their screen offset
 *   would need above the ground (plus a little at the core). A glow below a low, near-horizontal beam is otherwise cut
 *   off by the terrain it is drawn over; the original draws its lasers over the scene, so a generous lift is faithful.
 * - beamAgeKnee / beamAgeWhite / beamAgeCore / beamAgeClip (with beamFade): past 0.3 of its life (ramping over the
 *   next 0.2) a fading laser loses its additive-clipping white and 30% of its overbright, and its core widens up to
 *   2x, so the aged violet Prism beam is the original's 2-3 px deep violet line rather than a pale lavender thread.
 *   A beam younger than the knee (the fresh Prism beam at live SOV07TMD frame 165 is 0.13-0.27 through its life)
 *   keeps the look it had. Aged beam at frame 393: hue/saturation/value 273/0.68/0.71, original 273/0.77/0.72.
 */
export const LIGHT_LOOK_BEAM = {
  ...LIGHT_LOOK_NEW,
  smokeVolume: LIGHT_LOOK_CURRENT.smokeVolume,
  smokeWisp: LIGHT_LOOK_CURRENT.smokeWisp,
  smokeWarp: LIGHT_LOOK_CURRENT.smokeWarp,
  smokeStretch: LIGHT_LOOK_CURRENT.smokeStretch,
  smokeDetail: LIGHT_LOOK_CURRENT.smokeDetail,
  smokeEdge: LIGHT_LOOK_CURRENT.smokeEdge,
  smokeRidge: LIGHT_LOOK_CURRENT.smokeRidge,
  beamGlow: 2,
  beamTail: 0.4,
  beamOverbright: 1.64,
  beamKnee: 1,
  beamDepthLift: 2.5,
  beamFade: 1,
  beamOver: 1,
  beamAgeKnee: 0.3,
  beamAgeWhite: 1,
  beamAgeCore: 1,
  beamAgeClip: 0.3,
} as const;
export type LightLookPreset = 'new' | 'beam';
/** Parameters that are gains or widths rather than 0..1 shares. */
const WIDE = new Set<string>([
  'smokeDetail',
  'beamGlow',
  'beamSpread',
  'beamCoreGain',
  'beamImpact',
  'beamOverbright',
  'beamDepthLift',
  'beamAgeCore',
]);
export type LightLookValues = { [K in keyof typeof LIGHT_LOOK_NEW]: number };
const lookUniform = (v: number) => uniform(v);
export type LightLook = { [K in keyof LightLookValues]: ReturnType<typeof lookUniform> };
export type LightLookArgument = 'cur' | 'new' | 'beam' | Partial<LightLookValues>;
export function createLightLook(preset: LightLookPreset = 'new'): LightLook {
  const values: LightLookValues = preset === 'beam' ? LIGHT_LOOK_BEAM : LIGHT_LOOK_NEW;
  return Object.fromEntries(Object.entries(values).map(([k, v]) => [k, uniform(v)])) as LightLook;
}
/** Mutate existing uniforms, never graphs. Readback is a detached numeric snapshot for the development bridge. */
export function lightLook(look: LightLook, arg?: LightLookArgument): LightLookValues {
  const values =
    arg === 'cur' ? LIGHT_LOOK_CURRENT : arg === 'new' ? LIGHT_LOOK_NEW : arg === 'beam' ? LIGHT_LOOK_BEAM : arg;
  if (values)
    for (const [key, value] of Object.entries(values)) {
      if (!Object.hasOwn(LIGHT_LOOK_NEW, key) || !Number.isFinite(value))
        throw new Error(`Invalid light look parameter: ${key}`);
    }
  if (values)
    for (const [key, value] of Object.entries(values)) {
      const k = key as keyof LightLookValues;
      look[k].value = Math.max(0, Math.min(WIDE.has(k) ? 4 : 1, value!));
    }
  return Object.fromEntries(Object.entries(look).map(([k, v]) => [k, v.value])) as LightLookValues;
}
/** Smooth, calm 3 Hz wreck convection; bounded to +/-10%, independent of native simulation time. */
export function wreckFlicker(seed: number, seconds: number): number {
  return (
    1 + 0.1 * (0.7 * Math.sin(seconds * Math.PI * 6 + seed * 41) + 0.3 * Math.sin(seconds * Math.PI * 4 + seed * 17))
  );
}
