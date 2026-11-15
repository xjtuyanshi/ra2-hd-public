/** RA2/YR source policy. Pure conversions shared by every LiveView host and the deterministic bench. */
import { engineLightDelta, type EngineLight, type EngineLighting } from '../../../ra2-assets/src/lights';
import { EFFECT_TABLE, type EffectTableRow } from '../live/effectTable';
import type { LiveAnim, LiveBeam, LiveEffects } from '../live/effects';
import { beamColor, mindControlPoints } from '../live/beams';
import { engineAnchor } from '../live/effectSprites';
import type { BuildingLightPlacement } from '../live/buildingLights';
import {
  lightLuminance,
  lightMoment,
  reconstructLight,
  type PhysicalLight,
  type Point,
  type Rgb,
  type LightOccluders,
} from '../realLight';
import { LEVEL_HEIGHT } from '../world';

/**
 * Light colours are the source's own hue, strongly desaturated: at night the light should mostly brighten what it
 * falls on, keeping the surface's hue. Saturated warm light turned moonlit grass olive/yellow (round-1 review, live
 * SOV07TMD night). Saturation (1 - min / max) is about 0.3 for fire and explosions.
 */
export const LIGHT_COLOURS = {
  fire: [1, 0.82, 0.68],
  explosion: [1, 0.85, 0.72],
  hot: [1, 0.96, 0.92],
  muzzle: [1, 0.9, 0.8],
  tesla: capLightColour(beamColor([82, 80, 255]), 0.35),
  chrono: [0.8, 0.92, 1],
  psychic: capLightColour(beamColor([190, 90, 255]), 0.35),
  iron: [1, 0.6, 0.55],
} as const;
/** Saturation cap of light taken from a beam's or bolt's own colour. */
export const BEAM_LIGHT_SATURATION = 0.3;
export interface LightPolicy {
  lookClass?: PhysicalLight['lookClass'];
  height: number;
  reach?: number;
  /** Point/tube radius; for a Gaussian profile the falloff width (cells). */
  radius: number;
  ground: number;
  /** Calibrate a point at ground centre instead of losing its authored peak to source elevation. */
  groundPeak?: boolean;
  viewProjectedFootprint?: boolean;
  colour: Rgb;
  day: number;
  hold?: number;
  fade?: number;
  flicker?: number;
  continuous?: boolean;
  profile?: 'gaussian';
}
/** Live night beams need a 0.28-cell Gaussian centred on the strand, not a detached 0.44-cell road. */
const BEAM_SIGMA = 0.28;
const beam = (ground: number, colour: Rgb, day: number, radius = BEAM_SIGMA) =>
  ({
    reach: 1.2,
    height: 0.5,
    radius,
    ground,
    colour,
    day,
    profile: 'gaussian',
    viewProjectedFootprint: true,
  }) as const;
/**
 * Source-sized Gaussian falloffs (`radius` is the width, the window closes at 2.7 widths); `ground` is the peak
 * irradiance in noon units at the source, or at ground centre when groundPeak is set (including elevation/window
 * losses). Projected beams light a narrow soft line; fires and wrecks have a small flickering glow; superweapons
 * keep their design reach.
 */
export const REAL_LIGHT_POLICIES = {
  fire: {
    reach: 1.6,
    height: 0.4,
    radius: 0.6,
    ground: 0.8,
    colour: LIGHT_COLOURS.fire,
    day: 0,
    flicker: 0.3,
    continuous: true,
    profile: 'gaussian',
  },
  burning: {
    lookClass: 'wreck',
    reach: 1.2,
    height: 0.3,
    radius: 0.45,
    // At 0.1 ambient: 0.3*log1p(0.12/0.3) = 0.101; display ratio (2.01)^(1/2.2) = 1.37.
    ground: 0.12,
    groundPeak: true,
    colour: LIGHT_COLOURS.fire,
    day: 0,
    flicker: 0.15,
    continuous: true,
    profile: 'gaussian',
  },
  muzzle: {
    reach: 1.2,
    height: 0.5,
    radius: 0.44,
    ground: 0.4,
    colour: LIGHT_COLOURS.muzzle,
    day: 0.25,
    hold: 2 / 62.5,
    fade: 0.03,
    profile: 'gaussian',
  },
  // Chrono warps and the Chrono Legionnaire's erasure pulse repeatedly on one spot: a 6-cell pool lit a 5-cell
  // patch of night grass for 20 s around one erased tank (live SOV07TMD, frames 450-1050). At 2 cells and 0.5 it still
  // read as a flat white pool round the tank where the original shows none (round-6 review). The 1697702 faint
  // 0.18 pool remains the cur preset; chronoPool=0 suppresses it at presentation time by default (round 7).
  chrono: {
    lookClass: 'chrono',
    reach: 1.4,
    height: 0.5,
    radius: 0.5,
    ground: 0.18,
    colour: LIGHT_COLOURS.chrono,
    day: 0.15,
    profile: 'gaussian',
  },
  dominator: {
    reach: 10,
    height: 2,
    radius: 3.7,
    ground: 3,
    colour: LIGHT_COLOURS.psychic,
    day: 0.15,
    profile: 'gaussian',
  },
  iron: { reach: 6, height: 1, radius: 2.2, ground: 2, colour: LIGHT_COLOURS.iron, day: 0.15, profile: 'gaussian' },
  nuke: {
    reach: 12,
    height: 3,
    radius: 4.4,
    ground: 12,
    colour: LIGHT_COLOURS.hot,
    day: 0.3,
    hold: 0.15,
    fade: 2,
    profile: 'gaussian',
  },
  tesla: beam(0.12, LIGHT_COLOURS.tesla, 0.15),
  psychic: beam(0.12, LIGHT_COLOURS.psychic, 0.15),
  prism: beam(0.12, LIGHT_COLOURS.hot, 0.15),
  temporal: { ...beam(0.1, LIGHT_COLOURS.chrono, 0.1), lookClass: 'chrono' },
  rad: beam(0.1, [0.75, 1, 0.72], 0.1),
  eruption: beam(0.12, LIGHT_COLOURS.fire, 0.1),
  sonic: beam(0.1, [0.75, 0.92, 1], 0.1),
  magnetron: beam(0.1, LIGHT_COLOURS.psychic, 0.1),
  link: beam(0.04, LIGHT_COLOURS.psychic, 0, 0.35),
  storm: {
    reach: 4,
    height: 3,
    radius: 1.5,
    ground: 6,
    colour: LIGHT_COLOURS.tesla,
    day: 0.2,
    hold: 0.1,
    fade: 0.15,
    profile: 'gaussian',
  },
} as const satisfies Record<string, LightPolicy>;
const ANIM_POLICY: Record<string, keyof typeof REAL_LIGHT_POLICIES> = {
  FIRE01: 'fire',
  FIRE02: 'fire',
  FIRE03: 'fire',
  INITFIRE: 'fire',
  CHRONOSK: 'chrono',
  CHRONOTG: 'chrono',
  WARPIN: 'chrono',
  WARPOUT: 'chrono',
  PDFXLOC: 'dominator',
  PDFXCLD: 'dominator',
  IRONBLST: 'iron',
  IRONFX: 'iron',
  NUKETO: 'nuke',
  WCLBOLT1: 'storm',
  WCLBOLT3: 'storm',
};
export function capLightColour(colour: Rgb, saturation: number): [number, number, number] {
  const peak = Math.max(...colour, 1e-8);
  const current = 1 - Math.min(...colour) / peak;
  if (current <= saturation) return [...colour];
  // Interpolate toward equal-energy white, preserving HSV hue instead of clipping individual dark channels.
  const share = saturation / Math.max(current, 1e-8);
  const capped = colour.map((c) => peak + (c - peak) * share) as [number, number, number];
  const k = lightLuminance(colour) / Math.max(lightLuminance(capped), 1e-8);
  return capped.map((c) => c * k) as [number, number, number];
}
/** Warm flash, HSV saturation 0.32. Achromatic sources inherit the explosion's warm-white hue. */
export function warmFlashColour(source: Rgb): [number, number, number] {
  const hi = Math.max(...source),
    lo = Math.min(...source);
  const colour = hi - lo < 0.01 ? LIGHT_COLOURS.explosion : source;
  const peak = Math.max(...colour, 1e-8),
    range = peak - Math.min(...colour);
  return colour.map((c) => 1 - ((peak - c) / Math.max(range, 1e-8)) * 0.32) as [number, number, number];
}
const xyz = (p: { x: number; y: number; z: number }): Point => [p.x, p.y, p.z];
export const animWorld = (anim: Pick<LiveAnim, 'x' | 'y' | 'z'>): Point => xyz(engineAnchor(anim.x, anim.y, anim.z));
const energyCache = new Map<string, number>();
/** Convert source policy to physical radiance; point ground peaks include elevation and window losses. */
export function policyLight(policy: LightPolicy, position: Point, end?: Point): PhysicalLight {
  const light: PhysicalLight = {
    position,
    end,
    lookClass: policy.lookClass,
    radius: policy.radius,
    intensity: policy.colour,
    dayFactor: policy.day,
    reach: policy.reach,
    viewProjectedFootprint: policy.viewProjectedFootprint,
    occlude: !policy.hold || policy.hold >= 0.15,
    flicker: policy.flicker,
    ...(policy.profile ? { profile: policy.profile } : {}),
  };
  // Gaussian peaks are authored at the source or directly beneath it, never far out on a tail: that older
  // calibration made a 0.44-cell beam 13 times brighter at its core than its nominal value.
  if (policy.profile === 'gaussian') {
    const centre = policy.groundPeak
      ? lightLuminance(lightMoment({ ...light, position: [0, policy.height, 0], end: undefined }, [0, 0, 0]).s)
      : lightLuminance(policy.colour);
    const k = policy.ground / Math.max(centre, 1e-8);
    light.intensity = policy.colour.map((c) => c * k) as [number, number, number];
    return light;
  }
  // Author tube energy at a canonical 1-cell length and nominal height, independent of the observed beam length.
  const reference: PhysicalLight = end
    ? { ...light, position: [-0.5, policy.height, 0], end: [0.5, policy.height, 0] }
    : { ...light, position: [0, policy.height, 0] };
  const receiver: Point = end ? [0, 0, 0.5] : [0.5, 0, 0];
  const key = JSON.stringify([
    policy.height,
    policy.radius,
    policy.ground,
    policy.colour,
    policy.reach,
    !!end,
    policy.profile ?? '',
  ]);
  const cached = energyCache.get(key);
  if (cached !== undefined) {
    light.intensity = policy.colour.map((c) => c * cached) as [number, number, number];
    return light;
  }
  let lo = 0,
    hi = 1;
  const sample = (k: number) => {
    const l = { ...reference, intensity: policy.colour.map((c) => c * k) as [number, number, number] };
    const m = lightMoment(l, receiver);
    return lightLuminance(reconstructLight(m.s, m.d, [0, 1, 0]));
  };
  while (sample(hi) < policy.ground && hi < 1e8) hi *= 2;
  for (let i = 0; i < 48; i++) {
    const mid = (lo + hi) / 2;
    if (sample(mid) < policy.ground) lo = mid;
    else hi = mid;
  }
  if (energyCache.size >= 1024) energyCache.clear();
  energyCache.set(key, (lo + hi) / 2);
  light.intensity = policy.colour.map((c) => (c * (lo + hi)) / 2) as [number, number, number];
  return light;
}
export function animLightPolicy(
  type: string,
  row: EffectTableRow = EFFECT_TABLE[type]!,
  fps = 62.5,
): LightPolicy | null {
  const named = ANIM_POLICY[type.toUpperCase()];
  if (named) {
    const p: LightPolicy = REAL_LIGHT_POLICIES[named];
    return { ...p, hold: p.hold ?? Math.max(0.03, (row?.life ?? 30) / fps), fade: p.fade ?? 0.15 };
  }
  if (!row) return null;
  if (row.c === 'chrono') return { ...REAL_LIGHT_POLICIES.chrono, hold: row.life / fps, fade: 0.15 };
  if (row.c === 'fire') return { ...REAL_LIGHT_POLICIES.fire, continuous: !!row.loop, hold: row.life / fps };
  if (row.c === 'muzzle') return { ...REAL_LIGHT_POLICIES.muzzle, hold: 2 / fps };
  if (row.c === 'explosion' || row.c === 'collapse') {
    const rho = Math.max(0.2, row.light?.[3] ?? row.s);
    const reach = Math.min(3, Math.max(1.9, 1.7 + rho));
    return {
      // A short warm flash reaching 1.5-2.5 cells, not a pool that outlives the fireball: the art's
      // light radius rho sets the size and the fade (0.2-0.5 s). Raised a little so it reaches past its own art.
      height: 0.4 + 0.4 * Math.min(rho, 1.5),
      // Width half the reach: with the cosine of a low source the flash reads (> 6 % of its peak) out to about
      // 1.3 cells for EXPLOSML, 1.9 for EXPLOLRG and 2.3 for the largest blasts.
      radius: 0.5 * reach,
      // At ambient 0.1 noon, the logarithmic knee gives 0.416: (1 + 0.416/0.1)^(1/2.2) = 2.10
      // displayed night luminance. Compensate elevation rather than widening/overdriving the Gaussian.
      ground: 0.9,
      lookClass: 'flash',
      groundPeak: true,
      reach,
      colour: LIGHT_COLOURS.explosion,
      day: 0.15,
      hold: 0.08,
      fade: Math.min(0.5, 0.25 + 0.15 * rho),
      profile: 'gaussian',
    };
  }
  if (row.light)
    return {
      height: 0.5,
      radius: Math.min(0.8, Math.max(0.35, row.light[3] * 0.4)),
      reach: Math.min(2.2, Math.max(1, row.light[3] * 1.1)),
      ground: Math.min(0.8, row.s),
      colour: capLightColour([row.light[0], row.light[1], row.light[2]], BEAM_LIGHT_SATURATION),
      day: 0.15,
      hold: row.life / fps,
      fade: 0.15,
      profile: 'gaussian',
    };
  return null;
}
export function animPhysicalLight(anim: LiveAnim, policy: LightPolicy): PhysicalLight {
  const p = animWorld(anim);
  const row = EFFECT_TABLE[anim.type];
  if (row?.c === 'fire' && (anim.attachedTo?.kind === 'infantry' || anim.attachedTo?.kind === 'vehicle'))
    policy = { ...policy, ...REAL_LIGHT_POLICIES.burning };
  // Muzzle anims already carry a native FLH height when available; only a ground anchor needs the nominal fallback.
  const height = row?.c === 'muzzle' && p[1] > 0 ? p[1] : p[1] + policy.height;
  const position: Point = [p[0], height, p[2]];
  const light = policyLight(policy, position, /^WCLBOLT/.test(anim.type) ? [p[0], p[1], p[2]] : undefined);
  if (row?.c === 'explosion' || row?.c === 'collapse') {
    const gain = lightLuminance(light.intensity) / lightLuminance(LIGHT_COLOURS.hot);
    light.initialIntensity = LIGHT_COLOURS.hot.map((c) => c * gain) as [number, number, number];
    light.hotSeconds = 0.12;
    const colour = warmFlashColour(row.light ? [row.light[0], row.light[1], row.light[2]] : LIGHT_COLOURS.explosion);
    const warmGain = lightLuminance(light.intensity) / Math.max(lightLuminance(colour), 1e-8);
    light.warmIntensity = colour.map((c) => c * warmGain) as [number, number, number];
  }
  light.seed = anim.uid / 65536;
  return light;
}
/** Endpoints are native weapon coordinates, never the old target-only pulse. No decorative beams enter this list. */
export function nativeBeamLight(beam: LiveBeam): PhysicalLight {
  const psychic = beam.kind === 'bolt' && beam.colour[0] > 130;
  const policy: LightPolicy =
    beam.kind === 'bolt'
      ? {
          ...REAL_LIGHT_POLICIES[psychic ? 'psychic' : 'tesla'],
          colour: capLightColour(beamColor(beam.colour), BEAM_LIGHT_SATURATION),
        }
      : { ...REAL_LIGHT_POLICIES.prism, colour: capLightColour(beamColor(beam.colour), BEAM_LIGHT_SATURATION) };
  return policyLight(policy, animWorld(beam.source as LiveAnim), animWorld(beam.target as LiveAnim));
}
/** Coil-to-unit bolts receive a short, small target glow as well as their projected line. */
export function nativeBoltImpactLight(beam: LiveBeam): PhysicalLight | null {
  if (beam.kind !== 'bolt') return null;
  const p = animWorld(beam.target as LiveAnim);
  return policyLight(
    {
      height: 0.15,
      radius: 0.38,
      reach: 1.05,
      ground: 0.12,
      groundPeak: true,
      colour: capLightColour(beamColor(beam.colour), BEAM_LIGHT_SATURATION),
      day: 0.15,
      profile: 'gaussian',
    },
    [p[0], p[1] + 0.15, p[2]],
  );
}
export function auxiliaryBeamLights(effects: LiveEffects): { key: string; light: PhysicalLight }[] {
  const out: { key: string; light: PhysicalLight }[] = [];
  for (const b of effects.radBeams ?? []) {
    if (b.type === 'rad' && !b.sustained && b.age >= b.period) continue;
    const p: LightPolicy = {
      ...REAL_LIGHT_POLICIES[b.type],
      colour: b.type === 'temporal' ? LIGHT_COLOURS.chrono : capLightColour(beamColor(b.color), BEAM_LIGHT_SATURATION),
    };
    out.push({
      key: `rad:${b.id}`,
      light: policyLight(p, animWorld(b.source as LiveAnim), animWorld(b.target as LiveAnim)),
    });
  }
  for (const b of effects.waves ?? []) {
    const p = b.type === 'laser' || b.type === 'biglaser' ? REAL_LIGHT_POLICIES.prism : REAL_LIGHT_POLICIES[b.type];
    const light = policyLight(p, animWorld(b.source as LiveAnim), animWorld(b.target as LiveAnim));
    light.intensity = light.intensity.map((c) => c * (1 - Math.max(0, Math.min(1, b.progress)))) as [
      number,
      number,
      number,
    ];
    out.push({ key: `wave:${b.id}`, light });
  }
  for (const link of effects.mindControlLinks ?? []) {
    const points = mindControlPoints(link);
    // Four pieces follow the same captured arch, preserving faint per-length energy rather than 32 point pools.
    for (let i = 0; i < 4; i++)
      out.push({
        key: `link:${link.id}:${i}`,
        light: policyLight(REAL_LIGHT_POLICIES.link, xyz(points[i * 8]!), xyz(points[(i + 1) * 8]!)),
      });
  }
  return out;
}
/** A visible art head in world cells, independent of the invisible engine source dummy. */
export interface LampArtHead {
  x: number;
  z: number;
  height: number;
}
export const isLampArt = (type: string): boolean => /LAMP|LITE|LMP/.test(type);
export function lampArtHead(offsetY: number): number {
  return Math.max(0.3, (15 - offsetY - 5) / (Math.cos(Math.PI / 6) * (60 / Math.SQRT2)));
}
export function nearestLampHead(source: EngineLight, heads: readonly LampArtHead[]): number {
  let distance = 1,
    height = 1.25;
  for (const head of heads) {
    const d = Math.hypot(head.x - source.position[0] / 256, head.z - source.position[1] / 256);
    if (d <= distance) {
      distance = d;
      height = head.height;
    }
  }
  return height;
}

/**
 * Directional accent budget on top of the unchanged engine grade/pools. Calibrated at one cell in noon units;
 * the raw energy ceiling remains safe under every scene-relative knee.
 */
export const LAMP_GROUND_AT_ONE_CELL = { min: 0.04, max: 0.12 } as const;
/** Clamp the combined head/post energy before any receiver knee; terrain retains the legacy source only. */
export function engineLampLights(source: EngineLight, lighting: EngineLighting, head = 1.25): PhysicalLight[] {
  if (!(source.intensity > 0)) return [];
  const distance = 1;
  const grid = {
    width: 1,
    height: 1,
    exists: new Uint8Array([1]),
    z: new Uint8Array([Math.max(0, Math.round((source.position[2] ?? 0) / 104))]),
  };
  const delta = engineLightDelta(
    grid,
    lighting,
    [{ ...source, position: [128 - distance * 256, 128, source.position[2] ?? 0] }],
    lighting.ambient,
  );
  const colour = capLightColour([Math.max(0, delta[0]!), Math.max(0, delta[1]!), Math.max(0, delta[2]!)], 0.55);
  const target = lightLuminance(colour);
  if (!(target > 0)) return [];
  const base = ((source.position[2] ?? 0) / 104) * LEVEL_HEIGHT;
  const position: Point = [source.position[0] / 256, base + head, source.position[1] / 256];
  const lights: PhysicalLight[] = [
    { position, radius: 0.15, intensity: colour, dayFactor: 0, occlude: true, reach: 2.5, planarReach: true },
    {
      position,
      end: [position[0], base + Math.max(0.1, head - 0.6), position[2]],
      radius: 0.08,
      intensity: colour.map((c) => c * 0.08) as [number, number, number],
      dayFactor: 0,
      occlude: true,
      reach: 2.5,
      planarReach: true,
    },
  ];
  const receiver: Point = [position[0] + distance, base, position[2]];
  const sample = (k: number) => {
    const sum: [number, number, number] = [0, 0, 0],
      direction: [number, number, number] = [0, 0, 0];
    for (const light of lights) {
      const m = lightMoment(
        { ...light, intensity: light.intensity.map((c) => c * k) as [number, number, number] },
        receiver,
      );
      m.s.forEach((c, i) => (sum[i]! += c));
      m.d.forEach((c, i) => (direction[i]! += c));
    }
    return lightLuminance(reconstructLight(sum, direction, [0, 1, 0]));
  };
  let lo = 0,
    hi = 1;
  const reachable = Math.min(LAMP_GROUND_AT_ONE_CELL.max, Math.max(LAMP_GROUND_AT_ONE_CELL.min, target));
  while (sample(hi) < reachable && hi < 1e8) hi *= 2;
  for (let i = 0; i < 48; i++) {
    const mid = (lo + hi) / 2;
    if (sample(mid) < reachable) lo = mid;
    else hi = mid;
  }
  return lights.map((l) => ({
    ...l,
    intensity: l.intensity.map((c) => (c * (lo + hi)) / 2) as [number, number, number],
  }));
}
/** Existing cores/bloom stay intact; their ground emission becomes source-sized and night-only. Idle coil omitted. */
export function buildingPhysicalLight(type: string, placement: BuildingLightPlacement): PhysicalLight | null {
  if (type === 'TESLA') return null;
  const p: LightPolicy = {
    height: Math.max(0.2, placement.head ?? 0.5),
    radius: type === 'NAPOWR' ? 0.418 : 0.18,
    ground: type === 'NAPOWR' ? 0.15 : 0.12,
    colour: capLightColour(placement.colour, 0.55),
    day: 0,
    flicker: placement.flicker,
  };
  // Normalize old emitter gain: the new physical energy table is independent of legacy pool exaggeration.
  const peak = Math.max(...p.colour, 1e-8);
  p.colour = p.colour.map((c) => c / peak) as [number, number, number];
  const light = policyLight(p, xyz(placement.headPosition));
  light.intensity = light.intensity.map((c) => c * (placement.physicalGain ?? 1)) as [number, number, number];
  return { ...light, groundOnly: placement.groundOnly, seed: placement.seed };
}

/** Cell footprints and art-derived top heights, supplied by the same game library as the drawn building. */
export interface BuildingOccluder {
  x: number;
  z: number;
  width: number;
  depth: number;
  top: number;
}
export function ra2LightOccluders(
  width: number,
  height: number,
  _levels: ArrayLike<number>,
  buildings: readonly BuildingOccluder[],
): LightOccluders {
  const heights = new Float32Array(width * height * 4);
  for (const b of buildings)
    for (let z = Math.max(0, b.z * 2); z < Math.min(height * 2, (b.z + b.depth) * 2); z++)
      for (let x = Math.max(0, b.x * 2); x < Math.min(width * 2, (b.x + b.width) * 2); x++) {
        const i = z * width * 2 + x;
        heights[i] = Math.max(heights[i]!, b.top);
      }
  return { width: width * 2, height: height * 2, texelsPerCell: 2, heights };
}
