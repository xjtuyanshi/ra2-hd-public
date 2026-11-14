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
  irradianceKnee,
  type PhysicalLight,
  type Point,
  type Rgb,
  type LightOccluders,
} from '../realLight';
import { LEVEL_HEIGHT } from '../world';

export const LIGHT_COLOURS = {
  fire: [1, 0.62, 0.36],
  explosion: [1, 0.7, 0.45],
  hot: [1, 0.9, 0.8],
  tesla: [0.6, 0.7, 1],
  chrono: [0.75, 0.9, 1],
  psychic: [0.8, 0.55, 1],
  iron: [1, 0.45, 0.4],
} as const;
export interface LightPolicy {
  height: number;
  radius: number;
  ground: number;
  colour: Rgb;
  day: number;
  hold?: number;
  fade?: number;
  flicker?: number;
  continuous?: boolean;
}
export const REAL_LIGHT_POLICIES = {
  fire: { height: 0.4, radius: 0.25, ground: 0.5, colour: LIGHT_COLOURS.fire, day: 0, flicker: 0.18, continuous: true },
  burning: {
    height: 0.3,
    radius: 0.15,
    ground: 0.25,
    colour: LIGHT_COLOURS.fire,
    day: 0,
    flicker: 0.18,
    continuous: true,
  },
  muzzle: { height: 0.5, radius: 0.1, ground: 0.6, colour: LIGHT_COLOURS.hot, day: 0.25, hold: 2 / 62.5, fade: 0.03 },
  chrono: { height: 0.5, radius: 0.6, ground: 1, colour: LIGHT_COLOURS.chrono, day: 0.15 },
  dominator: { height: 2, radius: 2, ground: 3, colour: LIGHT_COLOURS.psychic, day: 0.15 },
  iron: { height: 1, radius: 1.5, ground: 2, colour: LIGHT_COLOURS.iron, day: 0.15 },
  nuke: { height: 3, radius: 3, ground: 12, colour: LIGHT_COLOURS.hot, day: 0.3, hold: 0.15, fade: 2 },
  tesla: { height: 0.5, radius: 0.12, ground: 0.9, colour: LIGHT_COLOURS.tesla, day: 0.2 },
  psychic: { height: 0.5, radius: 0.12, ground: 0.9, colour: LIGHT_COLOURS.psychic, day: 0.2 },
  prism: { height: 0.5, radius: 0.1, ground: 0.8, colour: LIGHT_COLOURS.hot, day: 0.2 },
  temporal: { height: 0.5, radius: 0.1, ground: 0.4, colour: LIGHT_COLOURS.chrono, day: 0.15 },
  rad: { height: 0.5, radius: 0.1, ground: 0.4, colour: [0.65, 1, 0.6], day: 0.15 },
  eruption: { height: 0.5, radius: 0.1, ground: 0.4, colour: LIGHT_COLOURS.fire, day: 0.15 },
  sonic: { height: 0.5, radius: 0.15, ground: 0.4, colour: [0.65, 0.9, 1], day: 0.15 },
  magnetron: { height: 0.5, radius: 0.15, ground: 0.4, colour: LIGHT_COLOURS.psychic, day: 0.15 },
  link: { height: 0.5, radius: 0.08, ground: 0.08, colour: LIGHT_COLOURS.psychic, day: 0 },
  storm: { height: 3, radius: 0.2, ground: 6, colour: LIGHT_COLOURS.tesla, day: 0.2, hold: 0.1, fade: 0.15 },
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
  const peak = Math.max(...colour, 1e-8),
    floor = peak * (1 - saturation);
  const capped = colour.map((c) => Math.max(floor, c)) as [number, number, number];
  const k = lightLuminance(colour) / Math.max(lightLuminance(capped), 1e-8);
  return capped.map((c) => c * k) as [number, number, number];
}
const xyz = (p: { x: number; y: number; z: number }): Point => [p.x, p.y, p.z];
export const animWorld = (anim: Pick<LiveAnim, 'x' | 'y' | 'z'>): Point => xyz(engineAnchor(anim.x, anim.y, anim.z));
const energyCache = new Map<string, number>();
/** Calibrate a policy's peak ground irradiance at half a cell; window is included, reach remains derived. */
export function policyLight(policy: LightPolicy, position: Point, end?: Point): PhysicalLight {
  const light: PhysicalLight = {
    position,
    end,
    radius: policy.radius,
    intensity: policy.colour,
    dayFactor: policy.day,
    occlude: !policy.hold || policy.hold >= 0.15,
    flicker: policy.flicker,
  };
  // Author tube energy at a canonical 1-cell length and nominal height, independent of the observed beam length.
  const reference: PhysicalLight = end
    ? { ...light, position: [-0.5, policy.height, 0], end: [0.5, policy.height, 0] }
    : { ...light, position: [0, policy.height, 0] };
  const receiver: Point = end ? [0, 0, 0.5] : [0.5, 0, 0];
  const key = JSON.stringify([policy.height, policy.radius, policy.ground, policy.colour, !!end]);
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
    return {
      height: 0.4 + 0.3 * rho,
      radius: 0.3 * rho,
      ground: 1.2 * rho ** 0.7,
      colour: LIGHT_COLOURS.explosion,
      day: 0.2,
      hold: 0.12,
      fade: row.life / fps + 0.6,
    };
  }
  if (row.light)
    return {
      height: 0.5,
      radius: Math.max(0.08, row.light[3] * 0.15),
      ground: Math.min(1, row.s),
      colour: capLightColour([row.light[0], row.light[1], row.light[2]], 0.55),
      day: 0.15,
      hold: row.life / fps,
      fade: 0.15,
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
  }
  light.seed = anim.uid / 65536;
  return light;
}
/** Endpoints are native weapon coordinates, never the old target-only pulse. No decorative beams enter this list. */
export function nativeBeamLight(beam: LiveBeam): PhysicalLight {
  const psychic = beam.kind === 'bolt' && beam.colour[0] > 130;
  const policy: LightPolicy =
    beam.kind === 'bolt'
      ? REAL_LIGHT_POLICIES[psychic ? 'psychic' : 'tesla']
      : { ...REAL_LIGHT_POLICIES.prism, colour: capLightColour(beamColor(beam.colour), 0.6) };
  return policyLight(policy, animWorld(beam.source as LiveAnim), animWorld(beam.target as LiveAnim));
}
export function auxiliaryBeamLights(effects: LiveEffects): { key: string; light: PhysicalLight }[] {
  const out: { key: string; light: PhysicalLight }[] = [];
  for (const b of effects.radBeams ?? []) {
    if (b.type === 'rad' && !b.sustained && b.age >= b.period) continue;
    const p: LightPolicy = {
      ...REAL_LIGHT_POLICIES[b.type],
      colour: b.type === 'temporal' ? LIGHT_COLOURS.chrono : capLightColour(beamColor(b.color), 0.55),
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

/** Match engine added brightness at one quarter LightVisibility after the irradiance knee. */
export function engineLampLights(source: EngineLight, lighting: EngineLighting, head = 1.25): PhysicalLight[] {
  if (!(source.intensity > 0)) return [];
  const distance = (source.radius / 256) * 0.25;
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
    { position, radius: 0.15, intensity: colour, dayFactor: 0, occlude: true },
    {
      position,
      end: [position[0], base + Math.max(0.1, head - 0.6), position[2]],
      radius: 0.08,
      intensity: colour.map((c) => c * 0.08) as [number, number, number],
      dayFactor: 0,
      occlude: true,
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
    return lightLuminance(irradianceKnee(reconstructLight(sum, direction, [0, 1, 0]), 1));
  };
  let lo = 0,
    hi = 1;
  const reachable = Math.min(target, 2.49);
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
  levels: ArrayLike<number>,
  buildings: readonly BuildingOccluder[],
): LightOccluders {
  const heights = new Float32Array(width * height * 4);
  for (let z = 0; z < height; z++)
    for (let x = 0; x < width; x++) {
      const top = (levels[z * width + x] ?? 0) * LEVEL_HEIGHT;
      for (let dz = 0; dz < 2; dz++)
        for (let dx = 0; dx < 2; dx++) heights[(z * 2 + dz) * width * 2 + x * 2 + dx] = top;
    }
  for (const b of buildings)
    for (let z = Math.max(0, b.z * 2); z < Math.min(height * 2, (b.z + b.depth) * 2); z++)
      for (let x = Math.max(0, b.x * 2); x < Math.min(width * 2, (b.x + b.width) * 2); x++) {
        const i = z * width * 2 + x;
        heights[i] = Math.max(heights[i]!, b.top);
      }
  return { width: width * 2, height: height * 2, texelsPerCell: 2, heights };
}
