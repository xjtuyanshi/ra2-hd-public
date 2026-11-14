import type { SceneLighting } from '../sceneLighting';
/** RA2/YR emissive policy; geometry and animation lifetime remain owned by their existing producers. */
import { Vector3 } from 'three/webgpu';
import { EFFECT_TABLE } from '../live/effectTable';
import type { LiveAnim } from '../live/effects';
import type { EmissivePrimitive } from '../realEmissive';
import { flicker } from '../lightField';
import { animWorld, animLightPolicy, REAL_LIGHT_POLICIES } from './realLights';
export const EMISSIVE_EVENTS = { seconds: 0.15, dayEV: 0.25, nightEV: 0.5, headDistance: 0.9, headRise: 1.5 } as const;
export function eventExposureKick(age: number, night: number): number {
  if (age < 0 || age >= EMISSIVE_EVENTS.seconds) return 0;
  const ev =
    (EMISSIVE_EVENTS.dayEV + (EMISSIVE_EVENTS.nightEV - EMISSIVE_EVENTS.dayEV) * Math.max(0, Math.min(1, night))) *
    (1 - age / EMISSIVE_EVENTS.seconds);
  return 2 ** ev - 1;
}
export function largeLightEvent(type: string): boolean {
  return (
    type === 'NUKETO' ||
    type === 'PDFXLOC' ||
    type === 'PDFXCLD' ||
    /^WCLBOLT[13]$/.test(type) ||
    EFFECT_TABLE[type]?.c === 'collapse'
  );
}
/**
 * A captured beam source must start at the building head: within headDistance on the ground plane and headRise in
 * height (the head is estimated from the art's top, which sits above the coil's orb and fire point; a 0.6 sphere
 * missed the sandbox coil firing from 700 leptons). No charge-frame inference.
 */
export function firingHead(head: Vector3, sources: readonly Vector3[]): boolean {
  return sources.some(
    (p) =>
      Math.hypot(head.x - p.x, head.z - p.z) <= EMISSIVE_EVENTS.headDistance &&
      Math.abs(head.y - p.y) <= EMISSIVE_EVENTS.headRise,
  );
}
export function animEmission(anim: LiveAnim, age: number, now: number, fps = 62.5): EmissivePrimitive | null {
  const row = EFFECT_TABLE[anim.type];
  const fire = row?.c === 'fire' || /^(FIRE0[123]|INITFIRE)$/.test(anim.type);
  const muzzle = row?.c === 'muzzle';
  const blast = row?.c === 'explosion' || row?.c === 'collapse' || largeLightEvent(anim.type);
  if (!fire && !muzzle && !blast) return null;
  const seconds = muzzle ? Math.min(0.15, 2 / fps + 0.03) : EMISSIVE_EVENTS.seconds;
  if (age < 0 || (!fire && age >= seconds)) return null;
  const at = new Vector3(...animWorld(anim));
  const radius = fire
    ? anim.attachedTo
      ? 0.1
      : 0.16
    : muzzle
      ? 0.1
      : Math.max(0.2, row?.light?.[3] ?? row?.s ?? animLightPolicy(anim.type)?.radius ?? 0.3);
  const policy = anim.attachedTo ? REAL_LIGHT_POLICIES.burning : REAL_LIGHT_POLICIES.fire;
  return {
    source: `anim:${anim.id}:${anim.uid}:${anim.type}`,
    from: at,
    to: at,
    width: radius * 2,
    disc: true,
    colour: fire ? [1, 0.3, 0.055] : [1, 0.5, 0.12],
    hot: fire ? [1, 0.65, 0.25] : [1, 0.95, 0.8],
    gain: fire ? 0.65 * flicker(anim.uid / 65536, now, policy.flicker) : 1 - age / seconds,
    // Pull along the view direction only, preserving the native on-screen centre.
    push: Math.max(0, radius * 0.5),
  };
}

/** Cheap semantic keys: animation frames and source-array identity do not invalidate static targets. */
export function physicalBuildingKey(
  buildings: readonly {
    kind: string;
    type: string;
    x: number;
    y: number;
    z: number;
    id?: number;
    uid?: number;
    nativeHidden?: boolean;
  }[],
  hidden: (building: { x: number; y: number; z: number }) => boolean,
  revision: number,
): string {
  let key = String(revision);
  for (const b of buildings)
    if (b.kind === 'building' || /LAMP|LITE/.test(b.type))
      key += `;${b.id},${b.uid},${b.type},${b.x},${b.y},${b.z},${b.nativeHidden ? 1 : 0},${hidden(b) ? 1 : 0}`;
  return key;
}
export function physicalLampKey(
  sources: readonly (
    | { position: readonly number[]; radius: number; intensity: number; tint: readonly number[] }
    | { x: number; y: number; z: number; radius: number; intensity: number; tint: readonly number[] }
  )[],
): string {
  let key = '';
  for (const s of sources) {
    key += 'position' in s ? `${s.position[0]},${s.position[1]},${s.position[2]}` : `${s.x},${s.y},${s.z}`;
    key += `,${s.radius},${s.intensity},${s.tint[0]},${s.tint[1]},${s.tint[2]};`;
  }
  return key;
}

/** Preserve the engine presentation; the short physical kick uses the same sky-flash multiplier path. */
export function physicalExposureFlash(scene: SceneLighting | null, kick: number): SceneLighting | null {
  if (!(kick > 0)) return scene;
  return {
    ...(scene ?? { terrain: [1, 1, 1], objects: [1, 1, 1], sunShadow: 1, overcast: 0 }),
    skyFlash: Math.max(scene?.skyFlash ?? 0, Math.min(Math.SQRT2 - 1, kick)),
  };
}
