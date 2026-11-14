import type { SceneLighting } from '../sceneLighting';
/** RA2/YR emissive policy; geometry and animation lifetime remain owned by their existing producers. */
import { Vector3 } from 'three/webgpu';
import { EFFECT_TABLE } from '../live/effectTable';
import type { LiveAnim } from '../live/effects';
import type { EmissivePrimitive } from '../realEmissive';
import { animWorld, animLightPolicy } from './realLights';
// LiveView.frame keeps zoom=1: 60/sqrt(2) engine pixels * device scale 2.5 = 106.07 device px/world unit.
// DPR is already in bufferScale (a cell is 150x75 device px); do not multiply it twice.
// Live SOV07TMD night at 1:1, the original Prism tower beam is 16-18 px with its glow and 11-12 px of saturated core;
// an 8 px calibration measured 8-11 px core and 11-16 px glow live. So: 0.22 world = 23.3 px strand, emissive core
// 0.11*1.4*0.72*106.07 = 11.8 saturated px, soft outer tail 0.11*2.5*106.07 = 29.2 px. Bolts keep their dimensions.
export const REAL_LASER_STYLE = { strandWidth: 0.22, emissionWidth: 0.11, coreWidth: 1.4 } as const;
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
export function animEmission(anim: LiveAnim, age: number, _now: number, fps = 62.5): EmissivePrimitive | null {
  const row = EFFECT_TABLE[anim.type];
  const fire = row?.c === 'fire' || /^(FIRE0[123]|INITFIRE)$/.test(anim.type);
  const muzzle = row?.c === 'muzzle';
  const blast = row?.c === 'explosion' || row?.c === 'collapse' || largeLightEvent(anim.type);
  // Fires get no added core: the flame art is drawn above the anim's coordinate, so a disc there read as a detached
  // glowing dot under the flame at 1:1 (sandbox SOV07/XMP29 night, 2026-11-14). Their physical light stays.
  if (fire || (!muzzle && !blast)) return null;
  const seconds = muzzle ? Math.min(0.15, 2 / fps + 0.03) : EMISSIVE_EVENTS.seconds;
  if (age < 0 || age >= seconds) return null;
  const at = new Vector3(...animWorld(anim));
  const radius = muzzle ? 0.1 : Math.max(0.2, row?.light?.[3] ?? row?.s ?? animLightPolicy(anim.type)?.radius ?? 0.3);
  return {
    source: `anim:${anim.id}:${anim.uid}:${anim.type}`,
    from: at,
    to: at,
    // A short bloom behind the fireball's own core, about its size: the 2-radius disc spread a large yellow halo.
    width: radius * 1.2,
    disc: true,
    // This additive attachment composites over translucent smoke too. A warm disc tinted grey smoke beige;
    // keep the invented bloom achromatic. Source art is untouched; faint warmth comes from ground irradiance.
    colour: [1, 1, 1],
    hot: [1, 1, 1],
    gain: 1 - age / seconds,
    // Pull along the view direction only, preserving the native on-screen centre. The pull must clear the whole outer
    // glow (outerWidth / 2 of the width), or the ground cut the disc's lower half with a straight edge (sandbox
    // EXPLOLRG at night, 0.05 s).
    push: muzzle ? radius * 0.5 : radius * 1.6,
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
