/** Generic additive emitter primitives. The caller owns paths, lifetimes, colours and source identity. */
import {
  AddEquation,
  CustomBlending,
  OneFactor,
  DoubleSide,
  MeshBasicNodeMaterial,
  Object3D,
  Vector3,
} from 'three/webgpu';
import {
  abs,
  cameraFar,
  cameraNear,
  depth,
  dot,
  exp,
  float,
  instancedBufferAttribute,
  length,
  max,
  min,
  mix,
  mrt,
  mx_noise_float,
  positionGeometry,
  pow,
  smoothstep,
  sqrt,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { createLightLook, LIGHT_LOOK_NEW, type LightLook, type LightLookValues } from './lightLook';
import { DirtyAttribute } from './dirtyAttribute';
import { InstancedQuad } from './instancedQuad';
import type { ShaderNode } from './tsl';
import { type Rgb } from './realLight';
export interface EmissivePrimitive {
  source: string;
  from: Vector3;
  to: Vector3;
  /** Full sheath width; a disc uses this as its diameter. */
  width: number;
  coreWidth?: number;
  /** Preserve source RGB ratios in a faithful laser sheath. */
  preserveHue?: boolean;
  /** Native straight laser only; auxiliary emitters never opt into this profile. */
  laserLight?: boolean;
  /** Last segment of a native laser: the hit point carries the small impact bloom. */
  impactEnd?: boolean;
  /** Share of a fading native laser's life already passed (0..1); dims its light by beamFade. */
  age?: number;
  colour: Rgb;
  hot?: Rgb;
  gain: number;
  disc?: boolean;
  push?: number;
  /** Existing electric noise, or the auxiliary beam's travelling sine modulation. */
  noise?: { mode: 1 | 2; rate: number; scale: number; amount: number; seed: number; distance: number };
}
/** Core colour: a pale tint of the source's own colour, so the hottest line and its bloom keep the hue (a white core read paler than the original bolt). */
export function emissiveCoreTint(colour: Rgb): [number, number, number] {
  const peak = Math.max(...colour, 1e-8);
  return colour.map((c) => 0.55 + 0.45 * (c / peak)) as [number, number, number];
}
export const EMISSIVE_BUDGET = {
  source: 6,
  frame: 32,
  core: 5,
  coreWidth: 0.2,
  outerWidth: 2.5,
  outerAlpha: 0.25,
  sheathCeiling: 0.8,
  sheathChroma: 1.2,
} as const;
/** Conservative radiance times world area, including the maximum modulation; invariant to path subdivision. */
export function emissiveEnergy(p: EmissivePrimitive): number {
  const area = p.disc ? Math.PI * (p.width * 0.5) ** 2 : p.width * p.from.distanceTo(p.to);
  return (
    Math.max(0, area * p.gain) *
    (EMISSIVE_BUDGET.core * (p.coreWidth ?? EMISSIVE_BUDGET.coreWidth) * Math.max(...(p.hot ?? [1, 1, 1])) +
      Math.max(...p.colour) * (1 + EMISSIVE_BUDGET.outerAlpha * EMISSIVE_BUDGET.outerWidth)) *
    Math.max(1.12, 1 + Math.abs(p.noise?.amount ?? 0))
  );
}
/** Hard per-source cap followed by a smooth frame cap; all pieces of one source receive the same scale. */
export function emissiveBudget(
  energies: ReadonlyMap<string, number>,
  sourceCap: number = EMISSIVE_BUDGET.source,
  frameCap: number = EMISSIVE_BUDGET.frame,
): { scales: Map<string, number>; total: number; gain: number } {
  const scales = new Map<string, number>();
  let total = 0;
  for (const [key, energy] of energies) {
    const safe = Math.max(0, energy);
    scales.set(key, safe > 0 ? Math.min(1, sourceCap / safe) : 1);
    total += Math.min(safe, sourceCap);
  }
  const gain = total > 0 ? (frameCap * -Math.expm1(-total / frameCap)) / total : 1;
  return { scales, total: total * gain, gain };
}
/** CPU radial profile twin; outer tail cannot cross the bloom threshold on its own. */
export function emissiveProfile(r: number): { core: number; sheath: number; glow: number } {
  const smooth = (a: number, b: number, x: number) => {
    const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  return {
    core: 1 - smooth(EMISSIVE_BUDGET.coreWidth * 0.72, EMISSIVE_BUDGET.coreWidth, r),
    sheath: 1 - smooth(0.7, 1, r),
    glow: EMISSIVE_BUDGET.outerAlpha * (1 - smooth(0, EMISSIVE_BUDGET.outerWidth, r)) ** 2,
  };
}
/** CPU twin of the native laser's radial radiance (r in half emission widths) before flicker and budgets. */
export function laserLightProfile(
  r: number,
  colour: Rgb,
  coreWidth = 1.4,
  look: Pick<
    LightLookValues,
    'beamCore' | 'beamGlow' | 'beamTail' | 'beamCoreGain' | 'beamSpread' | 'beamOverbright' | 'beamKnee'
  > = LIGHT_LOOK_NEW,
  gain = 1,
): Rgb {
  const core = Math.exp(-2.2 * (r / Math.max(1e-4, coreWidth * look.beamCore)) ** 2);
  const reach = EMISSIVE_BUDGET.outerWidth * look.beamSpread;
  const window = Math.max(0, Math.min(1, 1 - (r / reach) ** 2)) ** 2;
  const g = Math.max(1e-4, look.beamGlow);
  const raw = (1 - look.beamTail) * Math.exp(-1.6 * (r / g) ** 2) + look.beamTail * Math.exp((-1.3 * Math.abs(r)) / g);
  const profile = 1 - Math.exp(-4 * raw * gain);
  const amplitude = (1 - Math.exp(-4 * gain)) * raw;
  const glow = 0.85 * (profile + (amplitude - profile) * look.beamKnee) * window;
  const tint = laserGlowTint(colour, look.beamOverbright);
  return tint.map(
    (t) => (0.8 + 0.2 * t) * look.beamCoreGain * core + t * glow * (1 - 0.6 * core),
  ) as [number, number, number];
}
/** Display gamma used for the overbright emulation (the original adds colours in display space). */
const DISPLAY_GAMMA = 2.2;
/**
 * CPU twin of the laser glow's linear tint (peak channel 1): the source colour as the original shows it at the beam
 * centre, i.e. the absolute display colour scaled by `overbright` per channel and clipped there (a channel that clips
 * stops growing while the others still rise). overbright = 1 is exactly colour / peak for colours within range.
 */
export function laserGlowTint(colour: Rgb, overbright = 1): [number, number, number] {
  const display = colour.map((c) => Math.min(1, Math.max(0, c) ** (1 / DISPLAY_GAMMA) * overbright));
  const top = Math.max(...display, 1e-8);
  return display.map((d) => (d / top) ** DISPLAY_GAMMA) as [number, number, number];
}
/** CPU twin of the non-core luminance limit, applied after flicker and source gain. */
export function emissiveSheathColour(colour: Rgb, gain: number): [number, number, number] {
  const peak = Math.max(...colour);
  const boost = Math.min(EMISSIVE_BUDGET.sheathChroma, peak / Math.max(1e-8, peak - Math.min(...colour)));
  const rgb = colour.map((c) => Math.max(0, peak - (peak - c) * boost) * Math.max(0, gain)) as [number, number, number];
  const scale = Math.min(1, EMISSIVE_BUDGET.sheathCeiling / Math.max(1e-8, ...rgb));
  return rgb.map((c) => c * scale) as [number, number, number];
}
/** Fixed-capacity shared pool, updated after all producer channels; no game data or shape generation. */
export class RealEmissiveLayer {
  readonly root = new Object3D();
  readonly mesh: InstancedQuad;
  readonly frameGain = uniform(1);
  readonly now = uniform(0);
  readonly right = uniform(new Vector3(1, 0, 0));
  readonly up = uniform(new Vector3(0, 1, 0));
  readonly toCamera = uniform(new Vector3(0, 0, 1));
  private readonly channels = new Map<string, readonly EmissivePrimitive[]>();
  private readonly data: DirtyAttribute;
  totalEnergy = 0;
  sourceCount = 0;
  legacyEnergy = 0;
  constructor(
    readonly capacity = 16384,
    readonly look: LightLook = createLightLook(),
  ) {
    this.data = new DirtyAttribute(new Float32Array(capacity * 28), 28);
    const shared = this.data.interleaved();
    const attrs = Array.from(
      { length: 7 },
      (_, i) => instancedBufferAttribute(shared, 'vec4', 28, i * 4) as ShaderNode,
    );
    const [a, b, colour, hot, extra, noise, phase] = attrs as [
      ShaderNode,
      ShaderNode,
      ShaderNode,
      ShaderNode,
      ShaderNode,
      ShaderNode,
      ShaderNode,
    ];
    // Native lasers widen their quad (beamSpread) so the glow tail reaches zero inside it: a tail clipped at the quad
    // edge read as a hard-edged neon band at 1:1 (round-7 review). Exactly 1 in the cur preset and for every other
    // primitive, so their geometry and legacy profile are unchanged.
    const spread = phase.w.greaterThan(1.5).select(mix(float(1), this.look.beamSpread, this.look.beamLight), float(1));
    const width = a.w.max(1e-6).mul(EMISSIVE_BUDGET.outerWidth).mul(spread);
    const delta = b.xyz.sub(a.xyz);
    const projected = this.right.mul(dot(delta, this.right)).add(this.up.mul(dot(delta, this.up)));
    const span = length(projected).max(1e-6);
    const axis = projected.div(span);
    const side = this.right.mul(dot(axis, this.up).negate()).add(this.up.mul(dot(axis, this.right)));
    const disc = extra.x.greaterThan(0.5);
    const along = positionGeometry.x.add(0.5).mul(span.add(width)).sub(width.mul(0.5));
    const t = along.div(span).clamp(0, 1);
    const m = new MeshBasicNodeMaterial();
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = true;
    m.side = DoubleSide;
    // Emission carries alpha zero so it never covers scene art. Use ONE/ONE explicitly on both backends.
    m.blending = CustomBlending;
    m.blendEquation = m.blendEquationAlpha = AddEquation;
    m.blendSrc = m.blendSrcAlpha = m.blendDst = m.blendDstAlpha = OneFactor;
    m.positionNode = disc
      .select(
        a.xyz.add(this.right.mul(positionGeometry.x.mul(width))).add(this.up.mul(positionGeometry.y.mul(width))),
        mix(a.xyz, b.xyz, t)
          .add(axis.mul(along.sub(t.mul(span))))
          .add(side.mul(positionGeometry.y.mul(width))),
      )
      .add(this.toCamera.mul(extra.y));
    const cap = max(along.negate(), along.sub(span)).max(0).div(width.mul(0.5));
    const r = disc
      .select(length(uv().sub(0.5).mul(2)), length(vec2(cap, uv().y.sub(0.5).mul(2))))
      .mul(EMISSIVE_BUDGET.outerWidth)
      .mul(spread);
    const coreWidth = phase.z.greaterThan(0).select(phase.z, float(EMISSIVE_BUDGET.coreWidth));
    const core = float(1).sub(smoothstep(coreWidth.mul(0.72), coreWidth, r));
    const sheath = float(1).sub(smoothstep(0.7, 1, r));
    const glow = float(1)
      .sub(smoothstep(0, EMISSIVE_BUDGET.outerWidth, r))
      .pow(2)
      .mul(EMISSIVE_BUDGET.outerAlpha);
    const point = mix(a.xyz, b.xyz, t);
    const time: ShaderNode = this.now.mul(noise.y);
    const electric = float(1).add(mx_noise_float(point.mul(noise.z).add(vec3(time, 0, 0))).mul(noise.w));
    const angle = phase.y.add(along.clamp(0, span)).mul(9).sub(this.now.mul(8)).add(phase.x.mul(0.73));
    const travelling = angle.sin().mul(0.08).add(angle.mul(2.17).add(phase.x).sin().mul(0.04)).add(1);
    const modulation = noise.x.greaterThan(1.5).select(travelling, noise.x.greaterThan(0.5).select(electric, float(1)));
    const peak = max(colour.r, max(colour.g, colour.b));
    const boost = float(EMISSIVE_BUDGET.sheathChroma).min(
      peak.div(peak.sub(min(colour.r, min(colour.g, colour.b))).max(1e-8)),
    );
    const chromatic = phase.w
      .greaterThan(0.5)
      .select(colour.rgb, vec3(peak).sub(vec3(peak).sub(colour.rgb).mul(boost)).max(0));
    const body = chromatic.mul(sheath.add(glow)).mul(colour.w).mul(modulation);
    const bodyScale = float(EMISSIVE_BUDGET.sheathCeiling)
      .div(max(body.r, max(body.g, body.b)).max(1e-8))
      .min(1);
    const legacy = hot.rgb.mul(core.mul(EMISSIVE_BUDGET.core)).mul(colour.w).mul(modulation).add(body.mul(bodyScale));
    // Native laser as light (round 7b). r is in half emission widths (~4 device px at view 2.5). A Gaussian
    // white-hot core bright enough to seed the night bloom, a colour glow that is a Gaussian plus an exponential tail
    // (no plateau: the old per-fragment clamp made a flat band), windowed to zero at the widened quad edge, and a small
    // bloom at the hit point. The glow keeps the source's own RGB ratios, so its hue is the original house colour.
    const coreR = coreWidth.mul(this.look.beamCore).max(1e-4);
    const laserCore = exp(r.div(coreR).pow(2).mul(-2.2));
    const glowR = this.look.beamGlow.max(1e-4);
    const reach = float(EMISSIVE_BUDGET.outerWidth).mul(spread);
    const window = float(1).sub(r.div(reach).pow(2)).clamp(0, 1).pow(2);
    const rawGlow = mix(exp(r.div(glowR).pow(2).mul(-1.6)), exp(r.div(glowR).mul(-1.3)), this.look.beamTail);
    // The source budget scales the glow before the soft saturation, so a budgeted beam keeps a full house-colour body
    // near the core and only its faint tail shrinks (live round 7b: a long single beam read pink-white, not red).
    // beamKnee (hd-light=beam): saturate the amplitude only, keeping the radial falloff of a budgeted beam for a
    // full-gain one; the profile saturation grew a flat plateau ~16 px wide on the violet tower beam (round-7 review).
    const profileGlow = float(1).sub(exp(rawGlow.mul(colour.w).mul(-4)));
    const amplitudeGlow = float(1).sub(exp(colour.w.mul(-4))).mul(rawGlow);
    const laserGlow = mix(profileGlow, amplitudeGlow, this.look.beamKnee).mul(0.85).mul(window);
    const seed = phase.x.mul(1 / 65536).fract();
    const tick = this.now.mul(23).add(seed.mul(13));
    const flicker = mx_noise_float(vec3(tick, seed.mul(71), 0))
      .mul(2)
      .clamp(-1, 1)
      .mul(this.look.beamFlicker)
      .add(1);
    const shimmer = along.mul(29).sub(this.now.mul(17)).add(seed).sin().mul(0.025).add(1);
    const endR = along.sub(span).div(a.w.mul(0.5).max(1e-6));
    const sideR = uv().y.sub(0.5).mul(2).mul(reach);
    const impact = exp(endR.pow(2).add(sideR.pow(2)).mul(-0.32))
      .mul(this.look.beamImpact)
      .mul(extra.z)
      .mul(window);
    // Glow hue as the original shows it at the beam centre (beamOverbright; exactly colour / peak at 1).
    const display = pow(colour.rgb.max(0), vec3(1 / DISPLAY_GAMMA))
      .mul(this.look.beamOverbright)
      .min(1);
    const tint = pow(
      display.div(max(display.r, max(display.g, display.b)).max(1e-8)),
      vec3(DISPLAY_GAMMA),
    );
    const glowRgb = tint.mul(laserGlow).mul(flicker).mul(shimmer);
    // The core and the impact take 20%/45% of the source hue so violet stays violet-white, never pink.
    const hotTint = mix(vec3(1), tint, 0.2).mul(this.look.beamCoreGain);
    const lightRgb = hotTint
      .mul(laserCore)
      .mul(flicker)
      .mul(shimmer)
      .add(glowRgb.mul(float(1).sub(laserCore.mul(0.6))))
      .add(mix(vec3(1), tint, 0.45).mul(impact).mul(flicker).mul(colour.w));
    // beamFade (hd-light=beam): a fading native laser dims over its life like the original (b.w: share of the life
    // passed). The original's aged Prism beam is a darker, thinner violet; a fresh-looking glow there read pale and pink.
    const remaining = float(1).sub(b.w.clamp(0, 1).mul(this.look.beamFade));
    const rgb = mix(legacy, lightRgb.mul(remaining), phase.w.greaterThan(1.5).select(this.look.beamLight, float(0)));
    // beamDepthLift (hd-light=beam, a gain): the glow is depth-tested as if lifted towards the camera by the height
    // its screen-vertical offset would need above the ground (offset x cot(elevation)). A low near-horizontal beam's glow
    // below the core otherwise falls behind the very terrain it is drawn over and ends in a hard edge within 2 px
    // (round-7 review, live SOV07TMD frame 222), while the glow above survives. The core (offset ~0) keeps its depth.
    const offset = side.mul(uv().y.sub(0.5).mul(width));
    const elevation = this.toCamera.y.max(0.05);
    const cotangent = sqrt(float(1).sub(elevation.mul(elevation)).max(0)).div(elevation);
    // The small core term covers beams that touch or dip into the terrain where they hit the ground.
    const lift = abs(dot(offset, this.up))
      .add(width.mul(0.15))
      .mul(cotangent)
      .mul(phase.w.greaterThan(1.5).select(this.look.beamDepthLift.mul(this.look.beamLight), float(0)));
    // beamOver (hd-light=beam): native laser light is drawn over the scene, as the original draws its lasers after
    // everything else (a live SOV07TMD original shows a Prism beam crossing the smoke of the blast it hits). With the
    // depth test alone, the unit at a Prism volley's focus stood in front of the converging glow as a dark,
    // hard-edged shape over the blast (round-8 review, frame 225).
    const over = phase.w.greaterThan(1.5).select(this.look.beamOver.mul(this.look.beamLight), float(0));
    m.depthNode = mix(depth.sub(lift.div(cameraFar.sub(cameraNear))), float(0), over);
    // Budget only these new primitives, preserving legacy static overlays that share the attachment.
    m.colorNode = vec4(0);
    m.mrtNode = mrt({ output: vec4(0), emissive: vec4(rgb.mul(this.frameGain), 0) });
    this.mesh = new InstancedQuad(m, capacity);
    this.mesh.count = 0;
    this.mesh.renderOrder = 11;
    this.root.add(this.mesh);
    this.root.name = 'real-emissive';
  }
  set(channel: string, primitives: readonly EmissivePrimitive[]): void {
    this.channels.set(channel, primitives);
  }
  update(now: number, enabled: boolean): void {
    this.root.visible = enabled;
    this.now.value = now;
    if (!enabled) return;
    const energies = new Map<string, number>();
    let count = 0;
    for (const list of this.channels.values())
      for (const p of list) {
        if (count++ >= this.capacity) break;
        energies.set(p.source, (energies.get(p.source) ?? 0) + emissiveEnergy(p));
      }
    const budget = emissiveBudget(energies);
    // Legacy building overlays are already individually capped by their producer.
    const total = [...energies].reduce((sum, [key, e]) => sum + e * budget.scales.get(key)!, 0) + this.legacyEnergy;
    const gain = total > 0 ? (EMISSIVE_BUDGET.frame * -Math.expm1(-total / EMISSIVE_BUDGET.frame)) / total : 1;
    this.frameGain.value = gain;
    this.totalEnergy = total * gain;
    this.sourceCount = energies.size;
    let n = 0;
    for (const list of this.channels.values())
      for (const p of list) {
        if (n >= this.capacity) break;
        const values = [
          p.from.x,
          p.from.y,
          p.from.z,
          p.width,
          p.to.x,
          p.to.y,
          p.to.z,
          p.age ?? 0,
          ...p.colour,
          p.gain * (budget.scales.get(p.source) ?? 1),
          ...(p.hot ?? [1, 1, 1]),
          0,
          p.disc ? 1 : 0,
          p.push ?? 0,
          p.impactEnd ? 1 : 0,
          0,
          p.noise?.mode ?? 0,
          p.noise?.rate ?? 0,
          p.noise?.scale ?? 0,
          p.noise?.amount ?? 0,
          p.noise?.seed ?? 0,
          p.noise?.distance ?? 0,
          p.coreWidth ?? 0,
          p.laserLight ? 2 : p.preserveHue ? 1 : 0,
        ];
        for (let i = 0; i < 28; i++) this.data.setComponent(n, i, values[i]!);
        n++;
      }
    this.data.flush();
    this.mesh.count = n;
  }
  dispose(): void {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.channels.clear();
    this.root.clear();
    this.root.removeFromParent();
  }
}

/** Cap one existing source overlay (orb and all its filaments together) before the shared frame budget. */
export function capOverlaySource(
  discs: readonly import('./emissiveOverlay').EmissiveDisc[],
  strokes: readonly import('./emissiveOverlay').EmissiveStroke[],
  pixelsPerUnit: number,
): {
  discs: import('./emissiveOverlay').EmissiveDisc[];
  strokes: import('./emissiveOverlay').EmissiveStroke[];
  energy: number;
} {
  let energy = 0;
  for (const d of discs) {
    const o = d.orb;
    const peak = Math.max(...o.hot, ...o.mid, ...o.body, ...o.limb, ...o.filaments, ...o.halo);
    energy += Math.PI * (o.radius * Math.max(1, o.haloScale)) ** 2 * peak * d.brightness * d.alpha;
  }
  for (const s of strokes)
    for (let i = 1; i < s.points.length; i++) {
      const a = s.points[i - 1]!,
        b = s.points[i]!;
      energy +=
        ((Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) * s.width) / pixelsPerUnit) *
        Math.max(...s.core, ...s.edge) *
        s.alpha;
    }
  const scale = energy > 0 ? Math.min(1, EMISSIVE_BUDGET.source / energy) : 1;
  return {
    discs: discs.map((d) => ({ ...d, brightness: d.brightness * scale })),
    strokes: strokes.map((s) => ({
      ...s,
      core: s.core.map((c) => c * scale) as [number, number, number],
      edge: s.edge.map((c) => c * scale) as [number, number, number],
    })),
    energy: energy * scale,
  };
}
