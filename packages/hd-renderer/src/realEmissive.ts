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
  dot,
  float,
  instancedBufferAttribute,
  length,
  max,
  mix,
  mrt,
  mx_noise_float,
  positionGeometry,
  smoothstep,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { DirtyAttribute } from './dirtyAttribute';
import { InstancedQuad } from './instancedQuad';
import type { ShaderNode } from './tsl';
import { lightLuminance, type Rgb } from './realLight';
export interface EmissivePrimitive {
  source: string;
  from: Vector3;
  to: Vector3;
  /** Full sheath width; a disc uses this as its diameter. */
  width: number;
  colour: Rgb;
  hot?: Rgb;
  gain: number;
  disc?: boolean;
  push?: number;
  /** Existing electric noise, or the auxiliary beam's travelling sine modulation. */
  noise?: { mode: 1 | 2; rate: number; scale: number; amount: number; seed: number; distance: number };
}
export const EMISSIVE_BUDGET = {
  source: 6,
  frame: 32,
  core: 5,
  coreWidth: 0.25,
  outerWidth: 2.5,
  outerAlpha: 0.25,
  sheathCeiling: 0.8,
} as const;
/** Conservative radiance times world area, including the maximum modulation; invariant to path subdivision. */
export function emissiveEnergy(p: EmissivePrimitive): number {
  const area = p.disc ? Math.PI * (p.width * 0.5) ** 2 : p.width * p.from.distanceTo(p.to);
  return (
    Math.max(0, area * p.gain) *
    (EMISSIVE_BUDGET.core * EMISSIVE_BUDGET.coreWidth * Math.max(...(p.hot ?? [1, 1, 1])) +
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
/** CPU twin of the non-core luminance limit, applied after flicker and source gain. */
export function emissiveSheathColour(colour: Rgb, gain: number): [number, number, number] {
  const rgb = colour.map((c) => c * Math.max(0, gain)) as [number, number, number];
  const scale = Math.min(1, EMISSIVE_BUDGET.sheathCeiling / Math.max(1e-8, lightLuminance(rgb)));
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
  constructor(readonly capacity = 16384) {
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
    const width = a.w.max(1e-6).mul(EMISSIVE_BUDGET.outerWidth);
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
      .mul(EMISSIVE_BUDGET.outerWidth);
    const core = float(1).sub(smoothstep(EMISSIVE_BUDGET.coreWidth * 0.72, EMISSIVE_BUDGET.coreWidth, r));
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
    const body = colour.rgb.mul(sheath.add(glow)).mul(colour.w).mul(modulation);
    const bodyScale = float(EMISSIVE_BUDGET.sheathCeiling)
      .div(dot(body, vec3(0.2126, 0.7152, 0.0722)).max(1e-8))
      .min(1);
    const rgb = hot.rgb.mul(core.mul(EMISSIVE_BUDGET.core)).mul(colour.w).mul(modulation).add(body.mul(bodyScale));
    // Use the existing attachment: under physical mode it is composited before bloom, under legacy mode after it.
    m.colorNode = vec4(0);
    m.mrtNode = mrt({ output: vec4(0), emissive: vec4(rgb, 0) });
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
          0,
          ...p.colour,
          p.gain * (budget.scales.get(p.source) ?? 1),
          ...(p.hot ?? [1, 1, 1]),
          0,
          p.disc ? 1 : 0,
          p.push ?? 0,
          0,
          0,
          p.noise?.mode ?? 0,
          p.noise?.rate ?? 0,
          p.noise?.scale ?? 0,
          p.noise?.amount ?? 0,
          p.noise?.seed ?? 0,
          p.noise?.distance ?? 0,
          0,
          0,
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
