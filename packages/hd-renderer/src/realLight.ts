import {
  AddEquation,
  CustomBlending,
  Color,
  DataTexture,
  DataUtils,
  DoubleSide,
  FloatType,
  HalfFloatType,
  InstancedBufferAttribute,
  InstancedMesh,
  LinearFilter,
  Mesh,
  MeshBasicNodeMaterial,
  OneFactor,
  OrthographicCamera,
  PlaneGeometry,
  RedFormat,
  RenderTarget,
  RGBAFormat,
  Scene,
  Vector2,
  Vector3,
  Vector4,
  UnsignedByteType,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  Fn,
  If,
  atan,
  clamp,
  dot,
  exp,
  float,
  instancedBufferAttribute,
  length,
  max,
  mix,
  mrt,
  positionGeometry,
  positionWorld,
  texture,
  smoothstep,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { AmbientContactField, type AmbientContact } from './ambientContact';
import { LightCost } from './lightCost';
import { NIGHT_ART_DEFAULTS } from './nightArt';
import { flicker, heldPulseEnvelope } from './lightField';
import type { ShaderNode } from './tsl';

export type Rgb = readonly [number, number, number];
export type Point = readonly [number, number, number];
export const REAL_LIGHT_THRESHOLD = 0.006;
export const REAL_LIGHT_UPPER = 1.25;
const LUMA = [0.2126, 0.7152, 0.0722] as const;
export const lightLuminance = (v: Rgb): number => v.reduce((s, c, i) => s + c * LUMA[i]!, 0);
export interface PhysicalLight {
  position: Point;
  /** Absent for points; segments carry intensity per world-unit length. */
  end?: Point;
  radius: number;
  intensity: Rgb;
  /** Optional short hot-to-warm phase in linear radiance, independent of geometry. */
  initialIntensity?: Rgb;
  hotSeconds?: number;
  groundOnly?: number;
  occlude?: boolean;
  /** Static/kept sources use zero; short events may retain a subtle daylight share. */
  dayFactor: number;
  /** Source-policy window, in cells from the point or closest point on the segment. */
  reach?: number;
  /** Posts can bound the ground footprint independently of their elevation. */
  planarReach?: boolean;
  /** Ground-only footprint along the camera ray; upright receivers keep the actual 3D source. */
  viewProjectedFootprint?: boolean;
  flicker?: number;
  seed?: number;
  /**
   * 'gaussian': irradiance exp(-(d / radius)^2) of the 3D distance to the point or the closest point of the segment,
   * with `radius` the falloff width. Short events use it: an inverse-square core strong enough to read beside its own
   * art saturated the ground knee out to the window, which then cut a flat pool or a straight band along a beam with a
   * hard rim (live SOV07TMD night, Prism volley). The Gaussian is negligible (< 0.1 %) before the window closes.
   */
  profile?: 'gaussian';
}
/** Window, in Gaussian widths, beyond which a Gaussian light is not drawn (exp(-2.7^2) < 0.07 %). */
export const GAUSSIAN_REACH = 2.7;
export const segmentLength = (l: PhysicalLight): number =>
  l.end ? Math.hypot(...l.end.map((v, i) => v - l.position[i]!)) : 0;
/** Global dynamic default; source policy supplies tighter windows and explicit large-event exceptions. */
export const REAL_LIGHT_REACH = { cells: 3 } as const;
/** Conservative unwindowed scalar threshold, capped; the window reaches exactly zero at this derived distance. */
export function lightReach(l: PhysicalLight): number {
  if (l.profile === 'gaussian') return Math.min(l.reach ?? REAL_LIGHT_REACH.cells, GAUSSIAN_REACH * l.radius);
  return Math.min(
    l.reach ?? REAL_LIGHT_REACH.cells,
    Math.sqrt(
      Math.max(
        0,
        (lightLuminance(l.intensity) * (l.end ? segmentLength(l) : 1)) / REAL_LIGHT_THRESHOLD - l.radius ** 2,
      ),
    ),
  );
}
/**
 * Merge co-located continuous lights (several fire anims and fire particle systems on one burning vehicle) into one:
 * the brightest keeps its position, reach and colour; the others add only `share` of their intensity, and the sum is
 * capped at `cap` times the brightest. Summing them in full made a flat, hard-rimmed pool around each burning tank
 * (live SOV07TMD night, 28 kept lights for four tanks).
 */
export function mergeNearbyLights<K>(
  entries: readonly { key: K; light: PhysicalLight }[],
  radius = 1,
  share = 0.3,
  cap = 1.6,
): { key: K; light: PhysicalLight }[] {
  const sorted = [...entries].sort((a, b) => lightLuminance(b.light.intensity) - lightLuminance(a.light.intensity));
  const out: { key: K; light: PhysicalLight; base: number; sum: [number, number, number] }[] = [];
  for (const e of sorted) {
    const p = e.light.position;
    const host = e.light.end
      ? undefined
      : out.find((o) => !o.light.end && Math.hypot(o.light.position[0] - p[0], o.light.position[2] - p[2]) <= radius);
    if (!host) {
      out.push({ ...e, base: lightLuminance(e.light.intensity), sum: [...e.light.intensity] });
      continue;
    }
    for (let i = 0; i < 3; i++) host.sum[i]! += e.light.intensity[i]! * share;
    const k = Math.min(1, (cap * host.base) / Math.max(lightLuminance(host.sum), 1e-8));
    host.light = { ...host.light, intensity: host.sum.map((c) => c * k) as [number, number, number] };
  }
  return out.map(({ key, light }) => ({ key, light }));
}
export function lightWindow(t: number): number {
  return Math.max(0, 1 - Math.min(1, Math.max(0, t)) ** 2) ** 2;
}
/**
 * Window of a Gaussian light: flat until the Gaussian itself has fallen off, so it trims only a negligible tail
 * (the inverse-square window would also eat the visible shoulder of a short flash).
 */
export function gaussianWindow(t: number): number {
  return Math.max(0, 1 - Math.min(1, Math.max(0, t)) ** 6) ** 2;
}
export interface LightMoment {
  s: [number, number, number];
  d: [number, number, number];
  ground: number;
}
/** Project a source onto a receiver plane along the viewing ray, preserving its screen position. */
export function viewProjectedPoint(p: Point, height: number, direction: Point): Point {
  const t = (height - p[1]) / (Math.abs(direction[1]) > 1e-8 ? direction[1] : -1e-8);
  return [p[0] + t * direction[0], height, p[2] + t * direction[2]];
}
/** CPU twin of the separate ground footprint; objects always call lightMoment on the original source. */
export function groundLightMoment(l: PhysicalLight, p: Point, direction: Point): LightMoment {
  if (!l.viewProjectedFootprint) return lightMoment(l, p);
  const m = lightMoment(
    {
      ...l,
      position: viewProjectedPoint(l.position, p[1], direction),
      end: l.end ? viewProjectedPoint(l.end, p[1], direction) : undefined,
    },
    p,
  );
  // This is a screen-aligned diffuse footprint, not a grazing source at zero height.
  m.d = [0, lightLuminance(m.s), 0];
  return m;
}
/** Representative-point direction; finite-line integral normalises energy, including short and split tubes. */
export function lightMoment(l: PhysicalLight, p: Point, windowed = true): LightMoment {
  const ab = l.end?.map((v, i) => v - l.position[i]!) ?? [0, 0, 0];
  const len = segmentLength(l);
  const along = len > 0 ? ab.reduce((s, v, i) => s + v * (p[i]! - l.position[i]!), 0) / len : 0;
  const u = Math.max(0, Math.min(len, along));
  const delta = l.position.map((v, i) => v + (len ? (ab[i]! * u) / len : 0) - p[i]!);
  const d2 = delta.reduce((s, v) => s + v * v, 0);
  const distance = Math.sqrt(d2);
  let factor = 1 / Math.max(1e-8, d2 + l.radius ** 2);
  if (l.profile === 'gaussian') factor = Math.exp(-d2 / Math.max(1e-8, l.radius ** 2));
  else if (l.end) {
    // Integral of 1/(perpendicularDistance^2 + r_s^2 + t^2) over the actual finite segment.
    const a = Math.sqrt(Math.max(1e-8, d2 - (along - u) ** 2 + l.radius ** 2));
    factor = (Math.atan((len - along) / a) + Math.atan(along / a)) / a;
  }
  if (windowed) {
    const t = (l.planarReach ? Math.hypot(delta[0]!, delta[2]!) : distance) / Math.max(lightReach(l), 1e-8);
    factor *= l.profile === 'gaussian' ? gaussianWindow(t) : lightWindow(t);
  }
  const s = l.intensity.map((c) => c * factor) as [number, number, number];
  const lum = lightLuminance(s);
  return {
    s,
    d: delta.map((v) => (lum * v) / Math.max(distance, 1e-8)) as [number, number, number],
    ground: lum * (l.groundOnly ?? 0),
  };
}
export function reconstructLight(s: Rgb, d: Point, normal: Point): [number, number, number] {
  const lum = lightLuminance(s),
    dominant = Math.min(lum, Math.hypot(...d));
  const cosine = Math.max(0, d.reduce((v, c, i) => v + c * normal[i]!, 0) / Math.max(Math.hypot(...d), 1e-8));
  const k = (dominant * cosine + 0.5 * Math.max(0, lum - dominant)) / Math.max(lum, 1e-8);
  return s.map((c) => c * k) as [number, number, number];
}
/** Hue-preserving exponential knee in noon irradiance units. */
export function cappedIrradiance(e: Rgb, cap: number): [number, number, number] {
  const lum = lightLuminance(e);
  const k = cap > 0 && lum > 0 ? Math.min(1, (cap * -Math.expm1(-lum / cap)) / lum) : 0;
  return e.map((c) => c * k) as [number, number, number];
}
/** CPU twin of softKneeNode (ground). */
export function softCappedIrradiance(e: Rgb, cap: number): [number, number, number] {
  const lum = lightLuminance(e);
  const k = cap > 0 && lum > 0 ? Math.min(1, (cap * Math.log1p(lum / cap)) / lum) : 0;
  return e.map((c) => c * k) as [number, number, number];
}
export function irradianceKnee(e: Rgb, night: number, capScale = 1, ambientLum = 0.4): [number, number, number] {
  return softCappedIrradiance(e, (1.2 + (3 * ambientLum - 1.2) * Math.min(1, Math.max(0, night))) * capScale);
}
export function objectIrradianceKnee(e: Rgb, sceneLum: number, noon = 1): [number, number, number] {
  return cappedIrradiance(e, Math.max(0, 0.8 * noon - sceneLum));
}
/** Generic R16F contract. Row-major top heights, world y, 2 texels/cell; no object identity or game policy. */
export interface LightOccluders {
  width: number;
  height: number;
  texelsPerCell: 2;
  heights: Float32Array;
}

/** Eight absolute-world-height samples to the closest point of a light's segment. */
export function occlusionTransmission(
  receiver: Point,
  source: Point,
  heightAt: (x: number, z: number) => number,
  occlude = true,
): number {
  if (!occlude) return 1;
  let transmission = 1;
  for (let i = 1; i <= 8; i++) {
    const t = i / 8;
    const x = receiver[0] + (source[0] - receiver[0]) * t;
    const z = receiver[2] + (source[2] - receiver[2]) * t;
    if (Math.hypot(source[0] - x, source[2] - z) < 0.35) continue;
    const y = receiver[1] + (source[1] - receiver[1]) * t;
    const h = Math.max(0, Math.min(1, (heightAt(x, z) - y) / 0.4));
    transmission *= 1 - h * h * (3 - 2 * h);
  }
  return Math.max(0.1, transmission);
}
/** Unrolled graph twin. Receiver, source and uploaded top heights share absolute world y. */
export function occlusionNode(
  receiver: ShaderNode,
  source: ShaderNode,
  heights: ShaderNode,
  size: ShaderNode,
  enabled: ShaderNode,
): ShaderNode {
  let transmission: ShaderNode = float(1);
  for (let i = 1; i <= 8; i++) {
    const ray: ShaderNode = mix(receiver, source, i / 8);
    const h = texture(heights, ray.xz.div(size)).r;
    const blocked = float(1).sub(smoothstep(0, 0.4, h.sub(ray.y)));
    transmission = transmission.mul(length(source.xz.sub(ray.xz)).lessThan(0.35).select(float(1), blocked));
  }
  return enabled.greaterThan(0.5).select(transmission.max(0.1), float(1));
}

/** Allocated only by an opted-in world, before it constructs materials. */
export class RealLightNodes {
  readonly cost = new LightCost();
  logicFrame = -1;
  readonly enabled = uniform(1);
  readonly directional = uniform(1);
  readonly nightWeight = uniform(0);
  readonly scale = uniform(1);
  readonly artRolloff = uniform(NIGHT_ART_DEFAULTS.rolloff as number);
  readonly artBase = uniform(NIGHT_ART_DEFAULTS.base as number);
  readonly artWrap = uniform(NIGHT_ART_DEFAULTS.wrap as number);
  /** Luminance scale of projected redraw colour (page exposure and gain) into the sprite art scale. */
  readonly projectedArtScale = uniform(NIGHT_ART_DEFAULTS.projectedScale as number);
  readonly ambientOcclusion = uniform(NIGHT_ART_DEFAULTS.ao as number);
  contactField?: AmbientContactField;
  private readonly contacts = new Map<object, readonly AmbientContact[]>();
  setContacts(owner: object, contacts: readonly AmbientContact[]): void {
    if (!contacts.length) {
      this.deleteContacts(owner);
      return;
    }
    this.contacts.set(owner, contacts);
    this.contactField?.set(owner, contacts);
  }
  deleteContacts(owner: object): void {
    this.contacts.delete(owner);
    this.contactField?.delete(owner);
  }
  initContacts(width: number, depth: number): void {
    this.contactField?.dispose();
    this.contactField = new AmbientContactField(width, depth);
    for (const [owner, contacts] of this.contacts) this.contactField.set(owner, contacts);
    this.ambientContact.value = this.contactField.texture;
  }
  readonly size = uniform(new Vector2(1, 1));
  readonly dynamicBounds = uniform(new Vector4(0, 0, 1, 1));
  private readonly black = new DataTexture(new Float32Array(4), 1, 1, RGBAFormat, FloatType);
  private readonly flat = new DataTexture(new Uint16Array(1), 1, 1, RedFormat, HalfFloatType);
  /**
   * Ground height (world y) per cell, linearly filtered. Both field layers sit at this height (+0 and +UPPER) and
   * receivers measure their height above it: with absolute layers, elevated maps put the ground layer a level below
   * the ground and a fire 0.4 above the ground lit it at a tenth of its strength (sandbox r1, XMP29U2 at level 2).
   */
  readonly groundHeight = texture(this.flat);
  readonly groundHalfTexel = uniform(new Vector2(0.5, 0.5));
  readonly occluderHeight = texture(this.flat);
  readonly ambientContact = texture(this.flat);
  readonly accentGroundUv = uniform(new Vector4(1, 1, 0, 0));
  readonly accentUpperUv = uniform(new Vector4(1, 1, 0, 0));
  readonly accentGroundHalf = uniform(new Vector2(0.5, 0.5));
  readonly accentUpperHalf = uniform(new Vector2(0.5, 0.5));
  readonly accentGround = texture(this.black);
  readonly accentUpper = texture(this.black);
  readonly accentGroundDirection = texture(this.black);
  readonly accentUpperDirection = texture(this.black);
  /** Left half = actual source moments for objects; right half = view-projected ground moments. */
  readonly ground = texture(this.black);
  readonly groundDirection = texture(this.black);
  readonly upper = texture(this.black);
  readonly upperDirection = texture(this.black);
  constructor() {
    this.black.needsUpdate = true;
    this.flat.needsUpdate = true;
  }
  dispose(): void {
    this.contactField?.dispose();
    this.contacts.clear();
    this.black.dispose();
    this.flat.dispose();
  }
}
/** Object total-light ceiling in noon units; identifies the object receiver branch. */
export const OBJECT_KNEE = 0.8;
export function kneeNode(e: ShaderNode, cap: ShaderNode): ShaderNode {
  const lum = dot(e, vec3(...LUMA));
  return e.mul(
    cap
      .mul(float(1).sub(exp(lum.negate().div(cap.max(1e-8)))))
      .div(max(lum, 1e-8))
      .min(1),
  );
}
/**
 * Ground knee: logarithmic, so brightness keeps rising towards the source. The exponential ceiling flattened a burning
 * tank's light at SOV07TMD night (ambient about 0.1 noon) into a uniform yellow disc with a hard rim (live, 7a9107c).
 */
export function softKneeNode(e: ShaderNode, cap: ShaderNode): ShaderNode {
  const lum = dot(e, vec3(...LUMA));
  const c = cap.max(1e-8);
  return e.mul(c.mul(lum.div(c).add(1).log()).div(max(lum, 1e-8)).min(1));
}
/** S only for water/participating media; surfaces reconstruct the directional moments before the common knee. */
export function sampleRealLight(
  nodes: RealLightNodes,
  xz: ShaderNode,
  height: ShaderNode,
  normal?: ShaderNode,
  capScale = 1,
  ambientLum: ShaderNode = float(0.4),
): ShaderNode {
  const uv = xz.div(nodes.size);
  // Callers pass absolute world y; the layers follow the ground.
  height = max(height.sub(texture(nodes.groundHeight, uv).r), 0);
  const t = clamp(height.div(REAL_LIGHT_UPPER), 0, 1).mul(nodes.directional);
  const bounds = nodes.dynamicBounds;
  const inView = xz.x
    .greaterThanEqual(bounds.x)
    .and(xz.y.greaterThanEqual(bounds.y))
    .and(xz.x.lessThanEqual(bounds.z))
    .and(xz.y.lessThanEqual(bounds.w))
    .select(float(1), float(0));
  const groundUv = clamp(uv, nodes.groundHalfTexel, vec2(1).sub(nodes.groundHalfTexel))
    .mul(vec2(0.5, 1))
    .add(vec2(capScale === OBJECT_KNEE ? 0 : 0.5, 0));
  const g = texture(nodes.ground, groundUv);
  let s = mix(g.rgb, texture(nodes.upper, uv).rgb, t).mul(nodes.scale).mul(inView);
  // Static accents light upright art only. Ground keeps its legacy grade/pools exactly between events.
  const agUv = clamp(uv, nodes.accentGroundHalf, vec2(1).sub(nodes.accentGroundHalf))
    .mul(nodes.accentGroundUv.xy)
    .add(nodes.accentGroundUv.zw);
  const auUv = clamp(uv, nodes.accentUpperHalf, vec2(1).sub(nodes.accentUpperHalf))
    .mul(nodes.accentUpperUv.xy)
    .add(nodes.accentUpperUv.zw);
  if (capScale === OBJECT_KNEE)
    s = s.add(mix(texture(nodes.accentGround, agUv).rgb, texture(nodes.accentUpper, auUv).rgb, t).mul(nodes.scale));
  let e: ShaderNode = s;
  if (normal) {
    let d = mix(texture(nodes.groundDirection, groundUv).xyz, texture(nodes.upperDirection, uv).xyz, t)
      .mul(nodes.scale)
      .mul(inView);
    if (capScale === OBJECT_KNEE)
      d = d.add(
        mix(texture(nodes.accentGroundDirection, agUv).xyz, texture(nodes.accentUpperDirection, auUv).xyz, t).mul(
          nodes.scale,
        ),
      );
    const lum = dot(s, vec3(...LUMA));
    const magnitude = length(d);
    const dominant = magnitude.min(lum);
    const cosine = max(dot(normal, d).div(max(magnitude, 1e-8)), 0);
    const reconstructed = s.mul(dominant.mul(cosine).add(lum.sub(dominant).max(0).mul(0.5)).div(max(lum, 1e-8)));
    e = mix(s, reconstructed, nodes.directional);
    // Preserve the alpha contract: upright receivers take only the reduced ground-only share.
    const groundShare = mix(g.a, texture(nodes.upper, uv).a, t)
      .mul(nodes.scale)
      .mul(inView)
      .div(max(lum, 1e-8))
      .clamp(0, 1);
    e = e.mul(float(1).sub(groundShare.mul(max(normal.y.negate().add(1), 0).min(1)).mul(0.7)));
  }
  const cap =
    capScale === OBJECT_KNEE
      ? float(OBJECT_KNEE).sub(ambientLum).max(0)
      : mix(float(1.2), ambientLum.mul(3), nodes.nightWeight);
  const attenuated = e.mul(exp(max(height.sub(REAL_LIGHT_UPPER), 0).mul(-0.45)));
  return capScale === OBJECT_KNEE ? kneeNode(attenuated, cap) : softKneeNode(attenuated, cap);
}

interface TimedLight {
  light: PhysicalLight;
  born: number;
  seen: number;
  hold: number;
  fade: number;
  ttl?: number;
}
const CAPACITIES = { static: 896, explosion: 128, flash: 256, kept: 256, source: 256 } as const;
export type RealLightPool = keyof typeof CAPACITIES;
/** Independent pools preserve fire/source capacity during bursts. Legacy buffers are never repurposed. */
export class RealLightField {
  readonly targets: RenderTarget[];
  readonly scenes: Scene[];
  readonly camera: OrthographicCamera;
  readonly pools = new Map<RealLightPool, Map<string, TimedLight>>();
  private readonly attributes: InstancedBufferAttribute[][] = [];
  private readonly meshes: InstancedMesh[] = [];
  private readonly layers: number[];
  private readonly viewSlope = uniform(new Vector2());
  private readonly groundRange = uniform(new Vector2());
  private readonly footprintScene: Scene;
  private readonly tileClears: Mesh[] = [];
  private readonly cursors = new Map<RealLightPool, number>();
  private lastNight = -1;
  private staticDirty = true;
  readonly staticTargets: RenderTarget[];
  readonly staticScenes: Scene[];
  staticRenders = 0;
  private readonly viewport = { x0: 0, z0: 0, x1: Infinity, z1: Infinity };
  private viewKey = '';
  private lastViewKey = '';
  private lastCount = -1;
  private previousDynamic: Float32Array[][] = [];
  private groundMin = 0;
  private groundMax = 0;
  private readonly corner = new Vector3();
  /** Top-down bounds of the visible ground at both terrain extrema, expanded by the dynamic reach budget. */
  setView(camera: OrthographicCamera): void {
    let x0 = Infinity,
      z0 = Infinity,
      x1 = -Infinity,
      z1 = -Infinity;
    const direction = this.corner.set(0, 0, -1).transformDirection(camera.matrixWorld).clone();
    this.viewSlope.value.set(direction.x / Math.min(-1e-8, direction.y), direction.z / Math.min(-1e-8, direction.y));
    for (const y of [this.groundMin, this.groundMax])
      for (const x of [-1, 1])
        for (const z of [-1, 1]) {
          const p = this.corner.set(x, z, -1).unproject(camera);
          const t = (y - p.y) / Math.min(-1e-8, direction.y);
          const px = p.x + direction.x * t,
            pz = p.z + direction.z * t;
          x0 = Math.min(x0, px);
          z0 = Math.min(z0, pz);
          x1 = Math.max(x1, px);
          z1 = Math.max(z1, pz);
        }
    const target = this.targets[0]!;
    // Snap outwards to texels: subpixel camera motion need not re-render a stationary field.
    const sx = target.width / 2 / this.width,
      sz = target.height / this.height;
    Object.assign(this.viewport, {
      x0: Math.min(this.width, Math.max(0, Math.floor((x0 - 3) * sx) / sx)),
      z0: Math.min(this.height, Math.max(0, Math.floor((z0 - 3) * sz) / sz)),
      x1: Math.max(0, Math.min(this.width, Math.ceil((x1 + 3) * sx) / sx)),
      z1: Math.max(0, Math.min(this.height, Math.ceil((z1 + 3) * sz) / sz)),
    });
    this.nodes.dynamicBounds.value.set(this.viewport.x0, this.viewport.z0, this.viewport.x1, this.viewport.z1);
    const key = [...Object.values(this.viewport), ...this.viewSlope.value.toArray()].join(',');
    if (key !== this.viewKey) {
      this.viewKey = key;
    }
  }
  /** Empty dynamic field has exactly zero local irradiance on every terrain receiver. */
  get dynamicBounds() {
    return { ...this.viewport };
  }

  private occluders: DataTexture | null = null;
  private groundTexture: DataTexture | null = null;
  private groundHeights: ArrayLike<number> = [];
  private readonly clearColour = new Color();
  renders = 0;
  constructor(
    readonly width: number,
    readonly height: number,
    readonly nodes: RealLightNodes,
    directional = true,
    halfFloat = true,
  ) {
    nodes.directional.value = directional ? 1 : 0;
    nodes.scale.value = halfFloat ? 1 : 8;
    nodes.size.value.set(width, height);
    nodes.dynamicBounds.value.set(0, 0, width, height);
    nodes.initContacts(width, height);
    this.layers = directional ? [0, REAL_LIGHT_UPPER] : [0];
    this.camera = new OrthographicCamera(0, width, 0, height, -10, 10);
    this.camera.position.set(0, 0, 5);
    this.camera.updateMatrixWorld();
    const density = Math.max(1, Math.min(8, Math.floor(2048 / Math.max(width, height))));
    this.targets = this.layers.map((_, i) => {
      const target = new RenderTarget(
        Math.max(1, Math.floor((width * density) / (i ? 2 : 1))),
        Math.max(1, Math.floor((height * density) / (i ? 2 : 1))),
        {
          count: directional ? 2 : 1,
          type: halfFloat ? HalfFloatType : UnsignedByteType,
          depthBuffer: false,
          minFilter: LinearFilter,
          magFilter: LinearFilter,
          generateMipmaps: false,
        },
      );
      target.textures[0]!.name = 'output';
      if (directional) target.textures[1]!.name = 'direction';
      return target;
    });
    // Pack both cached heights into one MRT atlas: upright art must remain within the 16-texture stage limit.
    const ground = this.targets[0]!,
      upper = this.targets[1];
    const atlas = ground.clone();
    atlas.setSize(ground.width + (upper?.width ?? 0), ground.height);
    this.staticTargets = [atlas];
    nodes.accentGroundUv.value.set(ground.width / atlas.width, 1, 0, 0);
    nodes.accentGroundHalf.value.set(0.5 / ground.width, 0.5 / ground.height);
    nodes.accentUpperUv.value.set(
      (upper?.width ?? ground.width) / atlas.width,
      1,
      upper ? ground.width / atlas.width : 0,
      0,
    );
    nodes.accentUpperHalf.value.set(0.5 / (upper?.width ?? ground.width), 0.5 / ground.height);
    nodes.accentGround.value = nodes.accentUpper.value = atlas.textures[0]!;
    if (directional) nodes.accentGroundDirection.value = nodes.accentUpperDirection.value = atlas.textures[1]!;
    nodes.ground.value = this.targets[0]!.textures[0]!;
    nodes.upper.value = (this.targets[1] ?? this.targets[0])!.textures[0]!;
    if (directional) {
      nodes.groundDirection.value = this.targets[0]!.textures[1]!;
      nodes.upperDirection.value = this.targets[1]!.textures[1]!;
    }
    for (const pool of Object.keys(CAPACITIES) as RealLightPool[]) this.pools.set(pool, new Map());
    this.scenes = this.layers.map((y) => this.build(y, directional));
    this.staticScenes = this.layers.map((y) => this.build(y, directional));
    this.footprintScene = this.build(0, directional, true);
    this.addTileClear(this.footprintScene, directional);
    for (const scene of this.staticScenes.slice(1)) this.addTileClear(scene, directional);
    // Reuse the ground samplers: a second atlas tile avoids exceeding upright art's texture budget.
    nodes.groundHalfTexel.value.set(0.5 / ground.width, 0.5 / ground.height);
    ground.setSize(ground.width * 2, ground.height);
  }
  /**
   * WebGPU loadOp=clear clears the WHOLE attachment, regardless of scissor. Clear later atlas tiles by drawing zero
   * instead, with loadOp=load, so they cannot erase the first tile. This also resets stale texels on WebGL.
   */
  private addTileClear(scene: Scene, directional: boolean): void {
    const material = new MeshBasicNodeMaterial();
    material.depthTest = material.depthWrite = false;
    material.positionNode = vec3(positionGeometry.xy.add(0.5).mul(vec2(this.width, this.height)), 0);
    material.colorNode = vec4(0);
    if (directional) material.mrtNode = mrt({ output: vec4(0), direction: vec4(0) });
    const mesh = new Mesh(new PlaneGeometry(), material);
    mesh.frustumCulled = false;
    mesh.renderOrder = -1;
    mesh.name = 'real-light-tile-clear';
    scene.add(mesh);
    this.tileClears.push(mesh);
  }
  private build(y: number, directional: boolean, footprint = false): Scene {
    const capacity = Object.values(CAPACITIES).reduce((a, b) => a + b, 0);
    const attrs = Array.from({ length: 4 }, () => new InstancedBufferAttribute(new Float32Array(capacity * 4), 4));
    const [a, b, c, extra] = attrs.map((v) => instancedBufferAttribute(v, 'vec4') as ShaderNode) as [
      ShaderNode,
      ShaderNode,
      ShaderNode,
      ShaderNode,
    ];
    const m = new MeshBasicNodeMaterial();
    m.transparent = true;
    m.depthTest = false;
    m.depthWrite = false;
    m.side = DoubleSide;
    m.blending = CustomBlending;
    m.blendEquation = m.blendEquationAlpha = AddEquation;
    m.blendSrc = m.blendSrcAlpha = OneFactor;
    m.blendDst = m.blendDstAlpha = OneFactor;
    const projected = footprint ? extra.w.greaterThan(1.5) : float(0).greaterThan(1);
    const project = (v: ShaderNode, h: ShaderNode) => v.xz.add(this.viewSlope.mul(h.sub(v.y)));
    const pa = project(a, this.groundRange.x),
      pb = project(b, this.groundRange.x);
    const qa = project(a, this.groundRange.y),
      qb = project(b, this.groundRange.y);
    const lo = projected.select(pa.min(pb).min(qa).min(qb), a.xz.min(b.xz)).sub(b.w),
      hi = projected.select(pa.max(pb).max(qa).max(qb), a.xz.max(b.xz)).add(b.w);
    m.positionNode = vec3(mix(lo, hi, positionGeometry.xy.add(0.5)), 0);
    const p = vec3(
      positionWorld.x,
      texture(this.nodes.groundHeight, positionWorld.xy.div(this.nodes.size)).r.add(y),
      positionWorld.y,
    );
    const source = projected.select(vec3(project(a, p.y).x, p.y, project(a, p.y).y), a.xyz);
    const end = projected.select(vec3(project(b, p.y).x, p.y, project(b, p.y).y), b.xyz);
    const ab = end.sub(source),
      len = length(ab),
      axis = ab.div(max(len, 1e-8));
    const along = dot(p.sub(source), axis),
      u = clamp(along, 0, len);
    const delta = source.add(axis.mul(u)).sub(p),
      distance = length(delta);
    const d2 = dot(delta, delta).add(a.w.mul(a.w));
    const perpendicular = max(d2.sub(along.sub(u).pow(2)), 1e-8).sqrt();
    const integral = atan(len.sub(along).div(perpendicular))
      .add(atan(along.div(perpendicular)))
      .div(perpendicular);
    const windowDistance = extra.w.mod(2).greaterThan(0.5).select(length(delta.xz), distance);
    const windowT = clamp(windowDistance.div(max(b.w, 1e-8)), 0, 1);
    const weight = float(1)
      .sub(extra.z.greaterThan(1.5).select(windowT.pow(6), windowT.pow(2)))
      .pow(2);
    const transmission = Fn(() => {
      const value = float(1).toVar();
      If(extra.y.greaterThan(0.5), () => {
        value.assign(occlusionNode(p, p.add(delta), this.nodes.occluderHeight, this.nodes.size, extra.y));
      });
      return value;
    })();
    const gaussian = exp(
      dot(delta, delta)
        .div(max(a.w.mul(a.w), 1e-8))
        .negate(),
    );
    const scalar = extra.z
      .greaterThan(1.5)
      .select(gaussian, extra.z.greaterThan(0.5).select(integral, float(1).div(max(d2, 1e-8))))
      .mul(weight)
      .mul(transmission);
    const s = c.rgb.mul(scalar),
      lum = dot(s, vec3(...LUMA));
    const direction = projected.select(vec3(0, 1, 0), delta.div(max(distance, 1e-8)));
    const output = vec4(directional ? s : s.mul(max(direction.y, 0)), lum.mul(extra.x));
    m.colorNode = output;
    if (directional) m.mrtNode = mrt({ output, direction: vec4(direction.mul(lum), 0) });
    const mesh = new InstancedMesh(new PlaneGeometry(), m, capacity);
    mesh.frustumCulled = false;
    mesh.count = 0;
    const scene = new Scene();
    scene.add(mesh);
    this.attributes.push(attrs);
    this.meshes.push(mesh);
    return scene;
  }
  /** Game module supplies absolute top heights; this field owns the upload and cached shadow invalidation. */
  setOccluders(data: LightOccluders): void {
    if (
      data.heights.length !== data.width * data.height ||
      data.width !== this.width * 2 ||
      data.height !== this.height * 2
    )
      throw new Error('Invalid light occluder dimensions');
    this.occluders?.dispose();
    // HalfFloatType requires binary16 storage; avoid silently interpreting float32 bytes as half-floats.
    this.occluders = new DataTexture(
      Uint16Array.from(data.heights, (h) => DataUtils.toHalfFloat(h)),
      data.width,
      data.height,
      RedFormat,
      HalfFloatType,
    );
    this.occluders.needsUpdate = true;
    this.nodes.occluderHeight.value = this.occluders;
    this.staticDirty = true;
  }
  /** Ground height per cell (world y), row-major width x height cells. */
  setGround(heights: ArrayLike<number>): void {
    if (heights.length !== this.width * this.height) throw new Error('Invalid light ground dimensions');
    this.groundHeights = Float32Array.from(heights);
    this.groundMin = Infinity;
    this.groundMax = -Infinity;
    for (let i = 0; i < heights.length; i++) {
      this.groundMin = Math.min(this.groundMin, heights[i]!);
      this.groundMax = Math.max(this.groundMax, heights[i]!);
    }
    this.groundRange.value.set(this.groundMin, this.groundMax);
    this.nodes.contactField?.invalidate();
    this.groundTexture?.dispose();
    this.groundTexture = new DataTexture(
      Uint16Array.from(heights as ArrayLike<number>, (h) => DataUtils.toHalfFloat(h)),
      this.width,
      this.height,
      RedFormat,
      HalfFloatType,
    );
    this.groundTexture.minFilter = this.groundTexture.magFilter = LinearFilter;
    this.groundTexture.needsUpdate = true;
    this.nodes.groundHeight.value = this.groundTexture;
    this.staticDirty = true;
  }
  private readonly staticChannels = new Map<string, readonly PhysicalLight[]>();
  setStatic(lights: readonly PhysicalLight[], channel = 'engine'): void {
    this.staticChannels.set(channel, lights);
    lights = [...this.staticChannels.values()].flat();
    this.staticDirty = true;
    this.pools.get('static')!.clear();
    lights.slice(0, CAPACITIES.static).forEach((l, i) => this.put('static', String(i), 0, l));
  }
  put(pool: RealLightPool, key: string, now: number, light: PhysicalLight, hold = 0, fade = 0.15, ttl?: number): void {
    const entries = this.pools.get(pool)!;
    const old = entries.get(key);
    if (!old && entries.size >= CAPACITIES[pool]) {
      if (pool === 'kept' || pool === 'source' || pool === 'static') return;
      entries.delete(entries.keys().next().value!);
    }
    if (pool === 'static') this.staticDirty = true;
    entries.set(key, { light, born: old?.born ?? now, seen: now, hold, fade, ttl });
  }
  pulse(pool: 'flash' | 'explosion', now: number, light: PhysicalLight, hold: number, fade: number): void {
    const id = this.cursors.get(pool) ?? 0;
    this.cursors.set(pool, id + 1);
    this.put(pool, String(id), now, light, hold, fade);
  }
  retain(pool: 'source' | 'kept', keys: ReadonlySet<string>): void {
    for (const key of this.pools.get(pool)!.keys())
      if (!keys.has(key)) {
        this.pools.get(pool)!.delete(key);
      }
  }
  clearDynamic(): void {
    for (const [name, pool] of this.pools) if (name !== 'static') pool.clear();
  }
  clear(): void {
    this.staticChannels.clear();
    for (const p of this.pools.values()) p.clear();
    this.staticDirty = true;
  }
  update(now: number, renderer: WebGPURenderer, night: number): boolean {
    if (this.nodes.enabled.value && night > 0)
      this.nodes.cost.measure('contactRasterUpload', () => {
        const updated = this.nodes.contactField?.update(
          (x, z) =>
            this.groundHeights[
              Math.min(this.height - 1, Math.max(0, Math.floor(z))) * this.width +
                Math.min(this.width - 1, Math.max(0, Math.floor(x)))
            ] ?? 0,
          this.nodes.logicFrame >= 0 ? this.nodes.logicFrame : Math.floor(now * 60),
        );
        if (updated && this.nodes.contactField) renderer.initTexture?.(this.nodes.contactField.texture);
      });
    return this.nodes.cost.measure('fieldUpdate', () => this.updateField(now, renderer, night));
  }
  private updateField(now: number, renderer: WebGPURenderer, night: number): boolean {
    this.nodes.nightWeight.value = night;
    if (!this.nodes.enabled.value) {
      this.nodes.cost.lightsSplatted = this.nodes.cost.texelsSplattedEstimate = 0;
      return false;
    }
    const refreshStatic = this.staticDirty || this.lastNight !== night;
    let staticCount = 0;
    let count = 0;
    let texels = 0;
    for (const [pool, entries] of this.pools)
      for (const [key, entry] of entries) {
        if (pool === 'static' && !refreshStatic) continue;
        const { light: l, born, seen, hold, fade, ttl } = entry;
        const envelope =
          pool === 'static'
            ? 1
            : ttl === undefined
              ? heldPulseEnvelope(now - born, hold, fade)
              : Math.min(1, Math.max(0, (ttl - now + seen) / Math.max(ttl * 0.5, 1e-8)));
        if (now < born) continue;
        if (envelope <= 0) {
          entries.delete(key);

          continue;
        }
        const gain =
          (envelope / this.nodes.scale.value) *
          (l.dayFactor + (1 - l.dayFactor) * night) *
          flicker(l.seed ?? 0, now, l.flicker);
        if (!(gain > 0)) continue;
        const isStatic = pool === 'static';
        const reach = lightReach(l),
          v = this.viewport;
        // Projected elevated endpoints can lie beyond their vertical ground footprint.
        const padding = l.viewProjectedFootprint
          ? Math.max(
              ...[l.position[1], (l.end ?? l.position)[1]].flatMap((h) => [
                Math.abs(h - this.groundMin),
                Math.abs(h - this.groundMax),
              ]),
            ) * this.viewSlope.value.length()
          : 0;
        const x0 = Math.max(v.x0, Math.min(l.position[0], l.end?.[0] ?? l.position[0]) - reach - padding),
          z0 = Math.max(v.z0, Math.min(l.position[2], l.end?.[2] ?? l.position[2]) - reach - padding),
          x1 = Math.min(v.x1, Math.max(l.position[0], l.end?.[0] ?? l.position[0]) + reach + padding),
          z1 = Math.min(v.z1, Math.max(l.position[2], l.end?.[2] ?? l.position[2]) + reach + padding);
        if (!isStatic && (x1 <= x0 || z1 <= z0)) continue;
        if (!isStatic)
          texels +=
            (((((x1 - x0) * (z1 - z0) * this.targets[0]!.width) / this.width) * this.targets[0]!.height) /
              this.height) *
            (this.nodes.directional.value ? 1.125 : 1);
        const index = isStatic ? staticCount : count;
        const selected = isStatic
          ? this.attributes.slice(this.layers.length, this.layers.length * 2)
          : [...this.attributes.slice(0, this.layers.length), this.attributes[this.layers.length * 2]!];
        for (const attrs of selected) {
          attrs[0]!.setXYZW(index, ...l.position, l.radius);
          attrs[1]!.setXYZW(index, ...(l.end ?? l.position), lightReach(l));
          const intensity = l.initialIntensity && now - born < (l.hotSeconds ?? 0) ? l.initialIntensity : l.intensity;
          attrs[2]!.setXYZW(index, intensity[0] * gain, intensity[1] * gain, intensity[2] * gain, 0);
          attrs[3]!.setXYZW(
            index,
            l.groundOnly ?? 0,
            l.occlude === false || (!isStatic && (ttl ?? hold + fade) < 0.3) || lightReach(l) < 1.5 ? 0 : 1,
            l.profile === 'gaussian' ? 2 : l.end ? 1 : 0,
            (l.planarReach ? 1 : 0) + (l.viewProjectedFootprint ? 2 : 0),
          );
        }
        if (isStatic) staticCount++;
        else count++;
      }
    let changed = this.lastCount !== count || this.lastViewKey !== this.viewKey;
    for (const layer of [...this.layers.keys(), this.layers.length * 2]) {
      const previous = (this.previousDynamic[layer] ??= []);
      for (let i = 0; i < 4; i++) {
        const values = this.attributes[layer]![i]!.array;
        const old = (previous[i] ??= new Float32Array(values.length));
        for (let c = 0; c < count * 4; c++)
          if (old[c] !== values[c]) {
            changed = true;
            old[c] = values[c]!;
          }
      }
    }
    this.nodes.cost.lightsSplatted = changed || refreshStatic ? count + staticCount : 0;
    this.nodes.cost.texelsSplattedEstimate = changed || refreshStatic ? Math.ceil(texels) : 0;
    if (!changed && !refreshStatic) return false;
    this.lastCount = count;
    this.lastViewKey = this.viewKey;
    this.staticDirty = false;

    this.lastNight = night;
    const previous = renderer.getRenderTarget(),
      previousMrt = renderer.getMRT(),
      alpha = renderer.getClearAlpha();
    renderer.getClearColor(this.clearColour);
    const autoClear = renderer.autoClear;
    renderer.setClearColor(0x000000, 0);
    renderer.autoClear = true;
    try {
      const draw = (index: number, count: number, scene: Scene, target: RenderTarget, clear: boolean) => {
        const mesh = this.meshes[index]!;
        mesh.count = count;
        for (const attr of this.attributes[index]!) {
          attr.clearUpdateRanges();
          attr.addUpdateRange(0, Math.max(1, count) * 4);
          attr.needsUpdate = true;
        }
        renderer.autoClear = clear;
        renderer.setRenderTarget(target);
        renderer.setMRT(this.nodes.directional.value ? (mesh.material as MeshBasicNodeMaterial).mrtNode : null);
        renderer.render(scene, this.camera);
      };
      for (let i = 0; i < this.layers.length; i++) {
        if (refreshStatic) {
          const atlas = this.staticTargets[0]!,
            layer = this.targets[i]!,
            offset = i ? this.targets[0]!.width / 2 : 0;
          atlas.viewport.set(offset, 0, i ? layer.width : layer.width / 2, atlas.height);
          atlas.scissor.copy(atlas.viewport);
          atlas.scissorTest = true;
          draw(i + this.layers.length, staticCount, this.staticScenes[i]!, atlas, i === 0);
        }
        const target = this.targets[i]!,
          v = this.viewport;
        const tileWidth = i ? target.width : target.width / 2;
        target.viewport.set(0, 0, tileWidth, target.height);
        const sx = tileWidth / this.width,
          sz = target.height / this.height;
        const x0 = Math.floor(v.x0 * sx),
          z0 = Math.floor(v.z0 * sz),
          x1 = Math.min(tileWidth, Math.ceil(v.x1 * sx)),
          z1 = Math.min(target.height, Math.ceil(v.z1 * sz));
        target.scissor.set(x0, z0, Math.max(0, x1 - x0), Math.max(0, z1 - z0));
        target.scissorTest = true;
        draw(i, count, this.scenes[i]!, target, true);
        if (i === 0) {
          target.viewport.set(tileWidth, 0, tileWidth, target.height);
          target.scissor.x += tileWidth;
          draw(this.layers.length * 2, count, this.footprintScene, target, false);
        }
      }
      if (refreshStatic) this.staticRenders++;
    } finally {
      renderer.setRenderTarget(previous);
      renderer.setMRT(previousMrt);
      renderer.autoClear = autoClear;
      renderer.setClearColor(this.clearColour, alpha);
    }
    this.renders++;
    return true;
  }
  dispose(): void {
    for (const t of [...this.targets, ...this.staticTargets]) t.dispose();
    for (const m of this.meshes) {
      m.geometry.dispose();
      (m.material as MeshBasicNodeMaterial).dispose();
      m.dispose();
    }
    for (const m of this.tileClears) {
      m.geometry.dispose();
      (m.material as MeshBasicNodeMaterial).dispose();
    }
    for (const s of [...this.scenes, ...this.staticScenes, this.footprintScene]) s.clear();
    this.occluders?.dispose();
    this.groundTexture?.dispose();
    this.clear();
  }
}
