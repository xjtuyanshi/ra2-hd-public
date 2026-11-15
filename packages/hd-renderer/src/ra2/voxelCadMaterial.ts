import {
  DataTexture,
  Data3DTexture,
  EnvironmentNode,
  MeshStandardNodeMaterial,
  PhysicalLightingModel,
  LinearFilter,
  LinearMipmapLinearFilter,
  NearestFilter,
  Texture,
  ClampToEdgeWrapping,
  NoColorSpace,
  RGBAFormat,
  UnsignedByteType,
  Vector2,
  Vector3,
  type NodeBuilder,
} from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraWorldMatrix,
  float,
  fract,
  mix,
  modelPosition,
  mx_noise_float,
  normalView,
  normalWorld,
  normalWorldGeometry,
  output,
  positionGeometry,
  positionView,
  positionViewDirection,
  positionWorld,
  roughness,
  sRGBTransferEOTF,
  smoothstep,
  step,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { DEFAULT_ATMOSPHERE_NODES, moonlit, objectLight, type AtmosphereNodes } from '../atmosphere';
import { finishSceneLighting } from '../lighting';
import { inheritSlotBinder, slotMaterial, textureSlot } from '../materialSlots';
import { applyObjectTint, meshObjectTint } from '../objectTint';
import { setSilhouette } from '../silhouette';
import type { ShaderNode } from '../tsl';
import type { LookVolume } from '../hardSurfaceLookVolume';
import { RA2_HARD_SURFACE_LOOK as policy } from './hardSurfaceLook';
import { cadLookupBytes, CAD_CLASS } from './voxelCadLookup';
import { objectLevelAt, paletteGrade, paletteNodes, srgbScale, type MapLightGrade } from './mapLighting';
import { spriteLookFor } from './spriteLook';
import { meshLocalLight, voxelLightingFor, voxelLookFinish, voxelReflectionFinish } from './vpl';
import { voxelPbrFor } from './voxelPbr';
import { VOXEL_TO_WORLD } from './voxelScale';
import {
  registerVoxelReflectionColour,
  registerVoxelReflectionVariant,
  voxelReflectionAmbient,
} from './voxelReflectionColour';

/** Calibrated against clear-noon Lambert energy, without a VPL albedo lift. See HD_BRIDGE.md. */
export const VOXEL_CAD_DEFAULTS = Object.freeze({
  brightness: 1.1,
  sunGain: 1.6,
  fill: 0.6,
  envStrength: 0.4,
  specular: 0.35,
  wear: policy.wear,
  lineDark: 0.92,
  haloDark: 0.3,
  linkDark: 0.55,
  bump: 0.3,
  houseLevel: 0.6,
  wallOcclusion: policy.wallOcclusion,
  wallGrime: policy.wallGrime,
  aoStrength: 1,
  /** Baked donor detail (see hardSurfaceDetailAtlas.ts): normal relief, cavity AO, signed grime (<0) / wear (>0). */
  detailNormal: 1,
  detailAo: 1,
  detailGrime: 4,
  detailWear: 3,
});
export const CAD_PAINT_SPECULAR_LIMITS = Object.freeze({ direct: 0.04, environment: 0.02 });
export type VoxelCadKnob = keyof typeof VOXEL_CAD_DEFAULTS;
interface CadGraph {
  prototype: MeshStandardNodeMaterial;
  lookup: ShaderNode;
  volume: ShaderNode;
  origin: ShaderNode;
  scale: ShaderNode;
  bounds: ShaderNode;
  flags: ShaderNode;
  detail: ShaderNode;
  detailSize: ShaderNode;
  detailNeutral: Texture;
}
export class VoxelCadLookState {
  readonly knobs = Object.fromEntries(Object.entries(VOXEL_CAD_DEFAULTS).map(([k, v]) => [k, uniform(v)])) as Record<
    VoxelCadKnob,
    ReturnType<typeof uniform<'float', number>>
  >;
  readonly graphs = new Map<MapLightGrade | null, CadGraph>();
  readonly byPalette = new WeakMap<Uint8Array, Map<LookVolume, MeshStandardNodeMaterial>>();
  readonly lookups = new WeakMap<Uint8Array, DataTexture>();
  readonly volumes = new WeakMap<LookVolume, Data3DTexture>();
  readonly details = new WeakMap<object, Texture>();
  set(knob: string, value: number): void {
    if (!Object.hasOwn(VOXEL_CAD_DEFAULTS, knob) || !Number.isFinite(value))
      throw new Error(`Invalid CAD knob: ${knob}`);
    this.knobs[knob as VoxelCadKnob].value = Math.max(0, Math.min(8, value));
  }
  get(): Record<string, number> {
    return Object.fromEntries(Object.entries(this.knobs).map(([k, n]) => [k, n.value]));
  }
}
const states = new WeakMap<AtmosphereNodes, VoxelCadLookState>();
export function voxelCadLookFor(nodes: AtmosphereNodes = DEFAULT_ATMOSPHERE_NODES): VoxelCadLookState {
  let state = states.get(nodes);
  if (!state) states.set(nodes, (state = new VoxelCadLookState()));
  return state;
}
/** Detail atlases hold data channels: linear, unpremultiplied, top row first, mipmapped against 1:1 aliasing. */
export function cadDetailTexture(image: TexImageSource): Texture {
  const t = new Texture(image as HTMLImageElement);
  t.flipY = false;
  t.premultiplyAlpha = false;
  t.colorSpace = NoColorSpace;
  t.minFilter = LinearMipmapLinearFilter;
  t.magFilter = LinearFilter;
  t.wrapS = t.wrapT = ClampToEdgeWrapping;
  t.generateMipmaps = true;
  t.anisotropy = 4;
  t.needsUpdate = true;
  return t;
}
/** A neutral detail texel: flat normal, no occlusion, no grime. */
function neutralDetailTexture(): DataTexture {
  const t = new DataTexture(new Uint8Array([128, 128, 255, 128]), 1, 1, RGBAFormat, UnsignedByteType);
  t.colorSpace = NoColorSpace;
  t.minFilter = t.magFilter = LinearFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}
export function cadVolumeTexture(volume: LookVolume): Data3DTexture {
  const t = new Data3DTexture(volume.data, ...volume.size);
  t.format = RGBAFormat;
  t.type = UnsignedByteType;
  t.colorSpace = NoColorSpace;
  t.minFilter = t.magFilter = LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = ClampToEdgeWrapping;
  t.generateMipmaps = false;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}
class CadSky extends EnvironmentNode {
  override setup(builder: NodeBuilder): undefined {
    (builder.context as { radiance: ShaderNode }).radiance.addAssign(this.envNode);
    return undefined;
  }
}
class CadLighting extends PhysicalLightingModel {
  constructor(private readonly material: CadMaterial) {
    super();
  }
  override start(builder: NodeBuilder): void {
    super.start(builder);
    const r = (builder.context as { reflectedLight: Record<string, ShaderNode> }).reflectedLight,
      m = this.material,
      k = m.state.knobs;
    const gain = m.ao.mul(k.brightness);
    r.directDiffuse.mulAssign(gain.mul(k.sunGain));
    r.indirectDiffuse.mulAssign(gain.mul(k.fill));
    // Cap the accumulated GGX lobes before scene grading; cream cannot turn into a clipped white plate.
    r.directSpecular.assign(r.directSpecular.mul(gain.mul(k.sunGain).mul(k.specular)).min(m.specularCaps.x));
    r.indirectSpecular.assign(r.indirectSpecular.mul(gain.mul(k.specular)).min(m.specularCaps.y));
  }
  override finish(builder: NodeBuilder): void {
    super.finish(builder);
    finishSceneLighting(builder, this.material.sceneMultiplier, this.material.cloud);
  }
}
class CadMaterial extends MeshStandardNodeMaterial {
  state!: VoxelCadLookState;
  ao!: ShaderNode;
  sceneMultiplier!: ShaderNode;
  cloud!: ShaderNode;
  specularCaps!: ShaderNode;
  override copy(source: this): this {
    super.copy(source);
    this.state = source.state;
    this.ao = source.ao;
    this.sceneMultiplier = source.sceneMultiplier;
    this.cloud = source.cloud;
    this.specularCaps = source.specularCaps;
    inheritSlotBinder(this, source);
    return this;
  }
  override setupEnvironment(): EnvironmentNode | null {
    return this.envNode ? new CadSky(this.envNode) : null;
  }
  override setupLightingModel(): PhysicalLightingModel {
    return new CadLighting(this);
  }
}
const eotf = (colour: ShaderNode): ShaderNode => sRGBTransferEOTF(colour);
const linear = (rgb: readonly number[]): ShaderNode => sRGBTransferEOTF(vec3(...(rgb as [number, number, number])));
const noise = (p: ShaderNode): ShaderNode => mx_noise_float(p).mul(0.5).add(0.5).clamp();
function createGraph(grade: MapLightGrade | null, nodes: AtmosphereNodes): CadGraph {
  const state = voxelCadLookFor(nodes),
    k = state.knobs,
    material = nodes.own(setSilhouette(new CadMaterial()));
  material.name = 'voxel-cad-painted-steel';
  material.state = state;
  material.sceneMultiplier = nodes.sceneObjectMul;
  material.cloud = nodes.alive.shadow;
  const lookup = textureSlot(),
    placeholder = nodes.own(
      cadVolumeTexture({
        origin: [0, 0, 0],
        size: [1, 1, 1],
        res: 1,
        data: new Uint8Array(4),
        surfaceTexels: 0,
        grooveTexels: 0,
        seams: 0,
        dots: 0,
      }),
    );
  const volume = textureSlot(placeholder),
    origin = uniform(new Vector3()),
    scale = uniform(new Vector3(1, 1, 1)),
    bounds = uniform(new Vector2(0, 1)),
    flags = uniform(new Vector2(0, 0));
  const detailNeutral = nodes.own(neutralDetailTexture()),
    detail = textureSlot(detailNeutral),
    detailSize = uniform(new Vector2(1, 1));
  const pos: ShaderNode = attribute('vpos', 'vec3'),
    code: ShaderNode = attribute('lookCode', 'float'),
    pi: ShaderNode = attribute('lookIndex', 'float');
  const u: ShaderNode = pi.add(0.5).div(256),
    raw: ShaderNode = texture(lookup, vec2(u, 1 / 6)),
    params = texture(lookup, vec2(u, 0.5));
  const is = (value: number): ShaderNode => float(1).sub(step(0.5, code.sub(value).abs()));
  const cls = (value: number): ShaderNode => float(1).sub(step(0.5, params.b.mul(255).sub(value).abs()));
  const codedHouse = is(1).max(is(5)),
    house = codedHouse.max(cls(CAD_CLASS.house)).mul(float(1).sub(is(2).max(flags.x)));
  const steel = is(2).max(flags.x),
    hatch = is(6),
    wheel = is(4);
  const sampled: ShaderNode = volume.sample(pos.sub(origin).mul(scale));
  const tyre = float(1).sub(step(0.375, sampled.b)),
    hub = step(0.625, sampled.b);
  const wheelSrgb: ShaderNode = mix(mix(vec3(0.66, 0.66, 0.62), vec3(0.16), tyre), vec3(0.3, 0.31, 0.33), hub);
  const hue = texture(lookup, vec2(mix(u, float(24.5 / 256), codedHouse), 5 / 6)).rgb;
  const houseSrgb: ShaderNode = hue.mul(k.houseLevel.mul(1.25).clamp(0.35, 0.9));
  let base: ShaderNode = mix(eotf(raw.rgb), eotf(houseSrgb), codedHouse);
  base = mix(base, linear(policy.barrelSteel), steel);
  base = mix(base, linear(policy.hatchColour), hatch);
  const reflectionBase: ShaderNode = mix(base, linear([0.66, 0.66, 0.62]), wheel);
  base = mix(base, eotf(wheelSrgb), wheel);
  const paint = cls(CAD_CLASS.paint)
    .max(house)
    .mul(float(1).sub(steel.max(hatch).max(wheel)));
  material.specularCaps = mix(
    vec2(0.1, 0.08),
    vec2(CAD_PAINT_SPECULAR_LIMITS.direct, CAD_PAINT_SPECULAR_LIMITS.environment),
    paint,
  );
  const localHeight: ShaderNode = positionGeometry.y.div(VOXEL_TO_WORLD).sub(bounds.x).max(0);
  const heightFrac = localHeight.div(bounds.y.mul(0.3).max(0.001)).clamp();
  const wall = float(1).sub(normalWorld.y.abs().mul(1.5).clamp());
  const skyOcclusion = float(1).sub(wall.mul(float(1).sub(heightFrac).pow(1.5)).mul(k.wallOcclusion));
  const grime = float(1).sub(wall.mul(float(1).sub(heightFrac).pow(1.2)).mul(k.wallGrime));
  // Baked detail: texel coordinates (x < 0 = no chart on this triangle) into the part's atlas.
  const detailUv: ShaderNode = attribute('detailUv', 'vec2'),
    hasDetail = step(0, detailUv.x),
    detailTexel: ShaderNode = texture(detail, detailUv.max(0).div(detailSize)),
    detailRelief: ShaderNode = detailTexel.rg.mul(2).sub(1).mul(hasDetail).mul(k.detailNormal),
    detailAo: ShaderNode = mix(float(1), detailTexel.b, hasDetail.mul(k.detailAo.min(1))),
    detailGrime: ShaderNode = detailTexel.a.mul(2).sub(1).mul(hasDetail);
  material.ao = float(1).sub(attribute('occlusion', 'float').mul(k.aoStrength)).clamp().mul(skyOcclusion).mul(detailAo);
  const track = cls(CAD_CLASS.track)
    .mul(flags.y)
    .mul(float(1).sub(step(3.5, localHeight)))
    .max(wheel.mul(tyre));
  const phase = fract(pos.x),
    gap = float(1).sub(phase.sub(0.5).abs().div(0.14)).clamp(),
    ridge = float(1).sub(phase.sub(0.15).abs().div(0.1)).clamp();
  const links = float(1).sub(gap.mul(k.linkDark)).add(ridge.mul(0.45));
  const wear = attribute('curvature', 'float')
    .mul(0.5)
    .mul(normalWorld.y.mul(1.4).add(0.2).clamp())
    .mul(noise(pos.div(2.5).add(11.7)).sub(0.45).div(0.2).clamp())
    .mul(paint)
    .mul(k.wear)
    .clamp();
  const detailWear = detailGrime.max(0).mul(k.detailWear).mul(paint).clamp();
  let albedo: ShaderNode = mix(base, linear([0.8, 0.78, 0.72]).mul(0.75), wear.max(detailWear));
  albedo = albedo.mul(float(1).sub(detailGrime.negate().max(0).mul(k.detailGrime).clamp(0, 0.6)));
  albedo = albedo.mul(float(1).add(noise(pos.div(8)).sub(0.5).mul(0.12).mul(paint)));
  albedo = albedo.mul(mix(1, links, track));
  const low = float(1)
    .sub(localHeight.div(bounds.y.mul(0.3).max(0.001)).clamp())
    .pow(1.5)
    .mul(flags.y);
  albedo = albedo.mul(
    vec3(1).sub(
      vec3(0.2, 0.28, 0.42)
        .mul(low)
        .mul(0.28 * 2.2),
    ),
  );
  albedo = albedo
    .mul(float(1).sub(sampled.r.mul(k.lineDark)))
    .mul(float(1).sub(sampled.g.mul(k.haloDark)))
    .mul(grime);
  const r1: ShaderNode = positionView.dFdy().cross(normalView),
    r2: ShaderNode = normalView.cross(positionView.dFdx()),
    det: ShaderNode = positionView.dFdx().dot(r1);
  const bumpHeight: ShaderNode = sampled.r.negate().mul(k.bump).add(ridge.sub(gap).mul(track).mul(0.08));
  const gradient = r1.mul(bumpHeight.dFdx()).add(r2.mul(bumpHeight.dFdy()));
  const bumped: ShaderNode = normalView
    .mul(det.abs().max(1e-12))
    .sub(gradient.mul(det.sign()).mul(VOXEL_TO_WORLD))
    .normalize();
  // Cotangent frame from the chart coordinates (texels share one scale on both axes); chart seams inside a pixel
  // quad give huge derivatives, which shrink the perturbation there rather than inventing one.
  const dp1: ShaderNode = positionView.dFdx(),
    dp2: ShaderNode = positionView.dFdy(),
    duv1: ShaderNode = detailUv.dFdx(),
    duv2: ShaderNode = detailUv.dFdy();
  const p2: ShaderNode = dp2.cross(bumped),
    p1: ShaderNode = bumped.cross(dp1);
  const tangent: ShaderNode = p2.mul(duv1.x).add(p1.mul(duv2.x)),
    bitangent: ShaderNode = p2.mul(duv1.y).add(p1.mul(duv2.y));
  const frameScale: ShaderNode = tangent.dot(tangent).max(bitangent.dot(bitangent)).max(1e-30).inverseSqrt();
  material.normalNode = bumped
    .add(tangent.mul(detailRelief.x).add(bitangent.mul(detailRelief.y)).mul(frameScale))
    .normalize();
  let rough: ShaderNode = mix(params.r, 0.3, is(5).max(steel));
  rough = mix(rough, 0.42, hatch);
  rough = mix(rough, mix(0.42, 0.7, tyre), wheel);
  material.roughnessNode = rough.clamp(0.28, 1);
  let metal: ShaderNode = mix(params.g, 0, house);
  metal = mix(metal, 0.85, steel);
  metal = mix(metal, 0.55, hatch);
  const reflectionMetal: ShaderNode = mix(metal, 0.55, wheel);
  metal = mix(metal, mix(0.55, 0.15, tyre), wheel);
  material.metalnessNode = metal.clamp();
  const level: ShaderNode = objectLevelAt(grade, modelPosition.xz).add(voxelLightingFor(nodes).voxelExtraLight),
    lit = srgbScale(albedo, level);
  material.colorNode = mix(moonlit(lit, nodes), lit, house);
  material.emissiveNode = lit
    .mul(objectLight(positionWorld.xz, nodes))
    .mul(1 / Math.PI)
    .mul(meshLocalLight)
    .mul(material.ao)
    .mul(k.brightness);
  const look = spriteLookFor(nodes);
  material.outputNode = Fn(() =>
    vec4(
      applyObjectTint(
        voxelLookFinish(output.rgb, voxelLightingFor(nodes), normalWorldGeometry),
        meshObjectTint.x,
        meshObjectTint.y,
        level,
        look.unlitBaseLight,
      ),
      output.a,
    ),
  )();
  const reflection: ShaderNode = positionViewDirection.negate().reflect(normalView),
    direction = mix(reflection, normalView, roughness.pow(4)).normalize().transformDirection(cameraWorldMatrix);
  material.envNode = mix(
    vec3(0.3, 0.33, 0.22),
    mix(vec3(0.7, 0.8, 0.92), vec3(0.8, 0.85, 0.95), direction.y.max(0).pow(0.6)),
    smoothstep(-0.12, 0.12, direction.y),
  ).mul(k.envStrength);
  // A conservative ambient-only water twin; it intentionally does not sample the surface volume or derivatives.
  registerVoxelReflectionColour(material, (sourceMatrix) => {
    const sourceOrigin: ShaderNode = sourceMatrix.mul(vec4(0, 0, 0, 1)).xyz;
    const sourceLevel: ShaderNode = objectLevelAt(grade, sourceOrigin.xz).add(voxelLightingFor(nodes).voxelExtraLight);
    const graded: ShaderNode = srgbScale(reflectionBase, sourceLevel);
    const reflected: ShaderNode = mix(moonlit(graded, nodes), graded, house);
    const environment = voxelPbrFor(nodes);
    const ambient: ShaderNode = (environment.reflectionSky as ShaderNode)
      .min(environment.reflectionGround)
      .mul(k.fill)
      .mul(k.brightness);
    return voxelReflectionFinish(
      voxelReflectionAmbient(reflected.mul(float(1).sub(reflectionMetal)), ambient),
      voxelLightingFor(nodes),
    );
  });
  return { prototype: material, lookup, volume, origin, scale, bounds, flags, detail, detailSize, detailNeutral };
}
export interface CadMaterialPart {
  volume: LookVolume;
  ground: number;
  height: number;
  body: boolean;
  steel: boolean;
  /** Optional baked detail atlas and its size in texels. */
  detail?: { image: TexImageSource; size: readonly [number, number] } | null;
}
export function voxelCadMaterialFor(
  palette: Uint8Array,
  part: CadMaterialPart,
  nodes = paletteNodes(palette) ?? DEFAULT_ATMOSPHERE_NODES,
): MeshStandardNodeMaterial {
  const state = voxelCadLookFor(nodes);
  let materials = state.byPalette.get(palette);
  if (!materials) state.byPalette.set(palette, (materials = new Map()));
  let material = materials.get(part.volume);
  if (material) return material;
  const grade = paletteGrade(palette);
  let graph = state.graphs.get(grade);
  if (!graph) state.graphs.set(grade, (graph = createGraph(grade, nodes)));
  let lut = state.lookups.get(palette);
  if (!lut) {
    lut = nodes.own(new DataTexture(cadLookupBytes(palette), 256, 3, RGBAFormat, UnsignedByteType));
    lut.colorSpace = NoColorSpace;
    lut.minFilter = lut.magFilter = NearestFilter;
    lut.generateMipmaps = false;
    lut.needsUpdate = true;
    state.lookups.set(palette, lut);
  }
  let volume = state.volumes.get(part.volume);
  if (!volume) state.volumes.set(part.volume, (volume = nodes.own(cadVolumeTexture(part.volume))));
  let detailTex: Texture | undefined;
  if (part.detail) {
    detailTex = state.details.get(part.detail.image);
    if (!detailTex) state.details.set(part.detail.image, (detailTex = nodes.own(cadDetailTexture(part.detail.image))));
  }
  const slots = graph,
    lookup = lut,
    tex = volume,
    neutral = graph.detailNeutral,
    detailSize = part.detail?.size ?? [1, 1];
  material = nodes.own(
    slotMaterial(graph.prototype, () => {
      slots.lookup.value = lookup;
      slots.volume.value = tex;
      slots.origin.value.set(...part.volume.origin);
      slots.scale.value.set(...part.volume.size.map((v) => part.volume.res / v));
      slots.bounds.value.set(part.ground, part.height);
      slots.flags.value.set(part.steel ? 1 : 0, part.body ? 1 : 0);
      slots.detail.value = detailTex ?? neutral;
      slots.detailSize.value.set(detailSize[0], detailSize[1]);
    }),
  );
  registerVoxelReflectionVariant(material, graph.prototype);
  materials.set(part.volume, material);
  return material;
}
