import { PhysicalLightingModel, type MeshStandardNodeMaterial, type NodeBuilder } from 'three/webgpu';
import {
  Fn,
  attribute,
  float,
  mix,
  modelPosition,
  normalWorld,
  output,
  positionWorld,
  sRGBTransferEOTF,
  smoothstep,
  step,
  normalView,
  positionViewDirection,
  roughness,
  cameraWorldMatrix,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { type AtmosphereNodes, moonlit, objectLight } from '../atmosphere';
import type { ShaderNode } from '../tsl';
import { applyObjectTint, meshObjectTint } from '../objectTint';
import { setSilhouette } from '../silhouette';
import { slotMaterial, textureSlot } from '../materialSlots';
import { objectLevelAt, paletteGrade, srgbScale, type MapLightGrade } from './mapLighting';
import { poweredOcclusion, spriteLookFor, highlightRoughness } from './spriteLook';
import { meshLocalLight, voxelLightingFor, voxelLookFinish, voxelMeshGains, voxelReflectionFinish } from './vpl';
import { finishSceneLighting } from '../lighting';
import { RemasterMaterial, VoxelPbrState, voxelPbrLookupTexture, voxelPbrFor, VPL_ALBEDO_ROW } from './voxelPbr';
import {
  registerVoxelReflectionColour,
  registerVoxelReflectionVariant,
  voxelReflectionNormal,
} from './voxelReflectionColour';

/** Per atmosphere, independent of both the projected and legacy PBR vehicle controls. */
export class VehicleMeshKnobs {
  readonly bevel = uniform(1.1);
  readonly seam = uniform(0.42);
  readonly wear = uniform(0.06);
  constructor(readonly spec = uniform(1.15)) {}
  readonly gain = uniform(0.9);
  readonly sat = uniform(1.18);
  readonly formLight = uniform(1.1);
  readonly ao = uniform(1.25);
  readonly level = uniform(0.86);
}

/** Only the opt-in material uses this balance. All extra edge energy comes from actual shadowed sun/sky terms. */
class VehicleMeshLighting extends PhysicalLightingModel {
  constructor(private readonly material: VehicleMeshMaterial) {
    super();
  }
  override start(builder: NodeBuilder): void {
    super.start(builder);
    const { knobs, edge, ao, gains } = this.material;
    const { reflectedLight: light } = builder.context as { reflectedLight: Record<string, ShaderNode> };
    const form: ShaderNode = knobs.formLight.clamp(0, 2);
    const direct: ShaderNode = form.mul(0.6).add(0.55).mul(gains.sunGain);
    const fill: ShaderNode = float(0.7).div(form.mul(1.6).add(1));
    const bevel: ShaderNode = edge.mul(knobs.bevel.clamp(0, 2));
    // A subpixel rounded edge redirects the sun across a wider angle than the plate BRDF. Its coverage is one
    // device pixel; it cannot emit in darkness or receive sunlight through a cast shadow. The sky rim is faint.
    light.directDiffuse!.mulAssign(ao.mul(direct).mul(bevel.mul(1.35).add(1)));
    light.indirectDiffuse!.mulAssign(ao.mul(fill).mul(bevel.mul(0.12).add(1)));
    light.directSpecular!.mulAssign(ao.mul(direct).mul(knobs.spec.max(0)));
    light.indirectSpecular!.mulAssign(ao.mul(knobs.spec.max(0)));
  }
  override finish(builder: NodeBuilder): void {
    super.finish(builder);
    finishSceneLighting(builder, this.material.sceneMultiplier, this.material.cloudShadow);
  }
}
class VehicleMeshMaterial extends RemasterMaterial {
  knobs!: VehicleMeshKnobs;
  edge!: ShaderNode;
  override copy(source: this): this {
    super.copy(source);
    this.knobs = source.knobs;
    this.edge = source.edge;
    return this;
  }
  override setupLightingModel(): PhysicalLightingModel {
    return new VehicleMeshLighting(this);
  }
}
interface State {
  own<T extends { dispose(): void }>(resource: T): T;
  knobs: VehicleMeshKnobs;
  physical: VoxelPbrState;
  palettes: WeakMap<Uint8Array, MeshStandardNodeMaterial>;
  graphs: Map<MapLightGrade | null, { prototype: RemasterMaterial; lookup: ShaderNode }>;
}
const states = new WeakMap<AtmosphereNodes, State>();
function stateFor(nodes: AtmosphereNodes): State {
  let state = states.get(nodes);
  if (!state) {
    const physical = new VoxelPbrState();
    physical.specular.value = 1.15;
    const knobs = new VehicleMeshKnobs(physical.specular);
    physical.brightness.value = 1;
    physical.sunGain.value = 1;
    physical.fill.value = 0.65;
    state = { own: (resource) => nodes.own(resource), knobs, physical, palettes: new WeakMap(), graphs: new Map() };
    states.set(nodes, state);
  }
  return state;
}
export const vehicleMeshKnobsFor = (nodes: AtmosphereNodes): VehicleMeshKnobs => stateFor(nodes).knobs;

function graphFor(nodes: AtmosphereNodes, grade: MapLightGrade | null, state: State) {
  const { knobs, physical } = state;
  const material = state.own(setSilhouette(new VehicleMeshMaterial()));
  material.knobs = knobs;
  material.name = 'vehicle-mesh';
  material.state = physical;
  material.look = spriteLookFor(nodes);
  material.gains = voxelMeshGains(voxelLightingFor(nodes), material.look);
  material.sceneMultiplier = nodes.sceneObjectMul;
  material.cloudShadow = nodes.alive.shadow;
  const lut = textureSlot();
  const u: ShaderNode = attribute('pshade', 'float').add(0.5).div(256);
  const raw: ShaderNode = sRGBTransferEOTF(texture(lut, vec2(u, 1 / 6)).rgb);
  const params: ShaderNode = texture(lut, vec2(u, 0.5));
  const cls: ShaderNode = attribute('materialClass', 'float');
  const house: ShaderNode = float(1).sub(step(0.5, cls));
  const metal: ShaderNode = float(1).sub(step(0.5, cls.sub(2).abs()));
  // Distance in voxel lattice units divided by the fragment derivative gives device pixels. No atlas, mip, baked
  // lighting or facing blend is involved. A one-pixel band survives zoom and articulated turret rotation.
  const line = (name: string): ShaderNode => {
    const d: ShaderNode = attribute(name, 'float');
    const pixels = d.div(d.fwidth().max(1e-4));
    return float(1).sub(smoothstep(0.05, 0.55, pixels));
  };
  const edge = (material.edge = line('bevelDistance'));
  const seam = line('seamDistance').mul(knobs.seam.clamp(0, 0.9));
  const wear = edge.mul(knobs.wear.clamp()).mul(metal).mul(float(1).sub(house));
  const luma: ShaderNode = raw.dot(vec3(0.2126, 0.7152, 0.0722));
  let albedo: ShaderNode = mix(vec3(luma), raw, mix(knobs.sat, 1, house)).max(0);
  albedo = srgbScale(mix(albedo, vec3(0.5), wear).mul(float(1).sub(seam)), knobs.gain.max(0));
  const level: ShaderNode = objectLevelAt(grade, modelPosition.xz).add(voxelLightingFor(nodes).voxelExtraLight);
  const lit: ShaderNode = srgbScale(albedo, level);
  material.colorNode = mix(moonlit(lit, nodes), lit, house);
  material.roughnessNode = highlightRoughness(
    params.r.sub(edge.mul(knobs.bevel).mul(0.22)).max(0.18),
    material.gains.specularSize,
  );
  material.metalnessNode = params.g.add(wear.mul(0.2)).clamp();
  material.ao = poweredOcclusion(
    attribute('occlusion', 'float').mul(knobs.ao.max(0)).mul(-2.5).exp().max(0.1),
    material.gains.aoPower,
  );
  // The ordinary GGX sheen remains alongside the coverage-weighted edge response in VehicleMeshLighting.
  const local: ShaderNode = objectLight(positionWorld.xz, nodes)
    .mul(1 / Math.PI)
    .mul(normalWorld.y.max(0).mul(0.45).add(0.55))
    .mul(meshLocalLight);
  material.emissiveNode = lit.mul(local.add(params.a)).mul(material.ao);
  material.outputNode = Fn(() =>
    vec4(
      applyObjectTint(
        srgbScale(voxelLookFinish(output.rgb, voxelLightingFor(nodes), normalWorld), knobs.level.max(0)),
        meshObjectTint.x,
        meshObjectTint.y,
        level,
        material.look.unlitBaseLight,
      ),
      output.a,
    ),
  )();
  const reflection: ShaderNode = positionViewDirection.negate().reflect(normalView);
  const direction: ShaderNode = mix(reflection, normalView, roughness.pow(4))
    .normalize()
    .transformDirection(cameraWorldMatrix);
  const sceneSky = voxelPbrFor(nodes);
  const sky: ShaderNode = mix(material.look.skyHorizon, material.look.skyZenith, direction.y.max(0).pow(0.6));
  const ground: ShaderNode = physical.groundColour;
  material.envNode = mix(
    ground.mul(sceneSky.reflectionGround),
    sky.mul(sceneSky.reflectionSky),
    smoothstep(-0.12, 0.12, direction.y),
  ).mul(0.25);
  registerVoxelReflectionColour(material, (sourceMatrix) => {
    const origin: ShaderNode = sourceMatrix.mul(vec4(0, 0, 0, 1)).xyz;
    const normal = voxelReflectionNormal(sourceMatrix);
    const base = srgbScale(raw, objectLevelAt(grade, origin.xz).mul(knobs.gain).mul(knobs.level));
    return voxelReflectionFinish(
      mix(moonlit(base, nodes), base, house).mul(normal.y.max(0).mul(0.1).add(0.12)),
      voxelLightingFor(nodes),
    );
  });
  return { prototype: material, lookup: lut };
}

export function vehicleMeshMaterialFor(
  palette: Uint8Array,
  nodes: AtmosphereNodes,
  state = stateFor(nodes),
): MeshStandardNodeMaterial {
  let material = state.palettes.get(palette);
  if (material) return material;
  const grade = paletteGrade(palette);
  let graph = state.graphs.get(grade);
  if (!graph) state.graphs.set(grade, (graph = graphFor(nodes, grade, state)));
  const lookup = state.own(voxelPbrLookupTexture(palette, voxelLightingFor(nodes).loaded, VPL_ALBEDO_ROW));
  const slot = graph.lookup;
  material = state.own(
    slotMaterial(graph.prototype, () => {
      slot.value = lookup;
    }),
  );
  registerVoxelReflectionVariant(material, graph.prototype);
  state.palettes.set(palette, material);
  return material;
}

/** A controller owns its shader graphs and palettes; off releases them, while knobs retain their values. */
export class VehicleMeshMaterialLibrary {
  private readonly resources = new Set<{ dispose(): void }>();
  private readonly state: State;
  constructor(private readonly nodes: AtmosphereNodes) {
    this.state = {
      ...stateFor(nodes),
      palettes: new WeakMap(),
      graphs: new Map(),
      own: (resource) => {
        this.resources.add(resource);
        return resource;
      },
    };
  }
  get(palette: Uint8Array): MeshStandardNodeMaterial {
    return vehicleMeshMaterialFor(palette, this.nodes, this.state);
  }
  dispose(): void {
    for (const resource of this.resources) resource.dispose();
    this.resources.clear();
    this.state.palettes = new WeakMap();
    this.state.graphs.clear();
  }
}
