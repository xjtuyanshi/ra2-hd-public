import { describe, expect, it, vi } from 'vitest';
import {
  BoxGeometry,
  DataTexture,
  DirectionalLight,
  HemisphereLight,
  Mesh,
  MeshBasicNodeMaterial,
  OrthographicCamera,
  Scene,
  Vector3,
  WebGPURenderer,
} from 'three/webgpu';
import { float, lights, positionWorld, uniform, vec3, vec4 } from 'three/tsl';
import { AmbientContactField, ambientContactAt, ambientContactNode } from '../src/ambientContact';
import { AtmosphereNodes, moonlit, objectMoonlit, localSurfaceLight } from '../src/atmosphere';
import {
  alphaRim,
  nightArtColour,
  nightHighlightScale,
  nightSharpen,
  NIGHT_ART_DEFAULTS,
  nightArtNode,
  nightAmbient,
} from '../src/nightArt';
import { lightLuminance, reconstructLight, RealLightNodes } from '../src/realLight';
import { facingReceiverNormal, facingReceiverNormalNode, spriteContactFootprint } from '../src/ra2/nightObjects';
import { voxelAmbientContact } from '../src/ra2/voxelShadow';
import { SpriteLayer } from '../src/ra2/sprites';
import { SpritePool } from '../src/live/spritePool';
import { createVoxelProjectionPart, VoxelProjectionInputs, voxelProjectionFor } from '../src/ra2/voxelProjection';
import { projectedVoxelColour } from '../src/ra2/voxelProjectionNodes';
import { voxelProjectedMaterialFor } from '../src/ra2/vpl';
import { buildVoxelGeometry, createVoxelMaterial, instantiateVoxel } from '../src/ra2/voxels';
import { VoxelUnits } from '../src/live/voxelUnits';
import { painterObject } from '../src/ra2/painterOrder';
import { bindSlots } from '../src/materialSlots';
import { syntheticModel, syntheticSection } from './fixtures/syntheticVoxels';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';
import type { VoxelProjectionManifest } from '../src/ra2/voxelProjectionMath';

const lstar = (y: number) => 116 * Math.cbrt(y) - 16;
const rgb = (y: number): [number, number, number] => [y, y, y];
const shader = (colour: ReturnType<typeof vec3>) => {
  const material = new MeshBasicNodeMaterial();
  material.colorNode = vec4(colour, 0.73);
  const mesh = new Mesh(new BoxGeometry(), material);
  const text = shaderBytes(mesh);
  mesh.geometry.dispose();
  material.dispose();
  return text;
};

describe('night art scene-only transfer', () => {
  it('has an exact day identity, a smooth hue-preserving shoulder and no change below the base', () => {
    for (const y of [0, 0.01, 0.1, 0.24, 0.379, 0.38]) expect(nightHighlightScale(y)).toBe(1);
    expect(nightHighlightScale(0.380001)).toBeCloseTo(1, 8);
    const scene = [0.6, 0.3, 0.15] as const;
    expect(nightArtColour(scene, [1, 1, 1], [0, 0, 0], 1, 0)).toBe(scene);
    const graded = nightArtColour(scene, rgb(0.65), rgb(0), 0, 1);
    expect(graded[0] / graded[1]).toBe(2);
    expect(graded[1] / graded[2]).toBe(2);
    const shoulderDrop = lstar(0.65) - lstar(0.65 * nightHighlightScale(0.65));
    expect(shoulderDrop).toBeGreaterThan(8);
    expect(shoulderDrop).toBeLessThan(11);
    // Synthetic body: 94% midtone paint, 6% noon highlights. This is a transfer calibration, not asset QA.
    const body = [...Array(94).fill(0.18), ...Array(6).fill(0.65)] as number[];
    const mean = (values: number[]) => values.reduce((a, b) => a + b) / values.length;
    expect(lstar(mean(body)) - lstar(mean(body.map((y) => y * nightHighlightScale(y))))).toBeLessThan(2);
    const local = [0.9, 0.4, 0.1];
    const withLamp = graded.map((v, i) => v + local[i]!);
    withLamp.forEach((v, i) => expect(v - graded[i]!).toBeCloseTo(local[i]!));
  });
  it('wraps only occupied rim colour, never lifts a channel or changes coverage', () => {
    expect(alphaRim([1, 1, 1, 1])).toBe(0);
    expect(alphaRim([1, 0, 1, 1])).toBe(1);
    expect(alphaRim([-1, 2, 1, 1])).toBe(1);
    expect(alphaRim([])).toBe(0);
    const source = [0.3, 0.2, 0.1] as const;
    const result = nightArtColour(source, rgb(0.1), [0.01, 0.02, 0.8], 1, 1);
    expect(result).toEqual([0.3 + (0.01 - 0.3) * 0.35, 0.2 + (0.02 - 0.2) * 0.35, 0.1]);
    for (const n of [0, 0.25, 0.5, 1]) expect(nightSharpen(1.2, n)).toBe(1.2 * (1 - n));
  });
  it('emits a guarded assignment: the night=0 graph returns exactly its original scene and alpha', () => {
    const nodes = new RealLightNodes();
    const generated = shader(
      nightArtNode(vec3(0.2, 0.3, 0.4), vec3(0.9), vec3(0.01), float(1), nodes, float(0)) as ReturnType<typeof vec3>,
    );
    const result = generated.match(/(nodeVar\d+) = vec3<f32>\( 0\.2, 0\.3, 0\.4 \)/)![1]!;
    expect(result).toBeTruthy();
    expect(generated).toMatch(/if \(.*> 0\.0 \)/);
    expect(generated).toContain(`vec4<f32>( ${result}, 0.73 )`);
    // The only mutation is inside that branch; multiplying zero cannot reach the shoulder or wrap.
    expect(generated.match(new RegExp(`${result} =`, 'g'))).toHaveLength(2);
    nodes.dispose();
  });
  it('shares the terrain moon function, including house paint, while runtime-off restores the legacy expression', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const source = vec3(0.7, 0.1, 0.05);
    const generated = shader(objectMoonlit(source, float(1), nodes) as ReturnType<typeof vec3>);
    expect(generated).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(nodes.realLight!.nightWeight.value).toBe(0);
    nodes.realLight!.nightWeight.value = 1;
    const graph = nightAmbient(source, vec3(0.01, 0.02, 0.03), moonlit(vec3(1), nodes));
    expect(shader(graph as ReturnType<typeof vec3>)).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(nodes.realLight!.artRolloff.value).toBe(NIGHT_ART_DEFAULTS.rolloff);
    expect(nodes.realLight!.artWrap.value).toBe(NIGHT_ART_DEFAULTS.wrap);
    expect(nodes.realLight!.ambientOcclusion.value).toBe(NIGHT_ART_DEFAULTS.ao);
    nodes.dispose();
  });
});

describe('ambient contact receivers', () => {
  const pool = { x: 2, z: 2, width: 1, depth: 2, height: 0 };
  it('uses the contact-twin body bounds, oriented footprints, soft 0.4-cell edge and thin 0.3-cell building bands', () => {
    expect(ambientContactAt(pool, 2, 2)).toBe(1);
    expect(ambientContactAt(pool, 2.7, 2)).toBeCloseTo(0.5);
    expect(ambientContactAt(pool, 2.91, 2)).toBe(0);
    expect(ambientContactAt({ ...pool, yaw: Math.PI / 2 }, 2.9, 2)).toBe(1);
    expect(ambientContactAt({ ...pool, band: true }, 2, 2)).toBe(0);
    expect(ambientContactAt({ ...pool, band: true }, 2.65, 2)).toBeCloseTo(0.5);
    const geometry = new BoxGeometry(1, 0.5, 2);
    geometry.translate(0.3, 0, 0.1);
    const footprint = voxelAmbientContact(geometry, 4, 5, 0, Math.PI / 2)!;
    expect(footprint.x).toBeCloseTo(4.1);
    expect(footprint.z).toBeCloseTo(4.7);
    expect(footprint.width).toBeCloseTo(1);
    expect(footprint.depth).toBeCloseTo(2);
    geometry.dispose();
    expect(spriteContactFootprint(1, 2, 0, 16, true, 0).width).toBeCloseTo(16 / 60);
    expect(spriteContactFootprint(1, 2, 0, 200, true, 0).width).toBe(0.35);
  });
  it('max-unions duplicate parts, caches unchanged frames, removes owners, rejects ground below a bridge and disposes', () => {
    const field = new AmbientContactField(4, 4),
      owner = {};
    field.set(owner, [pool]);
    field.update(() => 0);
    const first = Array.from(field.texture.image.data),
      version = field.texture.version;
    field.set(owner, [{ ...pool }]);
    field.update(() => 0);
    expect(field.texture.version).toBe(version);
    field.set(owner, [pool, pool]);
    field.update(() => 0);
    expect(Array.from(field.texture.image.data)).toEqual(first);
    field.set(owner, [{ ...pool, height: 2 }]);
    field.update(() => 0);
    expect(Math.max(...field.texture.image.data)).toBe(0);
    field.set(owner, [pool]);
    field.update(() => 0);
    field.invalidate();
    field.update(() => 2);
    expect(Math.max(...field.texture.image.data)).toBe(0);
    field.delete(owner);
    field.update(() => 0);
    expect(field.groups.size).toBe(0);
    const dispose = vi.fn();
    field.texture.addEventListener('dispose', dispose);
    field.dispose();
    expect(dispose).toHaveBeenCalledTimes(1);
  });
  it('retains contact decals constructed before field allocation and keeps AO out of the local-light graph', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      real = nodes.realLight!,
      owner = {};
    real.setContacts(owner, [pool]);
    real.initContacts(4, 4);
    real.contactField!.update(() => 0);
    expect(Math.max(...real.contactField!.texture.image.data)).toBe(255);
    const ao = ambientContactNode(real, positionWorld.xz);
    const material = new MeshBasicNodeMaterial();
    const local = localSurfaceLight(vec3(0), vec3(0.3), nodes, positionWorld.xz, positionWorld.y, vec3(0, 1, 0));
    const inputs: unknown[] = [];
    local.traverse((node) => {
      if ('value' in node) inputs.push(node.value);
    });
    expect(inputs).not.toContain(real.contactField!.texture);
    expect(inputs).toContain(real.ground.value);
    material.colorNode = vec3(0.2).mul(ao).add(local);
    expect(shaderBytes(new Mesh(new BoxGeometry(), material))).not.toMatch(/\b(?:NaN|undefined)\b/);
    real.deleteContacts(owner);
    real.contactField!.update(() => 0);
    expect(Math.max(...real.contactField!.texture.image.data)).toBe(0);
    expect(shader(vec3(ambientContactNode(undefined, positionWorld.xz)))).toContain('vec3<f32>( 1.0 )');
    material.dispose();
    nodes.dispose();
  });
});

describe('physical unit receivers and night graph construction', () => {
  it('uses engine-facing normals for infantry, rather than the flat camera card, with the same directional moments as walls', () => {
    const north = facingReceiverNormal(0),
      south = facingReceiverNormal(Math.PI),
      east = facingReceiverNormal(Math.PI / 2);
    expect(Math.hypot(...north)).toBeCloseTo(1);
    expect(east[0]).toBeGreaterThan(0.9);
    const front = reconstructLight([1, 0.5, 0.2], [0, 0, -1], north);
    const back = reconstructLight([1, 0.5, 0.2], [0, 0, -1], south);
    expect(lightLuminance(front)).toBeGreaterThan(lightLuminance(back) * 10);
    expect(back).toEqual([0, 0, 0]);
    expect(shader(facingReceiverNormalNode(uniform(1)) as ReturnType<typeof vec3>)).toContain('normalize');
    const pool = new SpritePool();
    pool.physicalFacing = Math.PI;
    const body = pool.take(new Vector3(), { rect: [0, 0, 4, 8], offset: [0, 0] }, 'unit', 0, false);
    expect(body.physicalFacing).toBe(Math.PI);
    expect(pool.take(new Vector3(), body.frame, 'unit', 0, true, true).physicalFacing).toBe(Math.PI);
    pool.begin();
    expect(pool.take(new Vector3(), body.frame, 'unit', 0, false).physicalFacing).toBeUndefined();
  });
  it('builds sprite, PBR, VPL and terrain graphs with the shared scene-only transfer and preserves sprite geometry/coverage', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      scene = receiverScene(nodes);
    for (const mesh of scene.meshes) expect(shaderBytes(mesh)).not.toMatch(/\b(?:NaN|undefined)\b/);
    const layer = new SpriteLayer(scene.source, [
      {
        anchor: new Vector3(),
        frame: { rect: [0, 0, 10, 20], offset: [-5, -20] },
        physicalFacing: Math.PI / 2,
        palette: '',
        push: 0,
        shadow: false,
      },
    ]);
    const tint = layer.mesh.geometry.getAttribute('spriteTint');
    expect(tint.itemSize).toBe(3);
    expect(tint.getZ(0)).toBeCloseTo(1 + Math.PI / 2);
    // This is the actual sprite receiver graph (without a normal atlas), so the facing reaches directional sampling.
    const generated = shaderBytes(layer.mesh);
    expect(generated).toContain('sin(');
    expect(generated).toContain('cos(');
    const position = Array.from(layer.mesh.geometry.getAttribute('position').array);
    nodes.realLight!.nightWeight.value = 1;
    nodes.realLight!.artWrap.value = 0.5;
    nodes.realLight!.enabled.value = 0;
    expect(shaderBytes(layer.mesh)).toBe(generated);
    expect(Array.from(layer.mesh.geometry.getAttribute('position').array)).toEqual(position);
    layer.mesh.dispose();
    layer.mesh.geometry.dispose();
    layer.mesh.material.dispose();
    scene.dispose();
  });
  it('registers actual contact decals before allocation and removes hidden/airborne voxel contact pools', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      scene = receiverScene(nodes),
      real = nodes.realLight!;
    const painter = Object.assign(painterObject(2, 0), { x: 1, z: 1, foundation: [2, 3] as const });
    const contact = new SpriteLayer(scene.source, [
      {
        anchor: new Vector3(1, 0, 1),
        frame: { rect: [0, 0, 200, 100], offset: [0, 0] },
        kind: 'contact',
        flat: true,
        palette: '',
        push: 0,
        shadow: false,
        painter,
      },
    ]);
    real.initContacts(8, 8);
    expect(real.contactField!.groups.get(contact)).toEqual([
      { x: 2, z: 2.5, width: 2, depth: 3, height: 0, band: true },
    ]);
    contact.mesh.dispose();
    expect(real.contactField!.groups.has(contact)).toBe(false);
    const parts = buildVoxelGeometry(syntheticModel([syntheticSection(2)]), new Uint8Array(768), nodes);
    const unit = instantiateVoxel(parts, createVoxelMaterial(nodes), nodes);
    const receiver = { hd: { nodes } } as unknown as VoxelUnits;
    VoxelUnits.prototype.setGroundContact.call(receiver, unit, true, undefined, true);
    expect(real.contactField!.groups.get(unit.root)).toHaveLength(1);
    VoxelUnits.prototype.setGroundContact.call(receiver, unit, false);
    expect(real.contactField!.groups.has(unit.root)).toBe(false);
    unit.root.visible = false;
    VoxelUnits.prototype.setGroundContact.call(receiver, unit, true);
    expect(real.contactField!.groups.has(unit.root)).toBe(false);
    for (const part of parts) part.geometry.dispose();
    contact.mesh.geometry.dispose();
    contact.mesh.material.dispose();
    scene.dispose();
  });
  it.each([false, true])(
    'builds projected redraw alpha-neighbour/wrap/LOD graphs without a GPU, WebGL=%s',
    (forceWebGL) => {
      const nodes = new AtmosphereNodes({ realLight: true });
      const tex = () => nodes.own(new DataTexture(new Uint8Array(320 * 32 * 4), 320, 32));
      const spec: VoxelProjectionManifest = {
        version: 1,
        type: 'synthetic',
        frames: 16,
        facingStep: 16,
        scale: 1,
        jacobian: [
          [180, 0, -180],
          [90, -220, 90],
        ],
        house: { first: 16, count: 16 },
        registration: [],
        parts: {
          body: {
            colour: 'body.png',
            paint: 'paint.png',
            size: [320, 32],
            frames: Array.from({ length: 16 }, (_, i) => ({
              k: i * 2,
              facing: i * 16,
              pivot: [i * 20 + 8, 24],
              rect: [i * 20 + 1, 1, 16, 16],
            })),
          },
        },
      };
      const part = createVoxelProjectionPart(spec, 'body', tex(), tex());
      const coverage = (night?: ReturnType<typeof float>) => {
        const inputs = new VoxelProjectionInputs();
        inputs.bind(part);
        const art = projectedVoxelColour(
          inputs,
          false,
          voxelProjectionFor(nodes),
          () => vec3(0.3),
          vec3(0, 1, 0),
          vec3(0, 1, 0),
          { sunGain: float(1), specularSize: float(1) },
          undefined,
          night,
        );
        return shader(vec3(art.coverage));
      };
      // Full generated coverage graph, byte for byte: changing night RGB mip bias never changes the silhouette.
      expect(coverage(float(0))).toBe(coverage());
      expect(coverage(float(1))).toBe(coverage());
      const material = voxelProjectedMaterialFor(new Uint8Array(768), part, nodes);
      const geometry = buildVoxelGeometry(syntheticModel([syntheticSection(2)]), null, nodes)[0]!.geometry;
      const mesh = new Mesh(geometry, material);
      const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement, forceWebGL });
      renderer.hasFeature = () => false;
      renderer.hasCompatibility = () => false;
      const builder = renderer.backend.createNodeBuilder(mesh, renderer);
      builder.scene = new Scene();
      builder.camera = new OrthographicCamera();
      builder.lightsNode = lights([new DirectionalLight(), new HemisphereLight()]);
      bindSlots(material);
      builder.build();
      expect(builder.fragmentShader).not.toMatch(/\b(?:NaN|undefined)\b/);
      expect(builder.fragmentShader).toContain('0.12');
      expect(builder.fragmentShader).toMatch(/texture(?:SampleBias|Lod)?\(/);
      geometry.dispose();
      nodes.dispose();
    },
  );
});
