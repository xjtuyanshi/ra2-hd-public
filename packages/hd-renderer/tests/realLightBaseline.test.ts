import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Mesh, MeshBasicNodeMaterial, PlaneGeometry, Vector3, type Scene, type WebGPURenderer } from 'three/webgpu';
import { float, positionWorld, vec3 } from 'three/tsl';
import { AtmosphereNodes, objectLightAt, localLightHeadroom, sampleLightField } from '../src/atmosphere';
import { LightField, SHAPE_POOL } from '../src/lightField';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';
import expected from './fixtures/realLightDefault.json';
const hash = (v: string) => createHash('sha256').update(v).digest('hex');
/** Only three's process-global BufferNode id is unstable; preserve every operation, binding and buffer reference. */
const shaderHash = (mesh: Mesh): string => {
  const names = new Map<string, string>();
  return hash(
    shaderBytes(mesh).replace(/\bNodeBuffer_\d+/g, (name) => {
      if (!names.has(name)) names.set(name, `NodeBuffer_${names.size}`);
      return names.get(name)!;
    }),
  );
};
describe('real light opt-out parent baseline', () => {
  it('retains the parent terrain/sprite/VPL/PBR and legacy field graphs, bindings and buffer bytes', () => {
    const scene = receiverScene();
    const nodes = new AtmosphereNodes();
    expect('realLight' in nodes && nodes.realLight !== undefined).toBe(false);
    const material = new MeshBasicNodeMaterial(),
      albedo = vec3(0.2, 0.4, 0.1);
    material.colorNode = localLightHeadroom(albedo, albedo.mul(sampleLightField(positionWorld.xz, nodes))).add(
      objectLightAt(positionWorld.xz, positionWorld.xz, float(1), nodes),
    );
    const receiver = new Mesh(new PlaneGeometry(), material);
    const field = new LightField(2, 2, { nodes });
    field.setBase(new Float32Array([1, 0.5, 0.1, 0.2, 0.1, 0, 0, 0, 0, 1, 1, 1]));
    field.setStatic([{ x: 0.5, z: 1.5, radius: 1, shape: SHAPE_POOL, colour: [1, 0.6, 0.3] }]);
    field.pulse(0, new Vector3(1, 0.4, 1), [1, 0.7, 0.45], 1.5, 2, 0.7, 4, 'explosion');
    field.keep('fire', 0, new Vector3(0.5, 0.4, 1), [1, 0.62, 0.36], 0.3, 1, 0.18, 0.6);
    field.source('reactor', 0, { x: 1, z: 1, radius: 1, colour: [0.6, 0.7, 1], head: 1.4, groundOnly: 1 });
    const renderer = {
      getRenderTarget: () => null,
      getClearAlpha: () => 1,
      setClearAlpha() {},
      setRenderTarget() {},
      render() {},
    } as unknown as WebGPURenderer;
    field.update(0.2, renderer, 0.6);
    const internals = field as unknown as {
      scene: Scene;
      baseTexture: { image: { data: Uint16Array } };
      statics: { attributes: { array: Float32Array }[] };
      dynamics: { attributes: { array: Float32Array }[] };
    };
    const meshes: Mesh[] = [receiver];
    internals.scene.traverse((o) => {
      if (o instanceof Mesh) meshes.push(o);
    });
    const actual = {
      receivers: scene.meshes.map(shaderHash),
      legacy: meshes.map(shaderHash),
      buffers: [
        internals.baseTexture.image.data,
        ...internals.statics.attributes.map((a) => a.array),
        ...internals.dynamics.attributes.map((a) => a.array),
      ].map((a) => hash(Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64'))),
    };
    expect(actual).toEqual(expected);
    scene.dispose();
    field.dispose();
    nodes.dispose();
    receiver.geometry.dispose();
    material.dispose();
  });
});
