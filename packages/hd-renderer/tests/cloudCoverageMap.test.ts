import { expect, it, vi } from 'vitest';
import {
  DataUtils,
  HalfFloatType,
  LinearFilter,
  RedFormat,
  NodeMaterial,
  QuadMesh,
  OrthographicCamera,
  Scene,
  WebGPURenderer,
} from 'three/webgpu';
import { vec3, vec4 } from 'three/tsl';
import { CloudCoverageMap } from '../src/cloudCoverageMap';
import { AtmosphereNodes } from '../src/atmosphere';
import { cloudCoverage } from '../src/livingWorld';

function shader(material: NodeMaterial, webgl = false) {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement, forceWebGL: webgl });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const builder = renderer.backend.createNodeBuilder(new QuadMesh(material), renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  builder.build();
  return builder.fragmentShader;
}

it('owns a 4-texel/cell R16F cache and generates analytic pass / single-fetch receivers on both backends', () => {
  const nodes = new AtmosphereNodes();
  const map = new CloudCoverageMap(120, 143, nodes.alive);
  const receiver = new NodeMaterial();
  receiver.fragmentNode = vec4(vec3(nodes.alive.coverage), 1);
  try {
    expect([map.target.width, map.target.height]).toEqual([480, 572]);
    expect(map.target.texture.type).toBe(HalfFloatType);
    expect(map.target.texture.format).toBe(RedFormat);
    expect(map.target.texture.internalFormat).toBe('R16F');
    expect(map.target.texture.minFilter).toBe(LinearFilter);
    expect(map.target.texture.magFilter).toBe(LinearFilter);
    expect(map.target.depthBuffer).toBe(false);
    expect(nodes.alive.coverageBounds.value.toArray()).toEqual([0, 0, 120, 143]);
    expect(nodes.alive.coverageTexture.value).toBe(map.target.texture);
    for (const webgl of [false, true]) {
      const pass = shader(map.material, webgl),
        main = shader(receiver, webgl);
      expect(pass).toContain('aliveCloudShadow');
      expect(pass).not.toMatch(/texture(?:Sample|Lod)?\(/);
      expect(main).not.toContain('aliveCloudShadow');
      expect(main.match(/texture(?:Sample|Lod)?\(/g)).toHaveLength(1);
      expect(main).toContain('aliveCachedCoverage');
      expect(main).not.toMatch(/\b(?:NaN|undefined)\b/);
    }
    // The half-float storage is the only additional error at cache centres (away from centres filtering approximates).
    for (let z = 0; z < 572; z += 11)
      for (let x = 0; x < 480; x += 13) {
        const v = cloudCoverage((x + 0.5) / 4, (z + 0.5) / 4, 17, 0.6);
        expect(Math.abs(DataUtils.fromHalfFloat(DataUtils.toHalfFloat(v)) - v)).toBeLessThan(0.00049);
      }
  } finally {
    receiver.dispose();
    map.dispose();
    nodes.dispose();
  }
});

it('skips disabled frames, shares frozen frames, refreshes drift/shape, restores state after failure and disposes once', () => {
  const nodes = new AtmosphereNodes(),
    map = new CloudCoverageMap(3, 4, nodes.alive);
  const previous = {},
    mrt = {};
  const renderer = {
    getRenderTarget: () => previous,
    getActiveCubeFace: () => 2,
    getActiveMipmapLevel: () => 3,
    getMRT: () => mrt,
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
  };
  const gpu = renderer as unknown as WebGPURenderer;
  const targetDispose = vi.spyOn(map.target, 'dispose'),
    materialDispose = vi.spyOn(map.material, 'dispose');
  try {
    expect(map.update(gpu)).toBe(false);
    expect(renderer.render).not.toHaveBeenCalled();
    nodes.alive.set({ clouds: true });
    nodes.alive.update(7, 0.6, 0, 0, false);
    expect(map.update(gpu)).toBe(true);
    expect(renderer.render).toHaveBeenCalledTimes(1);
    expect(map.update(gpu)).toBe(false);
    nodes.alive.update(8, 0.6, 0, 0, false);
    expect(map.update(gpu)).toBe(true);
    nodes.alive.set({ preset: 'deep' });
    nodes.alive.update(8, 0.6, 0, 0, false);
    expect(map.update(gpu)).toBe(true);
    nodes.alive.set(false);
    nodes.alive.update(9, 0.6, 0, 0, false);
    expect(map.update(gpu)).toBe(false);
    nodes.alive.set({ clouds: true });
    nodes.alive.update(10, 0.6, 0, 0, false);
    renderer.render.mockImplementationOnce(() => {
      throw new Error('coverage draw failed');
    });
    expect(() => map.update(gpu)).toThrow('coverage draw failed');
    expect(renderer.setRenderTarget).toHaveBeenLastCalledWith(previous, 2, 3);
    expect(renderer.setMRT).toHaveBeenLastCalledWith(mrt);
    expect(map.update(gpu)).toBe(true);
  } finally {
    map.dispose();
    map.dispose();
    nodes.dispose();
  }
  expect(targetDispose).toHaveBeenCalledTimes(1);
  expect(materialDispose).toHaveBeenCalledTimes(1);
  expect(nodes.alive.coverageCached.value).toBe(0);
  expect(map.update(gpu)).toBe(false);
});
