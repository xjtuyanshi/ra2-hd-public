import { describe, expect, it, vi } from 'vitest';
import {
  DataTexture,
  HalfFloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  ClampToEdgeWrapping,
  Mesh,
  NodeMaterial,
  OrthographicCamera,
  QuadMesh,
  Scene,
  WebGPURenderer,
  type Texture,
} from 'three/webgpu';
import { float, vec2, vec3, vec4 } from 'three/tsl';
import { CloudToneMap } from '../src/cloudToneMap';
import { CloudToneUniforms, cloudToneGain, cloudToneRgb } from '../src/cloudTone';
import { lush2Defaults, lush2Mapping } from '../src/ra2/lush2Tone';
import { AtmosphereNodes } from '../src/atmosphere';
import { buildRa2Terrain } from '../src/ra2/terrain';
import { MapLightGrade } from '../src/ra2/mapLighting';
import { applyMaterialFallbacks } from '../src/materialFallback';
import type { Ra2Cache } from '../src/ra2/cache';

const params = () => lush2Mapping(lush2Defaults('temperate'), 'temperate').tone;
function shader(mesh: Mesh) {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const builder = renderer.backend.createNodeBuilder(mesh, renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  builder.build();
  return {
    code: builder.fragmentShader,
    textures: builder
      .getBindings()
      .flatMap((group) => group.bindings)
      .map((b) => (b as unknown as { texture?: Texture }).texture)
      .filter(Boolean),
  };
}
function fixture() {
  const uniforms = new CloudToneUniforms();
  uniforms.set(params());
  const source = new DataTexture(new Uint8Array(64 * 64 * 4), 64, 64),
    night = { value: 0 };
  source.minFilter = LinearMipmapLinearFilter;
  source.generateMipmaps = true;
  const map = new CloudToneMap({
    texture: source,
    width: 14400,
    height: 8610,
    blockWidth: 40,
    contextWidth: 192,
    uniforms,
    night,
    sourceVersion: () => source.version,
  });
  const previous = new Object(),
    mrt = new Object();
  const renderer = {
    getRenderTarget: () => previous,
    getActiveCubeFace: () => 2,
    getActiveMipmapLevel: () => 3,
    getMRT: () => mrt,
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
  };
  return { map, uniforms, source, night, renderer, previous, mrt };
}

describe('cached mean-only grading', () => {
  it('owns two block-sized filterable half-float targets and builds four explicit mip taps per endpoint', () => {
    const { map, source } = fixture();
    try {
      for (const target of [map.target]) {
        expect([target.width, target.height]).toEqual([720, 216]);
        expect(target.texture.type).toBe(HalfFloatType);
        expect(target.texture.minFilter).toBe(LinearFilter);
        expect(target.texture.magFilter).toBe(LinearFilter);
        expect(target.texture.wrapS).toBe(ClampToEdgeWrapping);
        expect(target.depthBuffer).toBe(false);
      }
      for (const material of [map.material]) {
        const result = shader(new QuadMesh(material));
        expect(result.code).toContain('@fragment');
        expect(result.code.match(/textureSampleLevel\(/g)).toHaveLength(4);
        expect(result.code).toContain('cos(');
        const main = result.code.slice(result.code.indexOf('@fragment'));
        for (const match of main.matchAll(/textureSampleLevel\(/g)) {
          const prefix = main.slice(0, match.index);
          expect((prefix.match(/\{/g)?.length ?? 0) - (prefix.match(/\}/g)?.length ?? 0)).toBe(1);
        }
        expect(result.code).not.toMatch(/\b(?:NaN|undefined)\b/);
      }
      const material = new NodeMaterial();
      material.fragmentNode = vec4(map.grade(vec3(0.3, 0.4, 0.1), vec2(0.5), float(0.5), float(0)), 1);
      const result = shader(new QuadMesh(material));
      expect(new Set(result.textures)).toEqual(new Set([map.target.texture]));
      expect(result.code.match(/textureSample\(/g)).toHaveLength(2);
      expect(result.code).not.toContain('cos(');
      expect(result.code).not.toContain('360.0');
      material.dispose();
    } finally {
      map.dispose();
      source.dispose();
    }
  });

  it('updates on changes, accumulating night deltas, restores renderer state and disposes exactly once', () => {
    const { map, uniforms, source, night, renderer, previous, mrt } = fixture();
    const gpu = renderer as unknown as WebGPURenderer;
    const dispose = vi.spyOn(map.target, 'dispose');
    try {
      expect(map.update(gpu)).toBe(true);
      expect(renderer.render).toHaveBeenCalledTimes(1);
      expect(renderer.setRenderTarget).toHaveBeenLastCalledWith(previous, 2, 3);
      expect(renderer.setMRT).toHaveBeenLastCalledWith(mrt);
      for (let i = 0; i < 100; i++) expect(map.update(gpu)).toBe(false);
      night.value = 0.004;
      expect(map.update(gpu)).toBe(false);
      night.value = 0.006;
      expect(map.update(gpu)).toBe(true);
      source.needsUpdate = true;
      expect(map.update(gpu)).toBe(true);
      uniforms.grassLight.value += 0.01;
      expect(map.update(gpu)).toBe(true);
      uniforms.moonTint.value.x += 0.01;
      expect(map.update(gpu)).toBe(true);
      map.invalidate();
      expect(map.update(gpu)).toBe(true);
      uniforms.set(null);
      expect(map.update(gpu)).toBe(false);
      uniforms.set(params());
      expect(map.update(gpu)).toBe(true);
      map.invalidate();
      renderer.render.mockImplementationOnce(() => {
        throw new Error('draw failed');
      });
      expect(() => map.update(gpu)).toThrow('draw failed');
      expect(renderer.setRenderTarget).toHaveBeenLastCalledWith(previous, 2, 3);
      expect(renderer.setMRT).toHaveBeenLastCalledWith(mrt);
      expect(map.update(gpu)).toBe(true);
      uniforms.snow.value = 1;
      expect(map.update(gpu)).toBe(true);
      expect(map.update(gpu)).toBe(false);
    } finally {
      map.dispose();
      map.dispose();
      source.dispose();
    }
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(map.update(gpu)).toBe(false);
  });

  it('reconstructs the CPU twin from channel gains and interpolates the same endpoint shade recipes', () => {
    const mean = [0.3, 0.4, 0.08] as const,
      context = [0.28, 0.35, 0.1] as const;
    const p = { ...params(), gold: 0, detail: 1 };
    for (const night of [0, 0.4, 1])
      for (const shade of [0, 0.25, 0.5, 1]) {
        const gain = cloudToneGain(mean, context, shade, p, night);
        const lit = cloudToneGain(mean, context, 0, p, night),
          shadow = cloudToneGain(mean, context, 1, p, night);
        gain.forEach((v, i) => expect(v).toBeCloseTo(lit[i]! + (shadow[i]! - lit[i]!) * shade, 12));
        for (const rgb of [
          [0.2, 0.3, 0.05],
          [0.4, 0.5, 0.1],
        ] as const) {
          const amount = p.strength * (1 + (p.nightKeep - 1) * night);
          const out = cloudToneRgb(rgb, shade, p, night, mean, false, 0, 0, context);
          out.forEach((v, i) => expect(v).toBeCloseTo(rgb[i]! * (1 + (gain[i]! - 1) * amount), 12));
        }
      }
  });
});

it.each(['temperate', 'snow'])(
  'ground, walls and both fallback materials share the grade atlas without pixel classification (%s)',
  (theater) => {
    const nodes = new AtmosphereNodes();
    nodes.cloudTone.set(lush2Mapping(lush2Defaults(theater), theater).tone);
    nodes.cloudTone.greenHue.setName('testMeanClassificationHue');
    nodes.cloudTone.contextHue.setName('testContextClassificationHue');
    const image = () => new DataTexture(new Uint8Array(64 * 64 * 4).fill(255), 64, 64);
    const source = image(),
      mask = image(),
      shore = image();
    const cache = {
      nodes,
      terrainAtlas: source,
      waterMask: mask,
      manifest: {
        grid: { width: 2, height: 2 },
        theater,
        terrainAtlas: { width: 64, height: 64 },
        tiles: [{ rect: [0, 0, 60, 30], anchor: [30, 0], x: 0, y: 0, rampType: 0, terrainType: 0, file: 'synthetic' }],
      },
      exists: new Uint8Array(4).fill(1),
      height: new Uint8Array([0, 1, 0, 1]),
      tile: new Uint16Array(4),
      terrainMap: {
        texture: source,
        waterMask: mask,
        shore,
        waterBed: true,
        ref: { width: 64, height: 64, originX: 0, originY: 0, scale: 1 },
      },
      mapLight: new MapLightGrade(
        { width: 2, height: 2, z: new Uint8Array(4), exists: new Uint8Array(4).fill(1) },
        { ambient: 1, level: 0, ground: 0, red: 1, green: 1, blue: 1 },
      ),
    } as unknown as Ra2Cache;
    const grade = vi.spyOn(CloudToneMap.prototype, 'grade');
    const terrain = buildRa2Terrain(cache);
    const assertMapped = (material: NodeMaterial) => {
      const result = shader(new Mesh(terrain.mesh.geometry, material));
      const map = grade.mock.contexts.at(-1)!;
      expect(result.textures).toContain(map.target.texture);
      expect(result.textures).toContain(map.target.texture);
      expect(result.code).not.toContain('360.0');
      expect(result.code).not.toContain('testMeanClassificationHue');
      expect(result.code).not.toContain('testContextClassificationHue');
      return map;
    };
    const materials = terrain.mesh.material as NodeMaterial[];
    let fallback: NodeMaterial | undefined;
    try {
      const map = assertMapped(materials[0]!);
      expect(assertMapped(materials[1]!)).toBe(map);
      if (theater === 'snow') {
        const code = shader(new QuadMesh(map.material));
        expect(code.textures).toContain(cache.mapLight!.texture);
        expect(code.code).toContain('sRGBTransferEOTF');
        expect(code.textures.length).toBeGreaterThan(1);
      }
      const original = materials[0]!;
      applyMaterialFallbacks(terrain.mesh, new Set([(original as unknown as { id: number }).id]));
      fallback = (terrain.mesh.material as NodeMaterial[])[0]!;
      expect(fallback).not.toBe(original);
      expect(assertMapped(fallback)).toBe(map);
      applyMaterialFallbacks(terrain.mesh, new Set([(fallback as unknown as { id: number }).id]));
      fallback = (terrain.mesh.material as NodeMaterial[])[0]!;
      expect(assertMapped(fallback)).toBe(map);
      const dispose = vi.spyOn(map, 'dispose');
      terrain.disposeWaterLooks?.();
      expect(dispose).toHaveBeenCalledTimes(1);
    } finally {
      grade.mockRestore();
      fallback?.dispose();
      terrain.mesh.geometry.dispose();
      terrain.heightTexture.dispose();
      nodes.dispose();
      source.dispose();
      mask.dispose();
      shore.dispose();
    }
  },
);
