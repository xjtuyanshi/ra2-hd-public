import { needsSurfaceClasses } from '../src/ra2/lookFlags';
import { moistTone } from '../src/ra2/moistTone';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DataTexture, MeshBasicNodeMaterial, OrthographicCamera, QuadMesh, Scene, WebGPURenderer } from 'three/webgpu';
import { texture, uniform, vec4 } from 'three/tsl';
import {
  objectOutlineFeather,
  objectPostCPU,
  objectPostMask,
  objectSurfaceFilter,
  ObjectMaskedSharpen,
} from '../src/objectPost';
import { SpriteIntegration } from '../src/spriteIntegration';
import { RECOMMENDED, configureSpriteIntegration } from '../src/ra2/spriteIntegrationTuning';
import { HdRenderer } from '../src/HdRenderer';
import { createScenario } from '../sandbox/scenario';

const defaults = { on: 1, objSoft: 1, objContrast: 0.3, outlineFeather: 1.5 };
const frame = (width = 25) => ({
  width,
  pixels: Array.from({ length: width * width }, () => [0.13, 0.37, 0.71] as [number, number, number]),
  codes: Array<number>(width * width).fill(0),
});
afterEach(() => vi.unstubAllGlobals());

describe('object-only post arithmetic', () => {
  it('copies distant ground bit for bit and feathers only the one-pixel mask band', () => {
    const f = frame();
    f.codes[12 * f.width + 12] = 1;
    const run = (x: number, y: number) => objectPostCPU(f.pixels, f.codes, f.width, x, y, defaults);
    const far = run(3, 3);
    expect(far).toEqual({ mask: 0, surface: f.pixels[3 * f.width + 3], feather: f.pixels[3 * f.width + 3] });
    expect(far.surface).toBe(f.pixels[3 * f.width + 3]);
    expect(run(12, 12).mask).toBe(1);
    expect(run(11, 12).mask).toBe(0.5);
    expect(run(11, 11).mask).toBe(0.25);
    expect(run(10, 12).mask).toBe(0);
  });
  it('blurs object microcontrast, never gathering another object or ground into the interior', () => {
    const f = frame();
    f.codes.fill(1);
    const k = 12 * f.width + 12;
    f.pixels[k] = [1, 0.5, 0.25];
    const a = objectPostCPU(f.pixels, f.codes, f.width, 12, 12, defaults);
    expect(a.surface[0]).toBeLessThan(0.8);
    expect(a.surface[0]).toBeGreaterThan(0.13);
    f.codes[k - 1] = 0;
    f.codes[k + 1] = 2;
    f.pixels[k - 1] = [0, 100, 0];
    f.pixels[k + 1] = [0, 0, 100];
    const b = objectPostCPU(f.pixels, f.codes, f.width, 12, 12, defaults);
    f.pixels[k - 1] = [100, 0, 0];
    f.pixels[k + 1] = [100, 0, 0];
    expect(objectPostCPU(f.pixels, f.codes, f.width, 12, 12, defaults).surface).toEqual(b.surface);
  });
  it('reduces local luminance contrast without changing channel ratios', () => {
    const f = frame();
    f.codes.fill(1);
    f.pixels.fill([0.1, 0.05, 0.025]);
    f.pixels[312] = [0.8, 0.4, 0.2];
    const { surface } = objectPostCPU(f.pixels, f.codes, f.width, 12, 12, { ...defaults, objSoft: 0, objContrast: 1 });
    expect(surface[0]).toBeLessThan(0.8);
    expect(surface[0]! / surface[1]!).toBeCloseTo(2, 14);
    expect(surface[1]! / surface[2]!).toBeCloseTo(2, 14);
  });
  it.each([0.5, 1, 1.5, 3])(
    'symmetrically mixes both sides of object/ground and object/object edges, radius %s',
    (radius) => {
      for (const leftCode of [0, 2]) {
        const f = frame();
        for (let y = 0; y < f.width; y++)
          for (let x = 0; x < f.width; x++) {
            const k = y * f.width + x;
            f.codes[k] = x < 12 ? leftCode : 1;
            f.pixels[k] = x < 12 ? [0, 0, 0] : [1, 1, 1];
          }
        const run = (x: number) =>
          objectPostCPU(f.pixels, f.codes, f.width, x, 12, { ...defaults, outlineFeather: radius }).feather[0]!;
        expect(run(11)).toBeGreaterThan(0);
        expect(run(12)).toBeLessThan(1);
        expect(run(11) + run(12)).toBeCloseTo(1, 14);
      }
    },
  );
  it('A/B off is exact even on the outline and when every strength is maximal', () => {
    const f = frame();
    f.codes[312] = 1;
    f.pixels[312] = [1, 0.5, 0.25];
    for (const x of [11, 12, 13]) {
      const result = objectPostCPU(f.pixels, f.codes, f.width, x, 12, {
        on: 0,
        objSoft: 1,
        objContrast: 1,
        outlineFeather: 3,
      });
      expect(result.surface).toBe(f.pixels[12 * f.width + x]);
      expect(result.feather).toBe(result.surface);
    }
  });
});

function build(fragment: unknown, forceWebGL: boolean) {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement, forceWebGL });
  renderer.hasFeature = () => false;
  const material = new MeshBasicNodeMaterial();
  material.fragmentNode = fragment as typeof material.fragmentNode;
  const mesh = new QuadMesh(material);
  const builder = renderer.backend.createNodeBuilder(mesh, renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  builder.build();
  return { builder, shader: builder.fragmentShader as string };
}

it.each([false, true])('generates object filters and masked RCAS with stable uniform A/B (WebGL=%s)', (webgl) => {
  const picture = texture(new DataTexture(new Uint8Array(16 * 16 * 4), 16, 16));
  const codes = texture(new DataTexture(new Uint8Array(16 * 16 * 4), 16, 16));
  const group = new SpriteIntegration();
  group.compiled = true;
  group.set(RECOMMENDED);
  const filters = [
    objectPostMask(codes),
    objectSurfaceFilter(picture, codes, group, uniform(2)),
    objectOutlineFeather(picture, codes, group),
  ];
  for (const filter of filters) {
    const first = build(vec4(filter), webgl).shader;
    expect(first).not.toMatch(/\b(?:NaN|undefined)\b/);
    group.ab(false);
    group.set({ objSoft: 0.1, outlineFeather: 3 });
    expect(build(vec4(filter), webgl).shader).toBe(first);
    group.ab(true);
  }
  const sharpen = new ObjectMaskedSharpen(picture, 1, codes, group);
  build(sharpen, webgl); // Builds the pass dependency; compile its actual fragment material as well.
  const material = (sharpen as unknown as { _material: MeshBasicNodeMaterial })._material;
  const shader = build(material.fragmentNode, webgl).shader;
  expect(shader).toContain('objectPostMask');
  expect(shader).not.toMatch(/\b(?:NaN|undefined)\b/);
  sharpen.dispose();
});

it('gates the post graph on compiled + painter depth, and keeps the key/graph through uniform edits', () => {
  vi.stubGlobal('window', { devicePixelRatio: 2 });
  const hd = new HdRenderer(
    { width: 1, height: 1, style: {}, addEventListener() {}, removeEventListener() {} } as unknown as HTMLCanvasElement,
    createScenario(2, 2),
  );
  const internal = hd as unknown as {
    settingsPipelineKey(): string;
    buildPipeline(): void;
    painterDepthEnabled: boolean;
    pipeline: { outputNode: unknown; dispose(): void };
    pipelineNodes: { dispose(): void }[];
  };
  const original = internal.settingsPipelineKey();
  configureSpriteIntegration(hd.nodes, true);
  expect(internal.settingsPipelineKey()).toBe(original);
  internal.painterDepthEnabled = true;
  expect(internal.settingsPipelineKey()).toContain('object-post');
  internal.buildPipeline();
  for (const webgl of [false, true]) {
    const generated = build(internal.pipeline.outputNode, webgl).shader;
    expect(generated).not.toMatch(/\b(?:NaN|undefined)\b/);
  }
  const key = internal.settingsPipelineKey(),
    output = internal.pipeline.outputNode;
  hd.spriteIntegration.ab(false);
  hd.spriteIntegration.set({ objSoft: 1, vCut: 0.2 });
  expect(internal.settingsPipelineKey()).toBe(key);
  expect(internal.pipeline.outputNode).toBe(output);
  internal.pipeline.dispose();
  internal.pipelineNodes.forEach((n) => n.dispose());
  hd.nodes.dispose();
});

it.each([0.5, 1, 2.5, 4])('keeps protected resource interiors and edges exact at view scale %s', (scale) => {
  const f = frame();
  f.codes.fill(1);
  for (let y = 10; y <= 14; y++)
    for (let x = 10; x <= 14; x++) {
      const k = y * f.width + x;
      f.codes[k] = 5;
      f.pixels[k] = [x % 2, y % 2, 0.17];
    }
  for (let y = 10; y <= 14; y++)
    for (let x = 10; x <= 14; x++) {
      const actual = objectPostCPU(f.pixels, f.codes, f.width, x, y, defaults, scale, 5);
      expect(actual).toEqual({ mask: 0, surface: f.pixels[y * f.width + x], feather: f.pixels[y * f.width + x] });
    }
  expect(objectPostCPU(f.pixels, f.codes, f.width, 9, 12, defaults, scale, 5).mask).toBe(1);
});

it.each([false, true])('compiles protected-category filtering in both GPU languages (WebGL=%s)', (webgl) => {
  const picture = texture(new DataTexture(new Uint8Array(16 * 16 * 4), 16, 16));
  const group = new SpriteIntegration();
  group.protectedCategory = 5;
  for (const node of [
    objectPostMask(picture, 5),
    objectSurfaceFilter(picture, picture, group, uniform(2.5)),
    objectOutlineFeather(picture, picture, group),
  ]) {
    const shader = build(vec4(node), webgl).shader;
    expect(shader).toMatch(/(?:%|mod(?:_float)?\s*\()/);
    expect(shader).toContain('5');
    expect(shader).not.toMatch(/\b(?:NaN|undefined)\b/);
  }
});

it.each([false, true])('compiles the reference filter resolve for protected final-tone pixels (WebGL=%s)', (webgl) => {
  vi.stubGlobal('window', { devicePixelRatio: 2 });
  const hd = new HdRenderer(
    {
      width: 16,
      height: 16,
      style: {},
      addEventListener() {},
      removeEventListener() {},
    } as unknown as HTMLCanvasElement,
    createScenario(2, 2),
  );
  hd.nodes.surfaceClasses = true;
  hd.spriteIntegration.protectedCategory = 5;
  hd.settings.tone = moistTone('temperate', 'moist3', hd.settings);
  hd.settings.clarity = 0.4;
  const internal = hd as unknown as {
    painterDepthEnabled: boolean;
    buildPipeline(): void;
    pipeline: { outputNode: unknown; dispose(): void };
    pipelineNodes: { dispose(): void }[];
  };
  internal.painterDepthEnabled = true;
  internal.buildPipeline();
  try {
    expect(build(internal.pipeline.outputNode, webgl).shader).not.toMatch(/\b(?:NaN|undefined)\b/);
  } finally {
    internal.pipeline.dispose();
    internal.pipelineNodes.forEach((n) => n.dispose());
    hd.nodes.dispose();
  }
});

// Exercise the entry policy and resource boundary for every independent and combined look.
it.each([
  'hd-sprite=1',
  'hd-tone=lush2',
  'hd-veh=1',
  'hd-blend=1',
  'hd-bld=1',
  'hd-bld-polish=1',
  'hd-sprite=1&hd-tone=lush2&hd-veh=1&hd-blend=1&hd-bld=1',
])('protects ore and gems with %s', (query) => {
  expect(needsSurfaceClasses(new URLSearchParams(query))).toBe(true);
  const f = frame();
  f.codes.fill(1);
  f.codes[12 * f.width + 12] = 5;
  for (const scale of [0.5, 1, 2.5, 4]) {
    const actual = objectPostCPU(f.pixels, f.codes, f.width, 12, 12, defaults, scale, 5);
    expect(actual.mask).toBe(0);
    expect(actual.surface).toBe(f.pixels[12 * f.width + 12]);
    expect(actual.feather).toBe(f.pixels[12 * f.width + 12]);
  }
});
it('keeps default and explicitly disabled looks on the original metadata path', () => {
  for (const query of ['', 'hd-tone=off', 'hd-bld=0&hd-blend=0&hd-veh=0&hd-sprite=0'])
    expect(needsSurfaceClasses(new URLSearchParams(query))).toBe(false);
});
