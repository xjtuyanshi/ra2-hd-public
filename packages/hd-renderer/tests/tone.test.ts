import { afterEach, describe, expect, it, vi } from 'vitest';
import { float, texture, vec2, vec3, vec4 } from 'three/tsl';
import {
  DataTexture,
  HalfFloatType,
  LinearFilter,
  NodeMaterial,
  OrthographicCamera,
  QuadMesh,
  RGBAFormat,
  Scene,
  WebGPURenderer,
  type RenderPipeline,
} from 'three/webgpu';
import { HdRenderer, applyLookPreset, LOOK_PRESETS } from '../src/HdRenderer';
import { FinalTone, finalToneRgb, type FinalToneParams, type Rgb } from '../src/tone';
import { cinematicGradeRgb } from '../src/post';
import { createScenario } from '../sandbox/scenario';
import { moistTone } from '../src/ra2/moistTone';
import type { ShaderNode } from '../src/tsl';

const params = (): FinalToneParams => ({
  strength: 1,
  grade: null,
  density: 0,
  contrast: 0,
  pivot: 0.5,
  chroma: [1, 1, 1],
  hueShifts: [],
  protect: null,
  objectShare: 0.5,
  objectDensity: 1,
  foliage: null,
  grain: 0.5,
  night: { colour: 0.35, grain: 1 },
});
const opts = { object: false, gradeShare: 0, night: 0 };
const encoded = (rgb: Rgb) => rgb.map((x) => x ** (1 / 2.2)) as [number, number, number];
const chroma = (rgb: Rgb) => {
  const [r, g, b] = encoded(rgb);
  return Math.hypot(r - g, (r + g) * 0.5 - b);
};
const hue = (rgb: Rgb) => {
  const [r, g, b] = encoded(rgb);
  return ((Math.atan2((r + g) * 0.5 - b, r - g) * 180) / Math.PI + 360) % 360;
};
const colour = (h: number, c = 0.12): [number, number, number] => {
  const a = c * Math.cos((h * Math.PI) / 180),
    b = c * Math.sin((h * Math.PI) / 180);
  const g = 0.5 - (0.2126 + 0.0722 * 0.5) * a + 0.0722 * b;
  return [g + a, g, g + a * 0.5 - b].map((x) => x ** 2.2) as [number, number, number];
};

describe('final tone colour', () => {
  it('returns the exact look at zero strength, including when a replacement grade is configured', () => {
    const p = { ...params(), strength: 0, density: 0.2, grade: LOOK_PRESETS.cinematic.settings };
    const look: Rgb = [0.17, 0.31, 0.73];
    expect(finalToneRgb(look, [0.1, 0.2, 0.3], p, opts)).toEqual(look);
  });
  it('fixes black/white and darkens middle grey with density alone', () => {
    const p = { ...params(), density: 0.16 };
    for (const n of [0, 1]) expect(finalToneRgb([n, n, n], [n, n, n], p, opts)).toEqual([n, n, n]);
    const grey: Rgb = [0.2, 0.2, 0.2];
    const y = 0.2 ** (1 / 2.2);
    const expected = (0.84 * y + 0.16 * y ** 6) ** 2.2;
    expect(expected).toBeLessThan(0.2);
    for (const x of finalToneRgb(grey, grey, p, opts)) expect(x).toBeCloseTo(expected, 12);
  });
  it('deepens like an exposure: linear-light ratios between mid-tones are kept, white stays', () => {
    const p = { ...params(), density: 0.2 };
    const lum = (rgb: Rgb) => rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
    const [dark, light] = ([0.04, 0.12] as const).map((v) => {
      const grey: Rgb = [v, v, v];
      return lum(finalToneRgb(grey, grey, p, opts));
    });
    expect(light! / dark!).toBeCloseTo(3, 1);
    expect(light!).toBeLessThan(0.12 * 0.75);
    expect(finalToneRgb([1, 1, 1], [1, 1, 1], p, opts)).toEqual([1, 1, 1]);
  });
  it('raises middle-band opponent chroma without rotating it', () => {
    const p = { ...params(), chroma: [1, 1.2, 1] as [number, number, number] };
    const c = colour(112);
    const out = finalToneRgb(c, c, p, opts);
    expect(chroma(out) / chroma(c)).toBeCloseTo(1.2, 12);
    expect(hue(out)).toBeCloseTo(hue(c), 12);
  });
  it('rotates green counter-clockwise, leaving sand, red and low-chroma grey untouched', () => {
    const p = { ...params(), hueShifts: [{ centre: 112, width: 28, minChroma: 0.04, shift: 3 }] };
    for (const h of [112, 72, 0, 359]) {
      const c = colour(h);
      expect(hue(finalToneRgb(c, c, p, opts))).toBeCloseTo(h === 112 ? 115 : h, 10);
    }
    const grey: Rgb = [0.2, 0.2, 0.2];
    finalToneRgb(grey, grey, p, opts).forEach((v, i) => expect(v).toBeCloseTo(grey[i]!, 14));
  });
  it('protects the look water exactly, even when the alternate grade would recolour it', () => {
    const p = moistTone('temperate', 'moist2', LOOK_PRESETS.cinematic.settings)!;
    const water = colour(215, 0.2);
    expect(finalToneRgb(water, [0.6, 0.2, 0.1], p, { ...opts, gradeShare: 1 })).toEqual(water);
  });
  it('uses half colour share on objects, full share on foliage, and fades colour at night', () => {
    const p = { ...params(), density: 0.16, foliage: { centre: 112, width: 30, minChroma: 0.05 } };
    for (const h of [0, 112]) {
      const c = colour(h);
      const full = finalToneRgb(c, c, p, opts);
      const object = finalToneRgb(c, c, p, { ...opts, object: true });
      const night = finalToneRgb(c, c, p, { ...opts, night: 1 });
      for (let i = 0; i < 3; i++) {
        expect(object[i]! - c[i]!).toBeCloseTo((full[i]! - c[i]!) * (h === 112 ? 1 : 0.5), 12);
        expect(night[i]! - c[i]!).toBeCloseTo((full[i]! - c[i]!) * 0.35, 12);
      }
    }
  });
});

/** Extend the constant/uniform TSL evaluator pattern from vehicleLook.test.ts; unknown node kinds fail. */
function evaluate(root: ShaderNode): number[] {
  const cache = new Map<ShaderNode, number[]>();
  const visit = (n: ShaderNode): number[] => {
    const prior = cache.get(n);
    if (prior) return prior;
    const value = compute(n);
    cache.set(n, value);
    return value;
  };
  function compute(n: ShaderNode): number[] {
    if (n.isConstNode || n.isUniformNode) return typeof n.value === 'number' ? [n.value] : n.value.toArray();
    if (n.isSplitNode)
      return [...n.components].map(
        (c: string) => visit(n.node)['xyzw'.includes(c) ? 'xyzw'.indexOf(c) : 'rgba'.indexOf(c)]!,
      );
    if (n.isConvertNode || n.type === 'ConvertNode') {
      const v = visit(n.node),
        size = ({ float: 1, vec2: 2, vec3: 3, vec4: 4 } as Record<string, number>)[n.convertTo] ?? v.length;
      return v.length === 1 ? Array<number>(size).fill(v[0]!) : v.slice(0, size);
    }
    if (n.isJoinNode || n.type === 'JoinNode') return n.nodes.flatMap(visit);
    if (n.isVarNode) return visit(n.node);
    if (n.condNode) return visit(visit(n.condNode)[0] ? n.ifNode : n.elseNode);
    const each = (args: number[][], fn: (...v: number[]) => number) =>
      Array.from({ length: Math.max(...args.map((v) => v.length)) }, (_, i) =>
        fn(...args.map((v) => (v.length === 1 ? v[0]! : v[i]!))),
      );
    if (n.isOperatorNode) {
      const f = (
        {
          '+': (x: number, y: number) => x + y,
          '-': (x: number, y: number) => x - y,
          '*': (x: number, y: number) => x * y,
          '/': (x: number, y: number) => x / y,
          '%': (x: number, y: number) => x - y * Math.floor(x / y),
          '>': (x: number, y: number) => Number(x > y),
        } as Record<string, (x: number, y: number) => number>
      )[n.op];
      if (!f) throw new Error(`operator ${String(n.op)}`);
      return each([visit(n.aNode), visit(n.bNode)], f);
    }
    if (n.isMathNode) {
      const a = visit(n.aNode),
        b = n.bNode ? visit(n.bNode) : [],
        c = n.cNode ? visit(n.cNode) : [];
      switch (n.method) {
        case 'mix':
          return each([a, b, c], (x, y, t) => x * (1 - t) + y * t);
        case 'smoothstep':
          return each([a, b, c], (lo, hi, x) => {
            const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
            return t * t * (3 - 2 * t);
          });
        case 'clamp':
          return each([a, b, c], (x, lo, hi) => Math.min(hi, Math.max(lo, x)));
        case 'max':
          return each([a, b], Math.max);
        case 'min':
          return each([a, b], Math.min);
        case 'pow':
          return each([a, b], Math.pow);
        case 'mod':
          return each([a, b], (x, y) => x - y * Math.floor(x / y));
        case 'atan':
          return each([a, b], Math.atan2);
        case 'sqrt':
          return a.map(Math.sqrt);
        case 'abs':
          return a.map(Math.abs);
        case 'cos':
          return a.map(Math.cos);
        case 'sin':
          return a.map(Math.sin);
        case 'dot':
          return [a.reduce((sum, x, i) => sum + x * b[i]!, 0)];
      }
    }
    throw new Error(`cannot evaluate ${String(n.type)}`);
  }
  return visit(root);
}

describe('TSL / JS agreement', () => {
  it('matches across grade, protection, two hue shifts, night and object uniforms without rebuilding', () => {
    const tone = new FinalTone();
    const p = moistTone('temperate', 'moist2', LOOK_PRESETS.cinematic.settings)!;
    p.hueShifts.push({ centre: 359, width: 25, minChroma: 0.04, shift: -8 });
    for (const grade of [p.grade, null])
      for (const night of [0, 0.4, 1])
        for (const object of [false, true]) {
          const configured = { ...p, grade };
          tone.set(configured);
          for (const lin of [
            [0, 0, 0],
            [1, 1, 1],
            colour(112),
            colour(215, 0.2),
            colour(359),
            colour(72),
            [0.8, 0.02, 0.3],
          ] as Rgb[]) {
            const look = cinematicGradeRgb(lin, LOOK_PRESETS.cinematic.settings);
            const actual = evaluate(
              tone.node(vec3(...lin), vec3(...look), float(0.7), float(night), object ? vec2(0.1, 0.2) : null),
            );
            const expected = finalToneRgb(look, lin, configured, { object, gradeShare: 0.7, night });
            actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 11));
          }
        }
    tone.set({ ...params(), foliage: { centre: 112, width: 30, minChroma: 0.05 } });
    expect(evaluate(tone.grainMask(vec3(...colour(112)), vec2(0.1, 0.2)))[0]).toBe(1);
    expect(evaluate(tone.grainMask(vec3(...colour(0)), vec2(0.1, 0.2)))[0]).toBe(0);
    expect(evaluate(tone.grainMask(vec3(...colour(0)), vec2(0, 0)))[0]).toBe(1);
  });
});

afterEach(() => vi.unstubAllGlobals());
describe('tone pipeline lifecycle', () => {
  it('keeps the legacy null key, preserves tone across looks, and only keys structural changes', () => {
    vi.stubGlobal('window', { devicePixelRatio: 1 });
    const canvas = {
      width: 1,
      height: 1,
      style: {},
      addEventListener() {},
      removeEventListener() {},
    } as unknown as HTMLCanvasElement;
    const hd = new HdRenderer(canvas, createScenario(2, 2));
    const state = hd as unknown as {
      pipelineKey: string;
      pipeline: RenderPipeline;
      pipelineNodes: { dispose(): void }[];
      finalTone: FinalTone;
      updateGrade(airless: boolean): void;
    };
    hd.settings.antialias = 'none';
    hd.settings.sharpness = -1;
    try {
      hd.applySettings();
      const originalKey = state.pipelineKey;
      expect(hd.settings.tone).toBeNull();
      expect(JSON.parse(originalKey)).toEqual(['none', 'faithful', -1, 'none', false, false]);
      const p = moistTone('snow', 'moist2', LOOK_PRESETS.cinematic.settings)!;
      p.grain = 0;
      hd.settings.tone = p;
      applyLookPreset(hd.settings, 'cinematic');
      expect(hd.settings.tone).toBe(p);
      hd.applySettings();
      expect(JSON.parse(state.pipelineKey).slice(-2)).toEqual(['tone', false]);
      const pipeline = state.pipeline;
      p.density = 0.2;
      p.grade = null;
      p.hueShifts = [];
      p.foliage = null;
      hd.applySettings();
      expect(state.pipeline).toBe(pipeline);
      expect(state.finalTone.density.value).toBe(0.2);
      expect(state.finalTone.useGrade.value).toBe(0);
      expect(state.finalTone.shifts[0].value.w).toBe(0);
      p.grade = LOOK_PRESETS.cinematic.settings;
      hd.applySettings();
      state.updateGrade(true);
      expect(state.finalTone.gradeUniforms.gradeBalance.value.toArray()).toEqual([1, 1, 1]);
      expect(state.finalTone.gradeUniforms.gradeShadowTint.value.toArray()).toEqual([1, 1, 1]);
      state.updateGrade(false);
      expect(state.finalTone.gradeUniforms.gradeBalance.value.toArray()).toEqual(p.grade.gradeBalance);
      p.grain = 0.5;
      hd.applySettings();
      expect(JSON.parse(state.pipelineKey).slice(-2)).toEqual(['tone', true]);
      hd.settings.tone = null;
      applyLookPreset(hd.settings, 'faithful');
      hd.applySettings();
      expect(state.pipelineKey).toBe(originalKey);
    } finally {
      state.pipeline?.dispose();
      state.pipelineNodes.forEach((node) => node.dispose());
      hd.nodes.dispose();
    }
  });
});

/** Build WGSL on the CPU, following silhouette.test.ts; no device initialization or drawing. */
function shader(node: ShaderNode): string {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const material = new NodeMaterial();
  material.fragmentNode = node;
  const quad = new QuadMesh(material);
  const builder = renderer.backend.createNodeBuilder(quad, renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  try {
    builder.build();
    return builder.fragmentShader;
  } finally {
    material.dispose();
  }
}

it('builds final colour/mask with one code tap, and grain with exactly five picture taps', () => {
  const tone = new FinalTone();
  tone.set(moistTone('temperate', 'moist2', LOOK_PRESETS.cinematic.settings)!);
  const data = new DataTexture(new Uint16Array(4 * 4 * 4), 4, 4, RGBAFormat, HalfFloatType);
  data.minFilter = data.magFilter = LinearFilter;
  const code: ShaderNode = texture(data).yz.toVar();
  const look = vec3(0.2, 0.3, 0.1);
  try {
    const colourShader = shader(vec4(tone.node(look, look, float(1), float(0), code), tone.grainMask(look, code)));
    expect(colourShader).toContain('@fragment');
    expect(colourShader).toContain('atan2');
    expect(colourShader.match(/textureSample(?:Level)?\(/g)).toHaveLength(1);
    const grainShader = shader(tone.grainCut(texture(data), float(0)));
    expect(grainShader.match(/textureSample(?:Level)?\(/g)).toHaveLength(5);
    // Offsets come from the screen size: a textureDimensions-based size left an untyped uniform (black frames).
    expect(grainShader).not.toContain('textureDimensions');
  } finally {
    data.dispose();
  }
});

it('cuts only low-amplitude luminance residuals, preserves RGB ratios and applies mask/night shares', () => {
  const tone = new FinalTone();
  tone.set({ ...params(), grain: 0.5, night: { colour: 0.35, grain: 0.25 } });
  const run = (centre: Rgb, blur: Rgb, mask: number, night: number) => {
    let tap = 0;
    // Constant sample results isolate the residual arithmetic; the real sampler is WGSL-built above.
    const picture = { sample: () => vec4(...(tap++ === 0 ? centre : blur), mask) };
    return evaluate(tone.grainCut(picture, float(night)));
  };
  const centre: Rgb = [0.3, 0.2, 0.1];
  const blur = centre.map((v) => v * 0.9) as [number, number, number];
  const day = run(centre, blur, 1, 0);
  const night = run(centre, blur, 1, 1);
  for (let i = 0; i < 3; i++) {
    expect(day[i]).toBeCloseTo(centre[i]! * 0.95, 12);
    expect(night[i]! - centre[i]!).toBeCloseTo((day[i]! - centre[i]!) * 0.25, 12);
    expect(run(centre, blur, 0, 0)[i]).toBeCloseTo(centre[i]!, 12);
    expect(run(centre, [0, 0, 0], 1, 0)[i]).toBeCloseTo(centre[i]!, 12);
  }
  expect(day[0]! / day[1]!).toBeCloseTo(centre[0] / centre[1], 12);
  expect(run([0, 0, 0], [0, 0, 0], 1, 0)).toEqual([0, 0, 0, 1]);
});

it('shares one picture between clarity and grain, omitting grain without painter depth', () => {
  vi.stubGlobal('window', { devicePixelRatio: 1 });
  const canvas = {
    width: 1,
    height: 1,
    style: {},
    addEventListener() {},
    removeEventListener() {},
  } as unknown as HTMLCanvasElement;
  const hd = new HdRenderer(canvas, createScenario(2, 2));
  const state = hd as unknown as {
    painterDepthEnabled: boolean;
    pipeline: RenderPipeline;
    pipelineNodes: { dispose(): void }[];
  };
  const targets = (node: ShaderNode) => {
    const seen = new Set<ShaderNode>(),
      copies = new Set<unknown>();
    const visit = (n: ShaderNode) => {
      if (seen.has(n)) return;
      seen.add(n);
      if (n.isRTTNode) copies.add(n.renderTarget);
      for (const child of n.getChildren()) visit(child);
    };
    visit(node);
    return copies.size;
  };
  hd.settings.antialias = 'none';
  hd.settings.sharpness = -1;
  hd.settings.tone = { ...params(), density: 0.1 };
  try {
    hd.applySettings();
    expect(targets(state.pipeline.outputNode)).toBe(0);
    state.painterDepthEnabled = true;
    // Painter depth is normally enabled by the renderer's owner before building the scene graph.
    hd.settings.clarity = 0.5;
    hd.applySettings();
    expect(targets(state.pipeline.outputNode)).toBe(1);
    hd.settings.tone.grain = 0;
    hd.applySettings();
    expect(targets(state.pipeline.outputNode)).toBe(1);
    hd.settings.tone.grain = 0.5;
    hd.settings.clarity = 0;
    hd.applySettings();
    expect(targets(state.pipeline.outputNode)).toBe(1);
  } finally {
    state.pipeline?.dispose();
    state.pipelineNodes.forEach((node) => node.dispose());
    hd.nodes.dispose();
  }
});
