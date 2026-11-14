import { describe, expect, it } from 'vitest';
import { float, vec2, vec3, vec4 } from 'three/tsl';
import { NodeMaterial, OrthographicCamera, QuadMesh, Scene, WebGPURenderer } from 'three/webgpu';
import {
  GroundToneUniforms,
  groundGreenness,
  groundPatchField,
  groundPatchNode,
  groundToneNode,
  groundToneRgb,
} from '../src/groundTone';
import { AtmosphereNodes } from '../src/atmosphere';
import { lushTheatre, LUSH_THEATRES } from '../src/ra2/lushTone';
import { applyMoistTone, parseToneStyle } from '../src/ra2/moistTone';
import { finalToneRgb, FinalTone } from '../src/tone';
import type { HdSettings } from '../src/HdRenderer';
import { neutralGrade } from '../src/post';
import { applyGroundOverrides } from '../sandbox/ground';
import type { ShaderNode } from '../src/tsl';

const params = () => lushTheatre('temperate')!;
const luma = (rgb: readonly number[]) => rgb.reduce((s, x, i) => s + x * [0.2126, 0.7152, 0.0722][i]!, 0);

describe('ground tone reference', () => {
  it('is exactly identity at strength zero or at full night without a night share, including HDR input', () => {
    for (const rgb of [
      [0, 0, 0],
      [0.1, 0.4, 0.2],
      [1.7, 0.3, 0.9],
    ] as const)
      for (const lit of [0, 0.3, 1]) {
        expect(groundToneRgb(rgb, lit, { ...params(), strength: 0 })).toEqual(rgb);
        expect(groundToneRgb(rgb, lit, { ...params(), nightKeep: 0 }, 1)).toEqual(rgb);
      }
  });
  it('keeps nightKeep of the dressing at full night with moonlit (not golden) patches, never brightening grass', () => {
    const p = params(),
      green = [0.2, 0.4, 0.1] as const;
    expect(p.nightKeep).toBeGreaterThan(0);
    const night = groundToneRgb(green, 1, p, 1);
    const full = groundToneRgb(green, 1, { ...p, sunTint: p.moonTint }, 0);
    night.forEach((v, i) => expect(v).toBeCloseTo(green[i]! + (full[i]! - green[i]!) * p.nightKeep, 12));
    for (const lit of [0, 0.5, 1]) expect(luma(groundToneRgb(green, lit, p, 1))).toBeLessThan(luma(green));
    // Moonlight is cooler than sunlight: blue over red rises in the lit patches.
    const day = groundToneRgb(green, 1, p, 0);
    expect(night[2] / night[0]).toBeGreaterThan(day[2] / day[0]);
  });
  it('normalises tints on coloured albedo, leaving depth/gains to control luminance', () => {
    const p = {
      ...params(),
      depth: 1,
      bareDepth: 1,
      bareChroma: 1,
      shadeGain: 1,
      bareShade: 1,
      sunGain: 1,
      sunChroma: 1,
      shadeChroma: 1,
    };
    for (const rgb of [
      [0.2, 0.5, 0.1],
      [0.3, 0.3, 0.3],
      [0.3, 0.1, 0.7],
    ] as const)
      for (const lit of [0, 0.4, 1]) expect(luma(groundToneRgb(rgb, lit, p))).toBeCloseTo(luma(rgb), 12);
    // Bare-ground depth and the dark-ground keep have their own tests; here grass takes depth times the patch gain.
    const g = { ...params(), bareDepth: params().depth, darkKeep: 1, darkPatch: 1 };
    const c = [0.3, 0.4, 0.25] as const;
    for (const lit of [0, 1]) {
      const out = groundToneRgb(c, lit, g);
      expect(luma(out) / luma(c)).toBeCloseTo(g.depth * (lit ? g.sunGain : g.shadeGain), 12);
    }
  });
  it('deepens grass more than bare ground, treats yellow-green tile stripes as grass, and spares dark ground', () => {
    const p = { ...params(), darkKeep: 1 };
    const ratio = (rgb: readonly [number, number, number], q = p) => luma(groundToneRgb(rgb, 1, q)) / luma(rgb);
    // Yellow-green and green stripes of the same lawn: one class, so the tile lattice is not drawn as bands.
    expect(groundGreenness([0.5, 0.52, 0.1])).toBe(1);
    expect(groundGreenness([0.3, 0.42, 0.08])).toBe(1);
    expect(groundGreenness([0.72, 0.51, 0.19])).toBe(0); // sand
    expect(groundGreenness([0.3, 0.3, 0.3])).toBe(0); // grey rock or paving
    expect(ratio([0.3, 0.42, 0.08])).toBeCloseTo(p.depth * p.sunGain, 12);
    expect(ratio([0.72, 0.51, 0.19])).toBeCloseTo(p.bareDepth * p.sunGain, 12);
    const keep = params();
    const dark = [0.03, 0.05, 0.02] as const;
    expect(ratio(dark, keep)).toBeCloseTo(
      (1 + (keep.depth - 1) * keep.darkKeep) * (1 + (keep.sunGain - 1) * keep.darkPatch),
      12,
    );
    // Bare ground in shade takes its own lighter gain, so shaded sand stays light instead of turning muddy.
    const shade = (rgb: readonly [number, number, number]) => luma(groundToneRgb(rgb, 0, p)) / luma(rgb);
    expect(shade([0.72, 0.51, 0.19])).toBeCloseTo(p.bareDepth * p.bareShade, 12);
    expect(shade([0.3, 0.42, 0.08])).toBeCloseTo(p.depth * p.shadeGain, 12);
    expect(p.bareDepth * p.bareShade).toBeGreaterThan(p.depth * p.shadeGain);
    expect(ratio([0.3, 0.42, 0.08], keep)).toBeCloseTo(keep.depth * keep.sunGain, 12);
  });
  it('gates the extra grass tint on greenness and only in shade; fades continuously at night', () => {
    const p = params(),
      neutral = { ...p, grassShadeTint: [1, 1, 1] as [number, number, number] };
    const grey = [0.3, 0.3, 0.3] as const,
      green = [0.2, 0.4, 0.1] as const;
    expect(groundToneRgb(grey, 0, p)).toEqual(groundToneRgb(grey, 0, neutral));
    expect(groundToneRgb(green, 1, p)).toEqual(groundToneRgb(green, 1, neutral));
    expect(groundToneRgb(green, 0, p)[1]).toBeGreaterThan(groundToneRgb(green, 0, neutral)[1]);
    const q = { ...p, nightKeep: 0, moonTint: p.sunTint };
    groundToneRgb(green, 0.4, q, 0.5).forEach((v, i) =>
      expect(v).toBeCloseTo((green[i]! + groundToneRgb(green, 0.4, q)[i]!) / 2, 12),
    );
  });
  it('is deterministic across negative world coordinates and camera-independent sampling order', () => {
    const p = params();
    const points = Array.from({ length: 100 }, (_, i) => [i * 0.17 - 8, i * -0.37]);
    const values = points.map(([x, z]) => groundPatchField(x!, z!, p));
    expect(
      [...points]
        .reverse()
        .map(([x, z]) => groundPatchField(x!, z!, p))
        .reverse(),
    ).toEqual(values);
    expect(Math.max(...values) - Math.min(...values)).toBeGreaterThan(0.9);
    for (const v of values) (expect(v).toBeGreaterThanOrEqual(0), expect(v).toBeLessThanOrEqual(1));
  });
  it('tracks area coverage and keeps the wavelength in world units, stretched along the authored angle', () => {
    const p = params();
    for (const coverage of [0, 0.2, 0.5, 0.8, 1]) {
      let sum = 0,
        count = 0;
      for (let z = -150; z < 150; z += 0.79)
        for (let x = -150; x < 150; x += 0.83) {
          sum += groundPatchField(x, z, { ...p, coverage });
          count++;
        }
      expect(sum / count).toBeCloseTo(coverage, 1);
    }
    const base = { ...p, angle: 0, detailMix: 0, stretch: 1 };
    const difference = (dx: number, dz: number, stretch = 1) => {
      let sum = 0;
      for (let i = 0; i < 2000; i++) {
        const x = i * 0.217,
          z = i * 0.591;
        sum += Math.abs(
          groundPatchField(x + dx, z + dz, { ...base, stretch }) - groundPatchField(x, z, { ...base, stretch }),
        );
      }
      return sum / 2000;
    };
    expect(difference(p.scale, 0)).toBeGreaterThan(difference(0.1, 0) * 8);
    expect(difference(p.scale, 0)).toBeGreaterThan(0.25);
    expect(difference(1, 0, 2)).toBeLessThan(difference(0, 1, 2));
    for (const [x, z] of [
      [0.1, 0.8],
      [-4, 3.1],
    ])
      expect(groundPatchField(x! * 2, z! * 2, { ...p, scale: p.scale * 2, detailScale: p.detailScale * 2 })).toBe(
        groundPatchField(x!, z!, p),
      );
  });
});

describe('lush policy and sandbox', () => {
  it('selects lush, clones tuning arrays, falls back to temperate and leaves lunar untouched', () => {
    const settings = { ...neutralGrade() } as HdSettings;
    expect(parseToneStyle('lush')).toBe('lush');
    expect(lushTheatre('unknown')).toEqual(params());
    expect(lushTheatre('lunar')).toBeNull();
    for (const theater of Object.keys(LUSH_THEATRES)) {
      applyMoistTone(settings, theater, 'lush');
      expect(settings.groundTone).toEqual(LUSH_THEATRES[theater as keyof typeof LUSH_THEATRES]);
    }
    expect(settings.groundTone).toBeNull();
    expect(settings.tone).toBeNull();
    const p = params();
    p.sunTint[0] = 99;
    expect(params().sunTint[0]).toBe(1.32);
    for (const style of ['off', 'moist1', 'moist2', 'moist3'] as const) {
      applyMoistTone(settings, 'temperate', 'lush');
      applyMoistTone(settings, 'temperate', style);
      expect(settings.groundTone).toBeNull();
    }
  });
  it('keeps final colour exactly identical at every night/object/grade share; only grain and clarity differ', () => {
    for (const theater of ['temperate', 'snow', 'urban', 'newurban', 'desert']) {
      const settings = { ...neutralGrade() } as HdSettings;
      applyMoistTone(settings, theater, 'lush');
      const p = settings.tone!;
      expect(p).toMatchObject({
        strength: 1,
        grade: null,
        density: 0,
        contrast: 0,
        pivot: 0.5,
        chroma: [1, 1, 1],
        hueShifts: [],
        keyRadius: 0,
        objectShare: 0,
        objectDensity: 0,
      });
      const u = new FinalTone();
      u.set(p);
      expect(u.colourActive.value).toBe(0);
      expect(u.grain.value).toBe(LUSH_THEATRES.temperate!.grain);
      expect(u.groundClarity.value).toBe(LUSH_THEATRES.temperate!.groundClarity);
      for (const rgb of [
        [0, 0, 0],
        [0.1, 0.3, 0.05],
        [0.2, 0.5, 0.9],
        [2, 0.001, 0.4],
      ] as const)
        for (const night of [0, 0.5, 1])
          for (const object of [false, true])
            for (const gradeShare of [0, 1])
              expect(finalToneRgb(rgb, [0.9, 0.1, 0.2], p, { night, object, gradeShare })).toEqual(rgb);
    }
  });
  it('parses every numeric field and all triples and rejects unknown or malformed overrides', () => {
    const p = params();
    applyGroundOverrides(
      p,
      'depth:0.7,scale:4,angle:-25,soften:0.8,sunTint:1.2/1/0.7,shadeTint:0.9/1/1,grassShadeTint:0.8/1/0.7,' +
        'bareShade:0.8,darkPatch:0.5,nightKeep:0.3,moonTint:0.9/1/1.1',
    );
    expect(p).toMatchObject({
      depth: 0.7,
      scale: 4,
      angle: -25,
      soften: 0.8,
      sunTint: [1.2, 1, 0.7],
      bareShade: 0.8,
      darkPatch: 0.5,
      nightKeep: 0.3,
      moonTint: [0.9, 1, 1.1],
    });
    for (const s of [
      'unknown:2',
      'grain:1',
      'depth:NaN',
      'depth:',
      'depth:1:2',
      'depth',
      'sunTint:1/2',
      'sunTint:1//2',
      'sunTint:1/2/NaN',
      'shadeTint:1',
    ])
      expect(() => applyGroundOverrides(p, s)).toThrow();
    expect(() => applyGroundOverrides(null, 'depth:1')).not.toThrow();
  });
  it('copies values without replacing uniforms, vectors, or mutating policy data', () => {
    const a = new AtmosphereNodes(),
      b = new AtmosphereNodes();
    const refs = [b.groundTone.depth, b.groundTone.sunTint, b.groundTone.sunTint.value];
    try {
      a.groundTone.set(params());
      b.copyValuesFrom(a);
      expect(b.groundTone.depth.value).toBe(params().depth);
      expect(b.groundTone.sunTint.value.toArray()).toEqual(params().sunTint);
      a.groundTone.sunTint.value.x = 2;
      expect(b.groundTone.sunTint.value.x).toBe(params().sunTint[0]);
      b.groundTone.set({ ...params(), depth: 0.8 });
      expect([b.groundTone.depth, b.groundTone.sunTint, b.groundTone.sunTint.value]).toEqual(refs);
      b.groundTone.set(null);
      expect(b.groundTone.strength.value).toBe(0);
    } finally {
      a.dispose();
      b.dispose();
    }
  });
});

function shader(node: ShaderNode): string {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  Object.assign(renderer.backend, {
    device: { limits: { maxUniformBufferBindingSize: 65536 }, features: new Set() },
    renderer,
  });
  renderer.hasFeature = renderer.hasCompatibility = () => false;
  const material = new NodeMaterial();
  material.fragmentNode = node;
  const builder = renderer.backend.createNodeBuilder(new QuadMesh(material), renderer);
  builder.scene = new Scene();
  builder.camera = new OrthographicCamera();
  try {
    builder.build();
    return builder.fragmentShader;
  } finally {
    material.dispose();
  }
}
it('builds WGSL with enabled/disabled ground and a uniform branch enclosing integer noise', () => {
  for (const p of [null, params()]) {
    const u = new GroundToneUniforms();
    u.set(p);
    const text = shader(vec4(groundToneNode(vec3(0.2, 0.4, 0.1), vec2(-2.4, 3.7), u, float(0)), 1));
    expect(text).toContain('@fragment');
    expect(text).toContain('if (');
    expect(text).toContain('>>');
    expect(text).toContain('u32');
    expect(text).not.toContain('textureSample');
  }
});

/** Evaluate the pure value-noise graph, including explicit uint overflow. Unknown operations fail the test. */
function evaluate(root: ShaderNode): number[] {
  const cache = new Map<ShaderNode, number[]>();
  const visit = (n: ShaderNode): number[] => {
    if (cache.has(n)) return cache.get(n)!;
    const result = compute(n);
    cache.set(n, result);
    return result;
  };
  const each = (args: number[][], fn: (...v: number[]) => number) =>
    Array.from({ length: Math.max(...args.map((v) => v.length)) }, (_, i) =>
      fn(...args.map((v) => (v.length === 1 ? v[0]! : v[i]!))),
    );
  function compute(n: ShaderNode): number[] {
    if (n.isConstNode || n.isUniformNode) return typeof n.value === 'number' ? [n.value] : n.value.toArray();
    if (n.isSplitNode) return [...n.components].map((c: string) => visit(n.node)['xyzw'.indexOf(c)]!);
    if (n.isVarNode) return visit(n.node);
    if (n.isConvertNode || n.type === 'ConvertNode') {
      const v = visit(n.node);
      if (n.convertTo === 'uint') return v.map((x) => x >>> 0);
      if (n.convertTo === 'int') return v.map((x) => x | 0);
      return v;
    }
    if (n.isJoinNode || n.type === 'JoinNode') return n.nodes.flatMap(visit);
    if (n.condNode) return visit(visit(n.condNode)[0] ? n.ifNode : n.elseNode);
    if (n.isOperatorNode) {
      const isUint = (v: ShaderNode): boolean =>
        v.convertTo
          ? v.convertTo === 'uint'
          : v.nodeType === 'uint' || (v.isVarNode ? isUint(v.node) : v.isOperatorNode && isUint(v.aNode));
      const uintOp = isUint(n);
      const op = (
        {
          '+': (a: number, b: number) => a + b,
          '-': (a: number, b: number) => a - b,
          '*': (a: number, b: number) => (uintOp ? Math.imul(a, b) >>> 0 : a * b),
          '/': (a: number, b: number) => a / b,
          '^': (a: number, b: number) => (a ^ b) >>> 0,
          '>>': (a: number, b: number) => a >>> b,
          '<=': (a: number, b: number) => Number(a <= b),
          '>=': (a: number, b: number) => Number(a >= b),
        } as Record<string, (a: number, b: number) => number>
      )[n.op];
      if (!op) throw new Error(`operator ${n.op}`);
      return each([visit(n.aNode), visit(n.bNode)], op);
    }
    if (n.isMathNode) {
      const a = visit(n.aNode),
        b = n.bNode ? visit(n.bNode) : [],
        c = n.cNode ? visit(n.cNode) : [];
      switch (n.method) {
        case 'floor':
          return a.map(Math.floor);
        case 'fract':
          return a.map((v) => v - Math.floor(v));
        case 'cos':
          return a.map(Math.cos);
        case 'sin':
          return a.map(Math.sin);
        case 'max':
          return each([a, b], Math.max);
        case 'pow':
          return each([a, b], Math.pow);
        case 'sqrt':
          return a.map(Math.sqrt);
        case 'mix':
          return each([a, b, c], (x, y, t) => x * (1 - t) + y * t);
        case 'smoothstep':
          return each([a, b, c], (lo, hi, x) => {
            const t = Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
            return t * t * (3 - 2 * t);
          });
      }
    }
    throw new Error(`cannot evaluate ${n.type}`);
  }
  return visit(root);
}
it('matches the JS patch twin against the actual TSL graph, including signed lattice positions', () => {
  const u = new GroundToneUniforms();
  for (const mix of [0, 0.35, 1])
    for (const coverage of [0, 0.3, 0.5, 0.8, 1]) {
      const p = { ...params(), detailMix: mix, coverage };
      u.set(p);
      for (const [x, z] of [
        [-17.3, 4.1],
        [0, 0],
        [2.6, -3.9],
        [90.1, 73.4],
      ])
        expect(evaluate(groundPatchNode(vec2(x!, z!), u))[0]).toBeCloseTo(groundPatchField(x!, z!, p), 10);
    }
});
