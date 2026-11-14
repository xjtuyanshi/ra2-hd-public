import { describe, expect, it, vi } from 'vitest';
import { Color, Mesh, Vector3, type WebGPURenderer } from 'three/webgpu';
import { HdRenderer } from '../src/HdRenderer';
import { LiveView } from '../src/live/liveView';
import { buildingLightSources } from '../src/live/buildingLights';
import { engineLightDelta } from '../../ra2-assets/src/lights';
import { AtmosphereNodes } from '../src/atmosphere';
import {
  RealLightField,
  RealLightNodes,
  lightMoment,
  lightReach,
  lightWindow,
  lightLuminance,
  reconstructLight,
  irradianceKnee,
  REAL_LIGHT_THRESHOLD,
  type PhysicalLight,
  type Rgb,
} from '../src/realLight';
import {
  REAL_LIGHT_POLICIES,
  LIGHT_COLOURS,
  policyLight,
  animLightPolicy,
  animPhysicalLight,
  engineLampLights,
  buildingPhysicalLight,
  nearestLampHead,
  ra2LightOccluders,
  capLightColour,
  nativeBeamLight,
  auxiliaryBeamLights,
} from '../src/ra2/realLights';
import { MapLightGrade } from '../src/ra2/mapLighting';
import { LiveEffectsDriver, type LiveAnim } from '../src/live/effects';
import { EffectsLayer } from '../src/effects';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';
const point: PhysicalLight = { position: [0, 0.5, 0], radius: 0.2, intensity: [0.8, 0.6, 0.4], dayFactor: 0 };
const rgbClose = (a: Rgb, b: Rgb, digits = 10) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, digits));
const irradiance = (l: PhysicalLight, distance: number) => {
  const m = lightMoment(l, [distance, 0, 0]);
  return reconstructLight(m.s, m.d, [0, 1, 0]);
};
const lighting = { ambient: 0.55, red: 0.4, green: 0.4, blue: 0.7, ground: 0, level: 0.003 };
function fakeRenderer() {
  return {
    getRenderTarget: () => null,
    getMRT: () => null,
    getClearAlpha: () => 1,
    getClearColor: (c: Color) => c.set(0x123456),
    setClearColor: vi.fn(),
    autoClear: false,
    setClearAlpha: vi.fn(),
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
  } as unknown as WebGPURenderer;
}
describe('physical light CPU twins', () => {
  it('is exact Lambert for one light, including back-facing normals', () => {
    const p = [0.3, 0, 0.2] as const,
      m = lightMoment(point, p);
    const distance = Math.hypot(0.3, 0.5, 0.2);
    for (const n of [
      [0, 1, 0],
      [1, 0, 0],
      [-1, 0, 0],
      [0, -1, 0],
    ] as const) {
      const cosine = Math.max(0, (-0.3 * n[0] + 0.5 * n[1] - 0.2 * n[2]) / distance);
      rgbClose(reconstructLight(m.s, m.d, n), m.s.map((v) => v * cosine) as [number, number, number]);
    }
  });
  it('opposing directions retain half-ambient energy without dividing by zero', () => {
    rgbClose(reconstructLight([2, 3, 4], [0, 0, 0], [0, 1, 0]), [1, 1.5, 2]);
    rgbClose(reconstructLight([0, 0, 0], [0, 0, 0], [0, 1, 0]), [0, 0, 0]);
  });
  it('derives cutoff from intensity and source radius, with a smooth UE4 window', () => {
    const r = lightReach(point);
    expect(lightLuminance(point.intensity) / (r * r + point.radius ** 2)).toBeCloseTo(REAL_LIGHT_THRESHOLD, 12);
    expect(lightReach({ ...point, intensity: [0, 0, 0] })).toBe(0);
    expect(lightReach({ ...point, intensity: [3.2, 2.4, 1.6] })).toBeGreaterThan(r);
    expect(lightWindow(0)).toBe(1);
    expect(lightWindow(1)).toBe(0);
    expect(lightWindow(2)).toBe(0);
    rgbClose(lightMoment(point, [r, 0.5, 0]).s, [0, 0, 0]);
  });
  it('normalizes finite segment energy by its actual length, including subdivision and the far field', () => {
    const tube = { ...point, position: [-2, 0.5, 0] as const, end: [2, 0.5, 0] as const };
    const a = { ...tube, end: [0, 0.5, 0] as const },
      b = { ...tube, position: [0, 0.5, 0] as const };
    for (const p of [
      [0, 0, 0.5],
      [3, 0, 1],
      [100, 0, 10],
    ] as const) {
      const whole = lightMoment(tube, p, false).s;
      const first = lightMoment(a, p, false).s,
        second = lightMoment(b, p, false).s;
      rgbClose(whole, first.map((v, i) => v + second[i]!) as [number, number, number]);
    }
    rgbClose(lightMoment({ ...tube, end: tube.position }, [0, 0, 0.5], false).s, [0, 0, 0]);
    const remote = lightMoment(tube, [0, 0, 1000], false).s;
    rgbClose(remote, point.intensity.map((c) => (c * 4) / 1e6) as [number, number, number], 10);
  });
  it('applies the knee to irradiance, preserving albedo ratios and illumination hue', () => {
    const e = irradianceKnee([12, 8, 5], 1);
    expect(lightLuminance(e)).toBeLessThanOrEqual(2.5);
    expect(e[0] / e[1]).toBeCloseTo(1.5);
    const lit = (albedo: number) => (albedo * (0.3 + e[0])) / Math.PI;
    expect(lit(0.4) / lit(0.1)).toBeCloseTo(4, 14);
    expect(lightLuminance(irradianceKnee([12, 8, 5], 0))).toBeLessThanOrEqual(1.2);
  });
  it('calibrates a night fire, retains convex physical falloff and caps source colours', () => {
    const light = policyLight(REAL_LIGHT_POLICIES.fire, [0, 0.4, 0]);
    expect(lightLuminance(irradiance(light, 0.5))).toBeCloseTo(0.5, 8);
    const profile = [0.5, 1, 2, 3].map((d) => lightLuminance(irradiance(light, d)));
    expect(profile[2]! / profile[0]!).toBeLessThan(0.15);
    expect(profile[0]).toBeGreaterThan(profile[1]!);
    expect(profile[1]).toBeGreaterThan(profile[2]!);
    expect(profile[2]).toBeGreaterThan(profile[3]!);
    const cap = capLightColour([1, 0.03, 0], 0.55);
    expect(1 - Math.min(...cap) / Math.max(...cap)).toBeCloseTo(0.55);
    expect(lightLuminance(cap)).toBeCloseTo(lightLuminance([1, 0.03, 0]));
  });
});
describe('physical field ownership and GPU graph generation', () => {
  it.each([true, false])('builds finite MRT/scalar field shaders (directional=%s)', (directional) => {
    const nodes = new RealLightNodes(),
      field = new RealLightField(2, 2, nodes, directional);
    expect(field.targets).toHaveLength(directional ? 2 : 1);
    for (let i = 0; i < field.scenes.length; i++)
      field.scenes[i]!.traverse((o) => {
        if (!(o instanceof Mesh)) return;
        const code = shaderBytes(o, field.targets[i]);
        expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
        expect(code).toContain('atan(');
        if (directional) expect(code).toContain('@location( 1 )');
      });
    field.dispose();
    nodes.dispose();
  });
  it('generates actual terrain, HD sprite with material normals, VPL and PBR receivers without a device', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const scene = receiverScene(nodes);
    for (const mesh of scene.meshes) {
      const code = shaderBytes(mesh);
      expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
      expect(code).toContain('exp(');
    }
    scene.dispose();
  });
  it('reserves pools, follows envelopes/day gain, clears expired lights and restores renderer state', () => {
    const nodes = new RealLightNodes(),
      field = new RealLightField(2, 2, nodes),
      renderer = fakeRenderer();
    field.setStatic([point]);
    field.put('kept', 'fire', 0, point, 0, 0, 0.3);
    for (let i = 0; i < 300; i++) field.pulse('flash', 0, { ...point, dayFactor: 0.2 }, 0.1, 0.1);
    expect(field.pools.get('flash')!.size).toBe(256);
    expect(field.pools.get('kept')!.size).toBe(1);
    field.update(0.05, renderer, 0);
    expect(renderer.render).toHaveBeenCalledTimes(2);
    field.update(0.4, renderer, 1);
    expect(field.pools.get('flash')!.size).toBe(0);
    expect(field.pools.get('kept')!.size).toBe(0);
    nodes.enabled.value = 0;
    expect(field.update(1, renderer, 1)).toBe(false);
    expect(renderer.setClearColor).toHaveBeenLastCalledWith(new Color(0x123456), 1);
    expect(renderer.autoClear).toBe(false);
    field.dispose();
    nodes.dispose();
  });
  it('rasterizes terrain levels and clipped building top heights at two texels per cell', () => {
    const o = ra2LightOccluders(2, 2, [0, 0, 0, 1], [{ x: -1, z: 0, width: 2, depth: 1, top: 3 }]);
    expect(o.heights).toHaveLength(16);
    expect(Array.from(o.heights.slice(0, 4))).toEqual([3, 3, 0, 0]);
    expect(o.heights[15]).toBeGreaterThan(0);
    expect(o.heights[8]).toBe(0);
  });
  it('validates the 2-texels/cell top-height contract', () => {
    const n = new RealLightNodes(),
      f = new RealLightField(2, 2, n);
    expect(() =>
      f.setOccluders({ width: 4, height: 4, texelsPerCell: 2, heights: new Float32Array(16) }),
    ).not.toThrow();
    expect(() => f.setOccluders({ width: 2, height: 2, texelsPerCell: 2, heights: new Float32Array(4) })).toThrow();
    f.dispose();
    n.dispose();
  });
});
describe('RA2 source policy and capture integration', () => {
  it('covers formerly unlit signature anims, damage fires and physical beams', () => {
    for (const name of [
      'FIRE01',
      'FIRE02',
      'FIRE03',
      'INITFIRE',
      'CHRONOSK',
      'WARPIN',
      'WARPOUT',
      'PDFXLOC',
      'PDFXCLD',
      'IRONBLST',
      'IRONFX',
      'NUKETO',
      'WCLBOLT1',
      'WCLBOLT3',
    ])
      expect(animLightPolicy(name)).not.toBeNull();
    expect(animLightPolicy('FIRE02')!.day).toBe(0);
    const beam = nativeBeamLight({
      kind: 'bolt',
      source: { x: 256, y: 512, z: 104 },
      target: { x: 1024, y: 512, z: 0 },
      colour: [82, 80, 255],
    });
    expect(beam.end).toBeDefined();
    expect(beam.position[0]).toBe(1);
    expect(
      auxiliaryBeamLights({
        anims: [],
        bullets: [],
        particleSystems: [],
        radBeams: [
          {
            id: 1,
            type: 'temporal',
            source: { x: 256, y: 0, z: 104 },
            target: { x: 512, y: 0, z: 0 },
            color: [100, 200, 255],
            period: 10,
            amplitude: 0,
            age: 1,
            sustained: true,
            progress: 0,
          },
        ],
      }),
    ).toHaveLength(1);
  });
  it('captures damage fires even when buildingAnim/garnish gates skip native cards, and preserves geometry', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      field = new RealLightField(2, 2, nodes.realLight!);
    const effects = new EffectsLayer(nodes);
    effects.physicalLights = field;
    const driver = new LiveEffectsDriver(effects, () => false);
    driver.garnishEnabled = false;
    const anim: LiveAnim = { id: 1, uid: 7, type: 'FIRE02', x: 256, y: 256, z: 0, buildingAnim: true };
    const snapshot = {
      anims: [anim],
      bullets: [],
      particleSystems: [],
      beams: [
        {
          kind: 'bolt' as const,
          source: { x: 0, y: 0, z: 104 },
          target: { x: 256, y: 256, z: 0 },
          colour: [82, 80, 255] as [number, number, number],
        },
      ],
    };
    const before = JSON.stringify(snapshot);
    driver.push(snapshot, 1, 0, new Vector3());
    driver.frame(1, new Vector3());
    expect(field.pools.get('kept')!.size).toBe(1);
    expect(field.pools.get('flash')!.get('beam:0')!.light.end).toBeDefined();
    expect(JSON.stringify(snapshot)).toBe(before);
    expect(animPhysicalLight(anim, REAL_LIGHT_POLICIES.fire).position[1]).toBe(0.4);
    effects.root.traverse((o) => {
      if (o instanceof Mesh) {
        o.geometry.dispose();
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
      }
    });
    field.dispose();
    nodes.dispose();
  });
  it('keeps negative engine sources and restores original grade bytes after an A/B switch', () => {
    const grid = { width: 4, height: 4, z: new Uint8Array(16), exists: new Uint8Array(16).fill(1) };
    const sources = [
      {
        position: [128, 128, 0] as [number, number, number],
        radius: 2000,
        intensity: 0.2,
        tint: [0.1, 0.1, 0.05] as [number, number, number],
      },
      {
        position: [384, 384, 0] as [number, number, number],
        radius: 1000,
        intensity: -0.1,
        tint: [0, 0, 0] as [number, number, number],
      },
    ];
    const grade = new MapLightGrade(grid, lighting, sources),
      negative = new MapLightGrade(grid, lighting, [sources[1]!]);
    const original = (grade.texture.image.data as Uint8Array).slice();
    grade.setPhysicalSources(true);
    expect(grade.texture.image.data).toEqual(negative.texture.image.data);
    grade.setPhysicalSources(false);
    expect(grade.texture.image.data).toEqual(original);
    const lamps = engineLampLights(sources[0]!, lighting);
    expect(lamps).toHaveLength(2);
    expect(lamps[1]!.end).toBeDefined();
    expect(lamps.every((l) => l.dayFactor === 0)).toBe(true);
    expect(engineLampLights(sources[1]!, lighting)).toEqual([]);
    grade.dispose();
    negative.dispose();
  });
  it('calibrates the combined lamp head/post moment to engineCellLight brightness', () => {
    const source = {
      position: [0, 0, 0] as [number, number, number],
      radius: 2000,
      intensity: 0.2,
      tint: [0.1, 0.1, 0.05] as [number, number, number],
    };
    const d = (source.radius / 256) * 0.25,
      lights = engineLampLights(source, lighting);
    const s: [number, number, number] = [0, 0, 0],
      direction: [number, number, number] = [0, 0, 0];
    for (const light of lights) {
      const m = lightMoment(light, [d, 0, 0]);
      m.s.forEach((c, i) => (s[i]! += c));
      m.d.forEach((c, i) => (direction[i]! += c));
    }
    const delta = engineLightDelta(
      { width: 1, height: 1, exists: new Uint8Array([1]), z: new Uint8Array(1) },
      lighting,
      [{ ...source, position: [128 - d * 256, 128, 0] }],
      lighting.ambient,
    );
    expect(lightLuminance(irradianceKnee(reconstructLight(s, direction, [0, 1, 0]), 1))).toBeCloseTo(
      lightLuminance([delta[0]!, delta[1]!, delta[2]!]),
      8,
    );
    expect(nearestLampHead(source, [{ x: 2, z: 0, height: 4 }])).toBe(1.25);
    expect(nearestLampHead(source, [{ x: 0.5, z: 0, height: 2 }])).toBe(2);
  });
  it('keeps the idle Tesla dark, preserves reactor cores and building damage/source gain', () => {
    const building = {
      id: 1,
      uid: 1,
      kind: 'building' as const,
      type: 'NAPOWR',
      x: 256,
      y: 256,
      z: 0,
      health: 1,
      buildState: 'done' as const,
    };
    const choose = (type: string, gain = 1, health = 1) =>
      buildingLightSources(
        [{ ...building, type, health }],
        { sourceLights: gain, night: 1, realLight: true },
        () => [3, 2],
        () => false,
      )[0]!;
    const reactor = choose('NAPOWR');
    expect(reactor.orb).toBeDefined();
    expect(buildingPhysicalLight('TESLA', choose('TESLA'))).toBeNull();
    const a = buildingPhysicalLight('NAPOWR', reactor)!;
    const b = buildingPhysicalLight('NAPOWR', choose('NAPOWR', 0.5))!;
    rgbClose(b.intensity, a.intensity.map((c) => c * 0.5) as [number, number, number]);
    const damaged = buildingPhysicalLight('NAPOWR', choose('NAPOWR', 1, 0.2))!;
    expect(lightLuminance(damaged.intensity)).toBeLessThan(lightLuminance(a.intensity));
    expect(a.dayFactor).toBe(0);
  });
  it('toggles an opted-in view without replacing state or resetting its frame clock', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const view = Object.assign(Object.create(LiveView.prototype), {
      hd: { nodes },
      current: null,
      engineLightKey: 'x',
      lightKey: 'x',
    });
    expect(view.setRealLight(false)).toBe(true);
    expect(nodes.realLight!.enabled.value).toBe(0);
    expect(view.setRealLight(true)).toBe(true);
    expect(nodes.realLight!.enabled.value).toBe(1);
    const hd = Object.assign(Object.create(HdRenderer.prototype), {
      nodes,
      settings: { relight: 0, occlusion: 0, specular: 0, glass: 0 },
    });
    expect(hd.materialsWanted()).toBe(true);
    const defaultNodes = new AtmosphereNodes();
    view.hd.nodes = defaultNodes;
    expect(view.setRealLight(true)).toBe(false);
    nodes.dispose();
    defaultNodes.dispose();
  });
  it('prints reproducible noon-unit calibration profiles for the round report', () => {
    const fire = policyLight(REAL_LIGHT_POLICIES.fire, [0, 0.4, 0]);
    const explosion = policyLight(animLightPolicy('EXPLOLRG')!, [0, animLightPolicy('EXPLOLRG')!.height, 0]);
    const tesla = policyLight(REAL_LIGHT_POLICIES.tesla, [-0.5, 0.5, 0], [0.5, 0.5, 0]);
    const lamp = engineLampLights(
      { position: [0, 0, 0], radius: 2000, intensity: 0.2, tint: [0.1, 0.1, 0.05] },
      lighting,
    )[0]!;
    const rounded = (n: number) => Number(n.toFixed(6));
    console.log(
      JSON.stringify({
        fireGround: [0.5, 1, 2, 3].map((d) => rounded(lightLuminance(irradiance(fire, d)))),
        reaches: {
          fire: rounded(lightReach(fire)),
          explosion: rounded(lightReach(explosion)),
          teslaOneCell: rounded(lightReach(tesla)),
          lamp: rounded(lightReach(lamp)),
        },
        fireColour: LIGHT_COLOURS.fire,
      }),
    );
  });
});
