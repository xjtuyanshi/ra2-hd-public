import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  Color,
  Mesh,
  MeshBasicNodeMaterial,
  PlaneGeometry,
  Vector3,
  type InstancedBufferAttribute,
  type WebGPURenderer,
} from 'three/webgpu';
import { float, positionWorld, vec3 } from 'three/tsl';
import { AtmosphereNodes, localSurfaceLight } from '../src/atmosphere';
import { AmbientContactField, ambientContactAt, contactHash } from '../src/ambientContact';
import { LightField, SHAPE_POOL } from '../src/lightField';
import { LightCost } from '../src/lightCost';
import {
  RealLightField,
  RealLightNodes,
  irradianceKnee,
  objectIrradianceKnee,
  lightMoment,
  lightLuminance,
  lightReach,
  reconstructLight,
  occlusionTransmission,
  type PhysicalLight,
  type Rgb,
} from '../src/realLight';
import {
  REAL_LIGHT_POLICIES,
  animLightPolicy,
  engineLampLights,
  nativeBeamLight,
  policyLight,
  ra2LightOccluders,
} from '../src/ra2/realLights';
import { MapLightGrade } from '../src/ra2/mapLighting';
import { Ra2NightDressing } from '../src/ra2/nightDressing';
import { LiveView, type LiveSnapshot } from '../src/live/liveView';
import { IsoCamera } from '../src/world';
import { faithfulToneMapRgb } from '../src/post';
import { EMISSIVE_BUDGET, emissiveSheathColour } from '../src/realEmissive';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';

const point: PhysicalLight = { position: [64, 0.5, 64], radius: 0.2, intensity: [1, 0.74, 0.52], dayFactor: 1 };
const renderer = () =>
  ({
    autoClear: false,
    getRenderTarget: () => null,
    getMRT: () => null,
    getClearAlpha: () => 1,
    getClearColor: (c: Color) => c.set(0),
    setClearColor: vi.fn(),
    setClearAlpha: vi.fn(),
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
    initTexture: vi.fn(),
  }) as unknown as WebGPURenderer;
const hue = (v: Rgb) => {
  const hi = Math.max(...v),
    lo = Math.min(...v),
    d = hi - lo;
  if (d === 0) return 0;
  return (
    ((hi === v[0] ? (v[1] - v[2]) / d : hi === v[1] ? (v[2] - v[0]) / d + 2 : (v[0] - v[1]) / d + 4) * 60 + 360) % 360
  );
};
afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('round 4 source fidelity and receiver bounds', () => {
  it('keeps positive and negative engine grade bytes at every presentation weight', () => {
    const grid = { width: 4, height: 4, z: new Uint8Array(16), exists: new Uint8Array(16).fill(1) };
    const sources = [
      {
        position: [128, 128, 0] as [number, number, number],
        radius: 2000,
        intensity: 0.4,
        tint: [0.2, 0.1, 0.05] as [number, number, number],
      },
      {
        position: [384, 384, 0] as [number, number, number],
        radius: 1000,
        intensity: -0.1,
        tint: [0, 0, 0] as [number, number, number],
      },
    ];
    const grade = new MapLightGrade(
      grid,
      { ambient: 0.55, red: 0.4, green: 0.4, blue: 0.7, ground: 0, level: 0 },
      sources,
    );
    for (const weight of [0, 0.25, 1])
      for (const share of [0, 0.4, 1]) {
        grade.setWeight(weight);
        grade.setSourceWeight(share);
        const baseline = (grade.texture.image.data as Uint8Array).slice();
        grade.setPhysicalSources(true);
        expect(grade.texture.image.data).toEqual(baseline);
        grade.setPhysicalSources(false);
        expect(grade.texture.image.data).toEqual(baseline);
      }
    grade.dispose();
  });
  it('keeps full receiver graphs within the sixteen-texture stage budget', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      field = new RealLightField(2, 2, nodes.realLight!);
    field.setGround(new Float32Array(4));
    field.setOccluders(ra2LightOccluders(2, 2, new Uint8Array(4), []));
    const scene = receiverScene(nodes);
    for (const mesh of scene.meshes) {
      const code = shaderBytes(mesh),
        names = new Set([...code.matchAll(/var\s+(\w+)\s*:\s*texture_2d/g)].map((m) => m[1]));
      expect(names.size).toBeGreaterThan(0);
      expect(names.size).toBeLessThanOrEqual(16);
    }
    field.dispose();
    scene.dispose();
  });
  it('retains every night dressing switch, map pool update and pole visibility under the flag', () => {
    for (const on of [true, false]) {
      const nodes = new AtmosphereNodes({ realLight: true });
      nodes.realLight!.enabled.value = Number(on);
      const rebuild = vi.fn(),
        update = vi.fn(),
        poles = { visible: false };
      const dressing = Object.assign(Object.create(Ra2NightDressing.prototype), {
        context: { nodes },
        lampsEnabled: () => true,
        rebuild,
        mapLights: { update },
        version: 1,
        poles,
      });
      const frame = { settings: { lamps: 'on', lights: true, windows: true }, night: 1, profile: { kind: 'night' } };
      dressing.update(frame);
      expect(rebuild.mock.calls[0]!.slice(1)).toEqual([true, true, true, true]);
      expect(update.mock.calls[0]!.at(-1)).toBe(true);
      expect(poles.visible).toBe(true);
      nodes.dispose();
    }
  });
  it('gates all legacy dynamic writers and restores buffered event ages for same-frame A/B', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      f = new LightField(8, 8, { nodes }),
      r = renderer();
    f.setStatic([{ x: 1, z: 1, radius: 2, shape: SHAPE_POOL, colour: [1, 0.8, 0.6] }]);
    f.source('window', 0, { x: 2, z: 2, radius: 2, colour: [0.5, 0.3, 0.1] });
    f.pulse(0, new Vector3(1, 0, 1), [1, 0.7, 0.4], 2, 1, 0.3, 0, 'explosion');
    f.pulse(0, new Vector3(1, 0, 1), [0.4, 0.4, 1], 1, 1, 0.2, 0, 'flash');
    f.keep('fire-system', 0, new Vector3(1, 0, 1), [1, 0.6, 0.3], 2, 1, 0, 0.6);
    expect(f.stats(0.1)).toMatchObject({ static: 1, explosion: 0, flash: 0, kept: 0, sources: 1 });
    f.update(0.1, r, 1);
    const attrs = (f as unknown as { dynamics: { mesh: { count: number } } }).dynamics;
    expect(attrs.mesh.count).toBe(1); // only the unchanged static window source
    nodes.realLight!.enabled.value = 0;
    f.update(0.1, r, 1);
    expect(attrs.mesh.count).toBe(4);
    expect(f.stats(0.1)).toMatchObject({ explosion: 1, flash: 1, kept: 1 });
    nodes.realLight!.enabled.value = 1;
    f.update(0.1, r, 1);
    expect(attrs.mesh.count).toBe(1);
    f.dispose();
    nodes.dispose();
  });
  it.each([1.25, 4, 10])(
    'keeps lamp accents nonnegative, at most 0.12 at one cell, and zero by 2.5 (head=%s)',
    (head) => {
      const lights = engineLampLights(
        { position: [0, 0, 0], radius: 5000, intensity: 1, tint: [1, 0.7, 0.3] },
        { ambient: 0.55, red: 0.4, green: 0.4, blue: 0.7, ground: 0, level: 0 },
        head,
      );
      const e = lights.map((l) => lightMoment(l, [1, 0, 0]));
      const sum = [0, 0, 0] as [number, number, number],
        d = [0, 0, 0] as [number, number, number];
      for (const m of e)
        for (let i = 0; i < 3; i++) {
          sum[i]! += m.s[i]!;
          d[i]! += m.d[i]!;
        }
      expect(lightLuminance(reconstructLight(sum, d, [0, 1, 0]))).toBeLessThanOrEqual(0.12000001);
      for (const l of lights) {
        expect(l.intensity.every((c) => c >= 0)).toBe(true);
        expect(lightMoment(l, [2.5, 0, 0]).s).toEqual([0, 0, 0]);
      }
    },
  );
  it('uses the actual scene-relative ground knee and remaining object headroom', () => {
    for (const ambient of [0.1, 0.3, 0.4, 0.5]) {
      // Ground: logarithmic knee scaled by 3x the scene ambient at night (1.2 by day); it keeps rising, never a plateau.
      expect(lightLuminance(irradianceKnee([100, 100, 100], 1, 1, ambient))).toBeCloseTo(
        3 * ambient * Math.log1p(100 / (3 * ambient)),
      );
      expect(lightLuminance(irradianceKnee([100, 100, 100], 0, 1, ambient))).toBeCloseTo(1.2 * Math.log1p(100 / 1.2));
      expect(lightLuminance(irradianceKnee([2, 2, 2], 1, 1, ambient))).toBeGreaterThan(
        lightLuminance(irradianceKnee([1, 1, 1], 1, 1, ambient)),
      );
      expect(lightLuminance(objectIrradianceKnee([100, 100, 100], ambient)) + ambient).toBeLessThanOrEqual(0.8000001);
    }
    expect(objectIrradianceKnee([3, 2, 1], 0.8)).toEqual([0, 0, 0]);
    const albedo = [0.6, 0.08, 0.06] as const,
      p = animLightPolicy('EXPLOLRG')!;
    const m = lightMoment(policyLight(p, [0, p.height, 0]), [1, 0, 0]);
    const local = objectIrradianceKnee(reconstructLight(m.s, m.d, [0, 1, 0]), 0.4);
    const lit = faithfulToneMapRgb(albedo.map((c, i) => c * (0.4 + local[i]!)) as [number, number, number]);
    expect(Math.max(...lit)).toBeLessThan(0.98);
    expect(Math.abs(hue(lit) - hue(albedo))).toBeLessThan(10);
    // An exposed projection page must use its noon art normalization before assigning local diffuse headroom.
    const page = albedo.map((c) => c / 0.21),
      normalized = page.map((c) => c * 0.21);
    expect(faithfulToneMapRgb(normalized.map((c, i) => c * (0.4 + local[i]!)) as [number, number, number])).toEqual(
      lit,
    );
  });
  it('enforces source-sized windows, captured bolt/Prism colours and short explosion/muzzle timing', () => {
    for (const [name, cap] of Object.entries({
      tesla: 2,
      psychic: 2,
      muzzle: 1.2,
      prism: 1.8,
      fire: 2.2,
      rad: 1.5,
      temporal: 1.5,
      sonic: 1.5,
      magnetron: 1.5,
    })) {
      const p = REAL_LIGHT_POLICIES[name as keyof typeof REAL_LIGHT_POLICIES];
      const l = policyLight(p, [0, p.height, 0]);
      expect(lightReach(l)).toBeLessThanOrEqual(cap);
      expect(lightMoment(l, [cap, 0, 0]).s).toEqual([0, 0, 0]);
    }
    expect(REAL_LIGHT_POLICIES.tesla.ground).toBe(1.2);
    expect(REAL_LIGHT_POLICIES.muzzle.ground).toBe(0.25);
    expect(animLightPolicy('VTMUZZLE')?.hold ?? REAL_LIGHT_POLICIES.muzzle.hold).toBe(2 / 62.5);
    // Explosion light lasts as long as its art (EXPLOLRG: 21 frames), never longer.
    const large = animLightPolicy('EXPLOLRG')!;
    expect(large.hold! + large.fade!).toBeLessThanOrEqual(Math.max(0.37, 21 / 62.5) + 1e-9);
    expect(large.reach).toBeLessThanOrEqual(4);
    expect(animLightPolicy('EXPLOSML')!.reach).toBeGreaterThanOrEqual(1.5);
    for (const [kind, colour, cap] of [
      ['bolt', [82, 80, 255], 0.55],
      ['bolt', [190, 90, 255], 0.55],
      ['laser', [255, 10, 0], 0.5],
    ] as const) {
      const l = nativeBeamLight({
        kind,
        source: { x: 0, y: 0, z: 104 },
        target: { x: 256, y: 0, z: 104 },
        colour: [...colour],
      });
      expect(1 - Math.min(...l.intensity) / Math.max(...l.intensity)).toBeCloseTo(cap);
      expect(hue(l.intensity)).toBeCloseTo(
        hue(colour.map((c) => new Color().setRGB(c / 255, c / 255, c / 255, 'srgb').r) as [number, number, number]),
        5,
      );
    }
    expect(EMISSIVE_BUDGET.coreWidth).toBeLessThanOrEqual(0.2);
    const sheath = emissiveSheathColour([0.08, 0.075, 1], 10),
      display = faithfulToneMapRgb(sheath);
    expect(Math.max(...sheath)).toBeLessThanOrEqual(0.8);
    expect(hue(display)).toBeCloseTo(hue(sheath));
    expect(hue(display)).toBeCloseTo(hue([0.08, 0.075, 1]), 10);
    expect(display[2]).toBeGreaterThan(10 * display[0]);
  });
  it('reduces every terrain added term to its original expression with zero dynamic irradiance', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const field = new RealLightField(2, 2, nodes.realLight!);
    const legacy = vec3(0.2, 0.1, 0.05),
      albedo = vec3(0.6, 0.3, 0.2);
    const graph = localSurfaceLight(
      legacy,
      albedo,
      nodes,
      positionWorld.xz,
      positionWorld.y,
      vec3(0, 1, 0),
      undefined,
      1,
      float(0.4),
    );
    const inputs: unknown[] = [];
    graph.traverse((n) => {
      if ('value' in n) inputs.push(n.value);
    });
    expect(inputs).not.toContain(nodes.realLight!.ambientContact.value);
    expect(inputs).not.toContain(nodes.realLight!.accentGround.value);
    const m = reconstructLight([0, 0, 0], [0, 0, 0], [0, 1, 0]),
      e = irradianceKnee(m, 1, 1, 0.4);
    expect([0.2, 0.1, 0.05].map((c, i) => c + ([0.6, 0.3, 0.2][i]! * e[i]!) / Math.PI)).toEqual([0.2, 0.1, 0.05]);
    const scene = receiverScene(nodes);
    for (const mesh of scene.meshes.slice(3)) {
      const colour = (mesh.material as MeshBasicNodeMaterial).colorNode;
      const values: unknown[] = [];
      colour?.traverse((n) => {
        if ('value' in n) values.push(n.value);
      });
      expect(values).not.toContain(nodes.realLight!.ambientContact.value);
      expect(shaderBytes(mesh)).not.toMatch(/\b(?:NaN|undefined)\b/);
    }
    field.dispose();
    scene.dispose();
  });
  it('does not turn terrain steps, craters or units into hard occluders', () => {
    const empty = ra2LightOccluders(4, 2, [0, 3, 2, 1, 2, 0, 3, 1], []);
    expect([...empty.heights].every((h) => h === 0)).toBe(true);
    expect(
      occlusionTransmission(
        [0.5, 0, 0.5],
        [3.5, 0.6, 0.5],
        (x, z) => empty.heights[Math.floor(z * 2) * 8 + Math.floor(x * 2)] ?? 0,
      ),
    ).toBe(1);
    const b = ra2LightOccluders(4, 2, [0, 3, 2, 1, 2, 0, 3, 1], [{ x: 1, z: 0, width: 1, depth: 1, top: 3 }]);
    expect(
      occlusionTransmission(
        [0.5, 0, 0.5],
        [3.5, 0.6, 0.5],
        (x, z) => b.heights[Math.floor(z * 2) * 8 + Math.floor(x * 2)] ?? 0,
      ),
    ).toBe(0.1);
  });
});

describe('round 4 cost and capture tooling', () => {
  it('quantises contacts and redraws only old/new dirty footprints once per logic frame, retaining overlap', () => {
    const field = new AmbientContactField(128, 128),
      a = {},
      b = {},
      p = { x: 20, z: 20, width: 1, depth: 1 };
    expect(field.density).toBe(4);
    expect(field.texture.image.width).toBeLessThanOrEqual(1024);
    field.set(a, [p]);
    field.set(b, [{ ...p, x: 21 }]);
    field.update(() => 0, 1);
    const stringify = vi.spyOn(JSON, 'stringify');
    field.set(a, [{ ...p, x: 20.001 }]);
    expect(stringify).not.toHaveBeenCalled();
    expect(contactHash([p])).toBe(contactHash([{ ...p, x: 20.001 }]));
    field.set(a, [{ ...p, x: 40 }]);
    const version = field.texture.version;
    expect(field.update(() => 0, 1)).toBe(false);
    expect(field.texture.version).toBe(version);
    expect(field.update(() => 0, 2)).toBe(true);
    expect(field.rasterTexels).toBeLessThan((512 * 512) / 10);
    field.delete(a);
    field.update(() => 0, 3);
    const { data, width: w, height: h } = field.texture.image;
    for (let z = 0; z < h; z++)
      for (let x = 0; x < w; x++)
        expect(data[z * w + x]).toBe(Math.round(255 * ambientContactAt({ ...p, x: 21 }, (x + 0.5) / 4, (z + 0.5) / 4)));
    field.dispose();
    const huge = new AmbientContactField(600, 300);
    expect(huge.texture.image.width).toBeLessThanOrEqual(1024);
    huge.dispose();
  });
  it('scissors dynamic splats to the viewport, caches static accents and skips unchanged frames/short-light marches', () => {
    const n = new RealLightNodes(),
      f = new RealLightField(128, 128, n),
      r = renderer(),
      camera = new IsoCamera();
    camera.target.set(64, 0, 64);
    camera.resize(600, 400);
    camera.update();
    f.setView(camera.camera);
    f.setStatic([{ ...point, position: [5, 1, 5] }]);
    f.put('kept', 'near', 0, point, 0, 0, 0.5);
    f.put('kept', 'outside', 0, { ...point, position: [2, 0.5, 2] }, 0, 0, 0.5);
    expect(f.update(0, r, 1)).toBe(true);
    expect(f.staticRenders).toBe(1);
    const internals = f as unknown as { attributes: InstancedBufferAttribute[][]; meshes: { count: number }[] };
    expect(internals.meshes[0]!.count).toBe(1);
    expect(f.targets[0]!.scissorTest).toBe(true);
    expect(f.targets[0]!.scissor.z).toBeLessThan(f.targets[0]!.width);
    expect(n.cost.texelsSplattedEstimate).toBeLessThan((f.targets[0]!.width * f.targets[0]!.height) / 10);
    f.put('kept', 'near', 0.1, point, 0, 0, 0.5);
    f.put('kept', 'outside', 0.1, { ...point, position: [2, 0.5, 2] }, 0, 0, 0.5);
    expect(f.update(0.1, r, 1)).toBe(false);
    expect(n.cost.lightsSplatted).toBe(0);
    f.pulse('flash', 0.1, point, 0.05, 0.1);
    expect(f.update(0.12, r, 1)).toBe(true);
    expect(internals.attributes[0]![3]!.getY(0)).toBe(0);
    f.put('kept', 'tiny', 0.12, { ...point, reach: 1.2 }, 0, 0, 1);
    f.update(0.13, r, 1);
    expect(internals.attributes[0]![3]!.getY(2)).toBe(0);
    expect(f.staticRenders).toBe(1);
    f.dispose();
    n.dispose();
  });
  it('reports separate moving CPU averages/counts and disables instrumentation in production', () => {
    vi.spyOn(performance, 'now').mockReturnValue(2000);
    const cost = new LightCost();
    cost.record('contactRasterUpload', 2);
    cost.record('fieldUpdate', 4);
    cost.record('emissiveBuild', 6);
    cost.record('physicalLampUpdate', 8);
    cost.finishFrame(true, 2000);
    cost.finishFrame(false, 2000);
    expect(cost.snapshot()).toMatchObject({
      samples: 2,
      fieldRendersPerSecond: 1,
      msPerFrame: { contactRasterUpload: 1, fieldUpdate: 2, emissiveBuild: 3, physicalLampUpdate: 4 },
    });
    vi.stubEnv('DEV', false);
    const off = new LightCost();
    expect(off.measure('fieldUpdate', () => 7)).toBe(7);
    off.finishFrame(true);
    expect(off.snapshot().samples).toBe(0);
  });
  it('holds the last accepted snapshot, drops intermediate publications and resumes the newest without touching the VM', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      publish = vi.fn();
    const view = Object.assign(Object.create(LiveView.prototype), {
      hd: { nodes },
      publish,
      withWarmList: (s: LiveSnapshot) => s,
    });
    const first = { frame: 1 },
      middle = { frame: 2 },
      last = { frame: 3 };
    view.push(first);
    expect(publish).toHaveBeenLastCalledWith(first, expect.any(Number));
    expect(view.holdLight(true)).toBe(true);
    view.push(middle);
    view.push(last);
    expect(publish).toHaveBeenCalledTimes(1);
    expect(nodes.realLight!.logicFrame).toBe(1);
    view.setRealLight(false);
    view.setRealLight(true);
    expect(publish).toHaveBeenCalledTimes(1);
    expect(view.holdLight(false)).toBe(true);
    expect(publish).toHaveBeenCalledTimes(2);
    expect(publish).toHaveBeenLastCalledWith(last, expect.any(Number));
    vi.stubEnv('DEV', false);
    expect(view.holdLight(true)).toBe(false);
    view.push({ frame: 4 });
    expect(publish).toHaveBeenCalledTimes(3);
    nodes.dispose();
  });
});

describe('round 4 live calibration', () => {
  it('merges co-located continuous lights so a burning vehicle does not sum into a flat pool', async () => {
    const { mergeNearbyLights, lightLuminance } = await import('../src/realLight');
    const fire = (x: number) => ({
      position: [x, 0.4, 0] as [number, number, number],
      radius: 0.25,
      intensity: [1, 0.7, 0.5] as [number, number, number],
    });
    const merged = mergeNearbyLights([0, 0.2, 0.4, 0.6, 0.8, 5].map((x, i) => ({ key: i, light: fire(x) })));
    expect(merged).toHaveLength(2);
    expect(lightLuminance(merged[0]!.light.intensity)).toBeLessThanOrEqual(1.6 * lightLuminance([1, 0.7, 0.5]) + 1e-9);
    expect(merged[1]!.light.position[0]).toBe(5);
  });
});
