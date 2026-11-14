import { describe, expect, it, vi } from 'vitest';
import { OneFactor, Color, Mesh, RenderTarget, Vector3, type WebGPURenderer } from 'three/webgpu';
import { uniform } from 'three/tsl';
import { RealLightField, RealLightNodes, lightLuminance, occlusionTransmission } from '../src/realLight';
import {
  EMISSIVE_BUDGET,
  RealEmissiveLayer,
  capOverlaySource,
  emissiveBudget,
  emissiveCoreTint,
  emissiveEnergy,
  emissiveSheathColour,
  emissiveProfile,
  type EmissivePrimitive,
} from '../src/realEmissive';
import {
  animEmission,
  eventExposureKick,
  firingHead,
  largeLightEvent,
  physicalBuildingKey,
  physicalLampKey,
  physicalExposureFlash,
} from '../src/ra2/realEmission';
import { BeamLayer, BeamSegmentBuffer, EffectsLayer } from '../src/effects';
import { AtmosphereNodes } from '../src/atmosphere';
import { LiveBeams, type LiveRadBeam } from '../src/live/beams';
import { LiveEffectsDriver } from '../src/live/effects';
import { LiveView } from '../src/live/liveView';
import { buildingLightOverlay, buildingLightPlacements } from '../src/live/buildingLights';
import { EFFECT_TABLE } from '../src/live/effectTable';
import { ra2SceneLighting } from '../src/ra2/sceneLighting';
import { shaderBytes } from './fixtures/realLightScene';
const primitive: EmissivePrimitive = {
  source: 'a',
  from: new Vector3(),
  to: new Vector3(2, 0, 0),
  width: 0.06,
  colour: [0.1, 0.3, 1],
  gain: 1,
};
const renderer = () =>
  ({
    getRenderTarget: () => null,
    getMRT: () => null,
    getClearAlpha: () => 1,
    getClearColor: (c: Color) => c.set(0),
    setClearColor: vi.fn(),
    autoClear: false,
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
  }) as unknown as WebGPURenderer;
describe('absolute-height occlusion march', () => {
  it('keeps open and elevated rays clear; blocks walls at a common absolute elevation', () => {
    const receiver = [0, 3, 0] as const,
      source = [4, 4, 0] as const;
    expect(occlusionTransmission(receiver, source, () => 3)).toBe(1);
    expect(occlusionTransmission(receiver, source, (x) => (x >= 1 && x <= 3 ? 5 : 3))).toBe(0.1);
    expect(occlusionTransmission(receiver, source, () => 20, false)).toBe(1);
    expect(occlusionTransmission([0, 0, 0], [4, 1, 0], (x) => (x >= 1 && x <= 3 ? 2 : 0))).toBe(0.1);
  });
  it('has the specified smooth attenuation, source exclusion, short-ray and coincident limits', () => {
    expect(occlusionTransmission([0, 1, 0], [4, 1, 0], (x) => (x === 2 ? 1.2 : 0))).toBeCloseTo(0.5);
    expect(occlusionTransmission([0, 1, 0], [4, 1, 0], (x) => (x > 3.65 ? 10 : 0))).toBe(1);
    expect(occlusionTransmission([0, 1, 0], [0.3, 1, 0], () => 10)).toBe(1);
    expect(occlusionTransmission([0, 1, 0], [0, 1, 0], () => 10)).toBe(1);
  });
});
describe('emissive budgets and finite graph', () => {
  it('caps a complete source, soft caps the frame, retains ratios and subdivision energy', () => {
    const a = emissiveEnergy(primitive);
    const b =
      emissiveEnergy({ ...primitive, to: new Vector3(1, 0, 0) }) +
      emissiveEnergy({ ...primitive, from: new Vector3(1, 0, 0) });
    expect(a).toBeCloseTo(b);
    expect(emissiveEnergy({ ...primitive, gain: 0 })).toBe(0);
    expect(emissiveEnergy({ ...primitive, disc: true })).toBeGreaterThan(0);
    const budget = emissiveBudget(
      new Map([
        ['a', 100],
        ['b', 50],
      ]),
      6,
      10,
    );
    expect(budget.scales.get('a')).toBe(0.06);
    expect(budget.scales.get('b')).toBe(0.12);
    expect(budget.total).toBeLessThan(10);
    expect(budget.gain).toBeGreaterThan(0);
    expect(emissiveBudget(new Map()).gain).toBe(1);
    expect(emissiveBudget(new Map([['a', -1]])).total).toBe(0);
  });
  it('uses a narrow white core, coloured sheath and a soft 2.5-width tail', () => {
    expect(emissiveProfile(0)).toEqual({ core: 1, sheath: 1, glow: 0.25 });
    expect(emissiveProfile(0.25).core).toBe(0);
    expect(emissiveProfile(1).sheath).toBe(0);
    expect(emissiveProfile(1.5).glow).toBeGreaterThan(0);
    expect(emissiveProfile(2.5)).toEqual({ core: 0, sheath: 0, glow: 0 });
    for (let r = 0; r <= 3; r += 0.01) expect(Object.values(emissiveProfile(r)).every(Number.isFinite)).toBe(true);
  });
  it('keeps non-core sheath/glow below both bloom thresholds even at peak flicker', () => {
    const original = [0.4, 1, 0.6] as const;
    const capped = emissiveSheathColour(original, 10);
    expect(Math.max(...capped)).toBeCloseTo(0.8);
    expect(capped[0] / capped[1]).toBeLessThan(original[0] / original[1]);
    expect(emissiveSheathColour(original, 0)).toEqual([0, 0, 0]);
    expect(emissiveSheathColour(original, -1)).toEqual([0, 0, 0]);
    expect(emissiveSheathColour(original, 0.1)).toEqual(original.map((c) => Math.max(0, 1 - (1 - c) * 1.2) * 0.1));
  });
  it('caps each reactor sphere and all its filaments as one source, without mutating art policy', () => {
    const source = buildingLightPlacements('NAPOWR', { x: 0, y: 0, z: 0, uid: 1, id: 1 }, [3, 2]);
    const original = buildingLightOverlay(source, 0, 0, {
      right: new Vector3(1, 0, 0),
      up: new Vector3(0, 1, 0),
      toCamera: new Vector3(0, 0, 1),
    });
    const before = JSON.stringify(original);
    const capped = capOverlaySource(
      original.discs.map((d) => ({ ...d, brightness: 1000 })),
      original.strokes,
      42,
    );
    expect(capped.energy).toBeLessThanOrEqual(6);
    expect(JSON.stringify(original)).toBe(before);
    expect(capOverlaySource([], [], 42).energy).toBe(0);
  });
  it('builds the additive MRT graph, including zero-length discs and camera-parallel strokes', () => {
    const layer = new RealEmissiveLayer(8),
      target = new RenderTarget(2, 2, { count: 2 });
    target.textures[0]!.name = 'output';
    target.textures[1]!.name = 'emissive';
    layer.set('a', [
      primitive,
      { ...primitive, from: new Vector3(), to: new Vector3(), disc: true },
      { ...primitive, to: new Vector3(0, 0, 1) },
    ]);
    layer.update(0, true);
    expect(layer.mesh.count).toBe(3);
    expect(layer.mesh.material.blendSrc).toBe(OneFactor);
    expect(layer.mesh.material.blendDst).toBe(OneFactor);
    expect(layer.totalEnergy).toBeLessThan(EMISSIVE_BUDGET.frame);
    const code = shaderBytes(layer.mesh, target);
    expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(code).toContain('smoothstep(');
    const data = (layer as unknown as { data: { array: Float32Array } }).data.array;
    expect([...data].every(Number.isFinite)).toBe(true);
    layer.update(1, false);
    expect(layer.root.visible).toBe(false);
    layer.set('a', []);
    layer.update(2, true);
    expect(layer.mesh.count).toBe(0);
    layer.dispose();
    target.dispose();
  });
});
describe('RA2 source lifetime and shape preservation', () => {
  it('adds short table-radius blast/muzzle cores and no added fire cores', () => {
    const anim = { id: 1, uid: 2, type: 'EXPLOLRG', x: 256, y: 512, z: 104 };
    expect(animEmission(anim, 0, 0)!.width).toBe(2 * EFFECT_TABLE.EXPLOLRG!.light![3]);
    expect(animEmission(anim, 0.15, 0.15)).toBeNull();
    expect(animEmission({ ...anim, type: 'GUNFIRE' }, 0, 0)).not.toBeNull();
    expect(animEmission({ ...anim, type: 'GUNFIRE' }, 0.1, 0.1)).toBeNull();
    const fire = { ...anim, type: 'FIRE02' };
    expect(animEmission(fire, 5, 5)).toBeNull();
    expect(animEmission({ ...fire, attachedTo: { id: 1, uid: 1, kind: 'vehicle' } }, 5, 5)).toBeNull();
    expect(animEmission({ ...anim, type: 'UNKNOWN' }, 0, 0)).toBeNull();
    expect(animEmission(fire, -1, 0)).toBeNull();
  });
  it('uses actual firing endpoints near the head, never idle/charge frames', () => {
    expect(firingHead(new Vector3(), [])).toBe(false);
    expect(firingHead(new Vector3(), [new Vector3(0.85, -1.2, 0)])).toBe(true);
    expect(firingHead(new Vector3(), [new Vector3(0, -1.51, 0)])).toBe(false);
    expect(firingHead(new Vector3(), [new Vector3(0.95, 0, 0)])).toBe(false);
  });
  it('caps the 150 ms sky-path kick in EV and recognizes only large events', () => {
    expect(eventExposureKick(0, 0)).toBeCloseTo(2 ** 0.25 - 1);
    expect(eventExposureKick(0, 1)).toBeCloseTo(Math.SQRT2 - 1);
    expect(eventExposureKick(0.075, 1)).toBeLessThan(eventExposureKick(0, 1));
    expect(eventExposureKick(-1, 1)).toBe(0);
    expect(eventExposureKick(0.15, 1)).toBe(0);
    for (const name of ['NUKETO', 'PDFXLOC', 'PDFXCLD', 'WCLBOLT1', 'WCLBOLT3'])
      expect(largeLightEvent(name)).toBe(true);
    expect(largeLightEvent('FIRE02')).toBe(false);
    const collapse = Object.keys(EFFECT_TABLE).find((key) => EFFECT_TABLE[key]!.c === 'collapse')!;
    expect(largeLightEvent(collapse)).toBe(true);
    const live = {
      ambient: 1,
      ground: 0,
      level: 0,
      tint: [1, 1, 1] as [number, number, number],
      event: null,
      override: null,
    };
    expect(physicalExposureFlash(ra2SceneLighting(null, live, 0), 0.4)!.skyFlash).toBe(0.4);
    expect(physicalExposureFlash(null, 0)).toBeNull();
    expect(physicalExposureFlash(null, 1)!.skyFlash).toBe(Math.SQRT2 - 1);
    expect(ra2SceneLighting(null, live, 0, 0.4)!.skyFlash).toBe(0);
  });
  it('publishes exact generated bolt vertices and original laser colours without changing baseline buffers', () => {
    const beams = [
      { from: new Vector3(), to: new Vector3(3, 1, 0), colour: new Vector3(0.1, 0.2, 1), bolt: true, seed: 42 },
      {
        from: new Vector3(0, 1, 0),
        to: new Vector3(2, 1, 0),
        colour: new Vector3(1, 0.1, 0),
        outerColour: new Vector3(0.2, 0, 0),
        bolt: false,
        seed: 10,
      },
    ];
    const base = new BeamSegmentBuffer();
    base.set(beams);
    const emission = new RealEmissiveLayer(),
      layer = new BeamLayer(uniform(new Vector3(1, 0, 0)), uniform(new Vector3(0, 1, 0)));
    layer.realEmissive = emission;
    layer.setBeams(beams);
    emission.update(0, true);
    const primitives = (emission as unknown as { channels: Map<string, EmissivePrimitive[]> }).channels.get('native')!;
    expect(primitives).toHaveLength(base.count);
    for (let i = 0; i < base.count; i++) {
      expect(primitives[i]!.from.toArray()).toEqual([...base.from.array.slice(i * 3, i * 3 + 3)]);
      expect(primitives[i]!.to.toArray()).toEqual([...base.to.array.slice(i * 3, i * 3 + 3)]);
    }
    // The core is a pale tint of the beam's own colour (a white core read paler than the original).
    expect(primitives.at(-1)!.hot).toEqual(emissiveCoreTint([1, 0.1, 0]));
    expect(primitives.at(-1)!.colour).toEqual([1, 0.1, 0]);
    layer.setBeams([]);
    emission.update(1, true);
    expect(emission.mesh.count).toBe(0);
    emission.dispose();
    layer.mesh.geometry.dispose();
    (layer.mesh.material as { dispose(): void }).dispose();
  });
  it('retains auxiliary ribbon/ring/arch vertex buffers, shroud and publication removal gates', () => {
    const aux = new RealEmissiveLayer(),
      right = uniform(new Vector3(1, 0, 0)),
      up = uniform(new Vector3(0, 1, 0)),
      now = uniform(0);
    const base = new LiveBeams(right, up, now),
      lit = new LiveBeams(right, up, now, aux);
    const beam: LiveRadBeam = {
      id: 1,
      type: 'temporal',
      source: { x: 0, y: 0, z: 0 },
      target: { x: 512, y: 256, z: 104 },
      color: [0, 200, 255],
      period: 10,
      amplitude: 20,
      age: 2,
      sustained: true,
      progress: 0.2,
    };
    const effects = {
      anims: [],
      bullets: [],
      particleSystems: [],
      radBeams: [beam],
      waves: [
        {
          id: 4,
          uid: 1,
          type: 'sonic' as const,
          source: beam.source,
          target: beam.target,
          intensity: 1,
          progress: 0.2,
        },
      ],
      mindControlLinks: [
        {
          id: 9,
          controller: 1,
          targetId: 2,
          source: beam.source,
          target: beam.target,
          color: [100, 0, 255] as [number, number, number],
          arcHeight: 512,
          remainingFrames: 10,
          selected: false,
        },
      ],
    };
    base.update(effects, 'faithful', () => false);
    lit.update(effects, 'faithful', () => false);
    const data = (l: LiveBeams) =>
      l as unknown as {
        from: { array: Float32Array };
        to: { array: Float32Array };
        appearance: { array: Float32Array };
      };
    for (const k of ['from', 'to', 'appearance'] as const)
      expect([...data(base)[k].array]).toEqual([...data(lit)[k].array]);
    aux.update(0, true);
    expect(aux.mesh.count).toBeGreaterThan(0);
    lit.update(null, 'faithful', () => false);
    aux.update(1, true);
    expect(aux.mesh.count).toBe(0);
    base.dispose();
    lit.dispose();
    aux.dispose();
  });
  it('removes transient cores on publication removal and preserves captured anim objects', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      field = new RealLightField(2, 2, nodes.realLight!),
      fx = new EffectsLayer(nodes),
      emission = new RealEmissiveLayer();
    fx.physicalLights = field;
    fx.realEmissive = emission;
    const driver = new LiveEffectsDriver(fx, () => false);
    driver.garnishEnabled = false;
    const effects = {
      anims: [
        { id: 1, uid: 1, type: 'NUKETO', x: 0, y: 0, z: 0 },
        { id: 2, uid: 2, type: 'FIRE02', x: 0, y: 0, z: 0, buildingAnim: true },
      ],
      bullets: [],
      particleSystems: [],
    };
    const before = JSON.stringify(effects);
    driver.push(effects, 1, 0, new Vector3());
    emission.update(0, true);
    // NUKETO's transient core only: fires add no core (animEmission), but keep their physical light.
    expect(emission.mesh.count).toBe(1);
    expect(field.pools.get('kept')?.size).toBe(1);
    expect(driver.skyFlash(0, 1)).toBeCloseTo(Math.SQRT2 - 1);
    expect(driver.skyFlash(0.15, 1)).toBe(0);
    driver.frame(0.2, new Vector3());
    emission.update(0.2, true);
    expect(emission.mesh.count).toBe(0);
    driver.push({ anims: [], bullets: [], particleSystems: [] }, 2, 0.2, new Vector3());
    emission.update(0.2, true);
    expect(emission.mesh.count).toBe(0);
    expect(JSON.stringify(effects)).toBe(before);
    emission.dispose();
    field.dispose();
    nodes.dispose();
  });
});
describe('static caching and semantic invalidation', () => {
  it('does not resplat static lamps on dynamic changes, and invalidates both layers on occluder/ground/night changes', () => {
    const nodes = new RealLightNodes(),
      field = new RealLightField(2, 2, nodes),
      r = renderer();
    const light = { position: [0, 1, 0] as const, radius: 0.1, intensity: [1, 1, 1] as const, dayFactor: 0 };
    field.setStatic([light]);
    field.update(0, r, 1);
    expect(field.staticRenders).toBe(1);
    vi.mocked(r.render).mockClear();
    field.put('kept', 'a', 0, light, 0, 0, 1);
    field.update(0.1, r, 1);
    field.update(0.2, r, 1);
    expect(field.staticRenders).toBe(1);
    for (const call of vi.mocked(r.render).mock.calls) expect(field.staticScenes).not.toContain(call[0]);
    field.setGround([2, 2, 2, 2]);
    field.update(0.3, r, 1);
    expect(field.staticRenders).toBe(2);
    field.setOccluders({ width: 4, height: 4, texelsPerCell: 2, heights: new Float32Array(16).fill(2) });
    field.update(0.4, r, 1);
    expect(field.staticRenders).toBe(3);
    field.update(0.5, r, 0);
    expect(field.staticRenders).toBe(4);
    for (const [i, scene] of field.scenes.entries())
      scene.traverse((m) => {
        if (m instanceof Mesh) expect(shaderBytes(m, field.targets[i])).not.toMatch(/\b(?:NaN|undefined)\b/);
      });
    field.dispose();
    nodes.dispose();
  });
  it('keys building membership/height/visibility and source contents, not publication or animation identity', () => {
    const b = { id: 1, uid: 1, kind: 'building', type: 'TESLA', x: 0, y: 0, z: 104, buildingFrame: 1 };
    const key = physicalBuildingKey([b], () => false, 1);
    expect(physicalBuildingKey([{ ...b, buildingFrame: 2 }], () => false, 1)).toBe(key);
    expect(physicalBuildingKey([{ ...b, z: 208 }], () => false, 1)).not.toBe(key);
    expect(physicalBuildingKey([b], () => true, 1)).not.toBe(key);
    expect(physicalBuildingKey([], () => false, 1)).not.toBe(key);
    expect(physicalBuildingKey([b], () => false, 2)).not.toBe(key);
    const light = { x: 1, y: 2, z: 3, radius: 100, intensity: 0.2, tint: [1, 0.5, 0] };
    expect(physicalLampKey([{ ...light }])).toBe(physicalLampKey([light]));
    expect(physicalLampKey([{ ...light, intensity: 0.3 }])).not.toBe(physicalLampKey([light]));
    expect(physicalLampKey([{ position: [1, 2, 3], radius: 100, intensity: 0.2, tint: [1, 0.5, 0] }])).toBe(
      physicalLampKey([light]),
    );
  });
  it('does not rebuild lamp/occluder lists for repeated equivalent snapshots', () => {
    const setStatic = vi.fn(),
      setOccluders = vi.fn(),
      setGround = vi.fn();
    const mapLight = {
      setPhysicalSources: vi.fn(),
      base: { ambient: 1, red: 1, green: 1, blue: 1, ground: 0, level: 0 },
    };
    const view = {
      hd: {
        realLightField: { setStatic, setOccluders, setGround },
        nodes: { realLight: { enabled: { value: 1 } } },
        atmosphere: { lights: true },
      },
      state: { map: {}, environment: { timeOfDay: 22 } },
      world: { mapLight, width: 1, height: 1 },
      cache: { height: [0], manifest: { objects: [], sprites: {} }, lights: { lights: [], types: {} } },
      shroud: { isLocationShrouded: () => false },
    };
    const update = (LiveView.prototype as unknown as { updatePhysicalLamps(this: unknown, s: unknown): void })
      .updatePhysicalLamps;
    update.call(view, { objects: [], lighting: { sources: [] } });
    update.call(view, { objects: [], lighting: { sources: [] } });
    expect(setStatic).toHaveBeenCalledTimes(1);
    expect(setOccluders).toHaveBeenCalledTimes(1);
    expect(setGround).toHaveBeenCalledTimes(1);
    update.call(view, {
      objects: [],
      lighting: { sources: [{ x: 0, y: 0, z: 0, radius: 256, intensity: 0, tint: [1, 1, 1] }] },
    });
    expect(setStatic).toHaveBeenCalledTimes(2);
    expect(setOccluders).toHaveBeenCalledTimes(1);
  });
});
