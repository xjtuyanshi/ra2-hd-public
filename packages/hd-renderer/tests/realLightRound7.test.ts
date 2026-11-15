import { describe, expect, it, vi } from 'vitest';
import { DataTexture, Mesh, Color, type WebGPURenderer, Vector3 } from 'three/webgpu';
import { AtmosphereNodes, groundLightAlbedoRgb } from '../src/atmosphere';
import { createLightLook, lightLook, LIGHT_LOOK_CURRENT, LIGHT_LOOK_NEW, wreckFlicker } from '../src/lightLook';
import {
  RealLightField,
  lightLuminance,
  lightMoment,
  irradianceKnee,
  reconstructLight,
  saturateAtLuminance,
  type Rgb,
} from '../src/realLight';
import {
  REAL_LIGHT_POLICIES,
  animLightPolicy,
  animPhysicalLight,
  auxiliaryBeamLights,
  nativeBeamLight,
  policyLight,
  warmFlashColour,
} from '../src/ra2/realLights';
import { laserLightProfile, RealEmissiveLayer } from '../src/realEmissive';
import { EffectsLayer } from '../src/effects';
import { EffectSpriteLayer, smokeCardAlpha, type EffectInstance } from '../src/live/effectSprites';
import { LiveView } from '../src/live/liveView';
import type { LiveAnim, LiveEffects } from '../src/live/effects';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';
import { createHash } from 'node:crypto';

const renderer = () =>
  ({
    autoClear: false,
    getRenderTarget: () => null,
    getMRT: () => null,
    getClearAlpha: () => 1,
    getClearColor: (c: Color) => c.set(0),
    setClearColor: vi.fn(),
    setRenderTarget: vi.fn(),
    setMRT: vi.fn(),
    render: vi.fn(),
  }) as unknown as WebGPURenderer;
const hue = (v: Rgb) => {
  const hi = Math.max(...v),
    lo = Math.min(...v),
    d = hi - lo;
  return d < 1e-8
    ? 0
    : ((hi === v[0] ? (v[1] - v[2]) / d : hi === v[1] ? (v[2] - v[0]) / d + 2 : (v[0] - v[1]) / d + 4) * 60 + 360) %
        360;
};
const saturation = (v: Rgb) => 1 - Math.min(...v) / Math.max(...v);
const display = (v: Rgb) => v.map((c) => c ** (1 / 2.2)) as [number, number, number];
const anim = (type: string): LiveAnim => ({ type, id: 1, uid: 123, x: 3 * 256, y: 3 * 256, z: 0 });
function fieldFixture() {
  const nodes = new AtmosphereNodes({ realLight: true });
  const field = new RealLightField(8, 8, nodes.realLight!);
  const r = renderer();
  // Real packed attributes consumed by the field shader, after current per-frame look evaluation.
  const data = () =>
    (field as unknown as { attributes: { array: Float32Array }[][] }).attributes[0]!.map((a) =>
      Array.from(a.array.slice(0, 4)),
    );
  return {
    nodes,
    field,
    r,
    data,
    dispose() {
      field.dispose();
      nodes.dispose();
    },
  };
}

describe('round 7: held-frame presentation presets', () => {
  it('restores every 1697702 parameter without replacing uniforms or moving the held clock', () => {
    const look = createLightLook();
    const smoke = look.smokeVolume,
      beam = look.beamLight;
    const view = { hd: { nodes: { smokeLook: look } } } as unknown as LiveView;
    expect(LiveView.prototype.lightLook.call(view)).toEqual(LIGHT_LOOK_NEW);
    expect(LiveView.prototype.lightLook.call(view, 'cur')).toEqual(LIGHT_LOOK_CURRENT);
    expect(LiveView.prototype.lightLook.call(view, { smokeWisp: 0.4 }).smokeWisp).toBe(0.4);
    expect(LiveView.prototype.lightLook.call(view, 'new')).toEqual(LIGHT_LOOK_NEW);
    expect(look.smokeVolume).toBe(smoke);
    expect(look.beamLight).toBe(beam);
    expect(() => lightLook(look, { beamCore: NaN })).toThrow();
  });
  it('switches look while held without accepting a queued snapshot or touching the frozen clock', () => {
    const f = fieldFixture(),
      publish = vi.fn(),
      freezeClock = vi.fn();
    const view = Object.assign(Object.create(LiveView.prototype), {
      hd: { nodes: f.nodes, freezeClock },
      publish,
      withWarmList: (s: unknown) => s,
    });
    view.push({ frame: 7 });
    view.holdLight(true);
    view.push({ frame: 8 });
    f.field.put('flash', 'chrono', 0, policyLight(REAL_LIGHT_POLICIES.chrono, [3, 0.5, 3]), 1, 0.15);
    view.lightLook('cur');
    f.field.update(0.05, f.r, 1);
    expect(lightLuminance(f.data()[2]!.slice(0, 3) as unknown as Rgb)).toBeGreaterThan(0);
    view.lightLook('new');
    f.field.update(0.05, f.r, 1);
    expect((f.field.scenes[0]!.children[0] as { count?: number }).count).toBe(0);
    expect(publish).toHaveBeenCalledTimes(1);
    expect(f.nodes.realLight!.logicFrame).toBe(7);
    expect(freezeClock.mock.calls).toEqual([[true]]);
    view.holdLight(false);
    expect(publish).toHaveBeenCalledTimes(2);
    expect(freezeClock.mock.calls).toEqual([[true], [false]]);
    f.dispose();
  });
  it.each(['CHRONOSK', 'CHRONOTG', 'WARPIN', 'WARPOUT'])(
    'zeros %s at the same timestamp, retaining the event for cur',
    (type) => {
      const f = fieldFixture(),
        a = anim(type),
        p = animLightPolicy(type)!;
      f.field.put('flash', type, 0, animPhysicalLight(a, p), 1, 0.15);
      expect(f.field.update(0.05, f.r, 1)).toBe(true);
      expect(f.data()[2]).toEqual([0, 0, 0, 0]);
      lightLook(f.nodes.smokeLook, 'cur');
      expect(f.field.update(0.05, f.r, 1)).toBe(true);
      expect(lightLuminance(f.data()[2]!.slice(0, 3) as unknown as Rgb)).toBeGreaterThan(0);
      lightLook(f.nodes.smokeLook, 'new');
      f.field.update(0.05, f.r, 1);
      expect(f.field.pools.get('flash')!.size).toBe(1);
      expect((f.field.scenes[0]!.children[0] as { count?: number }).count).toBe(0);
      f.dispose();
    },
  );
  it('tags unnamed chrono art and temporal ground light, without changing other auxiliary policies', () => {
    expect(animLightPolicy('custom', { c: 'chrono', life: 30 } as never)?.lookClass).toBe('chrono');
    const effects = {
      radBeams: [
        {
          type: 'temporal',
          id: 1,
          source: { x: 0, y: 0, z: 0 },
          target: { x: 256, y: 0, z: 0 },
          color: [255, 255, 255],
          sustained: true,
        },
      ],
    } as unknown as LiveEffects;
    expect(auxiliaryBeamLights(effects)[0]!.light.lookClass).toBe('chrono');
    expect(REAL_LIGHT_POLICIES.tesla.ground).toBe(0.12);
    expect(REAL_LIGHT_POLICIES.prism.ground).toBe(0.12);
  });
});

describe('round 7: source-class ground response', () => {
  it('uses source-coloured warm-white from the first flash frame; cur restores the old white phase', () => {
    for (const type of ['EXPLOSML', 'EXPLOLRG', 'TWLT070']) {
      const p = animLightPolicy(type)!;
      const light = animPhysicalLight(anim(type), p);
      expect(saturation(light.warmIntensity!)).toBeCloseTo(0.32, 8);
      expect(light.warmIntensity![0]).toBeGreaterThan(light.warmIntensity![2]);
      const f = fieldFixture();
      f.field.put('flash', 'blast', 0, light, 0.08, 0.3);
      f.field.update(0, f.r, 1);
      const now = f.data()[2]!.slice(0, 3) as unknown as Rgb;
      // Round 7b: the live warm flash is pushed to flashSat (0.32 summed with the blue night base to grey-white).
      expect(saturation(now)).toBeGreaterThan(0.45);
      expect(now[0]).toBeGreaterThan(now[1]);
      expect(now[1]).toBeGreaterThan(now[2]);
      expect(f.data()[2]![3]).toBeCloseTo(0.9);
      lightLook(f.nodes.smokeLook, 'cur');
      f.field.update(0, f.r, 1);
      expect(saturation(f.data()[2]!.slice(0, 3) as unknown as Rgb)).toBeCloseTo(0.08, 6);
      f.dispose();
    }
    expect(saturation(warmFlashColour([1, 1, 1]))).toBeCloseTo(0.32);
  });
  it('flashSat keeps luminance and hue order; flashSize shrinks only warm flash pools', () => {
    const warm = warmFlashColour([1, 0.85, 0.72]);
    const more = saturateAtLuminance(warm, LIGHT_LOOK_NEW.flashSat / 0.32);
    expect(lightLuminance(more)).toBeCloseTo(lightLuminance(warm), 8);
    expect(saturation(more)).toBeGreaterThan(saturation(warm));
    expect(saturateAtLuminance(warm, 1).map((c, i) => c - warm[i]!)).toEqual([0, 0, 0].map(() => expect.closeTo(0, 9)));
    const p = animLightPolicy('EXPLOLRG')!;
    const light = animPhysicalLight(anim('EXPLOLRG'), p);
    const f = fieldFixture();
    f.field.put('flash', 'blast', 0, light, 0.08, 0.3);
    f.field.update(0, f.r, 1);
    const small = [f.data()[0]![3]!, f.data()[1]![3]!];
    lightLook(f.nodes.smokeLook, 'cur');
    f.field.update(0, f.r, 1);
    const old = [f.data()[0]![3]!, f.data()[1]![3]!];
    expect(small[0]! / old[0]!).toBeCloseTo(LIGHT_LOOK_NEW.flashSize, 5);
    expect(small[1]! / old[1]!).toBeCloseTo(LIGHT_LOOK_NEW.flashSize, 5);
    f.dispose();
  });
  it('neutralizes graded road/grass only for flash contribution; beams have byte-identical packed values', () => {
    for (const [albedo, shown] of [
      [
        [0.3, 0.3, 0.32],
        [0.045, 0.025, 0.07],
      ],
      [
        [0.12, 0.22, 0.04],
        [0.02, 0.027, 0.05],
      ],
    ] as [Rgb, Rgb][]) {
      const neutral = groundLightAlbedoRgb(albedo, shown, 1, LIGHT_LOOK_NEW.flashNeutral);
      const lit = neutral.map((c, i) => c * warmFlashColour([1, 0.85, 0.72])[i]!) as [number, number, number];
      expect(lit[0]).toBeGreaterThan(lit[2]);
      expect(hue(lit)).not.toBeGreaterThan(270); // No lilac.
      expect(hue(lit)).toBeLessThan(60); // No olive/yellow-green.
    }
    for (const kind of ['laser', 'bolt'] as const) {
      const f = fieldFixture();
      const l = nativeBeamLight({
        kind,
        source: { x: 256, y: 256, z: 128 },
        target: { x: 1024, y: 256, z: 0 },
        colour: [137, 0, 197],
      });
      f.field.put('flash', 'beam', 0, l, 1, 0);
      f.field.update(0.05, f.r, 1);
      const newData = f.data();
      lightLook(f.nodes.smokeLook, 'cur');
      f.field.update(0.05, f.r, 1);
      expect(f.data()).toEqual(newData);
      expect(newData[2]![3]).toBe(0);
      f.dispose();
    }
  });
  it('raises wrecks fourfold inside the existing 0.45 cell radius with bounded calm flicker', () => {
    const f = fieldFixture(),
      l = policyLight(REAL_LIGHT_POLICIES.burning, [3, 0.3, 3]);
    l.seed = 0;
    f.field.put('kept', 'wreck', 0, l, 1, 0);
    f.field.update(0, f.r, 1);
    const bright = lightLuminance(f.data()[2]!.slice(0, 3) as unknown as Rgb);
    lightLook(f.nodes.smokeLook, 'cur');
    f.field.update(0, f.r, 1);
    // Old flicker is source-seeded and asymmetrical; compare with its unmodulated intensity.
    expect(bright / lightLuminance(l.intensity)).toBeCloseTo(4, 5);
    expect(f.data()[0]![3]).toBeCloseTo(0.45);
    const m = lightMoment(l, [3.3, 0, 3]);
    const old = reconstructLight(m.s, m.d, [0, 1, 0]);
    const ratio = (gain: number) =>
      lightLuminance(irradianceKnee(old.map((c) => c * gain) as [number, number, number], 1, 1, 0.1));
    // Compare the display lift on a dark, neutral road to the same source with the old gain.
    const base = 0.004;
    const lift = (gain: number) => 255 * ((base * (1 + ratio(gain) / 0.1)) ** (1 / 2.2) - base ** (1 / 2.2));
    expect(lift(4)).toBeGreaterThan(lift(1) * 2);
    expect(lift(4)).toBeGreaterThanOrEqual(10);
    expect(lift(4)).toBeLessThanOrEqual(15);
    const values = Array.from({ length: 1001 }, (_, i) => wreckFlicker(0.37, i / 100));
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0.9);
    expect(Math.max(...values)).toBeLessThanOrEqual(1.1);
    expect(Math.abs(wreckFlicker(0.37, 1.001) - wreckFlicker(0.37, 1))).toBeLessThan(0.003);
    f.dispose();
  });
});

describe('round 7: native laser light profile', () => {
  it.each([{ source: [137, 0, 197] as Rgb }, { source: [255, 0, 0] as Rgb }])(
    'keeps glow hue, a bright near-white core and a soft tail that reaches zero inside the quad for %j',
    ({ source }) => {
      const colour = source.map((c) => (c / 255) ** 2.2) as [number, number, number];
      const core = laserLightProfile(0, colour);
      // Hot core: near white (20% source hue) and at or above the night bloom threshold (~0.85-1), so it seeds bloom.
      expect(Math.min(...core)).toBeGreaterThan(0.85);
      const glowAt = (r: number) => laserLightProfile(r, colour);
      // Core clearly brighter than the glow beside it (round-7 review: neon tube with a weak core).
      expect(Math.max(...core)).toBeGreaterThan(1.2 * Math.max(...glowAt(1)));
      const reach = 2.5 * LIGHT_LOOK_NEW.beamSpread;
      let prev = Infinity;
      for (let r = 0.6; r < reach; r += 0.02) {
        const v = glowAt(r);
        const dh = Math.abs(hue(display(v)) - hue(display(colour)));
        if (Math.max(...v) > 0.01) expect(Math.min(dh, 360 - dh)).toBeLessThan(8);
        expect(Math.max(...v)).toBeLessThan(prev);
        prev = Math.max(...v);
      }
      // No step at the quad edge: the windowed tail is already below 1% of its peak there.
      expect(Math.max(...glowAt(reach - 0.05))).toBeLessThan(0.01 * Math.max(...glowAt(0.6)));
      // No plateau: the glow keeps falling across the band where the old clamp held it flat.
      expect(Math.max(...glowAt(2))).toBeLessThan(0.8 * Math.max(...glowAt(0.8)));
    },
  );
  it('does not opt auxiliary emitters or bolts into native laser treatment', () => {
    const layer = new RealEmissiveLayer(1);
    const from = new Vector3(),
      to = new Vector3(1, 0, 0);
    layer.set('aux', [{ source: 'aux', from, to, width: 0.1, colour: [0.5, 0, 1], gain: 1, preserveHue: true }]);
    layer.update(0, true);
    const data = (layer as unknown as { data: { array: Float32Array } }).data;
    expect(data.array[27]).toBe(1); // Only native lasers pack 2, independent of preserveHue.
    layer.dispose();
  });
});

describe('round 7: smoke and legacy boundaries', () => {
  it('bounds mean coverage and premultiplied grey luminance on every card texel, including edge erosion', () => {
    let before = 0,
      after = 0;
    for (let i = 1; i <= 100; i++)
      for (let j = 0; j <= 10; j++) {
        const a = i / 100,
          next = smokeCardAlpha(a, j / 10, ((i * j) % 19) / 19, 0.7, LIGHT_LOOK_NEW);
        expect(next).toBeGreaterThanOrEqual(a * 0.96 - 1e-12);
        expect(next).toBeLessThanOrEqual(a * 1.04 + 1e-12);
        expect(smokeCardAlpha(a, 0, 0, 1, LIGHT_LOOK_CURRENT)).toBe(a);
        expect(smokeCardAlpha(a, 0, 0, 1, LIGHT_LOOK_NEW, false)).toBe(a);
        before += a;
        after += next;
      }
    expect(Math.abs(after / before - 1)).toBeLessThan(0.05);
  });
  it('builds opted-in smoke/beam graphs with uniforms, and never rebuilds them after presets', () => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      effects = new EffectsLayer(nodes);
    const internal = effects as unknown as {
      billow: { mesh: import('three/webgpu').Mesh };
      column: { mesh: import('three/webgpu').Mesh };
    };
    for (const mesh of [internal.billow.mesh, internal.column.mesh]) {
      const code = shaderBytes(mesh);
      expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
      const taps = new Map<string, number>();
      for (const [, binding] of code.matchAll(/\btextureSample\w*\(\s*([\w]+)/g))
        taps.set(binding!, (taps.get(binding!) ?? 0) + 1);
      expect(Math.max(...taps.values())).toBeLessThanOrEqual(7); // Original three noise taps plus at most four.
      lightLook(nodes.smokeLook, 'cur');
      expect(shaderBytes(mesh).replace(/NodeBuffer_\d+/g, 'NodeBuffer')).toBe(
        code.replace(/NodeBuffer_\d+/g, 'NodeBuffer'),
      );
    }
    nodes.dispose();
  });
  it.each([false, true])('builds bounded smoke-card sampling for the %s RGBA path with two noise taps', (hd) => {
    const nodes = new AtmosphereNodes({ realLight: true }),
      scene = receiverScene(nodes);
    const texture = new DataTexture(new Uint8Array(32 * 32 * 4), 32, 32);
    texture.needsUpdate = true;
    const atlas = { texture, width: 32, height: 32 };
    const layer = new EffectSpriteLayer(
      scene.source,
      () => atlas,
      undefined,
      () => atlas,
    );
    const card: EffectInstance = {
      anchor: new Vector3(1, 1, 1),
      frame: { rect: [0, 0, 16, 16], offset: [-8, -8], page: 0 },
      palette: '',
      alpha: 0.5,
      push: 0,
      draw: 'card',
      pass: 'air',
      depth: 0,
      glow: false,
      category: 'smoke',
      ...(hd
        ? {
            hd: {
              frame: { page: 0, rect: [0, 0, 16, 16], offset: [-8, -8] },
              next: null,
              scale: 1,
              blend0: 0,
              t0: 0,
              stageSeconds: 0.1,
            },
          }
        : {}),
    };
    scene.source.paletteRow.set('', 0);
    layer.setInstances([card], 0);
    const mesh = layer.root.children.find((o) => o instanceof Mesh) as Mesh;
    const code = shaderBytes(mesh);
    expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(code).toContain('clamp('); // Both atlas accesses stay inside their own frame.
    expect(code).toContain('0.96');
    expect(code).toContain('1.04');
    // Each NoiseTexture RGB sample serves both warp and density, including crossfaded HD successors.
    const samples = new Map<string, number>();
    for (const [, binding] of code.matchAll(/\btextureSample\w*\(\s*([\w]+)/g))
      samples.set(binding!, (samples.get(binding!) ?? 0) + 1);
    expect([...samples.values()]).toContain(2); // Shared noise is sampled exactly twice.

    lightLook(nodes.smokeLook, 'cur');
    expect(shaderBytes(mesh).replace(/NodeBuffer_\d+/g, 'NodeBuffer')).toBe(
      code.replace(/NodeBuffer_\d+/g, 'NodeBuffer'),
    );
    layer.setInstances([], Infinity);
    texture.dispose();
    scene.dispose();
  });
  it('keeps hd-light-off receiver output fingerprints identical for all look presets', () => {
    const nodes = new AtmosphereNodes(),
      scene = receiverScene(nodes);
    const hashes = () =>
      scene.meshes.map((m) =>
        createHash('sha256')
          .update(shaderBytes(m).replace(/NodeBuffer_\d+/g, 'NodeBuffer'))
          .digest('hex'),
      );
    const original = hashes();
    lightLook(nodes.smokeLook, 'new');
    expect(hashes()).toEqual(original);
    lightLook(nodes.smokeLook, 'cur');
    expect(hashes()).toEqual(original);
    scene.dispose();
  });
});
