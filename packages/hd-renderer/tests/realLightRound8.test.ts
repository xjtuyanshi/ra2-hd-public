import { describe, expect, it, vi } from 'vitest';
import { DataTexture, Mesh, Vector3, type WebGPURenderer } from 'three/webgpu';
import { AtmosphereNodes, smokeLight } from '../src/atmosphere';
import { LightField } from '../src/lightField';
import { float, vec2, vec3 } from 'three/tsl';
import {
  createLightLook,
  lightLook,
  LIGHT_LOOK_BEAM,
  LIGHT_LOOK_CURRENT,
  LIGHT_LOOK_NEW,
  type LightLookValues,
} from '../src/lightLook';
import { laserGlowTint, laserLightProfile, RealEmissiveLayer } from '../src/realEmissive';
import { EffectsLayer } from '../src/effects';
import { EffectSpriteLayer, type EffectInstance } from '../src/live/effectSprites';
import { beamColor } from '../src/live/beams';
import { laserAge, laserRemaining } from '../src/live/effects';
import type { Rgb } from '../src/realLight';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';

const hue = (v: readonly number[]) => {
  const hi = Math.max(...v),
    lo = Math.min(...v),
    d = hi - lo;
  return d < 1e-8
    ? 0
    : ((hi === v[0] ? (v[1]! - v[2]!) / d : hi === v[1] ? (v[2]! - v[0]!) / d + 2 : (v[0]! - v[1]!) / d + 4) * 60 +
        360) %
        360;
};
const display = (v: readonly number[]) => v.map((c) => Math.max(0, c) ** (1 / 2.2));
const SMOKE_KEYS = ['smokeVolume', 'smokeWisp', 'smokeWarp', 'smokeStretch', 'smokeDetail', 'smokeEdge', 'smokeRidge'];
const textureTaps = (code: string) => [...code.matchAll(/\btextureSample\w*\(/g)].length;

describe('round 8: hd-light=beam preset', () => {
  it('keeps the round-7 chrono and flash values and Build 13 smoke values', () => {
    for (const [key, value] of Object.entries(LIGHT_LOOK_BEAM) as [keyof LightLookValues, number][]) {
      if (SMOKE_KEYS.includes(key)) expect(value, key).toBe(LIGHT_LOOK_CURRENT[key]);
      else if (!key.startsWith('beam')) expect(value, key).toBe(LIGHT_LOOK_NEW[key]);
    }
    expect(LIGHT_LOOK_BEAM.chronoPool).toBe(0);
    expect(LIGHT_LOOK_BEAM.flashWarm).toBe(1);
    expect(LIGHT_LOOK_BEAM.beamLight).toBe(1);
    // The new switches are inert in the earlier presets.
    for (const preset of [LIGHT_LOOK_CURRENT, LIGHT_LOOK_NEW]) {
      expect(preset.beamOverbright).toBe(1);
      expect(preset.beamKnee).toBe(0);
      expect(preset.beamDepthLift).toBe(0);
      expect(preset.beamFade).toBe(0);
    }
    expect(LIGHT_LOOK_BEAM.beamFade).toBe(1);
    const look = createLightLook('beam');
    expect(lightLook(look)).toEqual(LIGHT_LOOK_BEAM);
    expect(lightLook(look, 'new')).toEqual(LIGHT_LOOK_NEW);
    expect(lightLook(look, 'beam')).toEqual(LIGHT_LOOK_BEAM);
  });

  it('gives the violet Prism glow the hue the original shows at the beam centre, and leaves red red', () => {
    // Live SOV07TMD: inner colour (137, 0, 197) displayed (225, 43, 255) in the original, hue ~291 deg.
    const violet = beamColor([137, 0, 197]);
    const original = hue([225, 43, 255]);
    const round7 = hue(display(laserGlowTint(violet, 1)));
    const beam = hue(display(laserGlowTint(violet, LIGHT_LOOK_BEAM.beamOverbright)));
    expect(Math.abs(round7 - original)).toBeGreaterThan(8);
    expect(Math.abs(beam - original)).toBeLessThan(5);
    // overbright = 1 is exactly the round-7 tint (colour / peak).
    const peak = Math.max(...violet);
    laserGlowTint(violet, 1).forEach((c, i) => expect(c).toBeCloseTo(violet[i]! / peak, 6));
    // A single-channel house red keeps its hue within a few degrees and stays saturated.
    for (const red of [beamColor([255, 0, 0]), beamColor([255, 24, 0]), beamColor([200, 30, 20])]) {
      const t = laserGlowTint(red, LIGHT_LOOK_BEAM.beamOverbright);
      expect(Math.abs(hue(display(t)) - hue(display(red)))).toBeLessThan(5);
      expect(Math.max(t[1], t[2])).toBeLessThan(0.1);
    }
  });

  it('saturates the glow amplitude, not its profile: a full-gain beam is no wider than a budgeted one', () => {
    const violet = beamColor([137, 0, 197]) as Rgb;
    const width = (look: LightLookValues, gain: number, threshold: number) => {
      const peak = laserLightProfile(1.2, violet, 1.4, look, gain)[2];
      let r = 1.2;
      while (r < 8 && laserLightProfile(r, violet, 1.4, look, gain)[2] > peak * threshold) r += 0.01;
      return r;
    };
    // Round 7: the full-gain beam's relative falloff is wider (the plateau). Beam preset: one shape for every gain.
    expect(width(LIGHT_LOOK_NEW, 1, 0.25)).toBeGreaterThan(width(LIGHT_LOOK_NEW, 0.3, 0.25) + 0.2);
    expect(Math.abs(width(LIGHT_LOOK_BEAM, 1, 0.25) - width(LIGHT_LOOK_BEAM, 0.3, 0.25))).toBeLessThan(0.02);
    // And at full gain the beam preset's bright band is much narrower than round 7's, its faint tail no wider.
    const absolute = (look: LightLookValues, threshold: number) => {
      let r = 0.5;
      while (r < 8 && laserLightProfile(r, violet, 1.4, look, 1)[2] > threshold) r += 0.01;
      return r;
    };
    expect(absolute(LIGHT_LOOK_BEAM, 0.3)).toBeLessThan(absolute(LIGHT_LOOK_NEW, 0.3) * 0.8);
    expect(absolute(LIGHT_LOOK_BEAM, 0.02)).toBeLessThan(absolute(LIGHT_LOOK_NEW, 0.02));
    // White-hot core: near-white, keeps a violet tint (blue >= red > green).
    const core = laserLightProfile(0, violet, 1.4, LIGHT_LOOK_BEAM, 1);
    expect(Math.min(...core) / Math.max(...core)).toBeGreaterThan(0.6);
    expect(core[2]).toBeGreaterThanOrEqual(core[0]);
    expect(core[0]).toBeGreaterThan(core[1]);
  });

  it('lifts laser glow depth by uniform, without rebuilding the emissive graph', () => {
    const layer = new RealEmissiveLayer(4, createLightLook('beam'));
    const code = shaderBytes(layer.mesh);
    expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(code).toMatch(/frag_depth|FragDepth|fragDepth|gl_FragDepth/);
    lightLook(layer.look, 'new');
    expect(shaderBytes(layer.mesh).replace(/NodeBuffer_\d+/g, 'NodeBuffer')).toBe(
      code.replace(/NodeBuffer_\d+/g, 'NodeBuffer'),
    );
    layer.dispose();
  });

  it('builds no smoke-volume graph in the beam preset', () => {
    const off = new EffectsLayer(new AtmosphereNodes());
    const beamNodes = new AtmosphereNodes({ realLight: true, lightPreset: 'beam' });
    const newNodes = new AtmosphereNodes({ realLight: true });
    expect(beamNodes.smokeVolume).toBe(false);
    expect(newNodes.smokeVolume).toBe(true);
    expect(new AtmosphereNodes().smokeVolume).toBe(false);
    const beam = new EffectsLayer(beamNodes);
    const round7 = new EffectsLayer(newNodes);
    type Internal = { billow: { mesh: Mesh }; column: { mesh: Mesh } };
    for (const pool of ['billow', 'column'] as const) {
      const taps = (e: EffectsLayer) => textureTaps(shaderBytes((e as unknown as Internal)[pool].mesh));
      // Physical light on smoke may read the light field; the smoke's own noise taps are Build 13's.
      expect(taps(beam) - taps(off), pool).toBeLessThan(taps(round7) - taps(off));
    }
    beamNodes.dispose();
    newNodes.dispose();
  });

  it('keeps native smoke cards on the Build 13 graph in the beam preset', () => {
    const nodes = new AtmosphereNodes({ realLight: true, lightPreset: 'beam' }),
      scene = receiverScene(nodes);
    const texture = new DataTexture(new Uint8Array(32 * 32 * 4), 32, 32);
    texture.needsUpdate = true;
    const atlas = { texture, width: 32, height: 32 };
    const layer = new EffectSpriteLayer(scene.source, () => atlas);
    expect(layer.smokeNodes).toBeUndefined();
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
    };
    scene.source.paletteRow.set('', 0);
    layer.setInstances([card], 0);
    const mesh = layer.root.children.find((o) => o instanceof Mesh) as Mesh;
    const code = shaderBytes(mesh);
    expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
    expect(code).not.toContain('1.04'); // The round-7 card density clamp is absent.
    layer.setInstances([], Infinity);
    texture.dispose();
    scene.dispose();
  });

  it('lights smoke from a Build 13 mirror of the light field while real light mutes the legacy event lights', () => {
    const nodes = new AtmosphereNodes({ realLight: true, lightPreset: 'beam' });
    const field = new LightField(8, 8, { nodes, build13Mirror: true });
    const draws: { target: unknown; count: number }[] = [];
    let target: unknown = null;
    const dyn = (field as unknown as { dynamics: { mesh: { count: number } } }).dynamics;
    const r = {
      getRenderTarget: () => null,
      getClearAlpha: () => 1,
      setClearAlpha: vi.fn(),
      setRenderTarget: (t: unknown) => (target = t),
      render: () => draws.push({ target, count: dyn.mesh.count }),
    } as unknown as WebGPURenderer;
    expect(field.build13Target).toBeDefined();
    expect(nodes.build13FieldTexture.value).toBe(field.build13Target!.texture);
    field.source('window', 0, { x: 2, z: 2, radius: 2, colour: [0.5, 0.3, 0.1] });
    field.pulse(0, new Vector3(1, 0, 1), [1, 0.7, 0.4], 2, 1, 0.3, 0, 'explosion');
    field.pulse(0, new Vector3(1, 0, 1), [0.4, 0.4, 1], 1, 1, 0.2, 0, 'flash');
    field.keep('fire-system', 0, new Vector3(1, 0, 1), [1, 0.6, 0.3], 2, 1, 0, 0.6);
    field.update(0.1, r, 1);
    // The shared field keeps only the source (the physical field owns the events); the smoke mirror has all four.
    expect(draws).toEqual([
      { target: field.target, count: 1 },
      { target: field.build13Target, count: 4 },
    ]);
    // Once everything is over both are drawn once more (to clear them), then not again until something changes.
    draws.length = 0;
    field.update(5, r, 1);
    field.update(5.1, r, 1);
    expect(draws).toEqual([
      { target: field.target, count: 0 },
      { target: field.build13Target, count: 0 },
    ]);
    // Real light off: the shared field is Build 13's again and no mirror is drawn.
    nodes.realLight!.enabled.value = 0;
    draws.length = 0;
    field.pulse(6, new Vector3(1, 0, 1), [1, 0.7, 0.4], 2, 1, 0.3, 0, 'explosion');
    field.update(6.1, r, 1);
    expect(draws).toEqual([{ target: field.target, count: 1 }]);
    // Without the option (hd-light=1, no real light) smoke samples the shared field.
    const plain = new LightField(8, 8, { nodes: new AtmosphereNodes({ realLight: true }) });
    expect(plain.build13Target).toBeUndefined();
    field.dispose();
    plain.dispose();
    nodes.dispose();
  });

  it('builds the Build 13 smoke-light expression in the beam preset, gated by real light', () => {
    const build = (n: AtmosphereNodes) => smokeLight(vec2(1, 1), n, float(0.5), vec3(0, 0, 1));
    const beam = new AtmosphereNodes({ realLight: true, lightPreset: 'beam' });
    expect(build(beam)).toBeDefined();
    expect(build(new AtmosphereNodes())).toBeDefined();
    beam.dispose();
  });

  it('dims a fading native laser over its life like the original, only with beamFade', () => {
    const look = createLightLook('beam');
    const laser = (progress: number, fades = true) =>
      ({ kind: 'laser', fades, duration: 20, progress }) as Parameters<typeof laserRemaining>[0];
    expect(laserAge(laser(0))).toBe(0);
    expect(laserAge(laser(5))).toBeCloseTo(0.25);
    expect(laserAge(laser(30))).toBe(1);
    expect(laserAge(laser(10, false))).toBe(0);
    expect(laserAge({ fades: true })).toBe(0);
    expect(laserRemaining(laser(5), look)).toBeCloseTo(0.75);
    expect(laserRemaining({ ...laser(5), kind: 'bolt' }, look)).toBe(1);
    lightLook(look, 'new');
    expect(laserRemaining(laser(5), look)).toBe(1);
    // The emissive graph reads the age from the instance data; switching the look only changes a uniform.
    const layer = new RealEmissiveLayer(4, createLightLook('beam'));
    const code = shaderBytes(layer.mesh).replace(/NodeBuffer_\d+/g, 'NodeBuffer');
    lightLook(layer.look, 'cur');
    expect(shaderBytes(layer.mesh).replace(/NodeBuffer_\d+/g, 'NodeBuffer')).toBe(code);
    layer.dispose();
  });
});
