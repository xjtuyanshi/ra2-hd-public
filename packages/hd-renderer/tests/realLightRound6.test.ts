import { describe, expect, it, vi } from 'vitest';
import { Color, Mesh, MeshBasicNodeMaterial, PlaneGeometry, Vector3, type WebGPURenderer } from 'three/webgpu';
import { positionWorld, uniform, vec3 } from 'three/tsl';
import { AtmosphereNodes, groundLightAlbedo, groundLightAlbedoRgb, smokeLight } from '../src/atmosphere';
import { BeamLayer, EffectsLayer } from '../src/effects';
import { RealEmissiveLayer, EMISSIVE_BUDGET, type EmissivePrimitive } from '../src/realEmissive';
import {
  RealLightField,
  groundLightMoment,
  lightMoment,
  lightLuminance,
  irradianceKnee,
  reconstructLight,
  viewProjectedPoint,
  type PhysicalLight,
  type Point,
  type Rgb,
} from '../src/realLight';
import {
  REAL_LIGHT_POLICIES,
  animLightPolicy,
  nativeBeamLight,
  policyLight,
  capLightColour,
} from '../src/ra2/realLights';
import { animEmission, REAL_LASER_STYLE } from '../src/ra2/realEmission';
import { LiveEffectsDriver, type LiveAnim, type LiveBeam } from '../src/live/effects';
import { heldPulseEnvelope } from '../src/lightField';
import { IsoCamera, PIXELS_PER_CELL_EDGE } from '../src/world';
import { receiverScene, shaderBytes } from './fixtures/realLightScene';

const direction: Point = [-Math.sqrt(3 / 8), -0.5, -Math.sqrt(3 / 8)];
const sample = (l: PhysicalLight, p: Point, projected = true) => {
  const m = projected ? groundLightMoment(l, p, direction) : lightMoment(l, p);
  return reconstructLight(m.s, m.d, [0, 1, 0]);
};
const display = (rgb: Rgb) => rgb.map((c) => Math.max(0, c) ** (1 / 2.2)) as [number, number, number];
const hue = (v: Rgb) => {
  const hi = Math.max(...v),
    lo = Math.min(...v),
    d = hi - lo;
  return d === 0
    ? 0
    : ((hi === v[0] ? (v[1] - v[2]) / d : hi === v[1] ? (v[2] - v[0]) / d + 2 : (v[0] - v[1]) / d + 4) * 60 + 360) %
        360;
};
const fakeRenderer = () =>
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
const bolt: LiveBeam = {
  kind: 'bolt',
  source: { x: 256, y: 256, z: 128 },
  target: { x: 6 * 256, y: 256, z: 0 },
  colour: [82, 80, 255],
};

describe('round 6 A: the ground footprint follows the camera ray, independently of upright receivers', () => {
  it.each([0, 2])('has one symmetric narrow peak at the elevated strand on terrain height %s', (height) => {
    // Screen-horizontal strand: across it the ground runs along x+z. The elevated beam is half a cell up.
    const l = {
      ...nativeBeamLight({ ...bolt, kind: 'laser', colour: [255, 0, 0] }),
      position: [1, height + 0.5, 5] as Point,
      end: [5, height + 0.5, 1] as Point,
    };
    const centre = viewProjectedPoint([3, height + 0.5, 3], height, direction);
    const profile = (d: number) =>
      lightLuminance(
        irradianceKnee(sample(l, [centre[0] + d * Math.SQRT1_2, height, centre[2] + d * Math.SQRT1_2]), 1, 1, 0.1),
      );
    const values = Array.from({ length: 201 }, (_, i) => profile((i - 100) * 0.01));
    const peak = Math.max(...values),
      offset = (values.indexOf(peak) - 100) * 0.01;
    expect(Math.abs(offset)).toBeLessThan(0.1);
    expect(profile(0.35) / peak).toBeLessThan(0.5);
    expect(profile(0.8) / peak).toBeLessThan(0.05);
    expect(profile(-0.02) - 2 * peak + profile(0.02)).toBeLessThan(-0.0001);
    for (let i = 1; i <= 100; i++) {
      expect(profile(i * 0.01)).toBeLessThan(profile((i - 1) * 0.01) + 1e-12);
      expect(profile(i * 0.01)).toBeCloseTo(profile(-i * 0.01), 10);
    }
    // SOV07-like 0.1-noon ambient: display-luminance brightening stays gentle, below knee compression's shoulder.
    const ratio = ((0.1 + peak) / 0.1) ** (1 / 2.2) - 1;
    expect(ratio).toBeGreaterThan(0.35);
    expect(ratio).toBeLessThan(0.5);
    expect(l.intensity.reduce((n, c, i) => n + c * [0.2126, 0.7152, 0.0722][i]!, 0) / 0.3).toBeLessThan(0.5);
    const camera = new IsoCamera();
    camera.resize(1000, 800);
    camera.update();
    const a = new Vector3(3, height + 0.5, 3).project(camera.camera);
    const b = new Vector3(...centre).project(camera.camera);
    expect(a.x).toBeCloseTo(b.x, 12);
    expect(a.y).toBeCloseTo(b.y, 12);
    expect(lightLuminance(sample(l, [3, height + 0.4, 3], false))).toBeGreaterThan(0);
  });
  it.each([true, false])(
    'builds and renders the footprint atlas without changing the object source (directional=%s)',
    (directional) => {
      const nodes = new AtmosphereNodes({ realLight: true });
      const field = new RealLightField(16, 16, nodes.realLight!, directional);
      const camera = new IsoCamera();
      camera.target.set(8, 0, 8);
      camera.resize(800, 600);
      camera.update();
      field.setView(camera.camera);
      const light = nativeBeamLight(bolt),
        original = JSON.stringify(light);
      field.put('flash', 'bolt', 0, light, 0.08, 0.1);
      const renderer = fakeRenderer();
      field.update(0, renderer, 1);
      expect(JSON.stringify(light)).toBe(original);
      const internal = field as unknown as { footprintScene: import('three/webgpu').Scene };
      internal.footprintScene.traverse((o) => {
        if (o instanceof Mesh && o.name !== 'real-light-tile-clear') {
          const code = shaderBytes(o, field.targets[0]);
          expect(code).toContain('exp(');
          expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
        }
      });
      const target = field.targets[0]!;
      expect(target.viewport.x).toBe(target.width / 2);
      expect(target.scissor.x).toBeGreaterThanOrEqual(target.width / 2);
      expect(renderer.render).toHaveBeenCalledTimes(directional ? 5 : 3);
      const clear = internal.footprintScene.getObjectByName('real-light-tile-clear') as Mesh;
      expect(shaderBytes(clear, target)).not.toMatch(/\b(?:NaN|undefined)\b/);
      const modes: boolean[] = [];
      const loadRenderer = fakeRenderer();
      vi.mocked(loadRenderer.render).mockImplementation(() => {
        modes.push(loadRenderer.autoClear);
      });
      field.pulse('flash', 0.02, light, 0.08, 0.1);
      field.update(0.02, loadRenderer, 1);
      expect(modes[1]).toBe(false); // Second dynamic tile must load, never clear the whole attachment.
      expect(field.update(0.03, renderer, 1)).toBe(false);
      field.dispose();
      nodes.dispose();
    },
  );
});

describe('round 6 B: faithful laser widths and colour', () => {
  it('has an 8px saturated core and 16-20px soft outer footprint at the measured 2.5 device scale', () => {
    const pixels = PIXELS_PER_CELL_EDGE * 2.5;
    expect(REAL_LASER_STYLE.strandWidth * pixels).toBeGreaterThanOrEqual(16);
    expect(REAL_LASER_STYLE.strandWidth * pixels).toBeLessThanOrEqual(20);
    expect(REAL_LASER_STYLE.emissionWidth * EMISSIVE_BUDGET.outerWidth * pixels).toBeLessThanOrEqual(20);
    expect(REAL_LASER_STYLE.emissionWidth * REAL_LASER_STYLE.coreWidth * 0.72 * pixels).toBeCloseTo(8, 0);
    const layer = new BeamLayer(uniform(new Vector3(1, 0, 0)), uniform(new Vector3(0, 1, 0)), uniform(0), true);
    layer.laserStyle = REAL_LASER_STYLE;
    layer.realEmissive = new RealEmissiveLayer();
    layer.laserColour = true;
    const colour = new Vector3(...[225, 39, 255].map((c) => (c / 255) ** 2.2));
    layer.setBeams([{ from: new Vector3(), to: new Vector3(4, 0, 0), colour, bolt: false, seed: 1 }]);
    const primitives = (layer.realEmissive as unknown as { channels: Map<string, EmissivePrimitive[]> }).channels.get(
      'native',
    )!;
    expect(primitives[0]!.preserveHue).toBe(true);
    expect(hue(display(primitives[0]!.hot!))).toBeCloseTo(hue([225, 39, 255]), 10);
    expect(primitives[0]!.colour).toEqual(colour.toArray());
    expect(shaderBytes(layer.mesh)).not.toMatch(/\b(?:NaN|undefined)\b/);
    layer.realEmissive.dispose();
    layer.mesh.geometry.dispose();
    (layer.mesh.material as MeshBasicNodeMaterial).dispose();
  });
});

describe('round 6 C/D: night surface hue and neutral smoke', () => {
  it('keeps SOV07 blue night grass under red/fire/blue light rather than revealing olive daytime albedo', () => {
    const albedo: Rgb = [0.27, 0.3, 0.12];
    const shown = [42, 46, 49].map((c) => (c / 255) ** 2.2) as [number, number, number];
    // Includes map/scene blue grade and moon, as opposed to using moonlit(daytime grass) alone.
    const receiver = groundLightAlbedoRgb(albedo, shown, 1);
    expect(lightLuminance(receiver)).toBeCloseTo(lightLuminance(albedo), 12);
    expect(hue(display(receiver))).toBeCloseTo(hue(display(shown)), 12);
    for (const source of [
      [1, 0, 0],
      [1, 0.5, 0.1],
      [0.08, 0.075, 1],
    ] as Rgb[]) {
      const light = capLightColour(source, 0.3);
      const lit = display(shown.map((c, i) => c + receiver[i]! * light[i]! * 0.2) as [number, number, number]);
      const h = hue(lit);
      expect(h >= 40 && h <= 90).toBe(false);
      // Red moves through purple/neutral towards red; blue stays on the cool side.
      expect(Math.abs(h - hue(display(shown))) <= 25 || h > 200 || h < 25).toBe(true);
    }
    expect(groundLightAlbedoRgb(albedo, shown, 0)).toEqual(albedo);
    const nodes = new AtmosphereNodes({ realLight: true });
    const graph = groundLightAlbedo(
      vec3(...albedo),
      vec3(...shown)
        .mul(nodes.sceneLevel)
        .mul(nodes.sceneTerrainMul),
      nodes,
    );
    const values: unknown[] = [];
    graph.traverse((n) => {
      if ('value' in n) values.push(n.value);
    });
    expect(values).toContain(nodes.sceneTerrainMul.value);
    // All actual terrain/cliff/plain and flat-sprite receiver graphs must still compile within their texture budget.
    const scene = receiverScene(nodes);
    for (const mesh of scene.meshes) {
      const code = shaderBytes(mesh);
      expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
      expect(new Set([...code.matchAll(/var\s+(\w+)\s*:\s*texture_2d/g)].map((m) => m[1])).size).toBeLessThanOrEqual(
        16,
      );
    }
    scene.dispose();
    nodes.dispose();
  });
  it('uses a directional, <=0.1-saturation smoke fill and no persistent fire or smoke bloom discs', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const field = new RealLightField(4, 4, nodes.realLight!);
    const scene = smokeLight(positionWorld.xz, nodes, positionWorld.y, vec3(0, -1, 0));
    const material = new MeshBasicNodeMaterial();
    material.colorNode = scene;
    const mesh = new Mesh(new PlaneGeometry(), material);
    const code = shaderBytes(mesh);
    expect(code).toContain('0.04');
    expect(code).toContain('0.1');
    expect(code).not.toMatch(/\b(?:NaN|undefined)\b/);
    const fire = policyLight(REAL_LIGHT_POLICIES.fire, [0, 0.4, 0]);
    const m = lightMoment(fire, [0.1, 0.8, 0]);
    const facing = reconstructLight(m.s, m.d, [0, -1, 0]);
    expect(lightLuminance(facing)).toBeGreaterThan(0);
    expect(lightLuminance(reconstructLight(m.s, m.d, [0, 1, 0]))).toBeCloseTo(0);
    for (const type of ['FIRE01', 'FIRE02', 'INITFIRE', 'SMOKEY2'])
      expect(animEmission({ id: 1, uid: 1, type, x: 0, y: 0, z: 0 }, 0.01, 0)).toBeNull();
    const blast = animEmission({ id: 2, uid: 2, type: 'EXPLOLRG', x: 0, y: 0, z: 0 }, 0.01, 0)!;
    expect(new Set(blast.colour).size).toBe(1);
    expect(new Set(blast.hot).size).toBe(1); // Added anim bloom cannot colour a grey translucent receiver.
    field.dispose();
    nodes.dispose();
    mesh.geometry.dispose();
    material.dispose();
  });
});

describe('round 6 E/F: visible short flashes, calm wreck glow and coil-to-unit capture', () => {
  it('sets the night-centre display increase without flattening the Gaussian, and keeps source timing', () => {
    for (const type of ['EXPLOSML', 'EXPLOMED', 'EXPLOLRG', 'TWLT070', 'burning']) {
      const p = type === 'burning' ? REAL_LIGHT_POLICIES.burning : animLightPolicy(type)!;
      const l = policyLight(p, [0, p.height, 0]);
      const at = (x: number) => lightLuminance(irradianceKnee(sample(l, [x, 0, 0]), 1, 1, 0.1));
      const peak = at(0),
        increase = ((0.1 + peak) / 0.1) ** (1 / 2.2) - 1;
      expect(increase).toBeGreaterThanOrEqual(type === 'burning' ? 0.3 : 0.8);
      expect(increase).toBeLessThanOrEqual(type === 'burning' ? 0.4 : 1.2);
      expect(at(-0.05) - 2 * peak + at(0.05)).toBeLessThan(-0.00001);
      expect(at(0.4)).toBeLessThan(peak * 0.95);
      if (type !== 'burning') {
        expect(p.hold).toBeGreaterThanOrEqual(0.06);
        expect(p.hold).toBeLessThanOrEqual(0.1);
        expect(p.fade).toBeGreaterThanOrEqual(0.2);
        expect(p.fade).toBeLessThanOrEqual(0.5);
        expect(heldPulseEnvelope(p.hold! + p.fade! + 0.01, p.hold!, p.fade!)).toBe(0);
      } else expect(p.flicker).toBeLessThanOrEqual(0.15);
    }
  });
  it('adds line and target light for a captured coil-to-unit bolt with no coil-to-coil arc present', () => {
    const nodes = new AtmosphereNodes({ realLight: true });
    const field = new RealLightField(16, 16, nodes.realLight!);
    const effects = new EffectsLayer(nodes);
    effects.physicalLights = field;
    const driver = new LiveEffectsDriver(effects, () => false);
    driver.garnishEnabled = false;
    driver.push({ anims: [] as LiveAnim[], bullets: [], particleSystems: [], beams: [bolt] }, 1, 0, new Vector3());
    driver.frame(1, new Vector3());
    const line = field.pools.get('flash')!.get('beam:0')!.light;
    const impact = field.pools.get('flash')!.get('beam-impact:0')!.light;
    expect(line.viewProjectedFootprint).toBe(true);
    expect(line.intensity[2]).toBeGreaterThan(line.intensity[0]);
    expect(impact.reach).toBeCloseTo(1.05);
    expect(impact.end).toBeUndefined();
    expect(lightLuminance(sample(impact, [impact.position[0], 0, impact.position[2]]))).toBeGreaterThan(0.1);
    driver.push({ anims: [], bullets: [], particleSystems: [], beams: [] }, 1.1, 1, new Vector3());
    driver.frame(1.2, new Vector3());
    field.update(1.2, fakeRenderer(), 1);
    expect(field.pools.get('flash')!.has('beam-impact:0')).toBe(false);
    effects.root.traverse((o) => {
      if (o instanceof Mesh) {
        o.geometry.dispose();
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
      }
    });
    field.dispose();
    nodes.dispose();
  });
});
