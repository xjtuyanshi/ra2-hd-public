import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three/webgpu';
import { BeamSegmentBuffer, laserCoreColour } from '../src/effects';
import { EMISSIVE_BUDGET } from '../src/realEmissive';
import {
  GAUSSIAN_REACH,
  lightLuminance,
  lightMoment,
  lightReach,
  reconstructLight,
  type PhysicalLight,
  type Point,
} from '../src/realLight';
import {
  BEAM_LIGHT_SATURATION,
  LIGHT_COLOURS,
  REAL_LIGHT_POLICIES,
  animLightPolicy,
  nativeBeamLight,
  policyLight,
} from '../src/ra2/realLights';

/** Upward-facing ground irradiance (noon units) at a receiver. */
const ground = (l: PhysicalLight, p: Point) => {
  const m = lightMoment(l, p);
  return lightLuminance(reconstructLight(m.s, m.d, [0, 1, 0]));
};
const saturation = (c: readonly number[]) => 1 - Math.min(...c) / Math.max(...c);

describe('round 5: soft, source-sized light (round-1 review A and C)', () => {
  it('lights a beam as a narrow soft line with no edge where the window closes', () => {
    // A Prism beam at unit height, 6 cells long along x; the ground across it at the middle.
    const l = nativeBeamLight({
      kind: 'laser',
      source: { x: 0, y: 0, z: 104 },
      target: { x: 6 * 256, y: 0, z: 104 },
      colour: [255, 255, 255],
    });
    expect(l.profile).toBe('gaussian');
    const across = Array.from({ length: 31 }, (_, i) => ground(l, [3, 0, i * 0.05]));
    for (let i = 1; i < across.length; i++) expect(across[i]).toBeLessThanOrEqual(across[i - 1]! + 1e-12);
    // About one cell each side at most, and nothing left to cut where the window closes.
    expect(across[20]! / across[0]!).toBeLessThan(0.05);
    const reach = lightReach(l);
    expect(reach).toBeLessThanOrEqual(1.2);
    const inside = ground(l, [3, 0, Math.sqrt(Math.max(0, (reach * 0.98) ** 2 - 0.4 ** 2))]);
    expect(inside / across[0]!).toBeLessThan(0.01);
    // The same peak for a long and a short beam: no band that brightens with beam length.
    const short = nativeBeamLight({
      kind: 'laser',
      source: { x: 2.5 * 256, y: 0, z: 104 },
      target: { x: 3.5 * 256, y: 0, z: 104 },
      colour: [255, 255, 255],
    });
    expect(ground(short, [3, 0, 0])).toBeCloseTo(across[0]!, 10);
  });
  it('flashes explosions 1.5-2.5 cells wide for at most half a second, fires and wrecks as small glows', () => {
    for (const type of ['EXPLOSML', 'EXPLOMED', 'EXPLOLRG', 'TWLT070']) {
      const p = animLightPolicy(type)!;
      expect(p.profile).toBe('gaussian');
      expect(p.hold! + p.fade!).toBeLessThanOrEqual(0.6);
      expect(p.fade!).toBeGreaterThanOrEqual(0.2);
      const l = policyLight(p, [0, p.height, 0]);
      const peak = ground(l, [0, 0, 0]);
      // Visible radius: where the flash still adds 6 % of its peak.
      let r = 0;
      while (r < 4 && ground(l, [r, 0, 0]) >= 0.06 * peak) r += 0.01;
      expect(r).toBeGreaterThanOrEqual(1.2);
      expect(r).toBeLessThanOrEqual(2.6);
      expect(ground(l, [lightReach(l) * 0.99, 0, 0]) / peak).toBeLessThan(0.03);
    }
    for (const name of ['fire', 'burning'] as const) {
      const p = REAL_LIGHT_POLICIES[name];
      expect(lightReach(policyLight(p, [0, p.height, 0]))).toBeLessThanOrEqual(1.6);
      expect(p.flicker).toBeGreaterThanOrEqual(0.3);
    }
    expect(REAL_LIGHT_POLICIES.burning.ground).toBeLessThan(REAL_LIGHT_POLICIES.fire.ground);
  });
  it('takes the source hue strongly desaturated, so lit ground keeps its own hue', () => {
    for (const c of [LIGHT_COLOURS.fire, LIGHT_COLOURS.explosion, LIGHT_COLOURS.muzzle, LIGHT_COLOURS.hot])
      expect(saturation(c)).toBeLessThanOrEqual(0.33);
    for (const c of [LIGHT_COLOURS.tesla, LIGHT_COLOURS.psychic]) expect(saturation(c)).toBeLessThanOrEqual(0.351);
    const bolt = nativeBeamLight({
      kind: 'bolt',
      source: { x: 0, y: 0, z: 104 },
      target: { x: 256, y: 0, z: 104 },
      colour: [82, 80, 255],
    });
    expect(saturation(bolt.intensity)).toBeCloseTo(BEAM_LIGHT_SATURATION, 10);
  });
  it('keeps every Gaussian window at or inside 2.7 widths', () => {
    for (const p of Object.values(REAL_LIGHT_POLICIES)) {
      if (!('profile' in p)) continue;
      const l = policyLight(p, [0, p.height, 0]);
      expect(lightReach(l)).toBeLessThanOrEqual(GAUSSIAN_REACH * p.radius + 1e-12);
    }
  });
});

describe('round 5: lasers keep the original hue (round-1 review B)', () => {
  it('draws a laser strand white without real light and in its own colour with it; the core never turns white', () => {
    // A BadGuy3 Prism beam: inner colour (137, 0, 197), drawn by the original as (225, 43, 255).
    const colour = new Vector3(...[137, 0, 197].map((c) => (c / 255) ** 2.2));
    const beams = [{ from: new Vector3(0, 1, 0), to: new Vector3(4, 1, 0), colour, bolt: false, seed: 1 }];
    const plain = new BeamSegmentBuffer();
    plain.set(beams);
    expect([...plain.colour.array.slice(0, 3)]).toEqual([1, 1, 1]);
    const real = new BeamSegmentBuffer();
    real.set(beams, false, true);
    expect([...real.colour.array.slice(0, 3)]).toEqual([...new Float32Array(colour.toArray())]);
    const core = laserCoreColour(colour.toArray() as [number, number, number]);
    expect(Math.max(...core) * EMISSIVE_BUDGET.core).toBeCloseTo(1, 12);
    expect(core[0]! / core[2]!).toBeCloseTo(colour.x / colour.z, 12);
  });
  it('keeps chrono warps a small glow', () => {
    expect(REAL_LIGHT_POLICIES.chrono.reach).toBeLessThanOrEqual(2);
  });
});
