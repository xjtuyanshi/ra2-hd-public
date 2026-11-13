import { describe, expect, it } from 'vitest';
import { AtmosphereNodes } from '../src/atmosphere';
import {
  ALIVE,
  ALIVE_DEFAULTS,
  ALIVE_CLOUD_TINT,
  aliveDayWeight,
  aliveDrift,
  aliveGust,
  aliveHash,
  aliveNoise,
  aliveGradientNoise,
  grassRipple,
  cloudCoverage,
  cloudShadowColour,
  parseAliveParams,
  resolveAlive,
  treeSway,
} from '../src/livingWorld';
import { isLivingTree } from '../src/ra2/livingTrees';
import { canHaveLivingGrass } from '../src/ra2/livingGrass';

describe('living-world opt-in parameters', () => {
  it('keeps defaults off and requires an explicit 1 at the selected entry point', () => {
    expect(parseAliveParams(new URLSearchParams())).toEqual(ALIVE_DEFAULTS);
    for (const value of ['0', 'true', 'on', '', 'NaN', '-1']) {
      expect(parseAliveParams(new URLSearchParams({ alive: value })).clouds).toBe(false);
    }
    const params = new URLSearchParams('hd-alive=1&alive-cloud=.3&alive-sway=1.5&alive-speed=.5');
    expect(parseAliveParams(params, 'hd-alive')).toEqual({
      clouds: true,
      wind: true,
      debugCloud: false,
      cloud: 0.3,
      sway: 1.5,
      speed: 0.5,
    });
    expect(parseAliveParams(params).wind).toBe(false);
    expect(parseAliveParams(new URLSearchParams('alive-cloud=.4')).clouds).toBe(false);
  });

  it('bounds tunables, preserves explicit zero and ignores empty/nonfinite values', () => {
    expect(parseAliveParams(new URLSearchParams('alive=1&alive-cloud=0&alive-sway=-1&alive-speed=0'))).toEqual({
      clouds: true,
      wind: true,
      debugCloud: false,
      cloud: 0,
      sway: 0,
      speed: 0,
    });
    expect(parseAliveParams(new URLSearchParams('alive-cloud=Infinity&alive-sway=&alive-speed=NaN'))).toEqual(
      ALIVE_DEFAULTS,
    );
    expect(parseAliveParams(new URLSearchParams('alive-cloud=9&alive-sway=9&alive-speed=9'))).toMatchObject({
      cloud: 2,
      sway: 3,
      speed: 4,
    });
  });

  it('selects only the explicit raw cloud debug mode without implicitly enabling wind/clouds', () => {
    expect(parseAliveParams(new URLSearchParams('alive=1&alive-debug=cloud')).debugCloud).toBe(true);
    expect(parseAliveParams(new URLSearchParams('alive-debug=cloud'))).toMatchObject({
      clouds: false,
      wind: false,
      debugCloud: true,
    });
    expect(parseAliveParams(new URLSearchParams('alive-debug=true')).debugCloud).toBe(false);
  });

  it('toggles parts without losing tuning or changing global defaults', () => {
    const on = resolveAlive({ ...resolveAlive(true), cloud: 0.3, speed: 2 });
    const partial = resolveAlive({ wind: false }, on);
    expect(partial).toEqual({ ...on, wind: false });
    expect(resolveAlive(true, resolveAlive(false, partial))).toEqual(on);
    expect(ALIVE_DEFAULTS.clouds).toBe(false);
  });
});

describe('deterministic cloud field', () => {
  it('keeps a luminance-matched cool 0.64 core, exact clear gaps and a linear strength scale', () => {
    const luminance = (c: readonly number[]) => c[0]! * 0.2126 + c[1]! * 0.7152 + c[2]! * 0.0722;
    expect(luminance(ALIVE_CLOUD_TINT)).toBeCloseTo(1, 14);
    expect(luminance(cloudShadowColour(1))).toBeCloseTo(0.64, 14);
    expect(luminance(cloudShadowColour(1, 0.5))).toBeCloseTo(0.82, 14);
    expect(cloudShadowColour(1)[2]).toBeGreaterThan(cloudShadowColour(1)[0]!);
    for (const coverage of [0, 0.1, 0.5, 1]) {
      expect(cloudShadowColour(coverage, 0)).toEqual([1, 1, 1]);
      expect(cloudShadowColour(coverage, 1, 0)).toEqual([1, 1, 1]);
    }
    expect(cloudShadowColour(0)).toEqual([1, 1, 1]);
  });

  it('has bounded, continuous value noise across positive and negative cell boundaries', () => {
    for (let x = -12; x < 12; x++)
      for (let z = -12; z < 12; z++) {
        expect(aliveNoise(x, z)).toBe(aliveHash(x, z));
        expect(aliveNoise(x + 0.37, z - 0.51)).toBeGreaterThanOrEqual(0);
        expect(aliveNoise(x + 0.37, z - 0.51)).toBeLessThan(1);
        expect(Math.abs(aliveNoise(x - 1e-5, z + 0.3) - aliveNoise(x + 1e-5, z + 0.3))).toBeLessThan(1e-7);
      }
  });

  it('uses zero-centred continuous gradient noise with no lattice seam', () => {
    for (let x = -8; x <= 8; x++)
      for (let z = -8; z <= 8; z++) {
        expect(Math.abs(aliveGradientNoise(x, z))).toBe(0);
        expect(Math.abs(aliveGradientNoise(x + 0.37, z - 0.51))).toBeLessThan(0.71);
        expect(Math.abs(aliveGradientNoise(x - 1e-5, z + 0.3) - aliveGradientNoise(x + 1e-5, z + 0.3))).toBeLessThan(
          0.00003,
        );
      }
  });

  it('retains lit gaps and patch cores in a zoomed forest crop rather than global exposure drift', () => {
    for (const t of [0, 2, 4, 6, 8, 10, 12]) {
      const field: number[] = [];
      for (let x = 80; x < 90; x += 0.2) for (let z = 132; z < 138; z += 0.2) field.push(cloudCoverage(x, z, t, 0.6));
      expect(Math.min(...field)).toBe(0);
      expect(Math.max(...field)).toBe(1);
      expect(field.filter((v) => v === 0).length / field.length).toBeGreaterThan(0.1);
      expect(field.filter((v) => v === 1).length / field.length).toBeGreaterThan(0.1);
    }
    for (const t of [0, 4, 8, 12]) {
      const field: number[] = [];
      for (let x = 20; x < 60; x += 0.2) for (let z = 20; z < 60; z += 0.2) field.push(cloudCoverage(x, z, t, 0.6));
      const coverage = field.filter((v) => v > 0.5).length / field.length;
      expect(coverage).toBeGreaterThan(0.4);
      expect(coverage).toBeLessThan(0.5);
    }
  });

  it('drifts in scene wind at 0.85 cells/second and freezes at speed zero', () => {
    expect(aliveDrift(10, 0)).toEqual([8.5, 0]);
    expect(aliveDrift(10, Math.PI / 2)[1]).toBeCloseTo(8.5);
    expect(Math.hypot(...aliveDrift(10, 0.6, 2))).toBeCloseTo(17);
    expect(cloudCoverage(13, 42, 100, 0.6, 0)).toBe(cloudCoverage(13, 42, 0, 0.6, 0));
    expect(cloudCoverage(13, 42, 10, 0.6)).toBe(cloudCoverage(13, 42, 10, 0.6));
  });

  it('has both clear gaps and deep shadows in a 40-cell field, with no 40-cell repeat', () => {
    const field: number[] = [];
    let shiftedDifference = 0;
    for (let x = 20; x < 60; x++)
      for (let z = 20; z < 60; z++) {
        const a = cloudCoverage(x, z, 0, 0.6);
        field.push(a);
        shiftedDifference += Math.abs(a - cloudCoverage(x + 40, z + 40, 0, 0.6));
      }
    expect(Math.min(...field)).toBe(0);
    expect(Math.max(...field)).toBe(1);
    expect(field.filter((v) => v < 0.1).length).toBeGreaterThan(100);
    expect(field.filter((v) => v > 0.9).length).toBeGreaterThan(100);
    expect(shiftedDifference / field.length).toBeGreaterThan(0.1);
  });

  it('removes cloud shadows at night, under full overcast and on airless maps', () => {
    expect(aliveDayWeight(0, 0)).toBe(1);
    expect(aliveDayWeight(1, 0)).toBe(0);
    expect(aliveDayWeight(0, 1)).toBe(0);
    expect(aliveDayWeight(0.5, 0.4)).toBeCloseTo(0.3);
    expect(aliveDayWeight(0, 0, true)).toBe(0);
  });
});

describe('tree-only wind and travelling gusts', () => {
  it('does not confuse the cache terrain-object class with a botanical tree', () => {
    for (const type of ['TREE01', 'TREE36', 'tree05', 'TIBTRE01', 'TIBTRE03']) {
      expect(isLivingTree({ kind: 'tree', type })).toBe(true);
      expect(isLivingTree({ kind: 'building', type })).toBe(false);
    }
    for (const type of ['LT_GEN01', 'ROCK01', 'T01_ROCK', 'TIB01', 'TREEHOUSE', 'TREE', 'MOD_PALM']) {
      expect(isLivingTree({ kind: 'tree', type })).toBe(false);
    }
  });

  it('moves gusts with the same wind and speed as clouds', () => {
    const direction = 0.6;
    const [dx, dz] = aliveDrift(4, direction);
    expect(aliveGust(14 + dx, 27 + dz, 4, direction)).toBeCloseTo(aliveGust(14, 27, 0, direction), 12);
    expect(aliveGust(14, 27, 4, direction)).not.toBeCloseTo(aliveGust(14, 27, 0, direction), 3);
  });

  it('keeps phone-visible idle and gust crown excursions and a ten-second regional gust period', () => {
    const idle: number[] = [],
      peaks: number[] = [];
    for (let t = 0; t < 60; t += 0.01) {
      const gust = aliveGust(85, 135, t, 0.6);
      const crown = treeSway(85, 135, 1, t, 0.6);
      if (gust < 0.01) idle.push(crown);
      if (gust > 0.95) peaks.push(crown);
    }
    expect(Math.min(...idle)).toBeLessThan(-3.5);
    expect(Math.max(...idle)).toBeGreaterThan(3.5);
    expect(Math.min(...peaks)).toBeLessThan(-5.6);
    expect(Math.max(...peaks)).toBeGreaterThan(5.6);
    expect(aliveGust(85, 135, 10, 0.6)).toBeCloseTo(aliveGust(85, 135, 0, 0.6));
  });

  it('bounds grass luminance and moves its crest with the same gust, freezing with the renderer clock', () => {
    for (let t = 0; t < 12; t += 0.05) {
      expect(Math.abs(grassRipple(85, 135, t, 0.6))).toBeLessThanOrEqual(0.05);
      expect(grassRipple(85, 135, t, 0.6)).toBeCloseTo((2 * aliveGust(85, 135, t, 0.6) - 1) * 0.05);
    }
    const [dx, dz] = aliveDrift(4, 0.6);
    expect(grassRipple(85 + dx, 135 + dz, 4, 0.6)).toBeCloseTo(grassRipple(85, 135, 0, 0.6));
    expect(grassRipple(85, 135, 10, 0.6, 0)).toBe(grassRipple(85, 135, 0, 0.6, 0));
    for (const terrainType of [9, 10, 11, 12, 15]) expect(canHaveLivingGrass('temperate', { terrainType })).toBe(false);
    for (const theater of ['snow', 'desert', 'lunar'] as const)
      expect(canHaveLivingGrass(theater, { terrainType: 13 })).toBe(false);
    expect(canHaveLivingGrass('temperate', { terrainType: 13 })).toBe(true);
    expect(canHaveLivingGrass('urban', { terrainType: 14 })).toBe(true);
  });

  it('pins the foot, bounds the crown and retains smooth subpixel movement and per-tree phase', () => {
    for (let t = 0; t < 15; t += 0.1) {
      expect(treeSway(13, 42, 0, t, 0.6)).toBeCloseTo(0);
      expect(Math.abs(treeSway(13, 42, 1, t, 0.6))).toBeLessThanOrEqual(ALIVE.gustPixels);
      expect(treeSway(13, 42, 0.5, t, 0.6)).toBeCloseTo(treeSway(13, 42, 1, t, 0.6) * 0.5 ** 1.4);
      expect(Math.abs(treeSway(13, 42, 1, t + 0.001, 0.6) - treeSway(13, 42, 1, t, 0.6))).toBeLessThan(0.02);
    }
    expect(treeSway(13, 42, 1, 1, 0.6)).not.toBe(treeSway(14, 42, 1, 1, 0.6));
    expect(treeSway(13, 42, 1, 10, 0.6, 1, 0)).toBe(treeSway(13, 42, 1, 0, 0.6, 1, 0));
  });
});

describe('renderer-local living-world uniforms', () => {
  it('starts neutral before its first update and remains neutral after a tuned on/off cycle', () => {
    const nodes = new AtmosphereNodes();
    const alive = nodes.alive;
    try {
      const neutral = () => {
        expect(alive.cloudStrength.value).toBe(0);
        expect(alive.windAmplitude.value).toBe(0);
        expect(alive.debugCloud.value).toBe(0);
        expect(Number.isFinite(alive.clock.value)).toBe(true);
        expect(alive.windDirection.value.length()).toBeCloseTo(1);
        expect(cloudShadowColour(1, alive.cloudStrength.value / ALIVE.cloudDarkening)).toEqual([1, 1, 1]);
      };
      neutral();
      alive.set({ clouds: true, wind: true, cloud: 2, sway: 3, speed: 4, debugCloud: true });
      alive.update(17, -1.2, 0, 0, false);
      alive.set(false);
      alive.update(19, -1.2, 0, 0, false);
      neutral();
      alive.set({ debugCloud: true });
      alive.update(20, 0.6, 0, 0, false);
      neutral();
    } finally {
      nodes.dispose();
    }
  });

  it('updates on the supplied renderer clock, reuses its graph and restores exact neutral values', () => {
    const nodes = new AtmosphereNodes();
    const mirror = new AtmosphereNodes();
    const shadow = nodes.alive.shadow;
    nodes.alive.set({ ...resolveAlive(true), speed: 2, debugCloud: true });
    nodes.alive.update(5, 0.6, 0, 0, false);
    expect(nodes.alive.clock.value).toBe(10);
    expect(nodes.alive.cloudStrength.value).toBe(0.36);
    expect(nodes.alive.windAmplitude.value).toBe(1);
    expect(mirror.alive.cloudStrength.value).toBe(0);
    mirror.copyValuesFrom(nodes);
    expect(mirror.alive.clock.value).toBe(10);
    expect(mirror.alive.cloudStrength.value).toBe(0.36);
    expect(mirror.alive.debugCloud.value).toBe(1);
    nodes.alive.set(false);
    nodes.alive.update(5, 0.6, 0, 0, false);
    expect(nodes.alive.shadow).toBe(shadow);
    expect(nodes.alive.cloudStrength.value).toBe(0);
    expect(nodes.alive.windAmplitude.value).toBe(0);
    expect(nodes.alive.debugCloud.value).toBe(0);
    nodes.dispose();
    mirror.dispose();
  });
});
