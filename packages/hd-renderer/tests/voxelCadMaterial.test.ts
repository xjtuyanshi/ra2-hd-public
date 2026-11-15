import { describe, expect, it, vi } from 'vitest';
import { DirectionalLight, HemisphereLight, PerspectiveCamera, Scene, WebGPURenderer, Vector3 } from 'three/webgpu';
import { lights } from 'three/tsl';
import { AtmosphereNodes } from '../src/atmosphere';
import { bindSlots } from '../src/materialSlots';
import { cadHouseTone, cadLookupBytes, cadPaletteEntry, CAD_CLASS } from '../src/ra2/voxelCadLookup';
import {
  cadVolumeTexture,
  voxelCadLookFor,
  voxelCadMaterialFor,
  VOXEL_CAD_DEFAULTS,
} from '../src/ra2/voxelCadMaterial';
import { buildCadModel } from '../src/ra2/voxelCadMesh';
import { cadVoxelGeometry } from '../src/ra2/voxelCad';
import { instantiateVoxel, updateVoxelNativeDepth } from '../src/ra2/voxels';
import type { HardSurfaceSpec } from '../src/hardSurfaceMesh';
import type { LookVolume } from '../src/hardSurfaceLookVolume';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { syntheticSection } from './fixtures/syntheticVoxels';
const palette = new Uint8Array(768);
const volume = (): LookVolume => ({
  origin: [-1, -1, -1],
  size: [4, 4, 4],
  res: 4,
  data: new Uint8Array(4 * 4 * 4 * 4),
  surfaceTexels: 1,
  grooveTexels: 0,
  seams: 0,
  dots: 0,
});
const part = (v = volume()) => ({ volume: v, ground: 0, height: 2, body: true, steel: false });
describe('CAD lookup and shared material', () => {
  it('uses the live-calibrated lighting and visible halo defaults', () => {
    expect(VOXEL_CAD_DEFAULTS).toMatchObject({
      brightness: 1.1,
      sunGain: 1.6,
      fill: 0.6,
      envStrength: 0.4,
      haloDark: 0.3,
    });
  });
  it('folds baked glints, levels warm paint, separates neutral steel and ignores barrel palette tint', () => {
    const p = palette.slice(),
      set = (i: number, c: number[]) => p.set(c, i * 3);
    set(64, [138, 138, 113]);
    set(65, [255, 255, 255]);
    set(66, [120, 120, 120]);
    set(80, [80, 60, 40]);
    set(57, [0, 0, 0]);
    set(48, [0, 0, 0]);
    expect(cadPaletteEntry(p, 64).colour).toEqual([0.64, 0.615, 0.565]);
    expect(cadPaletteEntry(p, 65).colour).toEqual(cadPaletteEntry(p, 64).colour);
    expect(cadPaletteEntry(p, 66)).toMatchObject({ kind: 'bareSteel', metalness: 0.45, roughness: 0.45 });
    expect(cadPaletteEntry(p, 80).colour[0]).toBeLessThan(0.64);
    expect(cadPaletteEntry(p, 57)).toMatchObject({ kind: 'track', colour: [0.12, 0.12, 0.115] });
    expect(cadPaletteEntry(p, 48).colour).toEqual([0.3, 0.31, 0.34]);
    expect(cadPaletteEntry(p, 64, true)).toEqual(cadPaletteEntry(p, 80, true));
    expect(cadPaletteEntry(p, 64, true)).toMatchObject({ colour: [0.42, 0.44, 0.47], metalness: 0.85, roughness: 0.3 });
    const bytes = cadLookupBytes(p);
    expect(bytes[(256 + 66) * 4 + 2]).toBe(CAD_CLASS.bareSteel);
  });
  it('retains player hue while bounding chroma and levelling the house ramp', () => {
    const red = cadHouseTone([1, 0, 0]),
      blue = cadHouseTone([0, 0, 1]);
    expect(red[0]).toBeCloseTo(0.72);
    expect(red[1]).toBeCloseTo(0.17);
    expect(blue).toEqual([red[2], red[1], red[0]]);
    const p = palette.slice();
    p.set([255, 0, 0], 16 * 3);
    const h = cadPaletteEntry(p, 16);
    expect(h.kind).toBe('house');
    expect(Math.max(...h.colour)).toBeCloseTo(0.72 * 0.9);
    expect(h.metalness).toBe(0);
  });
  it('shares one prototype graph across two palettes and two parts, with independent texture slots and ownership', () => {
    const nodes = new AtmosphereNodes(),
      a = palette.slice(),
      b = palette.slice();
    b.fill(128);
    const p = part(),
      q = part();
    const mats = [
      voxelCadMaterialFor(a, p, nodes),
      voxelCadMaterialFor(a, q, nodes),
      voxelCadMaterialFor(b, p, nodes),
      voxelCadMaterialFor(b, q, nodes),
    ];
    const state = voxelCadLookFor(nodes);
    expect(state.graphs.size).toBe(1);
    expect(new Set(mats).size).toBe(4);
    for (const m of mats) {
      expect(m.colorNode).toBe(mats[0].colorNode);
      expect(m.normalNode).toBe(mats[0].normalNode);
    }
    const graph = [...state.graphs.values()][0];
    bindSlots(mats[0]);
    const first = graph.lookup.value,
      vol = graph.volume.value;
    bindSlots(mats[3]);
    expect(graph.lookup.value).not.toBe(first);
    expect(graph.volume.value).not.toBe(vol);
    expect(graph.volume.value.isData3DTexture).toBe(true);
    expect(graph.volume.value.image.data).toBe(q.volume.data);
    bindSlots(mats[0].clone());
    expect(graph.lookup.value).toBe(first);
    const disposed = vi.spyOn(state.volumes.get(p.volume)!, 'dispose');
    nodes.dispose();
    expect(disposed).toHaveBeenCalledOnce();
  });
  it.each(
    [false, true].flatMap((forceWebGL) =>
      ['physical', 'native-ground', 'translucent'].map((pass) => ({ forceWebGL, pass })),
    ),
  )('compiles volume sampling and groove derivatives ($pass, WebGL=$forceWebGL)', ({ forceWebGL, pass }) => {
    const nodes = new AtmosphereNodes();
    const model: VoxelModel = {
      id: 'SYNTHETIC',
      palette: 'unit',
      normalsMode: 4,
      parts: [{ role: 'body', sections: [syntheticSection(2, true)] }],
    };
    const spec: HardSurfaceSpec = {
      units: 'lattice',
      parts: { body: { primitives: [{ type: 'box', min: [0, 0, 0], max: [2, 2, 2] }] } },
    };
    const parts = cadVoxelGeometry(model, buildCadModel(model, spec, palette), palette, nodes),
      unit = instantiateVoxel(parts, parts[0].material!, nodes);
    if (pass !== 'physical') updateVoxelNativeDepth(unit, new Vector3());
    if (pass === 'translucent') unit.root.setOpacity(0.5);
    const mesh = unit.root.partMeshes[0];
    bindSlots(mesh.material);
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const canvas = {
      width: 64,
      height: 64,
      style: {},
      addEventListener() {},
      removeEventListener() {},
      setAttribute() {},
    };
    const renderer = new WebGPURenderer({ canvas: canvas as unknown as HTMLCanvasElement, forceWebGL });
    vi.spyOn(renderer, 'hasFeature').mockReturnValue(false);
    try {
      const builder = renderer.backend.createNodeBuilder(mesh, renderer);
      builder.scene = new Scene();
      builder.camera = new PerspectiveCamera();
      builder.lightsNode = lights([new DirectionalLight(), new HemisphereLight()]);
      builder.build();
      expect(errors.mock.calls).toEqual([]);
      expect(builder.fragmentShader).toMatch(/texture_3d|sampler3D/);
      expect(builder.fragmentShader).toMatch(/dpdx|dFdx/);
      expect(builder.fragmentShader).not.toMatch(/\b(?:NaN|undefined)\b/);
      expect(builder.vertexShader).toContain('lookCode');
    } finally {
      errors.mockRestore();
      unit.root.disposeOpacityMaterials();
      for (const p of parts) p.geometry.dispose();
      nodes.dispose();
    }
  });
  it('validates knobs and uses filtered, unclamped-colour-space 3D data', () => {
    const nodes = new AtmosphereNodes(),
      state = voxelCadLookFor(nodes);
    state.set('wear', 0.2);
    expect(state.get().wear).toBe(0.2);
    expect(() => state.set('bogus', 1)).toThrow();
    expect(() => state.set('wear', NaN)).toThrow();
    const v = volume(),
      t = cadVolumeTexture(v);
    expect(t.image.data).toBe(v.data);
    expect(t.image.depth).toBe(4);
    expect(t.colorSpace).toBe('');
    t.dispose();
    nodes.dispose();
  });
});
