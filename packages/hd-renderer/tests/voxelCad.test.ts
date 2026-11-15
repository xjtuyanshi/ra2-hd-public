import { afterEach, describe, expect, it, vi } from 'vitest';
import { Box3, Vector3 } from 'three/webgpu';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { AtmosphereNodes } from '../src/atmosphere';
import type { HdRenderer } from '../src/HdRenderer';
import type { Ra2Cache } from '../src/ra2/cache';
import type { HdWorld } from '../src/ra2/world';
import type { HardSurfaceSpec } from '../src/hardSurfaceMesh';
import { buildCadModel } from '../src/ra2/voxelCadMesh';
import { VoxelCadLibrary, cadVoxelGeometry, adoptCadVoxel } from '../src/ra2/voxelCad';
import { answerCadRequest } from '../src/ra2/voxelCadWorker';
import { buildStandInVoxelParts } from '../src/ra2/voxelSmoothGeometry';
import { buildVoxelGeometry, instantiateVoxel, updateVoxelNativeDepth, VoxelUnit } from '../src/ra2/voxels';
import { buildSmoothSection, type SmoothMeshData } from '../src/ra2/voxelSmooth';
import { VOXEL_TO_WORLD } from '../src/ra2/voxelScale';
import { VoxelUnits } from '../src/live/voxelUnits';
import { voxelLookParams } from '../sandbox/look';
import { syntheticSection } from './fixtures/syntheticVoxels';
import { silhouetteGeometry } from '../src/ra2/voxelShadow';

const roles = ['body', 'turret', 'barrel'] as const;
function fixture() {
  const model: VoxelModel = {
    id: 'SYNTHETIC',
    palette: 'unit',
    normalsMode: 4,
    turretOffset: 32,
    parts: roles.map((role, i) => {
      const section = syntheticSection(2, true);
      section.transform[3] += i;
      section.transform[7] -= i;
      section.transform[11] += i * 3;
      return { role, sections: [section] };
    }),
  };
  const spec: HardSurfaceSpec = {
    units: 'voxel lattice',
    parts: Object.fromEntries(
      roles.map((role) => [role, { primitives: [{ type: 'box', min: [0, 0, 0], max: [2, 2, 2], chamfer: 0 }] }]),
    ),
  };
  return { model, spec };
}
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it('matches stand-in model-frame bounds, HVA placement, roles and articulated pivots', () => {
  const { model, spec } = fixture();
  const nodes = new AtmosphereNodes(),
    palette = new Uint8Array(768);
  const meshes = buildCadModel(model, spec),
    cad = cadVoxelGeometry(model, meshes, palette, nodes),
    stand = buildStandInVoxelParts(model);
  for (let i = 0; i < cad.length; i++) {
    cad[i].geometry.computeBoundingBox();
    stand[i].geometry.computeBoundingBox();
    const a = cad[i].geometry.boundingBox!,
      b = stand[i].geometry.boundingBox!;
    expect(a.min.distanceTo(b.min)).toBeLessThan(VOXEL_TO_WORLD);
    expect(a.max.distanceTo(b.max)).toBeLessThan(VOXEL_TO_WORLD);
    expect(cad[i].role).toBe(roles[i]);
    expect(cad[i].turretOffset).toBe(32 / 256);
    expect(cad[i].material!.name).toBe('voxel-remaster');
    expect(silhouetteGeometry(cad[i].geometry)).not.toBe(cad[i].geometry);
    expect(meshes[i].pindex.every((index, t) => Math.floor(meshes[i].pshade[t] / 16) === Math.floor(index / 16))).toBe(
      true,
    );
    for (let t = 0; t < meshes[i].pindex.length; t += 3)
      expect(Array.from(meshes[i].pindex.subarray(t, t + 3))).toEqual(Array(3).fill(meshes[i].pindex[t]));
  }
  const unit = instantiateVoxel(cad, cad[0].material!, nodes);
  expect(unit.turret!.children).toEqual([unit.root.partMeshes[1], unit.root.partMeshes[2]]);
  expect(unit.turret!.position.x).toBe(32 / 256);
  for (const p of [...cad, ...stand]) p.geometry.dispose();
  nodes.dispose();
});

it('preserves unit pose, turret yaw, native depth, opacity, tint and source identity during adoption', () => {
  const { model, spec } = fixture(),
    nodes = new AtmosphereNodes(),
    palette = new Uint8Array(768);
  const old = buildVoxelGeometry(model, palette, nodes),
    next = cadVoxelGeometry(model, buildCadModel(model, spec), palette, nodes);
  const unit = instantiateVoxel(old, next[0].material!, nodes),
    sources = [...unit.root.partMeshes];
  unit.root.position.set(10, 3, 20);
  unit.root.rotation.y = 0.7;
  unit.turret!.rotation.y = 1.2;
  unit.root.setOpacity(0.4);
  unit.root.partMeshes[0].userData.keep = 'tint-marker';
  updateVoxelNativeDepth(unit, new Vector3(10, 3, 20), 0.12, true);
  adoptCadVoxel(unit.root, old, next);
  expect(unit.root.partMeshes).toEqual(sources);
  expect(unit.root.position.toArray()).toEqual([10, 3, 20]);
  expect(unit.turret!.rotation.y).toBe(1.2);
  expect(unit.root.bodyOpacity).toBe(0.4);
  expect(unit.root.partMeshes[0].material).toMatchObject({ transparent: true, opacity: 0.4 });
  expect(unit.root.partMeshes[0].userData.keep).toBe('tint-marker');
  expect(unit.root.nativeDepth.bonus).toBe(0.12);
  expect(unit.root.partMeshes.map((p) => p.geometry)).toEqual(next.map((p) => p.geometry));
  expect(new Box3().setFromObject(unit.root).isEmpty()).toBe(false);
  unit.root.disposeOpacityMaterials();
  for (const p of next) p.geometry.dispose();
  nodes.dispose();
});

describe('optional CAD library', () => {
  it.each([
    ['text/html', '<!doctype html><html>Vite fallback</html>', false],
    ['', '{"unexpected":"non-JSON media type"}', false],
    ['text/plain', 'not a spec', false],
    ['application/json', ' \n <html>fallback with the wrong header</html>', false],
    ['application/json; charset=utf-8', '{broken json', true],
    ['application/problem+json', '', true],
  ])('settles missing SPA responses silently but reports real JSON parse errors (%s)', async (type, body, failure) => {
    const { model } = fixture();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const fetch = vi.fn(async () => new Response(body, { headers: type ? { 'content-type': type } : {} }));
    const builder = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const lib = new VoxelCadLibrary('/cad', builder);
    expect(await lib.load(model)).toBeNull();
    expect(await lib.load({ ...model })).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(builder).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledTimes(failure ? 1 : 0);
    lib.dispose();
  });
  it('silently caches 404 and warns once for invalid JSON, HTTP and build failures', async () => {
    const { model, spec } = fixture(),
      warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    for (const [status, body, throws] of [
      [404, null, false],
      [200, {}, false],
      [500, null, false],
      [200, spec, true],
    ] as const) {
      const fetch = vi.fn(
        async () => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }),
      );
      vi.stubGlobal('fetch', fetch);
      const builder = vi.fn(async () => {
        if (throws) throw new Error('build');
        return buildCadModel(model, spec);
      });
      const lib = new VoxelCadLibrary('/cad', builder),
        before = warn.mock.calls.length;
      expect(await lib.load(model)).toBeNull();
      expect(await lib.load(model)).toBeNull();
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls.length - before).toBe(status === 404 ? 0 : 1);
      lib.dispose();
    }
  });
  it('uses the same mesh implementation in a Worker, terminates it and cancels outstanding work', async () => {
    const { model, spec } = fixture();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(spec), { headers: { 'content-type': 'application/json' } })),
    );
    class FakeWorker {
      static instances: FakeWorker[] = [];
      onmessage: ((event: { data: ReturnType<typeof answerCadRequest> }) => void) | null = null;
      onerror: unknown = null;
      request?: Parameters<typeof answerCadRequest>[0];
      terminated = false;
      constructor() {
        FakeWorker.instances.push(this);
      }
      postMessage(request: Parameters<typeof answerCadRequest>[0]) {
        this.request = request;
      }
      terminate() {
        this.terminated = true;
      }
      answer() {
        this.onmessage?.({ data: answerCadRequest(this.request!) });
      }
    }
    vi.stubGlobal('Worker', FakeWorker);
    const lib = new VoxelCadLibrary('/cad'),
      pending = lib.load(model);
    await vi.waitFor(() => expect(FakeWorker.instances).toHaveLength(1));
    FakeWorker.instances[0].answer();
    expect(await pending).toEqual(buildCadModel(model, spec));
    expect(FakeWorker.instances[0].terminated).toBe(true);
    const other = lib.load({ ...model });
    await vi.waitFor(() => expect(FakeWorker.instances).toHaveLength(2));
    lib.dispose();
    expect(await other).toBeNull();
    expect(FakeWorker.instances[1].terminated).toBe(true);
    expect(await lib.load(model)).toBeNull();
  });
});

it('keeps live engine geometry until CAD arrives, swaps all palettes in place, and leaves missing models unchanged', async () => {
  const { model, spec } = fixture(),
    nodes = new AtmosphereNodes();
  const hd = {
    nodes,
    settings: { voxelLook: 'engine' },
    terrainDepth: { setCaptured: vi.fn() },
  } as unknown as HdRenderer;
  const cache = {
    paletteRgb: new Map([
      ['unit', new Uint8Array(768)],
      ['house', new Uint8Array(768)],
    ]),
  } as Ra2Cache;
  let finish!: (meshes: ReturnType<typeof buildCadModel>) => void;
  const load = vi.spyOn(VoxelCadLibrary.prototype, 'load').mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, { voxelCad: '/cad' });
  const a = units.build(model, 'unit', 'a'),
    b = units.build(model, 'house', 'b');
  const root = a.root as VoxelUnit,
    old = root.partMeshes[0].geometry;
  expect(hd.settings.voxelLook).toBe('engine');
  expect(root.partMeshes[0].material).toBe(buildVoxelGeometry(model, cache.paletteRgb.get('unit')!, nodes)[0].material);
  a.root.position.set(1, 2, 3);
  a.turret!.rotation.y = 0.3;
  finish(buildCadModel(model, spec));
  await vi.waitFor(() => expect(units.revision).toBe(1));
  expect(root.partMeshes[0].geometry).not.toBe(old);
  expect(root.partMeshes[0].geometry).toBe((b.root as VoxelUnit).partMeshes[0].geometry);
  expect(root.partMeshes[0].material.name).toBe('voxel-remaster');
  expect(a.root.position.toArray()).toEqual([1, 2, 3]);
  expect(a.turret!.rotation.y).toBe(0.3);
  load.mockResolvedValue(null);
  const missing = { ...model, id: 'MISSING' },
    c = units.build(missing, 'unit', 'c'),
    before = (c.root as VoxelUnit).partMeshes[0].geometry;
  await Promise.resolve();
  expect((c.root as VoxelUnit).partMeshes[0].geometry).toBe(before);
  units.dispose(a);
  units.dispose(b);
  units.dispose(c);
  units.disposeAll();
  nodes.dispose();
});

it('gates sandbox CAD independently from other vehicle looks and sanitises overrides', () => {
  expect(voxelLookParams(new URLSearchParams('vcad=private')).voxelCad).toBeUndefined();
  expect(voxelLookParams(new URLSearchParams('veh=cad'))).toMatchObject({
    voxelLook: 'engine',
    voxelCad: '/_library/vehicles-cad',
  });
  expect(voxelLookParams(new URLSearchParams('veh=cad&vcad=/custom/cad/')).voxelCad).toBe('/custom/cad');
  for (const path of ['../x', 'x:y', 'x\\y'])
    expect(voxelLookParams(new URLSearchParams({ veh: 'cad', vcad: path })).voxelCad).toBe('/_library/vehicles-cad');
});

it('does not let a late redraw projection repaint CAD, and ignores CAD results after disposal', async () => {
  const voxels = await import('../src/ra2/voxels');
  const project = vi.spyOn(voxels, 'projectVoxelMaterials').mockReturnValue(false);
  const { model, spec } = fixture(),
    nodes = new AtmosphereNodes();
  const hd = {
    nodes,
    settings: { voxelLook: 'engine' },
    terrainDepth: { setCaptured: vi.fn() },
  } as unknown as HdRenderer;
  const cache = { paletteRgb: new Map([['unit', new Uint8Array(768)]]) } as Ra2Cache;
  let finish!: (meshes: ReturnType<typeof buildCadModel>) => void;
  let repaint!: (asset: null) => void;
  vi.spyOn(VoxelCadLibrary.prototype, 'load').mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const projections = {
    get: () => null,
    loadType: () =>
      new Promise<null>((resolve) => {
        repaint = resolve;
      }),
  };
  const units = new VoxelUnits(hd, { projections } as unknown as HdWorld, cache, undefined, { voxelCad: '/cad' });
  const unit = units.build(model, 'unit', 'one');
  expect(project).toHaveBeenCalledTimes(1);
  finish(buildCadModel(model, spec));
  await vi.waitFor(() => expect(units.revision).toBe(1));
  repaint(null);
  await Promise.resolve();
  expect(project).toHaveBeenCalledTimes(1);
  expect((unit.root as VoxelUnit).partMeshes[0].material.name).toBe('voxel-remaster');
  units.dispose(unit);
  units.disposeAll();
  const later = new VoxelUnits(hd, {} as HdWorld, cache, undefined, { voxelCad: '/cad' }),
    other = later.build(model, 'unit', 'two');
  const before = (other.root as VoxelUnit).partMeshes[0].geometry;
  later.dispose(other);
  later.disposeAll();
  finish(buildCadModel(model, spec));
  await Promise.resolve();
  expect((other.root as VoxelUnit).partMeshes[0].geometry).toBe(before);
  nodes.dispose();
});

it.each(['cad-first', 'smooth-first'])('keeps CAD final when smooth stand-in adoption races: %s', async (order) => {
  const { model, spec } = fixture(),
    nodes = new AtmosphereNodes();
  const hd = {
    nodes,
    settings: { voxelLook: 'engine' },
    terrainDepth: { setCaptured: vi.fn() },
  } as unknown as HdRenderer;
  const cache = { paletteRgb: new Map([['unit', new Uint8Array(768)]]) } as Ra2Cache;
  let finishCad!: (meshes: ReturnType<typeof buildCadModel>) => void;
  vi.spyOn(VoxelCadLibrary.prototype, 'load').mockImplementation(
    () =>
      new Promise((resolve) => {
        finishCad = resolve;
      }),
  );
  const jobs: (() => void)[] = [];
  const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, {
    voxelLook: 'remaster',
    voxelCad: '/cad',
    smoothVoxel: { aoRays: 0 },
    smoothBuilder: {
      build(section, options) {
        return new Promise<SmoothMeshData>((resolve) => {
          jobs.push(() => resolve(buildSmoothSection(section, options)));
        });
      },
    },
  });
  const unit = units.build(model, 'unit', 'one'),
    root = unit.root as VoxelUnit;
  expect(units.isStandIn(unit)).toBe(true);
  const cad = buildCadModel(model, spec);
  if (order === 'cad-first') {
    finishCad(cad);
    await vi.waitFor(() => expect(units.revision).toBe(1));
    const geometry = root.partMeshes[0].geometry;
    for (const job of jobs) job();
    await vi.waitFor(() => expect(units.revision).toBe(2));
    expect(root.partMeshes[0].geometry).toBe(geometry);
  } else {
    for (const job of jobs) job();
    await vi.waitFor(() => expect(units.revision).toBe(1));
    const geometry = root.partMeshes[0].geometry;
    finishCad(cad);
    await vi.waitFor(() => expect(units.revision).toBe(2));
    expect(root.partMeshes[0].geometry).not.toBe(geometry);
  }
  expect(root.partMeshes[0].geometry.getAttribute('position').array).toEqual(cad[0].position);
  expect(units.isStandIn(unit)).toBe(false);
  units.dispose(unit);
  units.disposeAll();
  nodes.dispose();
});
