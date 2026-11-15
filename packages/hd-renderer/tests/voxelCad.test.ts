import { afterEach, describe, expect, it, vi } from 'vitest';
import { Box3, Vector3, MeshStandardNodeMaterial } from 'three/webgpu';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { AtmosphereNodes } from '../src/atmosphere';
import type { HdRenderer } from '../src/HdRenderer';
import type { Ra2Cache } from '../src/ra2/cache';
import type { HdWorld } from '../src/ra2/world';
import type { HardSurfaceSpec } from '../src/hardSurfaceMesh';
import { buildCadModel } from '../src/ra2/voxelCadMesh';
import { VoxelCadLibrary, cadVoxelGeometry, adoptCadVoxel } from '../src/ra2/voxelCad';
import { answerCadRequest, cadReplyTransfers } from '../src/ra2/voxelCadWorker';
import { buildStandInVoxelParts } from '../src/ra2/voxelSmoothGeometry';
import { buildVoxelGeometry, instantiateVoxel, updateVoxelNativeDepth, VoxelUnit } from '../src/ra2/voxels';
import { buildSmoothSection, type SmoothMeshData } from '../src/ra2/voxelSmooth';
import { VOXEL_TO_WORLD } from '../src/ra2/voxelScale';
import { VoxelUnits } from '../src/live/voxelUnits';
import { voxelLookParams } from '../sandbox/look';
import { syntheticSection } from './fixtures/syntheticVoxels';
import { silhouetteGeometry } from '../src/ra2/voxelShadow';
import { buildHardSurfaceLiveMesh, hardSurfacePaintSampler, type VoxelPaint } from '../src/hardSurfaceLiveMesh';
import { gridIndex, hardSurfaceSolid } from '../src/hardSurfaceSolid';
import { voxelSectionFrame } from '../src/ra2/voxelSectionFrame';

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

it('retains the original sampler, topology and every old-PBR attribute independently of look folding', () => {
  const { model, spec } = fixture(),
    meshes = buildCadModel(model, spec, new Uint8Array(768));
  for (let part = 0; part < model.parts.length; part++) {
    const section = model.parts[part].sections[0],
      count = section.size.reduce((a, b) => a * b, 1);
    const paint: VoxelPaint = {
      size: section.size,
      data: new Uint8Array(count),
      index: new Int16Array(count).fill(-1),
      shade: new Float32Array(count),
    };
    const bytes = Uint8Array.from(atob(section.voxels), (c) => c.charCodeAt(0));
    for (let i = 0; i < bytes.length; i += 5) {
      const k = gridIndex(section.size, bytes[i], bytes[i + 1], bytes[i + 2]);
      paint.data[k] = 1;
      paint.index[k] = bytes[i + 3];
      paint.shade[k] = bytes[i + 3];
    }
    const raw = buildHardSurfaceLiveMesh(
        spec.parts[model.parts[part].role],
        0.5,
        hardSurfacePaintSampler(hardSurfaceSolid(paint), paint),
      ),
      frame = voxelSectionFrame(section),
      pbr = meshes[part].pbr;
    expect(pbr.vpos).toEqual(raw.lattice);
    expect(pbr.pindex).toEqual(raw.pindex);
    expect(pbr.pshade).toEqual(raw.pshade);
    expect(pbr.occlusion).toEqual(raw.occlusion);
    expect(pbr.curvature).toEqual(Float32Array.from(raw.chamfer, (v) => v * 2));
    const positions = raw.positions.slice(),
      normals = raw.normals.slice(),
      indices = raw.indices.slice();
    for (let v = 0; v < raw.pindex.length; v++) {
      positions.set(
        frame.position(...(Array.from(raw.positions.subarray(v * 3, v * 3 + 3)) as [number, number, number])),
        v * 3,
      );
      normals.set(
        frame.normal(...(Array.from(raw.normals.subarray(v * 3, v * 3 + 3)) as [number, number, number])),
        v * 3,
      );
    }
    if (frame.mirrored)
      for (let t = 0; t < indices.length; t += 3) [indices[t + 1], indices[t + 2]] = [indices[t + 2], indices[t + 1]];
    expect(pbr.position).toEqual(positions);
    expect(pbr.normal).toEqual(normals);
    expect(pbr.indices).toEqual(indices);
  }
});

it.each(['flat', 'hatch'])('applies the injected %s primitive policy before building look triangles', (paint) => {
  const { model, spec } = fixture(),
    palette = Uint8Array.from({ length: 768 }, (_, i) => i % 256);
  for (const part of Object.values(spec.parts)) part.primitives[0].paint = paint;
  for (const mesh of buildCadModel(model, spec, palette)) {
    const faces = new Map<string, Set<number>>();
    for (let i = 0; i < mesh.lookIndex.length; i++) {
      const normal = Array.from(mesh.normal.subarray(i * 3, i * 3 + 3), (v) => v.toFixed(5)).join(',');
      if (!faces.has(normal)) faces.set(normal, new Set());
      faces.get(normal)!.add(mesh.lookIndex[i]);
    }
    expect(faces.size).toBe(6);
    expect([...faces.values()].every((indices) => indices.size === 1)).toBe(true);
    expect(new Set(mesh.pbr.pindex).size).toBeGreaterThan(1);
  }
});

it('matches stand-in model-frame bounds, HVA placement, roles and articulated pivots', () => {
  const { model, spec } = fixture();
  const nodes = new AtmosphereNodes(),
    palette = new Uint8Array(768);
  const meshes = buildCadModel(model, spec, new Uint8Array(768)),
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
    expect(cad[i].material!.name).toBe('voxel-cad-painted-steel');
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
    next = cadVoxelGeometry(model, buildCadModel(model, spec, new Uint8Array(768)), palette, nodes);
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
    expect(await lib.load(model, new Uint8Array(768))).toBeNull();
    expect(await lib.load({ ...model }, new Uint8Array(768))).toBeNull();
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
      // Detail sidecars are absent here; only spec requests count.
      const fetch = vi.fn(async (url: string) =>
        url.includes('.detail/')
          ? new Response(null, { status: 404 })
          : new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }),
      );
      vi.stubGlobal('fetch', fetch);
      const builder = vi.fn(async () => {
        if (throws) throw new Error('build');
        return buildCadModel(model, spec, new Uint8Array(768));
      });
      const lib = new VoxelCadLibrary('/cad', builder),
        before = warn.mock.calls.length;
      expect(await lib.load(model, new Uint8Array(768))).toBeNull();
      expect(await lib.load(model, new Uint8Array(768))).toBeNull();
      expect(fetch.mock.calls.filter(([url]) => !url.includes('.detail/'))).toHaveLength(1);
      expect(warn.mock.calls.length - before).toBe(status === 404 ? 0 : 1);
      lib.dispose();
    }
  });
  it('threads optional baked detail charts into the look mesh only for parts that have them', async () => {
    const { model, spec } = fixture();
    const table = { res: 4, size: [32, 16], faces: { '0': [2, 3, 10, 10, -0.25, -0.25, 2, 1, 0, 0, 0, 1, 0] } };
    const fetch = vi.fn(async (url: string) =>
      url.endsWith('/SYNTHETIC.detail/body.json')
        ? new Response(JSON.stringify(table), { headers: { 'content-type': 'application/json' } })
        : url.includes('.detail/')
          ? new Response(null, { status: 404 })
          : new Response(JSON.stringify(spec), { headers: { 'content-type': 'application/json' } }),
    );
    vi.stubGlobal('fetch', fetch);
    const builder = vi.fn(async (...args: Parameters<typeof buildCadModel>) => buildCadModel(...args));
    const lib = new VoxelCadLibrary('/cad', (m, s, p, detail) => builder(m, s, p, undefined, detail));
    const meshes = (await lib.load(model, new Uint8Array(768)))!;
    expect(builder.mock.calls[0][4]).toEqual({ body: table });
    const [body, turret] = meshes;
    expect(body.detailSize).toEqual([32, 16]);
    expect(turret.detailSize).toBeNull();
    expect(turret.detailUv.every((v) => v === -1)).toBe(true);
    // Top-face vertices land in the chart (texels 3..11 x 4..12); every other face stays neutral.
    let mapped = 0;
    for (let v = 0; v < body.detailUv.length / 2; v++) {
      const [x, y] = [body.detailUv[v * 2], body.detailUv[v * 2 + 1]];
      if (x < 0) continue;
      mapped++;
      expect(body.vpos[v * 3 + 2]).toBeCloseTo(2);
      expect(x).toBeGreaterThanOrEqual(3 - 1e-4);
      expect(x).toBeLessThanOrEqual(11 + 1e-4);
      expect(y).toBeGreaterThanOrEqual(4 - 1e-4);
    }
    expect(mapped).toBeGreaterThan(0);
    expect(body.detailUv.length / 2).toBe(body.vpos.length / 3);
    // Node has no image decoder: coordinates stay, the atlas is absent and the image is never fetched.
    expect(body.detailImage).toBeNull();
    expect(fetch.mock.calls.some(([url]) => url.endsWith('.png'))).toBe(false);
    lib.dispose();
  });
  it('uses the same mesh implementation in a Worker, terminates it and cancels outstanding work', async () => {
    const { model, spec } = fixture();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        url.includes('.detail/')
          ? new Response(null, { status: 404 })
          : new Response(JSON.stringify(spec), { headers: { 'content-type': 'application/json' } }),
      ),
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
      pending = lib.load(model, new Uint8Array(768));
    await vi.waitFor(() => expect(FakeWorker.instances).toHaveLength(1));
    expect(FakeWorker.instances[0].request!.palette).toEqual(new Uint8Array(768));
    FakeWorker.instances[0].answer();
    expect(await pending).toEqual(buildCadModel(model, spec, new Uint8Array(768)));
    expect(FakeWorker.instances[0].terminated).toBe(true);
    const other = lib.load({ ...model }, new Uint8Array(768));
    await vi.waitFor(() => expect(FakeWorker.instances).toHaveLength(2));
    lib.dispose();
    expect(await other).toBeNull();
    expect(FakeWorker.instances[1].terminated).toBe(true);
    expect(await lib.load(model, new Uint8Array(768))).toBeNull();
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
  finish(buildCadModel(model, spec, new Uint8Array(768)));
  await vi.waitFor(() => expect(units.revision).toBe(1));
  expect(root.partMeshes[0].geometry).not.toBe(old);
  expect(root.partMeshes[0].geometry).toBe((b.root as VoxelUnit).partMeshes[0].geometry);
  expect(root.partMeshes[0].material.name).toBe('voxel-cad-painted-steel');
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
  finish(buildCadModel(model, spec, new Uint8Array(768)));
  await vi.waitFor(() => expect(units.revision).toBe(1));
  repaint(null);
  await Promise.resolve();
  expect(project).toHaveBeenCalledTimes(1);
  expect((unit.root as VoxelUnit).partMeshes[0].material.name).toBe('voxel-cad-painted-steel');
  units.dispose(unit);
  units.disposeAll();
  const later = new VoxelUnits(hd, {} as HdWorld, cache, undefined, { voxelCad: '/cad' }),
    other = later.build(model, 'unit', 'two');
  const before = (other.root as VoxelUnit).partMeshes[0].geometry;
  later.dispose(other);
  later.disposeAll();
  finish(buildCadModel(model, spec, new Uint8Array(768)));
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
  const cad = buildCadModel(model, spec, new Uint8Array(768));
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

it('switches look/pbr/off in place and restores exact source objects for old and newly created units', async () => {
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
      ['other', new Uint8Array(768)],
    ]),
  } as Ra2Cache;
  let finish!: (meshes: ReturnType<typeof buildCadModel>) => void;
  vi.spyOn(VoxelCadLibrary.prototype, 'load').mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, { voxelCad: '/cad' });
  const first = units.build(model, 'unit', 'unit'),
    root = first.root as VoxelUnit;
  const original = root.partMeshes.map((p) => ({ geometry: p.geometry, material: p.material }));
  finish(buildCadModel(model, spec, new Uint8Array(768)));
  await vi.waitFor(() => expect(units.revision).toBe(1));
  const look = root.partMeshes.map((p) => ({ geometry: p.geometry, material: p.material }));
  const duringLook = units.build(model, 'unit', 'unit');
  expect((duringLook.root as VoxelUnit).partMeshes[0].material).toBe(look[0].material);
  expect(units.vehicleCadMode('pbr')).toBe('pbr');
  const pbr = root.partMeshes.map((p) => ({ geometry: p.geometry, material: p.material }));
  expect(pbr[0].geometry).not.toBe(look[0].geometry);
  expect(pbr[0].material.name).toBe('voxel-remaster');
  const duringPbr = units.build(model, 'other', 'other');
  expect((duringPbr.root as VoxelUnit).partMeshes[0].material.name).toBe('voxel-remaster');
  units.vehicleCadMode('off');
  for (let i = 0; i < original.length; i++) {
    expect(root.partMeshes[i].geometry).toBe(original[i].geometry);
    expect(root.partMeshes[i].material).toBe(original[i].material);
  }
  const duringOff = units.build(model, 'unit', 'unit');
  expect((duringOff.root as VoxelUnit).partMeshes[0].geometry).toBe(original[0].geometry);
  expect((duringOff.root as VoxelUnit).partMeshes[0].material).toBe(original[0].material);
  units.vehicleCadMode('look');
  for (let i = 0; i < look.length; i++) {
    expect(root.partMeshes[i].geometry).toBe(look[i].geometry);
    expect(root.partMeshes[i].material).toBe(look[i].material);
  }
  units.vehicleCadMode('pbr');
  expect(root.partMeshes[0].material).toBe(pbr[0].material);
  units.vehicleCadSet('brightness', 1.2);
  expect(units.vehicleCadGet().brightness).toBe(1.2);
  for (const u of [first, duringLook, duringPbr, duringOff]) units.dispose(u);
  units.disposeAll();
  nodes.dispose();
});

it('transfers the look attributes and volume without detaching palette/model input', () => {
  const { model, spec } = fixture(),
    palette = new Uint8Array(768);
  palette[3] = 73;
  const reply = answerCadRequest(structuredClone({ model, spec, palette }));
  expect('meshes' in reply).toBe(true);
  if (!('meshes' in reply)) return;
  const bytes = reply.meshes[0].lookVolume.data.byteLength;
  const legacyIndices = reply.meshes[0].pbr.indices.slice();
  const copied = structuredClone(reply, { transfer: cadReplyTransfers(reply) });
  expect(reply.meshes[0].lookVolume.data.byteLength).toBe(0);
  expect(copied.meshes[0].lookVolume.data.byteLength).toBe(bytes);
  expect(reply.meshes[0].pbr.indices.byteLength).toBe(0);
  expect(copied.meshes[0].pbr.indices).toEqual(legacyIndices);
  expect(copied.meshes[0].lookCode.length).toBe(copied.meshes[0].pindex.length);
  expect(palette.byteLength).toBe(768);
  expect(palette[3]).toBe(73);
  expect(model.parts[0].sections[0].voxels.length).toBeGreaterThan(0);
  expect(answerCadRequest({ ...{ model, spec }, palette: new Uint8Array(0) })).toHaveProperty('error');
});

it.each(['off-before-smooth', 'off-after-smooth'])(
  'keeps the retained original alive through pending smooth adoption: %s',
  async (order) => {
    const { model, spec } = fixture(),
      nodes = new AtmosphereNodes();
    const hd = {
      nodes,
      settings: { voxelLook: 'engine' },
      terrainDepth: { setCaptured: vi.fn() },
    } as unknown as HdRenderer;
    const cache = { paletteRgb: new Map([['unit', new Uint8Array(768)]]) } as Ra2Cache;
    let finish!: (meshes: ReturnType<typeof buildCadModel>) => void;
    vi.spyOn(VoxelCadLibrary.prototype, 'load').mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const jobs: (() => void)[] = [];
    const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, {
      voxelCad: '/cad',
      voxelLook: 'remaster',
      smoothVoxel: { aoRays: 0 },
      smoothBuilder: {
        build(s, o) {
          return new Promise((resolve) => jobs.push(() => resolve(buildSmoothSection(s, o))));
        },
      },
    });
    const unit = units.build(model, 'unit', 'one'),
      root = unit.root as VoxelUnit;
    const stand = root.partMeshes.map((p) => p.geometry);
    finish(buildCadModel(model, spec, new Uint8Array(768)));
    await vi.waitFor(() => expect(units.revision).toBe(1));
    const cad = root.partMeshes.map((p) => p.geometry);
    if (order === 'off-before-smooth') units.vehicleCadMode('off');
    for (const job of jobs) job();
    await vi.waitFor(() => expect(units.loading).toBe(0));
    units.vehicleCadMode('off');
    const original = root.partMeshes.map((p) => p.geometry);
    for (let i = 0; i < original.length; i++) {
      expect(original[i]).not.toBe(stand[i]);
      expect(original[i]).not.toBe(cad[i]);
    }
    expect(units.isStandIn(unit)).toBe(false);
    units.vehicleCadMode('look');
    units.vehicleCadMode('off');
    for (let i = 0; i < original.length; i++) expect(root.partMeshes[i].geometry).toBe(original[i]);
    const later = units.build(model, 'unit', 'one');
    expect((later.root as VoxelUnit).partMeshes[0].geometry).toBe(original[0]);
    units.dispose(unit);
    units.dispose(later);
    units.disposeAll();
    nodes.dispose();
  },
);

it('builds shared seam data from the base unit palette even when the first unit has a house remap', () => {
  const { model } = fixture(),
    nodes = new AtmosphereNodes();
  const base = new Uint8Array(768).fill(100),
    house = new Uint8Array(768).fill(200);
  const hd = {
    nodes,
    settings: { voxelLook: 'engine' },
    terrainDepth: { setCaptured: vi.fn() },
  } as unknown as HdRenderer;
  const cache = {
    paletteRgb: new Map([
      ['unit', base],
      ['house', house],
    ]),
  } as Ra2Cache;
  const load = vi.spyOn(VoxelCadLibrary.prototype, 'load').mockResolvedValue(null);
  const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, { voxelCad: '/cad' }),
    unit = units.build(model, 'house', 'house');
  expect(load).toHaveBeenCalledExactlyOnceWith(model, base);
  units.dispose(unit);
  units.disposeAll();
  nodes.dispose();
});

it('retains an already cached source projection for new palettes created while CAD is active', async () => {
  const { model, spec } = fixture(),
    nodes = new AtmosphereNodes(),
    projectedMaterial = new MeshStandardNodeMaterial();
  const voxels = await import('../src/ra2/voxels');
  vi.spyOn(voxels, 'projectVoxelMaterials').mockImplementation((parts) => {
    for (const part of parts) part.material = projectedMaterial;
    return true;
  });
  const hd = {
    nodes,
    settings: { voxelLook: 'engine' },
    terrainDepth: { setCaptured: vi.fn() },
  } as unknown as HdRenderer;
  const cache = {
    paletteRgb: new Map([
      ['unit', new Uint8Array(768)],
      ['other', new Uint8Array(768)],
    ]),
  } as Ra2Cache;
  vi.spyOn(VoxelCadLibrary.prototype, 'load').mockResolvedValue(buildCadModel(model, spec, new Uint8Array(768)));
  const world = { projections: { get: () => null, loadType: vi.fn() } } as unknown as HdWorld;
  const units = new VoxelUnits(hd, world, cache, undefined, { voxelCad: '/cad' }),
    a = units.build(model, 'unit', 'one');
  await vi.waitFor(() => expect(units.revision).toBe(1));
  const b = units.build(model, 'other', 'two');
  units.vehicleCadMode('off');
  for (const unit of [a, b]) expect((unit.root as VoxelUnit).partMeshes[0].material).toBe(projectedMaterial);
  units.dispose(a);
  units.dispose(b);
  units.disposeAll();
  projectedMaterial.dispose();
  nodes.dispose();
});
