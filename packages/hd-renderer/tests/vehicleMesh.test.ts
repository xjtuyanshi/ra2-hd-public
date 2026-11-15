import { describe, expect, it } from 'vitest';
import { DirectionalLight, HemisphereLight, OrthographicCamera, Scene, WebGPURenderer, type Mesh } from 'three/webgpu';
import { lights } from 'three/tsl';
import type { VoxelSection } from '../../ra2-assets/src/manifest';
import { lookupNormal, normalTableFor } from '../../ra2-formats/src/vxl/vxlNormals';
import { AtmosphereNodes } from '../src/atmosphere';
import { buildSmoothSection } from '../src/ra2/voxelSmooth';
import {
  buildVehicleSection,
  refineVehicleSection,
  VEHICLE_MESH_OPTIONS,
  vehicleRamp,
  vehicleRegion,
  vehiclePlateNormal,
} from '../src/ra2/vehicleMesh';
import { vehiclePartGeometry } from '../src/ra2/vehicleMeshGeometry';
import { vehicleMeshMaterialFor, VehicleMeshMaterialLibrary } from '../src/ra2/vehicleMeshMaterial';
import { voxelShadowFor, voxelWaterMaterial } from '../src/ra2/voxelShadow';
import { VehicleMeshController } from '../src/ra2/vehicleMeshController';
import {
  buildVoxelGeometry,
  buildOutlinePartGeometry,
  instantiateVoxel,
  updateVoxelNativeDepth,
} from '../src/ra2/voxels';
import { shapeMesh, compareVehicleSilhouettes } from '../src/ra2/vehicleMeshShape';
import { answerSmoothSection } from '../src/ra2/voxelSmoothWorker';
import { VoxelUnits } from '../src/live/voxelUnits';
import { bindSlots } from '../src/materialSlots';
import { syntheticModel, syntheticSection } from './fixtures/syntheticVoxels';

function box(size = 4): VoxelSection {
  const raw: number[] = [];
  const table = normalTableFor(4);
  for (let z = 0; z < size; z++)
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const d = [x + 0.5 - size / 2, y + 0.5 - size / 2, z + 0.5 - size / 2];
        const axis = d.map(Math.abs).indexOf(Math.max(...d.map(Math.abs)));
        const plane = d.map((v, a) => (a === axis ? Math.sign(v) : 0));
        let normal = 0,
          best = -Infinity;
        for (let i = 0; i < table.length; i++) {
          const n = lookupNormal(table, i),
            dot = n[0] * plane[0]! + n[1] * plane[1]! + n[2] * plane[2]!;
          if (dot > best) {
            best = dot;
            normal = i;
          }
        }
        raw.push(x, y, z, x < 1 ? 20 : y === 1 && z === 1 ? 43 : 37, normal);
      }
  return {
    ...syntheticSection(1),
    size: [size, size, size],
    minBounds: [0, 0, 0],
    maxBounds: [size, size, size],
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    voxels: Buffer.from(raw).toString('base64'),
  };
}

describe('vehicle mesh geometry contract', () => {
  it('preserves two-cell seams, diagonal hatch lines and disconnected repeated track/rivet shades', () => {
    const section = box(9);
    const raw = Buffer.from(section.voxels, 'base64');
    for (let j = 0; j < raw.length; j += 5) {
      const x = raw[j]!,
        y = raw[j + 1]!,
        z = raw[j + 2]!;
      raw[j + 3] =
        y === 0 && ((x === 2 && z <= 1) || (x === z && x >= 4))
          ? 43
          : y === 0 && z === 6 && (x === 1 || x === 3)
            ? 44
            : 37;
    }
    section.voxels = raw.toString('base64');
    const mesh = buildVehicleSection(section);
    for (const shade of [43, 44]) {
      const indices = [...mesh.pindex.keys()].filter((i) => mesh.pindex[i] === shade);
      expect(indices.length).toBeGreaterThan(0);
      for (const i of indices) expect(mesh.pshade[i]).toBe(shade);
    }
    expect(mesh.seamDistance!.some((d, i) => d === 0 && mesh.pindex[i] === 43)).toBe(true);
  });

  it('fits stair-stepped plates without flattening a line or a curved surface', () => {
    const plane: [number, number, number][] = [];
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) plane.push([x, y, Math.floor(x / 2)]);
    const reference: [number, number, number] = [-1 / Math.sqrt(5), 0, 2 / Math.sqrt(5)];
    const fitted = vehiclePlateNormal(plane, reference)!;
    expect(fitted).not.toBeNull();
    expect(fitted.reduce((sum, n, i) => sum + n * reference[i]!, 0)).toBeGreaterThan(0.999);
    expect(
      vehiclePlateNormal(
        plane.map(([x]) => [x, 0, 0]),
        [0, 0, 1],
      ),
    ).toBeNull();
    expect(
      vehiclePlateNormal(
        plane.map(([x, y]) => [x, y, (x - 3.5) ** 2 / 2]),
        [0, 0, 1],
      ),
    ).toBeNull();
  });

  it('uses one normal across the interior of a sloped hull while retaining the original silhouette vertices', () => {
    const section = box(9),
      raw: number[] = [];
    const table = normalTableFor(4);
    const closest = (a: number, b: number, c: number) =>
      table.map((n, i) => ({ i, dot: n[0] * a + n[1] * b + n[2] * c })).sort((a, b) => b.dot - a.dot)[0]!.i;
    for (let x = 0; x < 9; x++)
      for (let y = 0; y < 9; y++)
        for (let z = 0; z <= x; z++) raw.push(x, y, z, 37, z === x ? closest(-1, 0, 1) : closest(0, y < 4 ? -1 : 1, 0));
    section.voxels = Buffer.from(raw).toString('base64');
    const original = buildSmoothSection(section, VEHICLE_MESH_OPTIONS);
    const positions = original.position.slice();
    const mesh = refineVehicleSection(section, original);
    const normals: number[][] = [];
    for (let i = 0; i < mesh.pindex.length; i++) {
      const [x, y, z] = mesh.vpos.subarray(i * 3, i * 3 + 3) as unknown as [number, number, number];
      if (x > 2 && x < 6 && y > 2 && y < 6 && Math.abs(z - x) < 0.6)
        normals.push([...mesh.normal.subarray(i * 3, i * 3 + 3)]);
    }
    expect(normals.length).toBeGreaterThan(10);
    for (const n of normals) expect((n[1]! - n[0]!) / Math.SQRT2).toBeGreaterThan(0.999);
    expect(mesh.position).toEqual(positions);
  });

  it('retains a connected within-ramp painted stripe while removing an isolated shade speckle', () => {
    const section = box(6);
    const raw = Buffer.from(section.voxels, 'base64');
    for (let j = 0; j < raw.length; j += 5) {
      raw[j + 3] = raw[j] === 2 && raw[j + 1] === 0 ? 43 : 37;
      if (raw[j] === 4 && raw[j + 1] === 0 && raw[j + 2] === 3) raw[j + 3] = 44;
    }
    section.voxels = raw.toString('base64');
    const mesh = buildVehicleSection(section);
    let stripe = 0,
      denoised = 0;
    for (let i = 0; i < mesh.pindex.length; i++) {
      if (mesh.pindex[i] === 43) {
        stripe++;
        expect(mesh.pshade[i]).toBe(43);
      }
      if (mesh.pindex[i] === 44 && mesh.pshade[i] !== 44) denoised++;
    }
    expect(stripe).toBeGreaterThan(0);
    expect(denoised).toBeGreaterThan(0);
  });

  it('keeps every position, face owner and house-colour index while filtering original ramps only', () => {
    const section = box();
    const mesh = buildSmoothSection(section, VEHICLE_MESH_OPTIONS);
    const positions = mesh.position.slice(),
      owners = mesh.pindex.slice();
    refineVehicleSection(section, mesh);
    expect(mesh.position).toEqual(positions);
    expect(mesh.pindex).toEqual(owners);
    for (let i = 0; i < mesh.pindex.length; i++) {
      expect(vehicleRamp(mesh.pshade[i]!)).toBe(vehicleRamp(owners[i]!));
      expect(vehicleRegion(mesh.pshade[i]!)).toBe(vehicleRegion(owners[i]!));
      if (vehicleRamp(owners[i]!) === 1) expect(mesh.pshade[i]).toBe(owners[i]);
      expect(Math.hypot(...mesh.normal.subarray(i * 3, i * 3 + 3))).toBeCloseTo(1, 5);
    }
    for (const name of ['bevelDistance', 'seamDistance', 'materialClass'] as const) {
      expect(mesh[name]!.length).toBe(owners.length);
      expect(mesh[name]!.every(Number.isFinite)).toBe(true);
    }
    expect(Math.min(...mesh.seamDistance!)).toBe(0);
    expect(Math.min(...mesh.bevelDistance!)).toBe(0);
  });
  it('is deterministic and transfers all new attributes from the existing worker protocol', () => {
    const section = box(3);
    const expected = buildVehicleSection(section);
    const { reply, transfer } = answerSmoothSection({
      id: 1,
      vehicleMesh: true,
      section,
      options: VEHICLE_MESH_OPTIONS,
    });
    if (!('mesh' in reply)) throw new Error(reply.error);
    expect(reply.mesh).toEqual(expected);
    for (const attribute of Object.values(reply.mesh)) expect(transfer).toContain(attribute.buffer);
  });
  it('stays within half an engine pixel of the original cube hull at all 16 facings', () => {
    const section = box(6),
      mesh = buildVehicleSection(section);
    const hull = buildOutlinePartGeometry([section], { cornerRelax: 0, normalBlend: 0 });
    for (let i = 0; i < 16; i++) {
      const result = compareVehicleSilhouettes([shapeMesh(hull)], [mesh], i * 16);
      expect(result.outward).toBeLessThanOrEqual(0.5);
      expect(result.inward).toBeLessThanOrEqual(0.5);
      expect(result.iou).toBeGreaterThan(0.8);
    }
    hull.dispose();
  });
});

describe('vehicle mesh ownership and articulation', () => {
  it('releases the derived colour-reflection material with its scoped source', () => {
    const nodes = new AtmosphereNodes();
    voxelShadowFor(nodes).reflection = 'colour';
    const library = new VehicleMeshMaterialLibrary(nodes);
    const material = library.get(new Uint8Array(768));
    const reflected = voxelWaterMaterial(nodes, 'reflection', material);
    let released = false;
    reflected.addEventListener('dispose', () => {
      released = true;
    });
    library.dispose();
    expect(released).toBe(true);
    nodes.dispose();
  });

  it('opts in vehicle roots without changing building turrets or baseline materials', async () => {
    const nodes = new AtmosphereNodes(),
      palette = new Uint8Array(768),
      model = syntheticModel([box(2)]);
    const units = new VoxelUnits(
      { nodes, settings: { voxelLook: 'engine' } } as never,
      { nodes, voxelLook: 'engine' } as never,
      { paletteRgb: new Map([['unit', palette]]) } as never,
      undefined,
      { vehicleMesh: true },
    );
    const vehicle = units.build(model, 'unit', 'vehicle');
    const building = units.build(model, 'unit', 'building');
    units.setBuildingPart(building);
    const original = vehicle.root.partMeshes[0]!.geometry;
    units.setVehiclePart(vehicle);
    await units.vehicleMeshes.setEnabled(true);
    expect(vehicle.root.partMeshes[0]!.geometry).not.toBe(original);
    expect(building.root.partMeshes[0]!.geometry).toBe(original);
    expect(building.root.partMeshes[0]!.material.name).not.toContain('vehicle-mesh');
    await units.vehicleMeshes.setEnabled(false);
    expect(vehicle.root.partMeshes[0]!.geometry).toBe(original);
    units.disposeAll();
    nodes.dispose();
  });

  it('restores the exact baseline on the same root and preserves turret pose, tint, opacity and native depth', async () => {
    const nodes = new AtmosphereNodes(),
      palette = new Uint8Array(768);
    const model = {
      ...syntheticModel([box(2)]),
      turretOffset: 32,
      parts: (['body', 'turret', 'barrel'] as const).map((role) => ({ role, sections: [box(2)] })),
    };
    const baseline = buildVoxelGeometry(model, palette, nodes);
    const unit = instantiateVoxel(baseline, baseline[0]!.material!, nodes);
    unit.turret!.rotation.y = 0.7;
    unit.root.position.set(2, 0, 3);
    unit.root.setOpacity(0.6);
    unit.root.setTint({ light: 0.5, add: [0.25, 0, 0] });
    const tint = unit.root.partMeshes.map((m) => m.userData.objectTint?.clone());
    updateVoxelNativeDepth(unit.root, unit.root.position, 0.01, true);
    const controller = new VehicleMeshController(nodes);
    controller.register(unit.root, model, palette, baseline);
    await controller.setEnabled(true);
    const made = unit.root.partMeshes.map((m) => m.geometry);
    const meshMaterial = unit.root.partMeshes[0]!.material;
    let materialDisposed = false;
    meshMaterial.addEventListener('dispose', () => (materialDisposed = true));
    const released: boolean[] = [];
    made.forEach((g, i) => g.addEventListener('dispose', () => (released[i] = true)));
    expect(made[0]).not.toBe(baseline[0]!.geometry);
    expect(unit.root.partMeshes.every((m) => m.material.name.startsWith('vehicle-mesh'))).toBe(true);
    expect(unit.root.bodyOpacity).toBe(0.6);
    expect(unit.root.nativeDepth.enabled).toBe(1);
    expect(unit.turret!.rotation.y).toBe(0.7);
    await controller.setEnabled(false);
    expect(unit.root.partMeshes.map((m) => m.geometry)).toEqual(baseline.map((p) => p.geometry));
    expect(released).toEqual([true, true, true]);
    expect(materialDisposed).toBe(true);
    expect(unit.root.partMeshes.map((m) => m.userData.objectTint?.clone())).toEqual(tint);
    expect(unit.root.partMeshes.every((m) => !m.material.name.startsWith('vehicle-mesh'))).toBe(true);
    const warm = controller.setEnabled(true);
    expect(unit.root.partMeshes[0]!.geometry).not.toBe(baseline[0]!.geometry);
    await warm;
    controller.dispose();
    nodes.dispose();
  });
  it('disposes the last model and ignores pending results after removal or off', async () => {
    const nodes = new AtmosphereNodes(),
      palette = new Uint8Array(768),
      model = syntheticModel([box(2)]);
    const baseline = buildVoxelGeometry(model, palette, nodes),
      unit = instantiateVoxel(baseline, baseline[0]!.material!, nodes);
    const controller = new VehicleMeshController(nodes);
    controller.register(unit.root, model, palette, baseline);
    const pending = controller.setEnabled(true);
    controller.unregister(unit.root);
    await pending;
    expect(unit.root.partMeshes[0]!.geometry).toBe(baseline[0]!.geometry);
    await controller.setEnabled(false);
    controller.dispose();
    nodes.dispose();
  });
});

function shaderFor(mesh: Mesh, forceWebGL = false) {
  const renderer = new WebGPURenderer({ forceWebGL, canvas: { width: 64, height: 64 } as HTMLCanvasElement });
  renderer.hasFeature = () => false;
  renderer.hasCompatibility = () => false;
  bindSlots(mesh.material);
  const builder = renderer.backend.createNodeBuilder(mesh, renderer);
  builder.camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  builder.scene = new Scene();
  builder.lightsNode = lights([new DirectionalLight(), new HemisphereLight()]);
  builder.build();
  return { renderer, builder };
}

it.each([false, true])(
  'keeps all console knobs bound to the active native-depth/opacity material after off/on (WebGL=%s)',
  async (forceWebGL) => {
    const nodes = new AtmosphereNodes(),
      palette = new Uint8Array(768),
      model = syntheticModel([box(2)]);
    const baseline = buildVoxelGeometry(model, palette, nodes);
    const unit = instantiateVoxel(baseline, baseline[0]!.material!, nodes);
    unit.root.setOpacity(0.7);
    updateVoxelNativeDepth(unit.root, unit.root.position, 0.01, true);
    const controller = new VehicleMeshController(nodes);
    controller.register(unit.root, model, palette, baseline);
    const knobs = controller.knobs;
    const values = {
      bevel: 1.2,
      seam: 0.6,
      wear: 0.1,
      spec: 0.8,
      gain: 0.7,
      sat: 0.9,
      formLight: 1.4,
      ao: 1.6,
      level: 0.7,
    };
    for (let cycle = 0; cycle < 2; cycle++) {
      await controller.setEnabled(true);
      const { builder } = shaderFor(unit.root.partMeshes[0]!, forceWebGL);
      const used = [...builder.uniforms.vertex, ...builder.uniforms.fragment].flatMap((entry) => [
        entry.node,
        (entry.node as unknown as { node?: unknown }).node,
      ]);
      for (const name of Object.keys(values) as (keyof typeof values)[]) {
        expect(used, name).toContain(knobs[name]);
        if (cycle === 0) knobs[name].value = values[name];
        expect(knobs[name].value).toBe(values[name]);
      }
      expect(controller.knobs).toBe(knobs);
      await controller.setEnabled(false);
    }
    controller.dispose();
    nodes.dispose();
  },
);

it.each([false, true])('builds the physical-light graph within device limits (WebGL=%s)', (forceWebGL) => {
  const nodes = new AtmosphereNodes(),
    palette = new Uint8Array(768),
    section = box(2);
  const geometry = vehiclePartGeometry([buildVehicleSection(section)], [section]);
  const material = vehicleMeshMaterialFor(palette, nodes);
  const unit = instantiateVoxel([{ role: 'body', geometry, material, turretOffset: 0 }], material, nodes);
  updateVoxelNativeDepth(unit, unit.root.position, 0.01, true);
  unit.root.setTint({ light: 0.8, add: [0.2, 0, 0] });
  const mesh = unit.root.partMeshes[0]!;
  const { renderer, builder } = shaderFor(mesh, forceWebGL);
  const attributes = builder.getAttributesArray().map((a) => a.node?.attribute ?? geometry.getAttribute(a.name));
  expect(attributes.every(Boolean)).toBe(true);
  if (!forceWebGL) {
    const buffers = (
      renderer.backend as never as { attributeUtils: { createShaderVertexBuffers(x: unknown): unknown[] } }
    ).attributeUtils.createShaderVertexBuffers({ getAttributes: () => attributes });
    expect(buffers.length).toBeLessThanOrEqual(8);
  }
  expect(attributes.length).toBeLessThanOrEqual(16);
  expect(builder.fragmentShader).toContain('fwidth');
  expect(builder.fragmentShader).not.toContain('voxelValueNoise');
  geometry.dispose();
  nodes.dispose();
});
