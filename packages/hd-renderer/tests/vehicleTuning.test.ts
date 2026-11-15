import { describe, expect, it } from 'vitest';
import type { MeshStandardNodeMaterial } from 'three/webgpu';
import { AtmosphereNodes } from '../src/atmosphere';
import type { HdRenderer } from '../src/HdRenderer';
import type { Ra2Cache } from '../src/ra2/cache';
import type { HdWorld } from '../src/ra2/world';
import { VoxelUnit as VoxelRoot } from '../src/ra2/voxels';
import { VoxelUnits } from '../src/live/voxelUnits';
import { voxelMaterialFor } from '../src/ra2/vpl';
import { voxelPbrMaterialFor } from '../src/ra2/voxelPbr';
import { parseVehicleParams, VEHICLE_TUNED, VEHICLE_DRAMATIC } from '../src/ra2/vehicleTuning';
import { syntheticModel, syntheticSection } from './fixtures/syntheticVoxels';

describe('native-resolution vehicle review', () => {
  it('requires an explicit flag, clamps finite knobs and retains tuned defaults for invalid values', () => {
    expect(parseVehicleParams(new URLSearchParams())).toBeNull();
    expect(parseVehicleParams(new URLSearchParams('hd-veh=0'))).toBeNull();
    expect(parseVehicleParams(new URLSearchParams('hd-veh=1'))).toEqual(VEHICLE_TUNED);
    expect(parseVehicleParams(new URLSearchParams('hd-veh=1&hd-veh-preset=dramatic'))).toEqual(VEHICLE_DRAMATIC);
    const result = parseVehicleParams(new URLSearchParams('hd-veh=1&hd-veh-brightness=9&hd-veh-wear=NaN&hd-veh-fill='));
    expect(result?.brightness).toBe(1.5);
    expect(result?.wear).toBe(VEHICLE_TUNED.wear);
    expect(result?.fill).toBe(VEHICLE_TUNED.fill);
  });

  it('restores the exact original material and preserves geometry, shadow, opacity and building turrets', () => {
    const nodes = new AtmosphereNodes();
    const hd = { nodes, settings: { voxelLook: 'engine' }, terrainDepth: { setCaptured() {} } } as HdRenderer;
    const palette = new Uint8Array(768).map((_, i) => Math.floor(i / 3));
    const cache = { paletteRgb: new Map([['unit', palette]]) } as Ra2Cache;
    const units = new VoxelUnits(hd, {} as HdWorld, cache, undefined, {
      vehicleRefinement: false,
      smoothBuilder: null,
    });
    const model = syntheticModel([syntheticSection(2, true)]);
    const vehicle = units.build(model, 'unit', 'test');
    expect(vehicle.root).toBeInstanceOf(VoxelRoot);
    const mesh = (vehicle.root as VoxelRoot).partMeshes[0]!;
    const geometry = mesh.geometry;
    const shadow = vehicle.root.children.find((child) => child.name === 'voxel-shadow');
    const original = voxelMaterialFor(palette, nodes);
    expect(mesh.material).toBe(original);
    units.setVehicleRefinement(true);
    expect(mesh.material).toBe(voxelPbrMaterialFor(palette, nodes));
    expect(mesh.geometry).toBe(geometry);
    expect(vehicle.root.children.find((child) => child.name === 'voxel-shadow')).toBe(shadow);
    units.setOpacity(vehicle, 0.5);
    expect((mesh.material as MeshStandardNodeMaterial).transparent).toBe(true);
    units.setVehicleRefinement(false);
    expect((mesh.material as MeshStandardNodeMaterial).transparent).toBe(true);
    units.setOpacity(vehicle, 1);
    expect(mesh.material).toBe(original);
    const turret = units.build(model, 'unit', 'test');
    units.setBuildingPart(turret);
    const buildingMaterial = (turret.root as VoxelRoot).partMeshes[0]!.material;
    units.setVehicleRefinement(true);
    expect((turret.root as VoxelRoot).partMeshes[0]!.material).toBe(buildingMaterial);
    units.dispose(vehicle);
    units.dispose(turret);
    units.disposeAll();
    nodes.dispose();
  });
});

it('applies production depth only to source non-door layers and clamps review strength', async () => {
  const { isProductionBuilding, productionLayerPush } = await import('../src/ra2/productionDepth');
  const { GROUND_DEPTH } = await import('../src/ra2/cardDepth');
  expect(isProductionBuilding([{ key: 'ActiveAnim' }])).toBe(false);
  expect(isProductionBuilding([{ key: 'DeployingAnim' }])).toBe(true);
  expect(isProductionBuilding([{ key: 'ProductionAnim' }])).toBe(true);
  expect(productionLayerPush({ key: 'ActiveAnim', zAdjust: -80 }, 1)).toBe(80 * GROUND_DEPTH);
  expect(productionLayerPush({ key: 'ActiveAnim', zAdjust: -80 }, 0)).toBe(0);
  expect(productionLayerPush({ key: 'ActiveAnim', zAdjust: -80 }, Infinity)).toBe(0);
  expect(productionLayerPush({ key: 'UnderDoorAnim', zAdjust: -45 }, 1)).toBe(0);
  expect(productionLayerPush({ key: 'ActiveAnim' }, 1)).toBe(0);
});
