import { Mesh, type BufferGeometry, type Group, type Vector3 } from 'three/webgpu';
import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import { localLightLift } from '../atmosphere';
import type { HdRenderer } from '../HdRenderer';
import type { ObjectTint } from '../objectTint';
import type { Ra2Cache } from '../ra2/cache';
import {
  buildVoxelGeometry,
  createVoxelMaterial,
  instantiateVoxel,
  projectVoxelMaterials,
  updateVoxelNativeDepth,
  VoxelUnit as VoxelRoot,
  voxelShapeFor,
  type VoxelGeometry,
} from '../ra2/voxels';
import type { HdWorld } from '../ra2/world';
import { NOT_CAPTURED_CODE } from '../objectDepth';
import { LEVEL_HEIGHT } from '../world';
import { smoothVoxelOptions, type SmoothVoxelOptions } from '../ra2/voxelSmooth';
import {
  buildSmoothVoxelGeometry,
  buildStandInVoxelParts,
  ensureSmoothVoxelGeometry,
  hasSmoothVoxelGeometry,
  prepareSmoothVoxelGeometry,
  smoothVoxelPartGeometries,
} from '../ra2/voxelSmoothGeometry';
import { voxelPbrMaterialFor } from '../ra2/voxelPbr';
import { createSmoothVoxelPool, type SmoothSectionBuilder } from '../ra2/voxelSmoothPool';
import { projectionParts, voxelProjectionLibrary, type VoxelProjectionLibrary } from '../ra2/voxelProjection';
import { setShadowContact } from '../ra2/voxelShadow';
import type { VoxelCardTies } from '../ra2/voxelNativeDepth';
import { VoxelCadLibrary, cadVoxelGeometry, adoptCadVoxel } from '../ra2/voxelCad';
import type { CadMeshData } from '../ra2/voxelCadMesh';
import { voxelCadLookFor } from '../ra2/voxelCadMaterial';
import { selectedVoxelTurretIndex } from '../ra2/voxelTurrets';
import { projectionSmoothVoxelOptions } from '../ra2/voxelProjectionMath';

export interface VoxelUnitsOptions {
  voxelLook?: 'engine' | 'remaster';
  smoothVoxel?: Partial<SmoothVoxelOptions>;
  voxelProjection?: string;
  /** Optional external hard-surface spec directory; absent leaves every vehicle unchanged. */
  voxelCad?: string;
  cadLook?: 'look' | 'pbr';
  /**
   * Where the remaster look's smooth meshes are built. Default: a pool of Workers when the host has Workers, else
   * (and with null) on the calling thread at the first instance of a model.
   */
  smoothBuilder?: (SmoothSectionBuilder & { dispose?(): void }) | null;
}

export type VehicleCadMode = 'look' | 'pbr' | 'off';
interface CadRecord {
  palette: Uint8Array;
  units: Set<VoxelUnit>;
  original?: VoxelGeometry[];
  look?: VoxelGeometry[];
  pbr?: VoxelGeometry[];
}

const LEPTONS_PER_CELL = 256;
const LEPTONS_PER_LEVEL = 104;

/** A placed voxel unit: the body root and, for Turret=yes types, the group the turret (and barrel) yaw in. */
export interface VoxelUnit {
  root: Group;
  turret?: Group;
}

/**
 * What draws a model while its smooth mesh is being built: the stand-in geometries (one per part, shared by every
 * house palette) and the roots of the placed units that show them (callers keep
 * their own record of a unit around the same root, so the root is what identifies it).
 */
interface StandIn {
  geometries: BufferGeometry[];
  units: Set<Group>;
}

/** Engine pose of a unit: lepton coordinates and a facing in radians (0 = north, clockwise). */
export interface VoxelPose {
  x: number;
  y: number;
  z: number;
  facing: number;
}

/**
 * Voxel vehicles and aircraft of the live view: geometry is shared across house palettes; each palette gets its own
 * LUT material. Places the models from engine poses. LiveView keeps
 * the bookkeeping (which object uses which unit); everything about how a voxel unit looks lives here.
 */
export class VoxelUnits {
  private readonly material: ReturnType<typeof createVoxelMaterial>;
  private readonly geometries = new Map<string, VoxelGeometry[]>();
  /** Caller keys point to the active option-keyed set; late page loads migrate that set in place. */
  private readonly geometryKeys = new Map<string, string>();
  private readonly smoothSets = new Map<
    VoxelModel,
    Map<VoxelGeometry[], { cacheKey: string; units: Set<VoxelUnit> }>
  >();
  private readonly voxelLook: 'engine' | 'remaster';
  private readonly smoothVoxel: Readonly<SmoothVoxelOptions>;
  private readonly projections?: VoxelProjectionLibrary;
  /** Instances waiting for one asynchronous type load; cleared once, never visited by pose()/the frame loop. */
  private readonly pending = new Map<VoxelGeometry[], Set<VoxelUnit>>();
  /** Builds the remaster look's meshes off this thread; null: built in place by the first instance. */
  private readonly smoothBuilder: (SmoothSectionBuilder & { dispose?(): void }) | null;
  /** Models whose mesh is being built by `smoothBuilder`. */
  private readonly meshing = new Map<VoxelModel, Set<string>>();
  /** Models drawn with a stand-in mesh until `smoothBuilder` answers (see build). */
  private readonly standIns = new Map<VoxelModel, StandIn>();
  /** The stand-in a geometry set was made from, while it is one. */
  private readonly standInOf = new Map<VoxelGeometry[], StandIn>();
  private readonly cad?: VoxelCadLibrary;
  private readonly cadLoading = new Set<VoxelModel>();
  private readonly cadGeometry = new Map<VoxelModel, BufferGeometry[]>();
  private readonly cadPbrGeometry = new Map<VoxelModel, BufferGeometry[]>();
  private readonly cadMeshes = new Map<VoxelModel, CadMeshData[]>();
  private cadMode: VehicleCadMode = 'look';
  private readonly cadParts = new Set<VoxelGeometry[]>();
  private readonly cadRecords = new Map<VoxelModel, Map<VoxelGeometry[], CadRecord>>();
  private disposed = false;
  /** Changes when a mesh or a redraw projection asked for earlier has arrived: what waited for it can draw now. */
  revision = 0;

  constructor(
    readonly hd: HdRenderer,
    readonly world: HdWorld,
    private readonly cache: Ra2Cache,
    readonly libraryBase?: string,
    options: VoxelUnitsOptions = {},
  ) {
    this.voxelLook = options.voxelLook ?? world.voxelLook ?? hd.settings.voxelLook;
    this.smoothVoxel = smoothVoxelOptions({ ...world.smoothVoxel, ...options.smoothVoxel });
    const projectionBase = options.voxelProjection ?? world.voxelProjection;
    this.projections = projectionBase ? voxelProjectionLibrary(projectionBase, hd.nodes) : world.projections;
    if (this.voxelLook === 'remaster') hd.settings.voxelLook = 'remaster';
    this.material = createVoxelMaterial(hd.nodes);
    const cadBase = options.voxelCad ?? world.voxelCad;
    this.cadMode = options.cadLook ?? world.cadLook ?? 'look';
    if (cadBase) this.cad = new VoxelCadLibrary(cadBase);
    this.smoothBuilder =
      this.voxelLook !== 'remaster'
        ? null
        : options.smoothBuilder !== undefined
          ? options.smoothBuilder
          : createSmoothVoxelPool();
  }

  /** Meshes being built and redraw projections being fetched: art that was asked for and is not on screen yet. */
  get loading(): number {
    return this.meshing.size + this.pending.size + this.cadLoading.size;
  }

  /** Successful section requests served by persistent storage or freshly built, for startup diagnostics. */
  get cacheHits(): number {
    return this.smoothBuilder?.cacheHits ?? 0;
  }
  get meshBuilds(): number {
    return this.smoothBuilder?.meshBuilds ?? 0;
  }

  /** Placed units still drawn with a stand-in mesh (diagnostics: a healthy match counts down to 0 within seconds). */
  get standInUnits(): number {
    let count = 0;
    for (const standIn of this.standIns.values()) count += standIn.units.size;
    return count;
  }

  /** Whether `unit` is still drawn with the stand-in mesh of its model (see build). */
  isStandIn(unit: VoxelUnit): boolean {
    for (const standIn of this.standIns.values()) if (standIn.units.has(unit.root)) return true;
    return false;
  }

  private smoothingFor(model: VoxelModel): Readonly<SmoothVoxelOptions> {
    const projected = selectedVoxelTurretIndex(model)
      ? model.parts.filter((part) => part.role === 'body')
      : model.parts;
    const asset = projected.length ? this.projections?.get(model.id, projectionParts(projected)) : null;
    return projectionSmoothVoxelOptions(this.smoothVoxel, asset?.manifest);
  }

  /**
   * Whether the model's final mesh exists, so that build() makes an instance of the final look. The remaster look's
   * smooth mesh costs 0.02 to 3.3 s of CPU per model (measured over the 94 library models; 3 s for the vehicles of
   * one skirmish start, once the longest freeze of a match's first seconds), so it is built by Workers: the first
   * call for a model starts that and answers false; `revision` changes when the mesh is there. That can take
   * seconds (measured: 7.2 s for a War Miner on a loaded machine, with the Workers started anew), so nothing that is
   * on screen may wait for it: build() draws a stand-in meanwhile. Callers ask this to hold back what nobody sees
   * yet (a view out of sight, the alternate image of a unit that still shows its usual one). Without Workers the
   * mesh is built by the first instance as before, and this is always true.
   */
  geometryReady(model: VoxelModel): boolean {
    const builder = this.smoothBuilder;
    const options = this.smoothingFor(model);
    if (!builder || this.voxelLook !== 'remaster' || hasSmoothVoxelGeometry(model, options)) return true;
    const key = JSON.stringify(options);
    let pending = this.meshing.get(model);
    if (!pending) this.meshing.set(model, (pending = new Set()));
    if (pending.has(key)) return false;
    pending.add(key);
    prepareSmoothVoxelGeometry(model, options, builder)
      .catch((error) => {
        if (this.disposed) return;
        // The Worker could not build it: build in place once rather than keeping the stand-in for good.
        console.warn(`[HD] smooth voxel mesh of ${model.id} built on the main thread`, error);
        ensureSmoothVoxelGeometry(model, options);
      })
      .finally(() => {
        pending.delete(key);
        if (!pending.size) this.meshing.delete(model);
        // A default build may finish after the page override: it must never restore the old mesh.
        if (!this.disposed && key === JSON.stringify(this.smoothingFor(model))) this.adoptSmooth(model, options);
        this.revision++;
      });
    return false;
  }

  /** A late page set replaces the type's mesh on both build paths, keeping existing units and CAD originals. */
  private refreshSmooth(model: VoxelModel): void {
    if (this.voxelLook !== 'remaster' || this.disposed) return;
    const options = this.smoothingFor(model);
    for (const [parts, { cacheKey }] of this.smoothSets.get(model) ?? []) {
      const key = JSON.stringify([model.id, cacheKey, options]);
      const previous = this.geometryKeys.get(cacheKey);
      if (previous !== undefined) this.geometries.delete(previous);
      this.geometries.set(key, parts);
      this.geometryKeys.set(cacheKey, key);
    }
    if (!this.geometryReady(model)) return;
    ensureSmoothVoxelGeometry(model, options);
    this.adoptSmooth(model, options);
  }

  /** Adopt the effective mesh in place, including palette/variant sets and retained pre-CAD geometry. */
  private adoptSmooth(model: VoxelModel, options: Readonly<SmoothVoxelOptions>): void {
    const smooth = smoothVoxelPartGeometries(model, options);
    if (!smooth) return;
    for (const [parts, { units }] of this.smoothSets.get(model) ?? []) {
      const cad = this.cadRecords.get(model)?.get(parts);
      const source = cad?.original ?? parts;
      const swap = new Map(source.map((part, index) => [part.geometry, smooth[index]!]));
      for (let i = 0; i < source.length; i++) source[i]!.geometry = smooth[i]!;
      this.standInOf.delete(parts);
      if (cad?.original && this.cadMode !== 'off') continue;
      if (cad?.original) for (let i = 0; i < parts.length; i++) Object.assign(parts[i], source[i]);
      for (const { root } of units) {
        if (!(root instanceof VoxelRoot) || !root.replaceGeometry(swap)) continue;
        // Resting units may receive no new pose: update the native card from the new bounds now.
        const { anchor, bonus, enabled } = root.nativeDepth;
        updateVoxelNativeDepth(root, anchor, bonus, enabled !== 0);
      }
    }
    const standIn = this.standIns.get(model);
    if (standIn) {
      this.standIns.delete(model);
      for (const geometry of standIn.geometries) geometry.dispose();
    }
  }

  /** Keep current parts on screen during fetch/build, then adopt the completed CAD set in place. */
  private requestCad(model: VoxelModel, parts: VoxelGeometry[], palette: Uint8Array): void {
    if (!this.cad) return;
    let records = this.cadRecords.get(model);
    if (!records) {
      records = new Map();
      this.cadRecords.set(model, records);
      this.cadLoading.add(model);
      void this.cad
        .load(model, this.cache.paletteRgb.get(model.palette) ?? palette)
        .then((meshes) => {
          this.cadLoading.delete(model);
          if (this.disposed || !meshes) return;
          const built = cadVoxelGeometry(model, meshes, palette, this.hd.nodes);
          const pbr = cadVoxelGeometry(model, meshes, palette, this.hd.nodes, 'pbr');
          this.cadMeshes.set(model, meshes);
          this.cadGeometry.set(
            model,
            built.map((part) => part.geometry),
          );
          this.cadPbrGeometry.set(
            model,
            pbr.map((part) => part.geometry),
          );
          for (const [target, record] of records!) this.adoptCad(model, target, record);
          this.revision++;
        })
        .catch((error) => {
          this.cadLoading.delete(model);
          if (!this.disposed) this.cad?.reportFailure(model.id, error);
        });
    }
    const record = { palette, units: new Set<VoxelUnit>() };
    records.set(parts, record);
    if (this.cadGeometry.has(model)) this.adoptCad(model, parts, record);
  }

  /** Development A/B switch: no rebuild, no new unit, and no change to articulation or per-object state. */
  vehicleCadMode(mode?: VehicleCadMode): string {
    if (mode !== undefined) {
      if (!['look', 'pbr', 'off'].includes(mode)) throw new Error('Invalid CAD mode');
      if (mode !== this.cadMode) {
        this.cadMode = mode;
        for (const [model, records] of this.cadRecords)
          if (this.cadGeometry.has(model)) for (const [parts, record] of records) this.adoptCad(model, parts, record);
        this.revision++;
      }
    }
    return this.cadMode;
  }
  vehicleCadSet(knob: string, value: number): void {
    voxelCadLookFor(this.hd.nodes).set(knob, value);
  }
  vehicleCadGet(): Record<string, number> {
    return voxelCadLookFor(this.hd.nodes).get();
  }

  private adoptCad(model: VoxelModel, parts: VoxelGeometry[], record: CadRecord): void {
    const standIn = this.standInOf.get(parts);
    if (!record.original) {
      record.original = parts.map((p) => ({ ...p }));
      record.look = cadVoxelGeometry(
        model,
        this.cadMeshes.get(model)!,
        record.palette,
        this.hd.nodes,
        'look',
        this.cadGeometry.get(model),
      );
      record.pbr = cadVoxelGeometry(
        model,
        this.cadMeshes.get(model)!,
        record.palette,
        this.hd.nodes,
        'pbr',
        this.cadPbrGeometry.get(model),
      );
    }
    const next = this.cadMode === 'off' ? record.original : this.cadMode === 'pbr' ? record.pbr! : record.look!;
    for (const unit of record.units) {
      if (this.cadMode === 'off') standIn?.units.add(unit.root);
      else standIn?.units.delete(unit.root);
      if (unit.root instanceof VoxelRoot) adoptCadVoxel(unit.root, parts, next);
    }
    for (let i = 0; i < parts.length; i++) Object.assign(parts[i], next[i]);
    this.cadParts.add(parts);
  }

  /** Geometry set of a model whose smooth mesh is still being built: its stand-in parts with the look's material. */
  private standInParts(model: VoxelModel, palette: Uint8Array): VoxelGeometry[] {
    let standIn = this.standIns.get(model);
    if (!standIn) {
      const built = buildStandInVoxelParts(model, voxelShapeFor(this.hd.nodes));
      standIn = { geometries: built.map((part) => part.geometry), units: new Set() };
      this.standIns.set(model, standIn);
    }
    const material = voxelPbrMaterialFor(palette, this.hd.nodes);
    const turretOffset = (model.turretOffset ?? 0) / LEPTONS_PER_CELL;
    const parts = model.parts.map((part, index) => ({
      role: part.role,
      geometry: standIn.geometries[index]!,
      turretOffset,
      material,
    }));
    this.standInOf.set(parts, standIn);
    return parts;
  }

  /**
   * The shared geometry set of `model` for palette row `paletteKey`, built once per `cacheKey`; its redraw projection
   * (when the look has one) is requested here and repaints every waiting instance when it arrives. `variant` names
   * the redraw's page variant the set is painted with (VoxelProjectionLibrary.get); callers put it in `cacheKey`.
   */
  private partsFor(model: VoxelModel, paletteKey: string, cacheKey: string, variant = ''): VoxelGeometry[] {
    const options = this.smoothingFor(model);
    const key = JSON.stringify([model.id, cacheKey, options]);
    let parts = this.geometries.get(this.geometryKeys.get(cacheKey) ?? key);
    if (parts) return parts;
    const palette = this.cache.paletteRgb.get(paletteKey)!;
    parts =
      this.voxelLook !== 'remaster'
        ? buildVoxelGeometry(model, palette, this.hd.nodes)
        : this.geometryReady(model)
          ? buildSmoothVoxelGeometry(model, palette, this.hd.nodes, options)
          : this.standInParts(model, palette);
    this.geometries.set(key, parts);
    this.geometryKeys.set(cacheKey, key);
    if (this.voxelLook === 'remaster') {
      let sets = this.smoothSets.get(model);
      if (!sets) this.smoothSets.set(model, (sets = new Map()));
      sets.set(parts, { cacheKey, units: new Set() });
    }
    const publish = (): VoxelGeometry[] => {
      // Capture the complete pre-CAD set, including an already cached redraw projection, before adoption.
      this.requestCad(model, parts!, palette);
      return parts!;
    };
    const projection = this.projections;
    if (!projection || this.cadParts.has(parts)) return publish();
    // Alternate native turrets have different silhouettes: retain the HD hull, but never paint the default gun
    // over a repair arm or another passenger's turret. Those parts keep their own palette/smooth material.
    const projected = selectedVoxelTurretIndex(model) ? parts.filter((part) => part.role === 'body') : parts;
    if (!projected.length) return publish();
    const required = projectionParts(projected);
    if (projectVoxelMaterials(projected, palette, projection.get(model.id, required, variant), this.hd.nodes))
      return publish();
    const waiting = new Set<VoxelUnit>();
    const target = parts;
    this.pending.set(target, waiting);
    void projection.loadType(model.id, required, variant).then((asset) => {
      if (asset?.manifest.smooth) this.refreshSmooth(model);
      if (
        !this.disposed &&
        !this.cadParts.has(target) &&
        projectVoxelMaterials(projected, palette, asset, this.hd.nodes)
      ) {
        const materials = new Map(target.map((part) => [part.geometry, part.material!]));
        for (const unit of waiting) {
          unit.root.traverse((mesh) => {
            // Shadow twins share geometry but own their shadow material and must never be repainted.
            if (mesh instanceof Mesh && mesh.userData.painterDepth) {
              const material = materials.get(mesh.geometry);
              if (material) mesh.material = material;
            }
          });
          // A resting unit may receive no new pose after this callback. Preserve its active color/depth
          // pass immediately instead of showing the new physical source until a later publication.
          if (unit.root instanceof VoxelRoot) {
            const { anchor, bonus, enabled } = unit.root.nativeDepth;
            updateVoxelNativeDepth(unit, anchor, bonus, enabled !== 0);
          }
        }
      }
      waiting.clear();
      this.pending.delete(target);
      this.revision++;
    });
    return publish();
  }

  /** Whether a geometry set (and its materials) already exists for `cacheKey`: an instance of it builds no shader. */
  built(cacheKey: string): boolean {
    return this.geometryKeys.has(cacheKey);
  }

  /**
   * Build the shared geometry and request its projection ahead of the first instance. True once a new instance
   * would draw its final look at once (the mesh is built; the projection has arrived, is absent or failed); callers
   * that swap one model for another on the same object keep the old one on screen until then.
   */
  prepare(model: VoxelModel, paletteKey: string, cacheKey: string, variant = ''): boolean {
    if (!this.geometryReady(model)) return false;
    return !this.pending.has(this.partsFor(model, paletteKey, cacheKey, variant));
  }

  /**
   * Whether the model's redraw has pages of its own for `variant`. False while the redraw loads and for a model
   * without that variant: such a unit is drawn (and keyed) as the base look, so it is never rebuilt for a variant
   * that would look the same.
   */
  hasVariant(model: VoxelModel, variant: string): boolean {
    return !this.cadGeometry.has(model) && !!variant && !!this.projections?.hasVariant(model.id, variant);
  }

  /**
   * A new instance of `model` drawn with palette row `paletteKey` (already resolved with paletteFor, so house-remapped
   * when the owner is known). `cacheKey` identifies the geometry set to share, e.g. `${type}@${paletteKey}`.
   *
   * Never waits and never builds a smooth mesh on this thread while Workers do that: until the model's own mesh is
   * there the instance is made from its stand-in (the plain exposed-face mesh, milliseconds to build), with the
   * materials the smooth mesh gets, and changes to the smooth mesh in place when it arrives (adoptSmooth). A unit
   * the game shows is therefore on screen from its first frame, where it belongs.
   */
  build(model: VoxelModel, paletteKey: string, cacheKey: string, variant = ''): VoxelUnit {
    const parts = this.partsFor(model, paletteKey, cacheKey, variant);
    const unit = instantiateVoxel(parts, this.material, this.hd.nodes);
    this.pending.get(parts)?.add(unit);
    this.smoothSets.get(model)?.get(parts)?.units.add(unit);
    if (!this.cadParts.has(parts) || this.cadMode === 'off') this.standInOf.get(parts)?.units.add(unit.root);
    this.cadRecords.get(model)?.get(parts)?.units.add(unit);
    return unit;
  }

  /**
   * Place a unit at an engine pose. Engine facing 0 = north (world -z), clockwise; the model's nose is local +x. The
   * turret yaws relative to the body (`turretFacing` in radians like `facing`; null = aligned with the body).
   */
  pose(unit: VoxelUnit, pose: VoxelPose, turretFacing: number | null, up?: Vector3 | null): void {
    unit.root.position.set(
      pose.x / LEPTONS_PER_CELL,
      (pose.z / LEPTONS_PER_LEVEL) * LEVEL_HEIGHT,
      pose.y / LEPTONS_PER_CELL,
    );
    unit.root.rotation.y = Math.PI / 2 - pose.facing;
    if (unit.turret) unit.turret.rotation.y = pose.facing - (turretFacing ?? pose.facing);
    // `up` (world axes) is the game's own body tilt; without it the root follows the registered ground.
    if (unit.root instanceof VoxelRoot) unit.root.setSurfaceUp(up ?? null);
  }

  /**
   * A unit standing on terrain or a bridge deck casts the engine's contact shadow at its own origin (see
   * voxelShadow.ts setShadowContact); `normal` is the plane of a tilted body, absent = the ground's own. `bottomFace`
   * lays the footprint on the bottom face of the body's bounds (vehicles).
   */
  setGroundContact(unit: VoxelUnit, grounded: boolean, normal?: Vector3, bottomFace = false): void {
    setShadowContact(unit.root, grounded, normal, bottomFace);
  }

  /**
   * A building's voxel turret takes the scene's look like the building's sprite body, not the vehicles' own gains
   * and finish (vpl.ts meshVehicleLook).
   */
  setBuildingPart(unit: VoxelUnit): void {
    if (unit.root instanceof VoxelRoot) unit.root.setBuildingPart(true);
  }

  /**
   * The unit's origin is `height` world units above the ground below it (0 standing on terrain or a bridge deck): it
   * then takes localLightLift(height) of the ground-level light field, as sprite pixels do above their ground line.
   */
  setLocalLightHeight(unit: VoxelUnit, height: number): void {
    if (unit.root instanceof VoxelRoot) unit.root.setLocalLightShare(localLightLift(height));
  }

  /** Painter bias remains additive to either physical flight depth or the native ground card. */
  setPainterBias(unit: VoxelUnit, bias: number): void {
    if (unit.root instanceof VoxelRoot) unit.root.setPainterBias(bias);
  }

  /** Native chrono blending changes body/turret/barrel alpha, never the separately rendered ground shadows. */
  /** The unit's picture tint (objectTint.ts); undefined draws it unchanged. */
  setTint(unit: VoxelUnit, tint: ObjectTint | undefined): void {
    if (unit.root instanceof VoxelRoot) unit.root.setTint(tint);
  }

  setOpacity(unit: VoxelUnit, opacity: number): void {
    if (unit.root instanceof VoxelRoot) unit.root.setOpacity(opacity);
  }

  /**
   * Call after pose/turret updates; anchor is the world-space ground draw point, including bridge elevation. `ties`
   * resolves near-ties with the map's depth-writing cards for this unit as a whole.
   */
  setNativeDepth(unit: VoxelUnit, anchor: Vector3, bonus = 0, enabled = true, ties?: VoxelCardTies | null): void {
    updateVoxelNativeDepth(unit, anchor, bonus, enabled, ties);
  }

  /** Whole-object capture membership belongs only to original part meshes, including articulated barrels. */
  setCaptureCode(unit: VoxelUnit, code: number): void {
    if (unit.root instanceof VoxelRoot)
      for (const mesh of unit.root.partMeshes) this.hd.terrainDepth.setCaptured(mesh, code !== NOT_CAPTURED_CODE, code);
  }

  captureRadius(unit: VoxelUnit): number {
    return unit.root instanceof VoxelRoot ? unit.root.captureRadius : 0;
  }

  /** Stop the mesh Workers; the view that owns this calls it once, after its last unit is gone. */
  disposeAll(): void {
    this.disposed = true;
    this.smoothBuilder?.dispose?.();
    this.cad?.dispose();
    for (const geometries of this.cadGeometry.values()) for (const geometry of geometries) geometry.dispose();
    this.cadGeometry.clear();
    for (const geometries of this.cadPbrGeometry.values()) for (const geometry of geometries) geometry.dispose();
    this.cadPbrGeometry.clear();
    this.cadMeshes.clear();
    this.cadRecords.clear();
    this.cadParts.clear();
    this.cadLoading.clear();
    this.meshing.clear();
    this.smoothSets.clear();
    this.geometryKeys.clear();
    this.geometries.clear();
    // Stand-ins belong to this instance alone (the smooth geometries they wait for follow their model).
    for (const standIn of this.standIns.values()) for (const geometry of standIn.geometries) geometry.dispose();
    this.standIns.clear();
    this.standInOf.clear();
  }

  /** Take a unit out of the scene (geometry stays cached for the next instance of its type). */
  dispose(unit: VoxelUnit): void {
    this.setCaptureCode(unit, NOT_CAPTURED_CODE);
    for (const waiting of this.pending.values()) waiting.delete(unit);
    for (const sets of this.smoothSets.values()) for (const record of sets.values()) record.units.delete(unit);
    for (const records of this.cadRecords.values()) for (const record of records.values()) record.units.delete(unit);
    for (const standIn of this.standIns.values()) standIn.units.delete(unit.root);
    if (unit.root instanceof VoxelRoot) unit.root.disposeOpacityMaterials();
    unit.root.removeFromParent();
  }
}
