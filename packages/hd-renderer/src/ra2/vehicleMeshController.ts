import type { BufferGeometry } from 'three/webgpu';
import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import type { SmoothMeshData } from './voxelSmooth';
import type { AtmosphereNodes } from '../atmosphere';
import { VoxelUnit, updateVoxelNativeDepth, type VoxelGeometry } from './voxels';
import { buildVehicleSection, VEHICLE_MESH_OPTIONS } from './vehicleMesh';
import { vehiclePartGeometry } from './vehicleMeshGeometry';
import { VehicleMeshMaterialLibrary, vehicleMeshKnobsFor } from './vehicleMeshMaterial';
import { createSmoothVoxelPool, type SmoothVoxelPool } from './voxelSmoothPool';

interface Record {
  root: VoxelUnit;
  model: VoxelModel;
  palette: Uint8Array;
  baseline: VoxelGeometry[];
}
interface Build {
  users: number;
  pending: boolean;
  parts?: BufferGeometry[];
  ready: Promise<void>;
}
/** One view owns the refinement, async work and live roots. Off restores the exact baseline part sets. */
export class VehicleMeshController {
  private enabled = false;
  private disposed = false;
  private generation = 0;
  private materials: VehicleMeshMaterialLibrary | null = null;
  private pool: SmoothVoxelPool | null = null;
  private readonly prepared = new Map<VoxelModel, SmoothMeshData[][]>();
  private readonly records = new Map<VoxelUnit, Record>();
  private readonly models = new Map<VoxelModel, Build>();
  constructor(
    readonly nodes: AtmosphereNodes,
    enabled = false,
  ) {
    if (enabled) this.setEnabled(true);
  }
  get knobs() {
    return vehicleMeshKnobsFor(this.nodes);
  }
  get loading() {
    return [...this.models.values()].filter((m) => m.pending).length;
  }
  get active() {
    return this.enabled;
  }
  register(root: VoxelUnit, model: VoxelModel, palette: Uint8Array, baseline: VoxelGeometry[]): void {
    const record = { root, model, palette, baseline };
    this.records.set(root, record);
    if (this.enabled) this.request(record);
  }
  unregister(root: VoxelUnit): void {
    const record = this.records.get(root);
    if (!record) return;
    this.records.delete(root);
    if (![...this.records.values()].some((r) => r.model === record.model)) this.prepared.delete(record.model);
    const built = this.models.get(record.model);
    if (built && --built.users === 0) {
      this.models.delete(record.model);
      for (const geometry of built.parts ?? []) geometry.dispose();
    }
    if (!this.records.size) {
      this.materials?.dispose();
      this.materials = null;
    }
  }
  /** Resolves after geometry adoption; rendering the next frame needs no reload or reconstruction of roots. */
  async setEnabled(enabled: boolean): Promise<boolean> {
    if (this.disposed) return false;
    if (enabled !== this.enabled) {
      this.enabled = enabled;
      this.generation++;
      if (enabled) {
        this.pool = createSmoothVoxelPool(true);
        for (const record of this.records.values()) this.request(record);
      } else {
        this.pool?.dispose();
        this.pool = null;
        for (const record of this.records.values()) this.apply(record, record.baseline);
        for (const built of this.models.values()) for (const geometry of built.parts ?? []) geometry.dispose();
        this.models.clear();
        this.materials?.dispose();
        this.materials = null;
      }
    }
    await Promise.all([...this.models.values()].map((m) => m.ready));
    return this.enabled;
  }
  private request(record: Record): void {
    let build = this.models.get(record.model);
    if (!build) {
      const generation = this.generation;
      const model = record.model;
      const pool = this.pool;
      const prepared = this.prepared.get(model);
      if (prepared) {
        const parts = prepared.map((meshes, i) => vehiclePartGeometry(meshes, model.parts[i]!.sections));
        build = { users: 0, pending: false, parts, ready: Promise.resolve() };
        this.models.set(model, build);
      } else {
        build = { users: 0, pending: true, ready: Promise.resolve() };
        const target = build;
        this.models.set(model, target);
        build.ready = Promise.all(
          model.parts.map(async (part) => {
            const meshes = await Promise.all(
              part.sections.map((section) =>
                pool ? pool.build(section, VEHICLE_MESH_OPTIONS) : Promise.resolve(buildVehicleSection(section)),
              ),
            );
            return { meshes, sections: part.sections };
          }),
        )
          .then((parts) => {
            if (this.disposed || generation !== this.generation || this.models.get(model) !== target) return;
            this.prepared.set(
              model,
              parts.map((part) => part.meshes),
            );
            target.parts = parts.map((part) => vehiclePartGeometry(part.meshes, part.sections));
            for (const next of this.records.values()) if (next.model === model) this.adopt(next, target.parts);
          })
          .catch((error) => {
            if (!this.disposed && generation === this.generation)
              console.warn(`[HD] vehicle mesh ${model.id} unavailable; retaining baseline`, error);
          })
          .finally(() => {
            target.pending = false;
          });
      }
    }
    build.users++;
    if (build.parts) this.adopt(record, build.parts);
  }
  private adopt(record: Record, parts: BufferGeometry[]): void {
    const materials = (this.materials ??= new VehicleMeshMaterialLibrary(this.nodes));
    this.apply(
      record,
      record.baseline.map((part, i) => ({
        ...part,
        geometry: parts[i]!,
        material: materials.get(record.palette),
      })),
    );
  }
  private apply(record: Record, parts: VoxelGeometry[]): void {
    record.root.replaceParts(parts);
    const { anchor, bonus, enabled } = record.root.nativeDepth;
    updateVoxelNativeDepth(record.root, anchor, bonus, enabled !== 0);
  }
  dispose(): void {
    void this.setEnabled(false);
    this.disposed = true;
    this.records.clear();
    this.prepared.clear();
  }
}
