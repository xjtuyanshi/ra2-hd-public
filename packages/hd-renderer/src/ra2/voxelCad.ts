import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import type { AtmosphereNodes } from '../atmosphere';
import { parseHardSurfaceSpec } from '../hardSurfaceLiveMesh';
import type { HardSurfaceSpec } from '../hardSurfaceMesh';
import { buildCadModel } from './voxelCadMesh';
import type { CadReply, CadRequest } from './voxelCadWorker';
import type { SmoothMeshData } from './voxelSmooth';
import { smoothPartGeometry, withSilhouette } from './voxelSmoothGeometry';
import { voxelPbrMaterialFor } from './voxelPbr';
import { selectedVoxelTurretIndex } from './voxelTurrets';
import { updateVoxelNativeDepth, VoxelUnit, type VoxelGeometry } from './voxels';

export type CadBuilder = (model: VoxelModel, spec: HardSurfaceSpec) => Promise<SmoothMeshData[]>;

/** Reuse source silhouette policies, turret pivots, palette slots and every existing surface twin. */
export function cadVoxelGeometry(
  model: VoxelModel,
  meshes: SmoothMeshData[],
  palette: Uint8Array,
  nodes?: AtmosphereNodes,
): VoxelGeometry[] {
  if (meshes.length !== model.parts.length) throw new Error('Incomplete CAD mesh set');
  const material = voxelPbrMaterialFor(palette, nodes);
  return model.parts.map((part, i) => ({
    role: part.role,
    turretOffset: (model.turretOffset ?? 0) / 256,
    geometry: withSilhouette(smoothPartGeometry([meshes[i]]), part.sections),
    material,
  }));
}

/** Exchange geometry and palette material without replacing the unit, articulation or per-object state. */
export function adoptCadVoxel(root: VoxelUnit, previous: VoxelGeometry[], next: VoxelGeometry[]): void {
  root.replaceGeometry(new Map(previous.map((part, i) => [part.geometry, next[i].geometry])));
  root.replacePartMaterials(next.map((part) => part.material!));
  const { anchor, bonus, enabled } = root.nativeDepth;
  updateVoxelNativeDepth(root, anchor, bonus, enabled !== 0);
}

/** Lazy optional specs. Missing specs are silent; all other failures settle once and retain the current geometry. */
export class VoxelCadLibrary {
  private readonly abort = new AbortController();
  private readonly specs = new Map<string, Promise<HardSurfaceSpec | null>>();
  private readonly pending = new Map<VoxelModel, Promise<SmoothMeshData[] | null>>();
  private readonly workers = new Map<Worker, (error: Error) => void>();
  private readonly warnings = new Set<string>();
  private disposed = false;
  constructor(
    readonly base: string,
    private readonly builder?: CadBuilder,
  ) {}
  get active(): boolean {
    return !this.disposed;
  }
  /** Geometry publication failures share the loader's once-per-model warning policy. */
  reportFailure(id: string, error: unknown): void {
    if (this.disposed || this.warnings.has(id)) return;
    this.warnings.add(id);
    console.warn(`[HD] voxel CAD ${id}: ${String(error)}; keeping the current look`);
  }
  load(model: VoxelModel): Promise<SmoothMeshData[] | null> {
    if (this.disposed || selectedVoxelTurretIndex(model)) return Promise.resolve(null);
    let pending = this.pending.get(model);
    if (!pending) {
      pending = this.spec(model.id)
        .then(async (spec) => {
          if (!spec || this.disposed) return null;
          const meshes = await (this.builder ? this.builder(model, spec) : this.build(model, spec));
          return this.disposed ? null : meshes;
        })
        .catch((error) => {
          this.reportFailure(model.id, error);
          return null;
        });
      this.pending.set(model, pending);
    }
    return pending;
  }
  private spec(id: string): Promise<HardSurfaceSpec | null> {
    let pending = this.specs.get(id);
    if (!pending) {
      pending = (async () => {
        const response = await fetch(`${this.base.replace(/\/$/, '')}/${encodeURIComponent(id)}.json`, {
          signal: this.abort.signal,
        });
        if (response.status === 404) return null;
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const type = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase();
        if (response.status === 200 && type !== 'application/json' && !type?.endsWith('+json')) return null;
        const body = await response.text();
        if (response.status === 200 && body.trimStart().startsWith('<')) return null;
        return parseHardSurfaceSpec(JSON.parse(body));
      })().catch((error) => {
        this.reportFailure(id, error);
        return null;
      });
      this.specs.set(id, pending);
    }
    return pending;
  }
  private build(model: VoxelModel, spec: HardSurfaceSpec): Promise<SmoothMeshData[]> {
    // Non-browser tooling has no presenting thread; it runs the same pure implementation directly.
    if (typeof Worker === 'undefined') {
      if (typeof document !== 'undefined') return Promise.reject(new Error('CAD requires a Worker in the browser'));
      return Promise.resolve(buildCadModel(model, spec));
    }
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./voxelCadWorker.ts', import.meta.url), { type: 'module' });
      this.workers.set(worker, reject);
      const finish = () => {
        worker.onmessage = worker.onerror = null;
        worker.terminate();
        this.workers.delete(worker);
      };
      worker.onmessage = (event: MessageEvent<CadReply>) => {
        finish();
        if ('error' in event.data) reject(new Error(event.data.error));
        else resolve(event.data.meshes);
      };
      worker.onerror = (event) => {
        finish();
        reject(new Error(event.message));
      };
      try {
        worker.postMessage({ model, spec } satisfies CadRequest);
      } catch (error) {
        finish();
        reject(error);
      }
    });
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.abort.abort();
    for (const [worker, reject] of this.workers) {
      worker.onmessage = worker.onerror = null;
      worker.terminate();
      reject(new Error('CAD library disposed'));
    }
    this.workers.clear();
    this.pending.clear();
    this.specs.clear();
  }
}
