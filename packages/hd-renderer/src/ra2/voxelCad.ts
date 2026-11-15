import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import type { AtmosphereNodes } from '../atmosphere';
import { parseHardSurfaceSpec } from '../hardSurfaceLiveMesh';
import type { HardSurfaceSpec } from '../hardSurfaceMesh';
import { buildCadModel, type CadMeshData } from './voxelCadMesh';
import type { CadReply, CadRequest } from './voxelCadWorker';
import { BufferAttribute, type BufferGeometry } from 'three/webgpu';
import { voxelCadMaterialFor } from './voxelCadMaterial';
import { VOXEL_TO_WORLD } from './voxelScale';
import { smoothPartGeometry, withSilhouette } from './voxelSmoothGeometry';
import { voxelPbrMaterialFor } from './voxelPbr';
import { selectedVoxelTurretIndex } from './voxelTurrets';
import { updateVoxelNativeDepth, VoxelUnit, type VoxelGeometry } from './voxels';

export type CadBuilder = (model: VoxelModel, spec: HardSurfaceSpec, palette: Uint8Array) => Promise<CadMeshData[]>;

/** Reuse source silhouette policies, turret pivots, palette slots and every existing surface twin. */
export function cadVoxelGeometry(
  model: VoxelModel,
  meshes: CadMeshData[],
  palette: Uint8Array,
  nodes?: AtmosphereNodes,
  mode: 'look' | 'pbr' = 'look',
  shared?: readonly BufferGeometry[],
): VoxelGeometry[] {
  if (meshes.length !== model.parts.length) throw new Error('Incomplete CAD mesh set');
  let ground = Infinity,
    top = -Infinity;
  const hull = meshes[model.parts.findIndex((p) => p.role === 'body')] ?? meshes[0];
  for (let i = 1; i < hull.position.length; i += 3) {
    ground = Math.min(ground, hull.position[i] / VOXEL_TO_WORLD);
    top = Math.max(top, hull.position[i] / VOXEL_TO_WORLD);
  }
  return model.parts.map((part, i) => {
    const geometry =
      shared?.[i] ?? withSilhouette(smoothPartGeometry([mode === 'pbr' ? meshes[i].pbr : meshes[i]]), part.sections);
    if (!shared && mode === 'look') {
      geometry.setAttribute('lookCode', new BufferAttribute(meshes[i].lookCode, 1));
      geometry.setAttribute('lookIndex', new BufferAttribute(meshes[i].lookIndex, 1));
    }
    return {
      role: part.role,
      turretOffset: (model.turretOffset ?? 0) / 256,
      geometry,
      material:
        mode === 'pbr'
          ? voxelPbrMaterialFor(palette, nodes)
          : voxelCadMaterialFor(
              palette,
              {
                volume: meshes[i].lookVolume,
                ground,
                height: top - ground,
                body: part.role === 'body',
                steel: part.role === 'barrel',
              },
              nodes,
            ),
    };
  });
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
  private readonly pending = new Map<VoxelModel, Promise<CadMeshData[] | null>>();
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
  load(model: VoxelModel, palette: Uint8Array): Promise<CadMeshData[] | null> {
    if (this.disposed || selectedVoxelTurretIndex(model)) return Promise.resolve(null);
    let pending = this.pending.get(model);
    if (!pending) {
      pending = this.spec(model.id)
        .then(async (spec) => {
          if (!spec || this.disposed) return null;
          const meshes = await (this.builder ? this.builder(model, spec, palette) : this.build(model, spec, palette));
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
  private build(model: VoxelModel, spec: HardSurfaceSpec, palette: Uint8Array): Promise<CadMeshData[]> {
    // Non-browser tooling has no presenting thread; it runs the same pure implementation directly.
    if (typeof Worker === 'undefined') {
      if (typeof document !== 'undefined') return Promise.reject(new Error('CAD requires a Worker in the browser'));
      return Promise.resolve(buildCadModel(model, spec, palette));
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
        worker.postMessage({ model, spec, palette } satisfies CadRequest);
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
