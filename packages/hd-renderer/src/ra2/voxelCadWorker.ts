import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import type { HardSurfaceSpec } from '../hardSurfaceMesh';
import type { SmoothMeshData } from './voxelSmooth';
import { buildCadModel } from './voxelCadMesh';

export interface CadRequest {
  model: VoxelModel;
  spec: HardSurfaceSpec;
}
export type CadReply = { meshes: SmoothMeshData[] } | { error: string };
/** One build per short-lived Worker. The owning library terminates it on completion, error or disposal. */
export function answerCadRequest(request: CadRequest): CadReply {
  try {
    return { meshes: buildCadModel(request.model, request.spec) };
  } catch (error) {
    return { error: String(error) };
  }
}
const scope = globalThis as unknown as {
  document?: unknown;
  onmessage: ((event: MessageEvent<CadRequest>) => void) | null;
  postMessage?(reply: CadReply, transfer: Transferable[]): void;
};
if (typeof scope.document === 'undefined' && typeof scope.postMessage === 'function')
  scope.onmessage = (event) => {
    const reply = answerCadRequest(event.data);
    const transfer: Transferable[] =
      'meshes' in reply
        ? reply.meshes.flatMap((mesh) => Object.values(mesh).map((array) => array.buffer as ArrayBuffer))
        : [];
    scope.postMessage!(reply, transfer);
  };
