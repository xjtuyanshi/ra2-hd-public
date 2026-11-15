import type { VoxelModel } from '../../../ra2-assets/src/manifest';
import type { HardSurfaceSpec } from '../hardSurfaceMesh';
import { buildCadModel, type CadMeshData } from './voxelCadMesh';

export interface CadRequest {
  model: VoxelModel;
  spec: HardSurfaceSpec;
  /** Unit RGB palette; cloned, never transferred from the shared palette cache. */
  palette: Uint8Array;
}
export type CadReply = { meshes: CadMeshData[] } | { error: string };
/** One build per short-lived Worker. The owning library terminates it on completion, error or disposal. */
export function answerCadRequest(request: CadRequest): CadReply {
  try {
    return { meshes: buildCadModel(request.model, request.spec, request.palette) };
  } catch (error) {
    return { error: String(error) };
  }
}
/** Transfer only newly owned output buffers, never the request's model or shared palette. */
export function cadReplyTransfers(reply: CadReply): ArrayBuffer[] {
  return 'meshes' in reply
    ? [
        ...new Set(
          reply.meshes.flatMap(({ lookVolume, pbr, ...mesh }) => [
            ...[...Object.values(mesh), ...Object.values(pbr)].map((array) => array.buffer as ArrayBuffer),
            lookVolume.data.buffer as ArrayBuffer,
          ]),
        ),
      ]
    : [];
}
const scope = globalThis as unknown as {
  document?: unknown;
  onmessage: ((event: MessageEvent<CadRequest>) => void) | null;
  postMessage?(reply: CadReply, transfer: Transferable[]): void;
};
if (typeof scope.document === 'undefined' && typeof scope.postMessage === 'function')
  scope.onmessage = (event) => {
    const reply = answerCadRequest(event.data);
    scope.postMessage!(reply, cadReplyTransfers(reply));
  };
