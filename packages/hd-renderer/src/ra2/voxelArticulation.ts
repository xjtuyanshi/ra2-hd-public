import type { Object3D } from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { ShaderNode } from '../tsl';

/** Owned by the voxel root; read live so direct joint rotations and geometry swaps need no pose notification. */
export interface VoxelArticulation {
  joint?: Object3D;
}

export function voxelRelativeYaw(object: Object3D | null | undefined): number {
  const source = (object as { source?: Object3D } | null | undefined)?.source ?? object;
  const articulation = source?.userData.voxelArticulation as VoxelArticulation | undefined;
  const yaw = articulation?.joint?.rotation.y;
  return yaw !== undefined && Number.isFinite(yaw) ? yaw : 0;
}

/** A per-object uniform of shared main, opacity, native-depth and source-colour reflection graphs. */
export const meshRelativeYaw: ShaderNode = uniform(0)
  .setName('voxelRelativeYaw')
  .onObjectUpdate(({ object }: { object?: Object3D | null }) => voxelRelativeYaw(object));
