import { BufferAttribute, type BufferGeometry } from 'three/webgpu';
import type { VoxelSection } from '../../../ra2-assets/src/manifest';
import type { SmoothMeshData } from './voxelSmooth';
import { smoothPartGeometry } from './voxelSmoothGeometry';
import { buildOutlinePartGeometry } from './voxels';
import { registerSilhouetteGeometry } from './voxelShadow';

/** Uncached: the vehicle mesh controller owns these arrays, GPU buffers and their silhouette twins. */
export function vehiclePartGeometry(meshes: SmoothMeshData[], sections: readonly VoxelSection[]): BufferGeometry {
  const geometry = smoothPartGeometry(meshes);
  geometry.name = sections.map((s) => s.name).join('+');
  geometry.userData.sectionNames = sections.map((s) => s.name);
  for (const name of ['bevelDistance', 'seamDistance', 'materialClass'] as const) {
    const data = new Float32Array(meshes.reduce((n, m) => n + m.pindex.length, 0));
    let offset = 0;
    for (const mesh of meshes) {
      if (!mesh[name]) throw new Error(`Vehicle mesh missing ${name}`);
      data.set(mesh[name], offset);
      offset += mesh.pindex.length;
    }
    geometry.setAttribute(name, new BufferAttribute(data, 1));
  }
  registerSilhouetteGeometry(geometry, () => buildOutlinePartGeometry(sections));
  return geometry;
}
