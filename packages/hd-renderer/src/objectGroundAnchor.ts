import { Vector3, type Object3D } from 'three/webgpu';
import { uniform } from 'three/tsl';

const anchors = new WeakMap<Object3D, Object3D>();
/** The model owner registers a common root for every part, including turret/barrel children. */
export function setObjectGroundAnchor(mesh: Object3D, root: Object3D): void {
  anchors.set(mesh, root);
}
export const objectGroundAnchor = uniform(new Vector3())
  .setName('objectGroundAnchor')
  .onObjectUpdate(({ object }): Vector3 => {
    if (!object) return objectGroundAnchor.value.set(0, 0, 0);
    const root = anchors.get(object) ?? object;
    return objectGroundAnchor.value.setFromMatrixPosition(root.matrixWorld);
  });
