import { forCells, type SolidGrid, type Vec3 } from '../src/hardSurfaceSolid';
/** Synthetic geometry only. Dense storage follows the reference's x/y/z iteration order. */
export function grid(size: Vec3, keep: (p: Vec3) => boolean = () => true): SolidGrid {
  const data = new Uint8Array(size[0] * size[1] * size[2]);
  forCells(size, (p, i) => {
    data[i] = keep(p) ? 1 : 0;
  });
  return { size, data };
}
