/** Dense x-major lattice: index = (x * sizeY + y) * sizeZ + z. No world-space transform is applied. */
export type Axis = 'x' | 'y' | 'z';
export type Vec3 = [number, number, number];
export interface SolidGrid {
  size: readonly [number, number, number];
  data: Uint8Array;
}
export const axisIndex = (axis: Axis): number => 'xyz'.indexOf(axis);
export const gridIndex = (size: SolidGrid['size'], x: number, y: number, z: number): number =>
  (x * size[1] + y) * size[2] + z;
export function gridCell(grid: SolidGrid, p: readonly number[]): boolean {
  return p.every((v, a) => v >= 0 && v < grid.size[a]) && !!grid.data[gridIndex(grid.size, p[0], p[1], p[2])];
}
export function validateGrid(grid: SolidGrid): void {
  if (
    grid.size.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    grid.data.length !== grid.size.reduce((a, b) => a * b, 1)
  )
    throw new Error('Invalid dense lattice dimensions');
}
export function forCells(size: SolidGrid['size'], fn: (p: Vec3, index: number) => void): void {
  for (let x = 0; x < size[0]; x++)
    for (let y = 0; y < size[1]; y++) for (let z = 0; z < size[2]; z++) fn([x, y, z], gridIndex(size, x, y, z));
}

/** Empty cells six-connected to the exterior, equivalent to a true one-cell padded flood. */
export function exterior(grid: SolidGrid): Uint8Array {
  const ext = new Uint8Array(grid.data.length);
  const queue: Vec3[] = [];
  const add = (p: Vec3, i: number) => {
    if (!grid.data[i] && !ext[i]) {
      ext[i] = 1;
      queue.push(p);
    }
  };
  forCells(grid.size, (p, i) => {
    if (p.some((v, a) => v === 0 || v === grid.size[a] - 1)) add(p, i);
  });
  for (let head = 0; head < queue.length; head++) {
    for (let a = 0; a < 3; a++)
      for (const s of [-1, 1]) {
        const p = [...queue[head]] as Vec3;
        p[a] += s;
        if (p[a] >= 0 && p[a] < grid.size[a]) add(p, gridIndex(grid.size, ...p));
      }
  }
  return ext;
}

/** Two-axis enclosure, unreachable fill, then one-cell recess restoration (negative down-axis is the floor). */
export function hardSurfaceSolid(occupied: SolidGrid, down: Axis = 'z'): SolidGrid {
  validateGrid(occupied);
  const { size, data } = occupied;
  const enc = new Uint8Array(data.length);
  for (let a = 0; a < 3; a++) {
    forCells(size, (p) => {
      if (p[a] !== 0) return;
      let first = -1,
        last = -1;
      for (let k = 0; k < size[a]; k++) {
        p[a] = k;
        if (data[gridIndex(size, ...p)]) {
          if (first < 0) first = k;
          last = k;
        }
      }
      for (let k = first; k >= 0 && k <= last; k++) {
        p[a] = k;
        enc[gridIndex(size, ...p)]++;
      }
    });
  }
  const s = new Uint8Array(data.length);
  for (let i = 0; i < s.length; i++) s[i] = data[i] || enc[i] >= 2 ? 1 : 0;
  const open = exterior({ size, data: s });
  for (let i = 0; i < s.length; i++) if (!open[i]) s[i] = 1;
  const ext = exterior({ size, data: s });
  const out = s.slice();
  forCells(size, (p, i) => {
    if (data[i] || enc[i] !== 2) return;
    for (let a = 0; a < 3; a++)
      for (const sign of [-1, 1]) {
        if (a === axisIndex(down) && sign === -1) continue;
        const q = [...p] as Vec3;
        q[a] += sign;
        if (q[a] < 0 || q[a] >= size[a] || ext[gridIndex(size, ...q)]) out[i] = 0;
      }
  });
  return { size: [...size], data: out };
}
