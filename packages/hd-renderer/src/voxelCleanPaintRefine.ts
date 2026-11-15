import type { CleanVertexPaint } from './voxelCleanPaint';
import type { VoxelShapeMesh } from './voxelShapeMesh';

export interface CleanPaintRefineOptions {
  paintedMaterial?: number;
  metalMaterial?: number;
  rubberMaterial?: number;
  floor?: number;
  fragmentCells?: number;
  fragmentPull?: number;
  brightPull?: number;
  decalPull?: number;
  decalCells?: number;
  darkRatio?: number;
  brightRatio?: number;
  lighterRatio?: number;
  midTolerance?: number;
}
export const DEFAULT_CLEAN_PAINT_REFINE = Object.freeze({
  paintedMaterial: 0,
  metalMaterial: 1,
  rubberMaterial: 2,
  floor: 0.3,
  fragmentCells: 6,
  fragmentPull: 0.6,
  brightPull: 0.9,
  decalPull: 0.45,
  decalCells: 30,
  darkRatio: 0.62,
  brightRatio: 1.3,
  lighterRatio: 0.8,
  midTolerance: 0.15,
});
export interface RefinedCleanVertexPaint extends CleanVertexPaint {
  darkFragment: Uint8Array;
  brightFragment: Uint8Array;
  decal: Uint8Array;
}
const luma = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
function roundEven(x: number): number {
  const lo = Math.floor(x);
  return x - lo === 0.5 ? lo + (lo % 2 === 0 ? 0 : 1) : Math.round(x);
}
function median(sorted: Float64Array): number {
  const n = sorted.length;
  sorted.sort();
  return n ? (sorted[(n - 1) >> 1]! + sorted[n >> 1]!) / 2 : 0;
}

/**
 * Refines sRGB paint without changing masks or seam metadata. Face-cells reproduce the reference's three indexed
 * triangle-centre assignments (corner slot first, then triangle order), followed by NumPy ties-to-even rounding.
 * Coordinate tuples avoid the reference's packed-4096 key collisions. Output is quantised once to the existing
 * byte albedo contract; all decisions and neighbourhood targets use the original, unrefined floating-point paint.
 */
export function refineCleanVertexPaint(
  paint: CleanVertexPaint,
  mesh: Pick<VoxelShapeMesh, 'lattice' | 'indices'>,
  opts: CleanPaintRefineOptions = {},
): RefinedCleanVertexPaint {
  const o = { ...DEFAULT_CLEAN_PAINT_REFINE, ...opts };
  const n = paint.material.length,
    lat = mesh.lattice,
    idx = mesh.indices;
  if (lat.length !== n * 3 || paint.albedo.length !== n * 3 || paint.mask.length !== n)
    throw new RangeError('Paint/mesh length mismatch');
  for (const key of ['floor', 'fragmentPull', 'brightPull', 'decalPull'] as const)
    if (!(o[key] >= 0 && o[key] <= 1)) throw new RangeError(`Invalid paint option ${key}`);
  for (const key of ['fragmentCells', 'decalCells', 'paintedMaterial', 'metalMaterial', 'rubberMaterial'] as const)
    if (!Number.isInteger(o[key]) || o[key] < 0) throw new RangeError(`Invalid paint option ${key}`);
  for (const key of ['darkRatio', 'brightRatio', 'lighterRatio', 'midTolerance'] as const)
    if (!(o[key] >= 0 && Number.isFinite(o[key]))) throw new RangeError(`Invalid paint option ${key}`);
  const vc = new Int32Array(n * 3);
  for (let j = 0; j < 3; j++)
    for (let t = 0; t < idx.length; t += 3)
      for (let k = 0; k < 3; k++) {
        vc[idx[t + j]! * 3 + k] = roundEven(
          (lat[idx[t]! * 3 + k]! + lat[idx[t + 1]! * 3 + k]! + lat[idx[t + 2]! * 3 + k]!) / 3,
        );
      }
  const cells = new Map<string, number>(),
    inverse = new Uint32Array(n),
    pos = new Int32Array(n * 3);
  const colour = new Float64Array(n * 3),
    cy = new Float64Array(n),
    count = new Uint32Array(n),
    metal = new Uint8Array(n);
  let total = 0;
  for (let v = 0; v < n; v++) {
    const i = v * 3,
      key = `${vc[i]},${vc[i + 1]},${vc[i + 2]}`;
    let c = cells.get(key);
    if (c === undefined) {
      c = total++;
      cells.set(key, c);
      for (let k = 0; k < 3; k++) pos[c * 3 + k] = vc[i + k]!;
    }
    inverse[v] = c;
    if (paint.mask[v]) continue;
    if (paint.material[v] === o.metalMaterial) metal[c] = 1;
    if (paint.material[v] !== o.paintedMaterial) continue;
    count[c]++;
    for (let k = 0; k < 3; k++) colour[c * 3 + k] += paint.albedo[i + k]! / 255;
  }
  for (let c = 0; c < total; c++)
    if (count[c]) {
      for (let k = 0; k < 3; k++) colour[c * 3 + k] /= count[c]!;
      cy[c] = luma(colour[c * 3]!, colour[c * 3 + 1]!, colour[c * 3 + 2]!);
    }
  const target = colour.slice(),
    dark = new Uint8Array(total),
    bright = new Uint8Array(total),
    decalCell = new Uint8Array(total),
    decalTarget = new Float64Array(total * 3);
  const nb = new Uint32Array(125),
    values = new Float64Array(125);
  for (let c = 0; c < total; c++) {
    if (!count[c] && !metal[c]) continue;
    let size = 0;
    for (let dx = -2; dx <= 2; dx++)
      for (let dy = -2; dy <= 2; dy++)
        for (let dz = -2; dz <= 2; dz++) {
          const other = cells.get(`${pos[c * 3]! + dx},${pos[c * 3 + 1]! + dy},${pos[c * 3 + 2]! + dz}`);
          if (other !== undefined && count[other]) {
            nb[size] = other;
            values[size++] = cy[other]!;
          }
        }
    const med = median(values.subarray(0, size));
    if (count[c]) {
      let darker = 0,
        brighter = 0;
      for (let j = 0; j < size; j++) {
        const y = cy[nb[j]!]!;
        if (y < med * o.darkRatio) darker++;
        if (y > med * o.brightRatio) brighter++;
      }
      if (cy[c]! < med * o.darkRatio && darker <= o.fragmentCells) dark[c] = 1;
      else if (cy[c]! > med * o.brightRatio && brighter <= o.fragmentCells * 2) bright[c] = 1;
      if (dark[c] || bright[c]) {
        let used = 0,
          r = 0,
          g = 0,
          b = 0;
        for (let j = 0; j < size; j++) {
          const other = nb[j]!,
            y = cy[other]!;
          if (dark[c] ? y >= med * o.lighterRatio : Math.abs(y - med) < med * o.midTolerance) {
            used++;
            r += colour[other * 3]!;
            g += colour[other * 3 + 1]!;
            b += colour[other * 3 + 2]!;
          }
        }
        if (used) {
          target[c * 3] = r / used;
          target[c * 3 + 1] = g / used;
          target[c * 3 + 2] = b / used;
        }
      }
    }
    if (metal[c] && size >= o.decalCells) {
      decalCell[c] = 1;
      for (let k = 0; k < 3; k++) {
        for (let j = 0; j < size; j++) values[j] = colour[nb[j]! * 3 + k]!;
        decalTarget[c * 3 + k] = median(values.subarray(0, size));
      }
    }
  }
  const albedo = paint.albedo.slice(),
    material = paint.material.slice();
  const darkFragment = new Uint8Array(n),
    brightFragment = new Uint8Array(n),
    decal = new Uint8Array(n);
  for (let v = 0; v < n; v++) {
    if (paint.mask[v]) continue;
    const c = inverse[v]!,
      i = v * 3;
    const painted = material[v] === o.paintedMaterial;
    darkFragment[v] = painted ? dark[c]! : 0;
    brightFragment[v] = painted ? bright[c]! : 0;
    decal[v] = material[v] === o.metalMaterial ? decalCell[c]! : 0;
    const pull = darkFragment[v] ? o.fragmentPull : brightFragment[v] ? o.brightPull : decal[v] ? o.decalPull : 0;
    const tc = decal[v] ? decalTarget : target;
    let r = (albedo[i]! / 255) * (1 - pull) + tc[c * 3]! * pull;
    let g = (albedo[i + 1]! / 255) * (1 - pull) + tc[c * 3 + 1]! * pull;
    let b = (albedo[i + 2]! / 255) * (1 - pull) + tc[c * 3 + 2]! * pull;
    if (decal[v]) material[v] = o.paintedMaterial;
    const y = luma(r, g, b),
      lift = paint.material[v] !== o.rubberMaterial ? Math.max(0, o.floor - y) : 0;
    r += lift;
    g += lift;
    b += lift;
    // Round upward only for lifted colours so byte quantisation cannot undo the luma floor.
    const quantise = lift ? Math.ceil : Math.round;
    albedo[i] = Math.min(255, Math.max(0, quantise(r * 255)));
    albedo[i + 1] = Math.min(255, Math.max(0, quantise(g * 255)));
    albedo[i + 2] = Math.min(255, Math.max(0, quantise(b * 255)));
  }
  return { ...paint, albedo, material, darkFragment, brightFragment, decal };
}
