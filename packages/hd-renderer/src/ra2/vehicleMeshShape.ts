import type { BufferGeometry } from 'three/webgpu';
import { VOXEL_TO_WORLD } from './voxelScale';

export interface ShapeMesh {
  position: Float32Array;
  indices: Uint16Array | Uint32Array;
}
export function shapeMesh(geometry: BufferGeometry): ShapeMesh {
  return {
    position: geometry.getAttribute('position').array as Float32Array,
    indices: geometry.index!.array as Uint16Array | Uint32Array,
  };
}
/** Engine orthographic projection, pixels per original voxel; includes each section's original HVA transform. */
function projected(meshes: ShapeMesh[], facing: number) {
  const angle = Math.PI / 2 - (facing * Math.PI * 2) / 256;
  const c = Math.cos(angle),
    s = Math.sin(angle);
  return meshes.map((mesh) => {
    const xy = new Float32Array((mesh.position.length / 3) * 2);
    for (let i = 0; i < mesh.position.length / 3; i++) {
      const x = mesh.position[i * 3]! / VOXEL_TO_WORLD;
      const y = mesh.position[i * 3 + 1]! / VOXEL_TO_WORLD;
      const z = mesh.position[i * 3 + 2]! / VOXEL_TO_WORLD;
      const rx = c * x + s * z,
        rz = -s * x + c * z;
      xy[i * 2] = (rx - rz) / Math.SQRT2;
      xy[i * 2 + 1] = ((rx + rz) / Math.SQRT2) * 0.5 - y * Math.sqrt(0.75);
    }
    return { xy, indices: mesh.indices };
  });
}
/** Coverage raster, without shading, culling, depth bias or shadow twins. Supersampling bounds contour precision. */
export function compareVehicleSilhouettes(original: ShapeMesh[], refined: ShapeMesh[], facing: number, samples = 8) {
  if (!Number.isInteger(samples) || samples < 1 || samples > 16) throw new RangeError('Invalid silhouette samples');
  const sets = [projected(original, facing), projected(refined, facing)];
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const meshes of sets)
    for (const { xy } of meshes)
      for (let i = 0; i < xy.length; i += 2) {
        x0 = Math.min(x0, xy[i]!);
        y0 = Math.min(y0, xy[i + 1]!);
        x1 = Math.max(x1, xy[i]!);
        y1 = Math.max(y1, xy[i + 1]!);
      }
  if (!Number.isFinite(x0)) throw new Error('Empty silhouette');
  x0 = Math.floor(x0) - 1;
  y0 = Math.floor(y0) - 1;
  const width = (Math.ceil(x1) - x0 + 1) * samples,
    height = (Math.ceil(y1) - y0 + 1) * samples;
  const masks = sets.map((meshes) => {
    const mask = new Uint8Array(width * height);
    for (const { xy, indices } of meshes)
      for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i]! * 2,
          b = indices[i + 1]! * 2,
          c = indices[i + 2]! * 2;
        const ax = (xy[a]! - x0) * samples,
          ay = (xy[a + 1]! - y0) * samples;
        const bx = (xy[b]! - x0) * samples,
          by = (xy[b + 1]! - y0) * samples;
        const cx = (xy[c]! - x0) * samples,
          cy = (xy[c + 1]! - y0) * samples;
        const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (Math.abs(area) < 1e-9) continue;
        for (
          let y = Math.max(0, Math.floor(Math.min(ay, by, cy)));
          y <= Math.min(height - 1, Math.ceil(Math.max(ay, by, cy)));
          y++
        )
          for (
            let x = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
            x <= Math.min(width - 1, Math.ceil(Math.max(ax, bx, cx)));
            x++
          ) {
            const u = ((bx - x - 0.5) * (cy - y - 0.5) - (by - y - 0.5) * (cx - x - 0.5)) / area;
            const v = ((cx - x - 0.5) * (ay - y - 0.5) - (cy - y - 0.5) * (ax - x - 0.5)) / area;
            if (u >= -1e-7 && v >= -1e-7 && u + v <= 1 + 1e-7) mask[y * width + x] = 1;
          }
      }
    return mask;
  });
  const [base, next] = masks as [Uint8Array, Uint8Array];
  const deviation = (from: Uint8Array, to: Uint8Array) => {
    const boundary: number[] = [];
    for (let k = width; k < to.length - width; k++)
      if (to[k] && (!to[k - 1] || !to[k + 1] || !to[k - width] || !to[k + width])) boundary.push(k);
    let max = 0;
    for (let k = 0; k < from.length; k++) {
      if (!from[k] || to[k]) continue;
      let best = Infinity;
      const x = k % width,
        y = Math.floor(k / width);
      for (const j of boundary) best = Math.min(best, (x - (j % width)) ** 2 + (y - Math.floor(j / width)) ** 2);
      max = Math.max(max, Math.sqrt(best) / samples);
    }
    return max;
  };
  let intersection = 0,
    union = 0,
    engineIntersection = 0,
    engineUnion = 0;
  for (let k = 0; k < base.length; k++) {
    intersection += base[k]! & next[k]!;
    union += base[k]! | next[k]!;
  }
  // Engine-pixel coverage majority, with the same origin and raster phase for both surfaces.
  for (let y = 0; y < height; y += samples)
    for (let x = 0; x < width; x += samples) {
      let a = 0,
        b = 0;
      for (let dy = 0; dy < samples; dy++)
        for (let dx = 0; dx < samples; dx++) {
          const k = (y + dy) * width + x + dx;
          a += base[k]!;
          b += next[k]!;
        }
      const aa = a >= (samples * samples) / 2,
        bb = b >= (samples * samples) / 2;
      engineIntersection += Number(aa && bb);
      engineUnion += Number(aa || bb);
    }
  return {
    facing,
    outward: deviation(next, base),
    inward: deviation(base, next),
    iou: intersection / union,
    engineIoU: engineIntersection / engineUnion,
    precision: 1 / samples,
  };
}
