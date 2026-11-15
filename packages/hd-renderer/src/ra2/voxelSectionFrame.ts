import type { VoxelSection } from '../../../ra2-assets/src/manifest';
import { VOXEL_TO_WORLD } from './voxelScale';

/** Shared bounds -> frame-0 HVA -> (x,z,-y) -> cell-unit placement. */
export function voxelSectionFrame(section: VoxelSection) {
  const m = section.transform;
  const step = section.size.map((n, a) => (section.maxBounds[a]! - section.minBounds[a]!) / Math.max(1, n));
  const [a, b, c, , d, e, f, , g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = [
    e * i - f * h,
    f * g - d * i,
    d * h - e * g,
    c * h - b * i,
    a * i - c * g,
    b * g - a * h,
    b * f - c * e,
    c * d - a * f,
    a * e - b * d,
  ].map((v) => v / (det || 1));
  const legacy = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  return {
    mirrored: det < 0,
    position(x: number, y: number, z: number, lattice = true): [number, number, number] {
      if (lattice) {
        x = section.minBounds[0] + x * step[0];
        y = section.minBounds[1] + y * step[1];
        z = section.minBounds[2] + z * step[2];
      }
      return [
        (m[0] * x + m[1] * y + m[2] * z + m[3]) * VOXEL_TO_WORLD,
        (m[8] * x + m[9] * y + m[10] * z + m[11]) * VOXEL_TO_WORLD,
        -(m[4] * x + m[5] * y + m[6] * z + m[7]) * VOXEL_TO_WORLD,
      ];
    },
    normal(
      x: number,
      y: number,
      z: number,
      mode: 'lattice' | 'physical' | 'legacy' = 'lattice',
    ): [number, number, number] {
      if (mode === 'lattice') {
        x /= step[0] || 1;
        y /= step[1] || 1;
        z /= step[2] || 1;
      }
      const n = mode === 'legacy' ? legacy : inv;
      const X = n[0] * x + n[1] * y + n[2] * z,
        Y = n[3] * x + n[4] * y + n[5] * z,
        Z = n[6] * x + n[7] * y + n[8] * z;
      const l = Math.hypot(X, Y, Z) || 1;
      return [X / l, Z / l, -Y / l];
    },
  };
}
