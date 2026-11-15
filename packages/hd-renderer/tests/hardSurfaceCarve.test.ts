import { describe, expect, it } from 'vitest';
import {
  add,
  buildPart,
  carve,
  chamferPlanes,
  dot,
  expand,
  insideAny,
  partPrimitives,
  polytopeFaces,
  scale,
  unit,
  type PartSpec,
} from '../src/hardSurfaceMesh';
import { evaluateHardSurface, revoxelise } from '../src/hardSurfaceMetrics';
import { grid } from './hardSurfaceFixtures';

// A 6x4x4 block with a 2-wide, 2-deep slot cut into its top (x 2..4, all y, z 2..4).
const slotted: PartSpec = {
  primitives: [{ type: 'box', name: 'block', min: [0, 0, 0], max: [6, 4, 4], chamfer: 0.15 }],
  carve: [{ min: [2, -1, 2], max: [4, 5, 5] }],
};

describe('hard surface carve', () => {
  it('keeps a sloped plate continuous across convex carve pieces while beveling its outer edges', () => {
    const part: PartSpec = {
      primitives: [
        {
          type: 'poly',
          name: 'plate',
          chamfer: 0.2,
          planes: [
            [-1, 0, 0, 0],
            [1, 0, 0, 12],
            [0, -1, 0, 0],
            [0, 1, 0, 8],
            [0, 0, -1, 0],
            [1, 0, 3, 18],
          ],
        },
      ],
      carve: [{ min: [3, 2, 1], max: [5, 4, 2] }],
    };
    const prims = partPrimitives(part),
      slope = unit([1, 0, 3]);
    expect(prims.length).toBeGreaterThan(2);
    let seams = 0,
      outer = 0;
    prims.forEach((p, self) => {
      const { vertices, faces } = polytopeFaces(p.planes),
        top = faces.find((f) => dot(p.planes[f.plane], slope) > 1 - 1e-8)!,
        ch = chamferPlanes(p.planes, 0.2, prims, self),
        isolated = chamferPlanes(p.planes, 0.2);
      if (!top) return;
      for (const f of faces) {
        if (f === top) continue;
        const edge = f.vertices.filter((k) => top.vertices.includes(k));
        if (edge.length !== 2) continue;
        const mid = scale(add(vertices[edge[0]], vertices[edge[1]]), 0.5),
          n = unit(add(slope, p.planes[f.plane].slice(0, 3) as [number, number, number])),
          hasBevel = (planes: typeof ch) => planes.some((q) => dot(q, n) > 1 - 1e-8);
        if (mid[0] > 0.1 && mid[0] < 11.9 && mid[1] > 0.1 && mid[1] < 7.9 && mid[2] > 2.1) {
          // These edges lie above the small carve: only the decomposition splits this continuous slope.
          expect(hasBevel(isolated)).toBe(true);
          expect(hasBevel(ch)).toBe(false);
          seams++;
        } else if (mid[0] < 0.1 || mid[0] > 11.9 || mid[1] < 0.1 || mid[1] > 7.9) {
          expect(hasBevel(ch)).toBe(true);
          outer++;
        }
      }
    });
    expect(seams).toBeGreaterThan(0);
    expect(outer).toBeGreaterThan(0);
  });
  it('splits a primitive into convex pieces whose union is the difference', () => {
    const prims = partPrimitives(slotted);
    expect(prims.length).toBeGreaterThan(1);
    expect(prims.every((p) => p.name?.startsWith('block#c'))).toBe(true);
    expect(insideAny([3, 2, 3], prims)).toBe(false);
    expect(insideAny([3, 2, 1], prims)).toBe(true);
    expect(insideAny([1, 2, 3], prims)).toBe(true);
    expect(insideAny([5, 2, 3], prims)).toBe(true);
    // pieces do not overlap: every interior point lies in exactly one piece
    for (const p of [
      [0.5, 1, 1],
      [3, 1, 1],
      [5.5, 3, 3],
    ] as const)
      expect(prims.filter((q) => insideAny([...p], [q])).length).toBe(1);
  });

  it('re-voxelises exactly onto a slotted voxel solid', () => {
    const occ = (x: number, _y: number, z: number) => !(x >= 2 && x < 4 && z >= 2);
    const s = grid([6, 4, 4], ([x, y, z]) => occ(x, y, z));
    const vox = revoxelise(partPrimitives(slotted), s.size);
    let diff = 0;
    for (let x = 0; x < 6; x++)
      for (let y = 0; y < 4; y++)
        for (let z = 0; z < 4; z++) if (vox[(x * 4 + y) * 4 + z] >= 0.5 !== occ(x, y, z)) diff++;
    expect(diff).toBe(0);
    const m = evaluateHardSurface(s, slotted);
    expect(m.iou).toBe(1);
  });

  it('leaves untouched and kept primitives whole and bevels only the carved walls', () => {
    const far = expand({ type: 'box', name: 'far', min: [10, 0, 0], max: [12, 2, 2] });
    const kept = expand({ type: 'box', name: 'kept', min: [0, 0, 0], max: [6, 4, 4] });
    const out = carve([far, kept], { min: [2, -1, 2], max: [4, 5, 5], keep: ['kept'] });
    expect(out.map((p) => p.name)).toEqual(['far', 'kept']);
    const { mesh } = buildPart(slotted);
    // chamfer bevels exist on the slot rims (z = 4 at x = 2 and x = 4) but none on the internal piece junctions
    let rim = 0,
      junction = 0;
    for (let i = 0; i < mesh.chamfer.length; i++) {
      if (!mesh.chamfer[i]) continue;
      const x = mesh.positions[i * 3],
        z = mesh.positions[i * 3 + 2];
      if (z > 3.7 && ((x > 1.7 && x < 2.01) || (x > 3.99 && x < 4.3))) rim++;
      if (z < 1.9 && z > 0.2 && x > 2.2 && x < 3.8) junction++;
    }
    expect(rim).toBeGreaterThan(0);
    expect(junction).toBe(0);
  });
});
