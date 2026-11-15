import { describe, it, expect } from 'vitest';
import { buildPart, insideAny, dot, cross, sub, mirror, expand, type Vec3 } from '../src/hardSurfaceMesh';
import {
  taperedCylinder,
  sweptCylinders,
  sphericalCap,
  capsule,
  torusSegments,
  edgeFillet,
} from '../src/hardSurfaceRound';
import {
  detectRoundFeatures,
  fitDome,
  fitEllipsoidDome,
  fitTaperedTube,
  fitCapsule,
  fitFillet,
  fitPipeBend,
  fittedTubeAxis,
} from '../src/hardSurfaceRoundFit';
import { occupiedCells, proposeHardSurface } from '../src/hardSurfacePropose';
import { grid } from './hardSurfaceFixtures';
import { parseHardSurfaceSpec } from '../src/hardSurfaceLiveMesh';
import { coplanarBoundaries, shiftCoplanarBoundary } from '../src/hardSurfaceRoundPatch';
import { fitHardSurface } from '../src/hardSurfaceFit';

describe('source-supported curved patches', () => {
  it('builds a tapered, oblique cylinder with outward smooth normals and planar ends', () => {
    const q = taperedCylinder([1, 2, 3], [9, 10, 3], [2, 1]);
    const { mesh, primitives } = buildPart({ primitives: [q] });
    expect(insideAny([5, 6, 3], primitives)).toBe(true);
    expect(insideAny([5, 6, 6], primitives)).toBe(false);
    for (let i = 0; i < mesh.indices.length; i += 3) {
      const [a, b, c] = [0, 1, 2].map(
        (k) => Array.from(mesh.positions.slice(mesh.indices[i + k] * 3, mesh.indices[i + k] * 3 + 3)) as Vec3,
      );
      const n = Array.from(mesh.normals.slice(mesh.indices[i] * 3, mesh.indices[i] * 3 + 3));
      expect(dot(cross(sub(b, a), sub(c, a)), n)).toBeGreaterThan(0);
    }
  });
  it('keeps sweep bends continuous and counts every convex piece', () => {
    const built = buildPart({
      primitives: sweptCylinders(
        [
          [0, 0, 0],
          [8, 0, 0],
          [8, 8, 0],
        ],
        1,
      ),
    });
    expect(built.primitives).toHaveLength(2);
    expect(insideAny([8, 0, 0], built.primitives)).toBe(true);
    expect(insideAny([7.5, 0.5, 0], built.primitives)).toBe(true);
    expect(insideAny([4, 4, 0], built.primitives)).toBe(false);
    expect(() =>
      sweptCylinders(
        [
          [0, 0, 0],
          [0, 0, 0],
        ],
        1,
      ),
    ).toThrow();
  });
  it('clips a spherical dome and mirrors its analytic normals with the geometry', () => {
    const p = expand(sphericalCap([4, 4, 4], 3, [0, 0, 1], 3)),
      reflected = mirror(p, 'z', 10);
    expect(insideAny([4, 4, 3], [p])).toBe(false);
    expect(insideAny([4, 4, 6], [p])).toBe(true);
    expect(insideAny([4, 4, 14], [reflected])).toBe(true);
    expect(reflected.round).toEqual({ type: 'sphere', centre: [4, 4, 16] });
    expect(buildPart({ primitives: [sphericalCap([4, 4, 4], 3, [0, 0, 1], 3)] }).mesh.indices.length).toBeGreaterThan(
      0,
    );
  });
  it('models capsules, bounded torus bends and clipped fillets without filling their holes', () => {
    const c = buildPart({ primitives: [capsule([0, 0, 0], [5, 0, 0], 1)] });
    expect(insideAny([-0.8, 0, 0], c.primitives)).toBe(true);
    expect(insideAny([-1.2, 0, 0], c.primitives)).toBe(false);
    const t = buildPart({ primitives: torusSegments([0, 0, 0], [0, 0, 1], [1, 0, 0], 5, 1, Math.PI / 2) });
    expect(t.primitives).toHaveLength(6);
    expect(insideAny([0, 0, 0], t.primitives)).toBe(false);
    expect(insideAny([5, 0, 0], t.primitives)).toBe(true);
    const f = buildPart({
      primitives: [
        edgeFillet([0, 0, 0], [8, 0, 0], 2, [
          [0, 1, 0, 0],
          [0, 0, 1, 0],
        ]),
      ],
    });
    expect(insideAny([4, -1, -1], f.primitives)).toBe(true);
    expect(insideAny([4, 1, 1], f.primitives)).toBe(false);
  });
  it('finds a round pipe from cross sections while rejecting a flat rectangular plate', () => {
    const pipe = grid([18, 12, 12], ([x, y, z]) => x < 16 && Math.hypot(y + 0.5 - 6, z + 0.5 - 6) < 2.5);
    const features = detectRoundFeatures(pipe);
    expect(features.some((f) => f.axis === 'x' && f.evidence.slices >= 14)).toBe(true);
    const plate = grid([18, 12, 12], ([x, y, z]) => x < 16 && y >= 3 && y < 9 && z >= 4 && z < 8);
    expect(detectRoundFeatures(plate)).toHaveLength(0);
  });
  it('detects coherent taper and spherical cross-section curvature but rejects steps and boxes', () => {
    const taper = occupiedCells(grid([16, 14, 14], ([x, y, z]) => Math.hypot(y + 0.5 - 7, z + 0.5 - 7) < 4 - x * 0.15));
    expect(fitTaperedTube(taper, 'x', [7, 7])?.round?.type).toBe('taper');
    const step = occupiedCells(
      grid([16, 14, 14], ([x, y, z]) => Math.hypot(y + 0.5 - 7, z + 0.5 - 7) < (x < 8 ? 4 : 2)),
    );
    expect(fitTaperedTube(step, 'x', [7, 7])).toBeUndefined();
    const dome = occupiedCells(
      grid([16, 16, 16], ([x, y, z]) => z >= 8 && Math.hypot(x + 0.5 - 8, y + 0.5 - 8, z + 0.5 - 8) < 6),
    );
    expect(fitDome(dome, 'z')?.round?.type).toBe('sphere');
    expect(fitDome(occupiedCells(grid([8, 8, 8])), 'z')).toBeUndefined();
  });
  it('infers rounded ends and a quarter-circle fillet from the solid, and exposes seed round fitting', () => {
    const capGrid = grid(
      [26, 14, 14],
      ([x, y, z]) => Math.hypot(x + 0.5 - Math.max(6, Math.min(20, x + 0.5)), y + 0.5 - 7, z + 0.5 - 7) < 4,
    );
    const cells = occupiedCells(capGrid);
    expect(fitCapsule(cells)?.round?.type).toBe('capsule');
    const fitted = fitHardSurface(
      {
        parts: {
          shape: {
            components: [
              {
                type: 'round',
                region: [
                  [0, 0, 0],
                  [26, 14, 14],
                ],
              },
            ],
          },
        },
      },
      { shape: capGrid },
    );
    expect(fitted.parts.shape.primitives[0].round?.type).toBe('capsule');
    const fillet = occupiedCells(grid([20, 8, 8], ([, y, z]) => Math.hypot(y + 0.5, z + 0.5) < 8));
    expect(fitFillet(fillet)?.round?.type).toBe('taper');
    expect(() =>
      fitHardSurface(
        {
          parts: {
            shape: {
              components: [
                {
                  type: 'round',
                  region: [
                    [0, 0, 0],
                    [8, 8, 8],
                  ],
                },
              ],
            },
          },
        },
        { shape: grid([8, 8, 8]) },
      ),
    ).toThrow('insufficient source curvature');
  });
  it('infers torus curvature only when the intervening source arc exists', () => {
    const source = grid(
      [28, 28, 9],
      ([x, y, z]) => x >= 7 && y >= 7 && Math.hypot(Math.hypot(x + 0.5 - 8, y + 0.5 - 8) - 10, z + 0.5 - 4) < 2,
    );
    const points: Vec3[] = [
      [18, 8, 4],
      [8 + 10 / Math.sqrt(2), 8 + 10 / Math.sqrt(2), 4],
      [8, 18, 4],
    ];
    expect(fitPipeBend(points, 1.5, source)?.every((q) => q.round?.type === 'torus')).toBe(true);
    expect(
      fitPipeBend(
        points,
        1.5,
        grid([28, 28, 9], () => false),
      ),
    ).toBeUndefined();
  });
  it('derives a common elevated tube axis from source slice centres', () => {
    const cells = occupiedCells(
      grid([25, 12, 10], ([x, y, z]) => Math.hypot(y + 0.5 - (3 + x * 0.2), z + 0.5 - 5) < 1.7),
    );
    const fitted = fittedTubeAxis(cells, 'x');
    expect(fitted.direction[1] / fitted.direction[0]).toBeCloseTo(0.2, 1);
    expect(fitted.residual).toBeLessThan(0.6);
  });
  it('fits anisotropic dome curvature and preserves its schema and mirrored normals', () => {
    const cells = occupiedCells(
      grid(
        [24, 24, 18],
        ([x, y, z]) => z >= 7 && ((x + 0.5 - 12) / 7) ** 2 + ((y + 0.5 - 12) / 5) ** 2 + ((z + 0.5 - 7) / 8) ** 2 < 1,
      ),
    );
    const q = fitEllipsoidDome(cells, 'z');
    expect(q?.round?.type).toBe('ellipsoid');
    const spec = { units: 'synthetic', parts: { shape: { primitives: [q!] } } };
    expect(parseHardSurfaceSpec(spec).parts.shape.primitives[0].round).toEqual(q?.round);
    const bad = structuredClone(spec);
    if (bad.parts.shape.primitives[0].round?.type === 'ellipsoid') bad.parts.shape.primitives[0].round.radii[1] = 0;
    expect(() => parseHardSurfaceSpec(bad)).toThrow('Invalid CAD spec');
  });
  it('keeps paired partition faces flush and leaves round tangent planes untouched', () => {
    const left = expand({ type: 'box', min: [0, 0, 0], max: [4, 4, 4] }).planes;
    const right = expand({ type: 'box', min: [4, 0, 0], max: [8, 4, 4] }).planes;
    const part = {
      primitives: [
        { type: 'poly' as const, planes: left },
        { type: 'poly' as const, planes: right },
        sphericalCap([20, 20, 20], 2, [0, 0, 1], 4),
      ],
    };
    const seam = coplanarBoundaries(part).find((g) => g.key === '1000,0,0,4000')!;
    expect(seam.members).toHaveLength(2);
    const shifted = shiftCoplanarBoundary(part, seam, 0.5);
    expect(shifted.primitives[2]).toEqual(part.primitives[2]);
    if (shifted.primitives[0].type === 'poly' && shifted.primitives[1].type === 'poly') {
      expect(shifted.primitives[0].planes[1][3]).toBe(4.5);
      expect(shifted.primitives[1].planes[0][3]).toBe(-4.5);
    }
  });
  it('exposes source-only pipe seeds and opt-in generic pipe proposals without making box covers', () => {
    const source = grid([18, 12, 12], ([x, y, z]) => x < 16 && Math.hypot(y + 0.5 - 6, z + 0.5 - 6) < 2.5);
    const fitted = fitHardSurface(
      {
        parts: {
          shape: {
            components: [
              {
                type: 'pipe',
                region: [
                  [0, 0, 0],
                  [18, 12, 12],
                ],
              },
            ],
          },
        },
      },
      { shape: source },
    );
    expect(fitted.parts.shape.primitives.every((q) => q.round?.type === 'taper')).toBe(true);
    const model = proposeHardSurface(source, { category: 'chassis', budget: 20, roundPipes: true });
    expect(model.part.primitives.some((q) => q.round?.type === 'taper')).toBe(true);
    expect(buildPart(model.part).primitives.length).toBeLessThanOrEqual(20);
  });
});
