import { describe, expect, it } from 'vitest';
import { grid } from './hardSurfaceFixtures';
import { partPrimitives, buildPart } from '../src/hardSurfaceMesh';
import { proposeRecesses } from '../src/hardSurfaceRecess';
import { evaluateHardSurface } from '../src/hardSurfaceMetrics';

describe('source-empty broad recess proposals', () => {
  const shell = {
    primitives: [
      {
        type: 'box' as const,
        min: [0, 0, 0] as [number, number, number],
        max: [12, 10, 5] as [number, number, number],
        chamfer: 0,
      },
    ],
  };
  it('recovers a long open deck pocket from occupancy and improves formal surface fidelity', () => {
    const source = grid([12, 10, 5], ([x, y, z]) => !(x >= 2 && x < 10 && y >= 3 && y < 7 && z >= 3));
    const proposals = proposeRecesses(source, partPrimitives(shell));
    expect(proposals).toHaveLength(1);
    expect(proposals[0].sourceOverlap).toBe(0);
    const carved = { ...shell, carve: [proposals[0].carve] };
    expect(buildPart(carved).primitives.length).toBeLessThanOrEqual(5);
    expect(evaluateHardSurface(source, carved).iou).toBe(1);
    expect(evaluateHardSurface(source, carved).dev_vox_to_cad.p99).toBeLessThan(
      evaluateHardSurface(source, shell).dev_vox_to_cad.p99,
    );
  });
  it('does not use tiny residuals, paint-only marks or unconstrained exterior space as a cell-cover fallback', () => {
    expect(proposeRecesses(grid([12, 10, 5]), partPrimitives(shell))).toHaveLength(0);
    const dimple = grid([12, 10, 5], ([x, y, z]) => !(x === 6 && y === 5 && z === 4));
    expect(proposeRecesses(dimple, partPrimitives(shell))).toHaveLength(0);
    const exterior = grid([12, 10, 5], ([x]) => x < 6);
    expect(proposeRecesses(exterior, partPrimitives(shell))).toHaveLength(0);
  });
  it('finds broad safe sub-pockets when connected empty space wraps around a source island', () => {
    const source = grid(
      [26, 18, 5],
      ([x, y, z]) => !(z >= 3 && ((x >= 2 && x < 24 && y >= 3 && y < 7) || (x >= 2 && x < 10 && y >= 3 && y < 15))),
    );
    const envelope = {
      primitives: [
        {
          type: 'box' as const,
          min: [0, 0, 0] as [number, number, number],
          max: [26, 18, 5] as [number, number, number],
          chamfer: 0,
        },
      ],
    };
    const proposals = proposeRecesses(source, partPrimitives(envelope));
    expect(proposals.length).toBeGreaterThanOrEqual(2);
    expect(proposals.length).toBeLessThanOrEqual(8);
    expect(proposals.every((p) => p.cells >= 12 && p.sourceOverlap <= 0.01 && p.enclosedAxes.length > 0)).toBe(true);
    const carved = { ...envelope, carve: proposals.map((p) => p.carve) };
    expect(evaluateHardSurface(source, carved).iou).toBeGreaterThan(evaluateHardSurface(source, envelope).iou);
    expect(buildPart(carved).primitives.length).toBeLessThan(35);
  });
  it('trims a broad thin overshoot only when an adjacent source face supports it', () => {
    const source = grid([12, 10, 5], ([, y]) => y < 9);
    const proposals = proposeRecesses(source, partPrimitives(shell));
    expect(proposals).toHaveLength(1);
    expect(proposals[0]).toMatchObject({ enclosedAxes: [], trimAxes: [1], sourceOverlap: 0 });
    const carved = { ...shell, carve: [proposals[0].carve] };
    const metrics = evaluateHardSurface(source, carved);
    expect(metrics.iou).toBe(1);
    expect(metrics.extra).toBe(0);
    expect(metrics.missing).toBe(0);
    expect(metrics.dev_cad_to_vox.p99).toBeLessThanOrEqual(0.85);
  });
});
