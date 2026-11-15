import { describe, expect, it } from 'vitest';
import { exterior, gridCell, hardSurfaceSolid } from '../src/hardSurfaceSolid';
import { grid } from './hardSurfaceFixtures';
describe('hard surface solid', () => {
  it('fills a two-axis enclosed shell despite a leaking deck and preserves a one-cell recess', () => {
    const shell = grid(
      [5, 5, 3],
      ([x, y, z]) => (x === 0 || x === 4 || y === 0 || y === 4) && !(x === 0 && y === 2 && z === 1),
    );
    const before = shell.data.slice(),
      s = hardSurfaceSolid(shell);
    expect(gridCell(s, [2, 2, 1])).toBe(true);
    expect(gridCell(s, [0, 2, 1])).toBe(false);
    expect(gridCell(s, [2, 2, 2])).toBe(false); // Open deck is a recess.
    expect(gridCell(s, [2, 2, 0])).toBe(true); // Open belly is not.
    expect(shell.data).toEqual(before);
    expect(hardSurfaceSolid(shell)).toEqual(s);
  });
  it('rotates the down-axis exception with the caller coordinate system', () => {
    const shell = grid([3, 5, 5], ([, y, z]) => y === 0 || y === 4 || z === 0 || z === 4);
    const s = hardSurfaceSolid(shell, 'x');
    expect(gridCell(s, [0, 2, 2])).toBe(true);
    expect(gridCell(s, [2, 2, 2])).toBe(false);
    expect(gridCell(hardSurfaceSolid(shell), [0, 2, 2])).toBe(false);
  });
  it('floods only six-connected exterior emptiness and fills an unreachable cavity', () => {
    const shell = grid([5, 5, 5], ([x, y, z]) => x === 0 || x === 4 || y === 0 || y === 4 || z === 0 || z === 4);
    expect(exterior(shell).some(Boolean)).toBe(false);
    expect(hardSurfaceSolid(shell).data.every(Boolean)).toBe(true);
    const opened = grid(
      [5, 5, 5],
      ([x, y, z]) =>
        (x === 0 || x === 4 || y === 0 || y === 4 || z === 0 || z === 4) && !(x === 0 && y === 2 && z === 2),
    );
    expect(exterior(opened).filter(Boolean).length).toBe(28);
  });
  it('preserves occupied cells, handles empty and single-cell grids, and rejects malformed storage', () => {
    expect(hardSurfaceSolid(grid([1, 1, 1])).data).toEqual(new Uint8Array([1]));
    expect(hardSurfaceSolid(grid([2, 2, 2], () => false)).data.some(Boolean)).toBe(false);
    expect(() => hardSurfaceSolid({ size: [2, 2, 2], data: new Uint8Array(1) })).toThrow('dimensions');
  });
});
