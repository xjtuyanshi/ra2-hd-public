import { describe, expect, it } from 'vitest';
import type { CleanVertexPaint } from '../src/voxelCleanPaint';
import { refineCleanVertexPaint } from '../src/voxelCleanPaintRefine';

/** Independent triangle face-cells centred on integer lattice coordinates; no original art. */
function fixture(nx = 9, ny = 9, nz = 1) {
  const n = nx * ny * nz * 3;
  const lattice = new Float32Array(n * 3),
    indices = Uint32Array.from({ length: n }, (_, i) => i);
  const paint: CleanVertexPaint = {
    albedo: new Uint8Array(n * 3).fill(160),
    material: new Uint8Array(n),
    mask: new Uint8Array(n),
    seam: new Float32Array(n).fill(0.5),
    seamKind: new Uint8Array(n).fill(1),
    seamAxis: new Uint8Array(n).fill(2),
    region: new Int32Array(n).fill(7),
  };
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        const v = ((x * ny + y) * nz + z) * 3;
        lattice.set([x - 0.1, y - 0.1, z, x + 0.1, y - 0.1, z, x, y + 0.2, z], v * 3);
      }
  const at = (x: number, y: number, z = 0) => ((x * ny + y) * nz + z) * 3;
  const colour = (v: number, c: number) => paint.albedo.fill(c, v * 3, (v + 3) * 3);
  return { paint, mesh: { lattice, indices }, at, colour };
}
describe('clean paint refinement', () => {
  it('pulls isolated dark fragments 60% and baked bright highlights 90% towards the panel', () => {
    for (const [c, pull, flag] of [
      [40, 0.6, 'darkFragment'],
      [240, 0.9, 'brightFragment'],
    ] as const) {
      const f = fixture(),
        v = f.at(4, 4);
      f.colour(v, c);
      const before = f.paint.albedo.slice(),
        out = refineCleanVertexPaint(f.paint, f.mesh);
      expect(out.albedo[v * 3]).toBe(Math.round(c * (1 - pull) + 160 * pull));
      expect(out[flag][v]).toBe(1);
      expect(f.paint.albedo).toEqual(before);
      expect(out.albedo[0]).toBe(160);
      expect(out.seam).toBe(f.paint.seam);
      expect(out.region).toBe(f.paint.region);
    }
  });
  it('keeps a large dark panel rather than treating it as a fragment', () => {
    const f = fixture();
    for (let x = 2; x <= 6; x++) for (let y = 2; y <= 6; y++) f.colour(f.at(x, y), 90);
    const out = refineCleanVertexPaint(f.paint, f.mesh);
    expect(out.albedo[f.at(4, 4) * 3]).toBe(90);
    expect(out.darkFragment[f.at(4, 4)]).toBe(0);
    // A thin dark strip has a bright neighbourhood median but exceeds the six-cell fragment budget.
    const strip = fixture();
    for (let x = 0; x < 9; x++) for (let y = 3; y <= 4; y++) strip.colour(strip.at(x, y), 90);
    const other = refineCleanVertexPaint(strip.paint, strip.mesh);
    expect(other.albedo[strip.at(4, 3) * 3]).toBe(90);
    expect(other.darkFragment[strip.at(4, 3)]).toBe(0);
  });
  it('softens metal only with at least 30 painted face-cells in its 5x5x5 neighbourhood', () => {
    const f = fixture(5, 5, 5),
      v = f.at(2, 2, 2);
    f.colour(v, 80);
    f.paint.material.fill(1, v, v + 3);
    const out = refineCleanVertexPaint(f.paint, f.mesh);
    expect(out.albedo[v * 3]).toBe(116);
    expect(out.material[v]).toBe(0);
    expect(out.decal[v]).toBe(1);
    expect(f.paint.material[v]).toBe(1);
    const sparse = fixture(5, 5),
      s = sparse.at(2, 2);
    sparse.colour(s, 80);
    sparse.paint.material.fill(1, s, s + 3);
    const kept = refineCleanVertexPaint(sparse.paint, sparse.mesh);
    expect(kept.albedo[s * 3]).toBe(80);
    expect(kept.material[s]).toBe(1);
    expect(kept.decal[s]).toBe(0);
  });
  it('applies the sRGB luma floor except to rubber and preserves every masked byte', () => {
    const f = fixture();
    f.paint.albedo.fill(10);
    const rubber = f.at(4, 4),
      house = f.at(4, 5);
    f.paint.material.fill(2, rubber, rubber + 3);
    f.paint.mask.fill(1, house, house + 3);
    f.paint.albedo.set([20, 5, 1], house * 3);
    const out = refineCleanVertexPaint(f.paint, f.mesh);
    expect(out.albedo[0]).toBe(77);
    expect(out.albedo[rubber * 3]).toBe(10);
    expect([...out.albedo.subarray(house * 3, house * 3 + 9)]).toEqual([
      ...f.paint.albedo.subarray(house * 3, house * 3 + 9),
    ]);
    expect(out.material[house]).toBe(f.paint.material[house]);
    expect(out.darkFragment[house] + out.brightFragment[house] + out.decal[house]).toBe(0);
    expect(out.mask).toBe(f.paint.mask);
  });
  it('accepts caller material ids and thresholds and is deterministic', () => {
    const f = fixture();
    f.paint.material.fill(12);
    const v = f.at(4, 4);
    f.colour(v, 40);
    const opts = { paintedMaterial: 12, metalMaterial: 13, rubberMaterial: 14, fragmentPull: 0.5 };
    const out = refineCleanVertexPaint(f.paint, f.mesh, opts);
    expect(out.albedo[v * 3]).toBe(100);
    expect(out).toEqual(refineCleanVertexPaint(f.paint, f.mesh, opts));
  });
});
