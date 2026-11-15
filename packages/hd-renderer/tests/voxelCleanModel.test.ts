import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { VoxelModel, VoxelSection } from '../../ra2-assets/src/manifest';
import { buildVoxelCleanModel } from '../src/ra2/voxelCleanModel';
import { buildVoxelShapeModel } from '../src/ra2/voxelShapeModel';
import { smoothVoxelShapeNormals } from '../src/voxelShapeRelax';

const palette = new Uint8Array(768).fill(40);
function section(offset = 0): VoxelSection {
  const raw: number[] = [];
  for (let x = 0; x < 3; x++)
    for (let y = 0; y < 3; y++) for (let z = 0; z < 8; z++) raw.push(x, y, z, x === 0 && y === 0 ? 16 : 80, 0);
  return {
    name: 'synthetic',
    size: [3, 3, 8],
    scale: 1,
    minBounds: [0, 0, 0],
    maxBounds: [3, 3, 8],
    transform: [1, 0, 0, offset, 0, 1, 0, 0, 0, 0, 1, 0],
    voxels: Buffer.from(raw).toString('base64'),
  };
}
const model: VoxelModel = {
  id: 'SYNTHETIC',
  palette: 'unit',
  normalsMode: 4,
  turretOffset: 64,
  parts: [
    { role: 'body', sections: [section(), section(5)] },
    { role: 'barrel', sections: [section()] },
  ],
};

describe('clean model round-two wiring and export', () => {
  it('defaults to refinement, chooses small zones from masks/barrels and retains broad-panel shading', () => {
    const parts = buildVoxelCleanModel(model, palette);
    expect(parts.map((p) => p.role)).toEqual(['body', 'barrel']);
    expect(parts[1]!.turretOffset).toBe(0.25);
    for (const part of parts) {
      const r = part.relaxation!;
      expect(r).toBeDefined();
      expect(r.stats.flips).toBe(0);
      expect(part.mesh.positions).toBe(r.positions);
      expect(part.mesh.indices).toBe(part.originalMesh.indices);
      expect(part.mesh.lattice).toBe(part.originalMesh.lattice);
      const originalNormals = smoothVoxelShapeNormals(part.originalMesh);
      const relaxedNormals = smoothVoxelShapeNormals({
        ...part.originalMesh,
        positions: r.positions,
        normals: r.normals,
      });
      let changedPaint = 0;
      for (let v = 0; v < part.paint.material.length; v++) {
        if (part.role === 'barrel' || part.paint.mask[v]) expect(r.zone[v]).toBe(2);
        for (let k = 0; k < 3; k++)
          expect(part.mesh.normals[v * 3 + k]).toBe((r.zone[v] ? relaxedNormals : originalNormals)[v * 3 + k]);
        if (part.paint.mask[v])
          expect([...part.paint.albedo.subarray(v * 3, v * 3 + 3)]).toEqual([
            ...part.originalPaint.albedo.subarray(v * 3, v * 3 + 3),
          ]);
        else if (part.paint.albedo[v * 3] !== part.originalPaint.albedo[v * 3]) changedPaint++;
      }
      expect(changedPaint).toBeGreaterThan(0);
      expect(part.mesh.positions).not.toEqual(part.originalMesh.positions);
    }
    expect([...parts[0]!.relaxation!.zone]).toContain(0);
  });
  it('keeps the legacy mesh, paint and metadata intact when disabled', () => {
    const parts = buildVoxelCleanModel(model, palette, undefined, { refine: false }),
      shapes = buildVoxelShapeModel(model);
    parts.forEach((p, i) => {
      expect(p.relaxation).toBeUndefined();
      expect(p.mesh).toBe(p.originalMesh);
      expect(p.paint).toBe(p.originalPaint);
      expect(p.mesh).toEqual(shapes[i]!.mesh);
      for (const value of p.paint.albedo) expect(value).toBe(40);
    });
  });
  it('exports aligned, backwards-compatible buffers and additional relaxed data', () => {
    const dir = mkdtempSync(join(process.cwd(), '.tmp-voxel-clean-export-'));
    try {
      writeFileSync(join(dir, 'SYNTHETIC.json'), JSON.stringify(model));
      writeFileSync(join(dir, 'palette.json'), JSON.stringify(Array.from({ length: 256 }, () => [40, 40, 40])));
      execFileSync(
        process.execPath,
        [
          '--import',
          'tsx',
          'packages/hd-renderer/scripts/voxelCleanExport.ts',
          dir,
          join(dir, 'palette.json'),
          dir,
          'SYNTHETIC',
        ],
        { cwd: process.cwd(), stdio: 'pipe' },
      );
      const header = JSON.parse(readFileSync(join(dir, 'SYNTHETIC.clean.json'), 'utf8'));
      const bytes = readFileSync(join(dir, 'SYNTHETIC.clean.bin'));
      const expected = buildVoxelCleanModel(model, palette);
      expect(header.parts).toHaveLength(expected.length);
      expected.forEach((p, i) => {
        const h = header.parts[i];
        expect(h.vertices).toBe(p.paint.material.length);
        expect(h.indices).toBe(p.mesh.indices.length);
        const arrays = {
          positions: p.originalMesh.positions,
          normals: p.originalMesh.normals,
          lattice: p.mesh.lattice,
          indexData: p.mesh.indices,
          direction: p.mesh.direction,
          albedo: p.originalPaint.albedo,
          material: p.originalPaint.material,
          mask: p.paint.mask,
          seam: p.paint.seam,
          seamKind: p.paint.seamKind,
          seamAxis: p.paint.seamAxis,
          relaxedPositions: p.mesh.positions,
          relaxedNormals: p.mesh.normals,
          relaxedFaceNormals: p.relaxation!.normals,
          zone: p.relaxation!.zone,
          refinedAlbedo: p.paint.albedo,
          refinedMaterial: p.paint.material,
        };
        for (const [key, array] of Object.entries(arrays)) {
          expect(h[key] % 4).toBe(0);
          expect(bytes.subarray(h[key], h[key] + array.byteLength)).toEqual(
            Buffer.from(array.buffer, array.byteOffset, array.byteLength),
          );
        }
        expect(h.relaxationStats).toEqual(p.relaxation!.stats);
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
