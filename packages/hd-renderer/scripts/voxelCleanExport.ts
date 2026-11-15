import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { buildVoxelCleanModel } from '../src/ra2/voxelCleanModel';
import { PAINT_MATERIALS } from '../src/voxelPaint';

// Writes clean-voxel meshes (shape mesh + flat paint per vertex) for the offline still renderer. Game-derived output
// goes to an explicit directory outside the checkout.
// Usage: pnpm exec tsx packages/hd-renderer/scripts/voxelCleanExport.ts <voxel-json-dir> <palette.json> <out-dir> ID...
// <palette.json> is 256 [r, g, b] triples (8-bit sRGB) of the unit palette.
const [input, paletteFile, output, ...ids] = process.argv.slice(2);
if (!input || !paletteFile || !output || !ids.length)
  throw new Error('Usage: voxelCleanExport.ts <voxel-json-dir> <palette.json> <out-dir> ID...');
await mkdir(output, { recursive: true });
const palette = (JSON.parse(await readFile(paletteFile, 'utf8')) as number[][]).flat();
for (const id of ids) {
  const model = JSON.parse(await readFile(join(input, `${id}.json`), 'utf8')) as VoxelModel;
  const parts = buildVoxelCleanModel(model, palette);
  const header: unknown[] = [];
  const chunks: Uint8Array[] = [];
  let offset = 0;
  const push = (array: ArrayBufferView) => {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    // Keep every chunk 4-byte aligned so readers can view floats in place.
    const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
    padded.set(bytes);
    chunks.push(padded);
    const at = offset;
    offset += padded.byteLength;
    return at;
  };
  for (const { role, turretOffset, mesh, paint, analyses } of parts) {
    header.push({
      role,
      turretOffset,
      vertices: mesh.material.length,
      indices: mesh.indices.length,
      positions: push(mesh.positions),
      normals: push(mesh.normals),
      lattice: push(mesh.lattice),
      indexData: push(mesh.indices),
      direction: push(mesh.direction),
      albedo: push(paint.albedo),
      material: push(paint.material),
      mask: push(paint.mask),
      seam: push(paint.seam),
      seamKind: push(paint.seamKind),
      seamAxis: push(paint.seamAxis),
      seams: analyses.reduce((s, a) => s + a.seams.length, 0),
      regions: analyses.reduce((s, a) => s + a.regions.length, 0),
    });
  }
  const blob = new Uint8Array(offset);
  let at = 0;
  for (const c of chunks) {
    blob.set(c, at);
    at += c.byteLength;
  }
  await writeFile(join(output, `${id}.clean.bin`), blob);
  await writeFile(join(output, `${id}.clean.json`), JSON.stringify({ id, materials: PAINT_MATERIALS, parts: header }));
  console.log(id, parts.map((p) => `${p.role}:${p.mesh.material.length}v`).join(' '));
}
