import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { buildVoxelShapeModel } from '../src/ra2/voxelShapeModel';
import { voxelShapeOptions } from '../src/voxelShapeMesh';

// Explicit read-only input and an external output directory: game-derived meshes never land in the checkout.
// Usage: pnpm exec tsx packages/hd-renderer/scripts/voxelShapeReport.ts <voxel-json-dir> <out-dir> [ID ...]
const [input, output, ...ids] = process.argv.slice(2);
if (!input || !output)
  throw new Error(
    'Usage: pnpm exec tsx packages/hd-renderer/scripts/voxelShapeReport.ts <voxel-json-dir> <out-dir> [ID ...]',
  );
await mkdir(output, { recursive: true });
const names = ids.length
  ? ids
  : (await readdir(input)).filter((f) => /^[A-Z0-9]+\.json$/.test(f)).map((f) => f.slice(0, -5));
// Review variants: VOXEL_SHAPE='{"stairFit":0.8}' and VOXEL_SHAPE_TAG=name write <ID>.<tag>.json instead of <ID>.shape.json.
const shaped = voxelShapeOptions(JSON.parse(process.env.VOXEL_SHAPE ?? '{}'));
const tag = process.env.VOXEL_SHAPE_TAG ?? 'shape';
const cubes = voxelShapeOptions({ maxOffset: 0, bevel: 0, creaseDegrees: 1 });
console.log(
  '| Model | Part | Cells | Quads | Triangles | Welded verts | Stair faces | Max move (vox) | Mean move | ms |',
);
console.log('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
for (const id of names) {
  const model = JSON.parse(await readFile(join(input, `${id}.json`), 'utf8')) as VoxelModel;
  for (const [variant, options] of [
    [tag, shaped],
    ['cubes', cubes],
  ] as const) {
    const start = performance.now();
    const parts = buildVoxelShapeModel(model, options);
    const ms = performance.now() - start;
    const header: unknown[] = [];
    const chunks: Uint8Array[] = [];
    let offset = 0;
    const push = (array: ArrayBufferView) => {
      const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
      chunks.push(bytes);
      const at = offset;
      offset += bytes.byteLength;
      return at;
    };
    for (const { role, turretOffset, mesh } of parts) {
      header.push({
        role,
        turretOffset,
        stats: mesh.stats,
        vertices: mesh.material.length,
        indices: mesh.indices.length,
        positions: push(mesh.positions),
        normals: push(mesh.normals),
        lattice: push(mesh.lattice),
        indexData: push(mesh.indices),
        material: push(mesh.material),
        direction: push(mesh.direction),
        stair: push(mesh.stair),
      });
      if (variant === tag) {
        const s = mesh.stats;
        console.log(
          `| ${id} | ${role} | ${s.cells} | ${s.faces} | ${s.triangles} | ${s.weldedVertices} | ${s.stairFaces} | ${s.maxOffset.toFixed(3)} | ${s.meanOffset.toFixed(3)} | ${ms.toFixed(0)} |`,
        );
      }
    }
    const blob = new Uint8Array(offset);
    let at = 0;
    for (const c of chunks) {
      blob.set(c, at);
      at += c.byteLength;
    }
    await writeFile(join(output, `${id}.${variant}.bin`), blob);
    await writeFile(join(output, `${id}.${variant}.json`), JSON.stringify({ id, options, parts: header }));
  }
}
