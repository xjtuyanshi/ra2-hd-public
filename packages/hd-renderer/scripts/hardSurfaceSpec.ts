/** Offline research entry: cache decoding and coordinate policy stay outside the generic fitter. */
import { readFile, writeFile, realpath, lstat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { fitHardSurface, type FitGrid, type HardSurfaceSeed } from '../src/hardSurfaceFit';
import type { HardSurfaceSpec } from '../src/hardSurfaceMesh';
import { evaluateHardSurface } from '../src/hardSurfaceMetrics';
import { gridIndex, hardSurfaceSolid, validateGrid } from '../src/hardSurfaceSolid';

const args = process.argv.slice(2),
  metrics = args.includes('--metrics'),
  evaluateOnly = args.includes('--evaluate'),
  files = args.filter((a) => a !== '--metrics' && a !== '--evaluate');
if (evaluateOnly ? files.length !== 2 : files.length !== 3 || files.some((a) => a.startsWith('--')))
  throw new Error(
    'Usage: hardSurfaceSpec.ts <voxel-model.json> <seed.json> <out-spec.json> [--metrics]\n' +
      '       hardSurfaceSpec.ts --evaluate <voxel-model.json> <spec.json>',
  );
const [modelPath, seedPath, outPath] = files;
const model = JSON.parse(await readFile(modelPath, 'utf8')) as VoxelModel;
const grids = decodeGrids(model);
if (evaluateOnly) {
  // Metrics of an existing (hand-corrected) spec; writes nothing.
  printMetrics(JSON.parse(await readFile(seedPath, 'utf8')) as HardSurfaceSpec);
  process.exit(0);
}
// Resolve the parent too: symlinked resource directories must not be mistaken for repository files.
const repo = await realpath(resolve(dirname(fileURLToPath(import.meta.url)), '../../..'));
const candidate = resolve(await realpath(dirname(resolve(outPath))), basename(outPath));
const existing = await lstat(candidate).catch((error: NodeJS.ErrnoException) => {
  if (error.code !== 'ENOENT') throw error;
  return undefined;
});
const target = existing ? await realpath(candidate) : candidate;
const rel = relative(repo, target);
if (rel === '' || (!(rel === '..' || rel.startsWith('..' + sep)) && !isAbsolute(rel)))
  throw new Error('Game-derived specs must be written outside the repository');
if (target === (await realpath(modelPath)) || target === (await realpath(seedPath)))
  throw new Error('Output must not overwrite either input');
const seed = JSON.parse(await readFile(seedPath, 'utf8')) as HardSurfaceSeed;
function decodeGrids(model: VoxelModel): Record<string, FitGrid> {
  const grids: Record<string, FitGrid> = {};
  for (const part of model.parts) {
    const section = part.sections[0];
    if (!section) throw new Error(`Missing first section: ${part.role}`);
    const bytes = Buffer.from(section.voxels, 'base64'),
      size = section.size;
    const grid: FitGrid = {
      size,
      data: new Uint8Array(size[0] * size[1] * size[2]),
      colours: new Int16Array(size[0] * size[1] * size[2]).fill(-1),
    };
    validateGrid(grid);
    if (bytes.length % 5) throw new Error('Voxel cache is not a sequence of quintuples');
    for (let i = 0; i < bytes.length; i += 5) {
      const [x, y, z] = bytes.subarray(i, i + 3);
      if (x >= size[0] || y >= size[1] || z >= size[2]) throw new Error('Voxel outside section dimensions');
      const k = gridIndex(size, x, y, z);
      grid.data[k] = 1;
      grid.colours![k] = bytes[i + 3];
    }
    grids[part.role] = grid;
  }
  return grids;
}
const spec = fitHardSurface(seed, grids, {
  units: 'voxel lattice, corner coordinates (voxel i spans [i, i+1]); x nose, y left->right, z up',
  onEmpty: (role, comp) => console.error('empty', role, comp.name),
});
await writeFile(target, JSON.stringify(spec, null, 1) + '\n', { flag: 'w' });
console.log(
  JSON.stringify({
    wrote: target,
    parts: Object.fromEntries(Object.entries(spec.parts).map(([r, p]) => [r, p.primitives.length])),
  }),
);
if (metrics) printMetrics(spec);
function printMetrics(spec: HardSurfaceSpec): void {
  for (const [role, part] of Object.entries(spec.parts)) {
    if (!grids[role]) throw new Error(`Spec part has no voxel part: ${role}`);
    const solid = hardSurfaceSolid(grids[role]);
    if (!solid.data.some(Boolean)) throw new Error(`Cannot evaluate an empty voxel part: ${role}`);
    console.log(JSON.stringify({ part: role, ...evaluateHardSurface(solid, part) }));
  }
}
