/** Local RA2 cache policy for the offline CAD tool; never opens a VM or fetches resources. */
import { createHash } from 'node:crypto';
import { readdir, readFile, realpath, mkdir, lstat } from 'node:fs/promises';
import { resolve, relative, sep, isAbsolute } from 'node:path';
import type { VoxelModel, VoxelSection } from '../../../packages/ra2-assets/src/manifest';
import { gridIndex, validateGrid, type SolidGrid, type Vec3 } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
import { voxelSectionFrame } from '../../../packages/hd-renderer/src/ra2/voxelSectionFrame';

export const sha256 = (bytes: string | Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
export async function discoverModel(root: string, id: string): Promise<string> {
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new Error('Invalid model ID');
  const hits: string[] = [];
  async function visit(dir: string, depth: number): Promise<void> {
    for (const e of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (e.isDirectory() && depth < 4 && !e.name.startsWith('.') && !['node_modules', 'vehicles-hd'].includes(e.name))
        await visit(resolve(dir, e.name), depth + 1);
      if (e.isFile() && e.name.toUpperCase() === id.toUpperCase() + '.JSON') {
        const p = resolve(dir, e.name);
        const m = JSON.parse(await readFile(p, 'utf8')) as Partial<VoxelModel>;
        if (
          m.id?.toUpperCase() === id.toUpperCase() &&
          Array.isArray(m.parts) &&
          m.parts.every((p) => Array.isArray(p.sections))
        )
          hits.push(p);
      }
    }
  }
  await visit(await realpath(root), 0);
  if (!hits.length) throw new Error(`No source voxel cache found for ${id}`);
  const hashes = await Promise.all(hits.map(async (p) => sha256(await readFile(p))));
  if (new Set(hashes).size > 1)
    throw new Error(`Ambiguous local caches for ${id}; narrow --cache-root: ${hits.join(', ')}`);
  return hits[0];
}
function isInside(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel === '' || (!(rel === '..' || rel.startsWith('..' + sep)) && !isAbsolute(rel));
}
/** Check existing ancestors before creating anything, then resolve again to reject symlink escapes. */
export async function externalOutputDirectory(path: string, repository: string): Promise<string> {
  const repo = await realpath(repository),
    target = resolve(path);
  let ancestor = target;
  const suffix: string[] = [];
  while (true) {
    const stat = await lstat(ancestor).catch((error: NodeJS.ErrnoException) => {
      if (error.code !== 'ENOENT') throw error;
      return undefined;
    });
    if (stat) break;
    const parent = resolve(ancestor, '..');
    suffix.unshift(relative(parent, ancestor));
    ancestor = parent;
  }
  const actual = resolve(await realpath(ancestor), ...suffix);
  if (isInside(repo, actual)) throw new Error('Game-derived output must be outside the repository');
  await mkdir(actual, { recursive: true });
  const final = await realpath(actual);
  if (isInside(repo, final)) throw new Error('Game-derived output must be outside the repository');
  for (const e of await readdir(final, { withFileTypes: true }))
    if (e.isSymbolicLink()) throw new Error('Output directory must not contain symbolic links');
  return final;
}
export function decodeSection(section: VoxelSection): SolidGrid {
  const bytes = Buffer.from(section.voxels, 'base64'),
    size = section.size;
  const data = new Uint8Array(size.reduce((a, b) => a * b, 1));
  validateGrid({ size, data });
  if (bytes.length % 5) throw new Error('Voxel cache must contain quintuples');
  for (let i = 0; i < bytes.length; i += 5) {
    const [x, y, z] = bytes.subarray(i, i + 3);
    if (x >= size[0] || y >= size[1] || z >= size[2]) throw new Error('Voxel outside section dimensions');
    data[gridIndex(size, x, y, z)] = 1;
  }
  if (!data.some(Boolean)) throw new Error('Empty source section');
  return { size, data };
}
/** Row-major affine from lattice corners, including section bounds and the pre-scaled HVA translation. */
export function sectionAffine(s: VoxelSection, turretOffset: number): number[] {
  const step = s.size.map((n, a) => (s.maxBounds[a] - s.minBounds[a]) / n),
    m = s.transform;
  if (m.length !== 12 || ![...step, ...m, ...s.minBounds].every(Number.isFinite))
    throw new Error('Invalid section transform');
  const frame = voxelSectionFrame(s),
    origin = frame.position(0, 0, 0);
  const columns = [frame.position(1, 0, 0), frame.position(0, 1, 0), frame.position(0, 0, 1)].map((p) =>
    p.map((v, a) => v - origin[a]),
  );
  return [0, 1, 2].flatMap((r) => [...columns.map((p) => p[r]), origin[r] + (r === 0 ? turretOffset : 0)]);
}
export function transformPoint(p: Vec3, m: readonly number[]): Vec3 {
  return [0, 1, 2].map((r) => p.reduce((s, v, a) => s + v * m[r * 4 + a], m[r * 4 + 3])) as Vec3;
}
