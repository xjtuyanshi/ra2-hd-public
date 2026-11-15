/** Offline residual evidence; no changes to the acceptance metric or fitted geometry. */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPart, type HardSurfaceSpec, type Vec3 } from '../../../packages/hd-renderer/src/hardSurfaceMesh';
import {
  cadSurfaceSamples,
  voxelSurfaceSamples,
  nearestDistances,
  rayVisible,
  deviation,
  evaluateHardSurface,
} from '../../../packages/hd-renderer/src/hardSurfaceMetrics';
import { solidSurfaceMesh } from '../../../packages/hd-renderer/src/hardSurfaceSilhouette';
import { surfaceDistanceQuery } from '../../../packages/hd-renderer/src/hardSurfaceResidual';
import type { SolidGrid } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
import { externalOutputDirectory } from './cache';

const args = process.argv.slice(2);
if (args.length !== 4 || args[0] !== '--input' || args[2] !== '--out')
  throw new Error('residual.ts --input DELIVERY --out DIRECTORY');
const input = resolve(args[1]),
  repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..'),
  out = await externalOutputDirectory(args[3], repo),
  exported = JSON.parse(await readFile(join(input, 'export.json'), 'utf8')),
  spec = JSON.parse(await readFile(join(input, 'spec.json'), 'utf8')) as HardSurfaceSpec,
  summary: Record<string, unknown> = {},
  samples: Record<string, unknown> = {};
for (const [key, part] of Object.entries(spec.parts)) {
  console.log(new Date().toISOString(), 'residual', key);
  const solid = exported.parts[key].solid as SolidGrid;
  const built = buildPart(part),
    vs = voxelSurfaceSamples(solid),
    visible = vs.filter((s) => rayVisible(solid, s)),
    cs = cadSurfaceSamples(built.primitives, built.meshes),
    cadQuery = surfaceDistanceQuery(built.meshes),
    voxelQuery = surfaceDistanceQuery([solidSurfaceMesh(solid)]),
    d1 = nearestDistances(
      visible.map((s) => s.point),
      cs.map((s) => s.point),
    ),
    d2 = nearestDistances(
      cs.map((s) => s.point),
      vs.map((s) => s.point),
    ),
    exact1 = visible.map((s) => cadQuery(s.point)),
    exact2 = cs.map((s) => voxelQuery(s.point));
  const alternate = cadSurfaceSamples(built.primitives, built.meshes, 0.2, 11),
    phase1 = nearestDistances(
      visible.map((s) => s.point),
      alternate.map((s) => s.point),
    ),
    phase2 = nearestDistances(
      alternate.map((s) => s.point),
      vs.map((s) => s.point),
    );
  const zero = structuredClone(part);
  for (const q of zero.primitives) q.chamfer = 0;
  const zeroMetrics = evaluateHardSurface(solid, zero);
  // Translation probes use explicitly reported stratified subsets, not acceptance results.
  const strideV = Math.max(1, Math.ceil(visible.length / 3000)),
    strideC = Math.max(1, Math.ceil(cs.length / 3000)),
    probeV = visible.filter((_, i) => i % strideV === 0),
    probeC = cs.filter((_, i) => i % strideC === 0);
  const translations = [];
  for (let a = 0; a < 3; a++)
    for (const delta of [-0.5, 0.5]) {
      const shift = (p: Vec3, sign: number) => p.map((v, k) => v + (k === a ? sign * delta : 0)) as Vec3;
      translations.push({
        axis: 'xyz'[a],
        delta,
        voxelToCadLowerBound: deviation(probeV.map((s) => cadQuery(shift(s.point, -1)))),
        cadToVoxel: deviation(probeC.map((s) => voxelQuery(shift(s.point, 1)))),
      });
    }
  const high = cs
    .map((s, i) => ({ ...s, formal: d2[i], exact: exact2[i], name: built.primitives[s.primitive].name }))
    .filter((s) => s.exact > 0.85)
    .sort((a, b) => b.exact - a.exact)
    .slice(0, 30);
  summary[key] = {
    formal: { voxelToCad: deviation(d1), cadToVoxel: deviation(d2) },
    exact: { voxelToCadLowerBound: deviation(exact1), cadToVoxel: deviation(exact2) },
    seed11: { voxelToCad: deviation(phase1), cadToVoxel: deviation(phase2) },
    chamferZero: { voxelToCad: zeroMetrics.dev_vox_to_cad, cadToVoxel: zeroMetrics.dev_cad_to_vox },
    translations,
    translationSubset: { strideV, strideC, voxelSamples: probeV.length, cadSamples: probeC.length },
    highResiduals: high,
    note: 'Acceptance is unchanged. Exact CAD-to-voxel distances use the exposed source triangle surface. Voxel-to-CAD exact is a lower bound because the input CAD meshes can contain buried primitive faces. Translation probes are diagnostic subsets, not acceptance.',
  };
  samples[key] = {
    voxel: visible.map((s, i) => [...s.point, ...s.normal, d1[i], exact1[i]]),
    cad: cs.map((s, i) => [...s.point, ...s.normal, d2[i], exact2[i]]),
  };
}
const json = (value: unknown) =>
  JSON.stringify(value, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? String(v) : v), 1) + '\n';
await writeFile(
  join(out, 'residual_summary.json'),
  json({ input, generatedAt: new Date().toISOString(), parts: summary }),
);
await writeFile(join(out, 'residual_samples.json'), json(samples));
