/** Read-only private-model audit. Run from the repository root with node --import tsx; see README. */
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import * as current from '../src/ra2/voxelSmooth';
import { VOXEL_TO_WORLD } from '../src/ra2/voxelScale';
import { auditSmoothMesh, projectedMeshArea } from '../tests/fixtures/smoothMeshAudit';

const [modelPath, baselinePath] = process.argv.slice(2);
if (!modelPath)
  throw new Error(
    'Usage: node --import tsx packages/hd-renderer/benchmarks/voxelSmoothAudit.mts <model.json> [baseline-module.ts]',
  );
const model = JSON.parse(readFileSync(modelPath, 'utf8')) as VoxelModel;
const baseline: typeof current = baselinePath ? await import(pathToFileURL(resolve(baselinePath)).href) : current;
const median = (n: number[]) => n.sort((a, b) => a - b)[Math.floor(n.length / 2)]!;
console.log(
  JSON.stringify({ node: process.version, model: model.id, iterations: 9, areaFacings: 16, pixelsPerSourceUnit: 8 }),
);
for (const shape of ['occupancy', 'imls'] as const)
  for (const part of model.parts)
    for (const [sectionIndex, section] of part.sections.entries()) {
      const options = current.smoothVoxelOptions({ shape });
      const before = baseline.buildSmoothSection(section, options);
      const after = current.buildSmoothSection(section, options);
      // Same options (AO included), no geometry caches; warm both and alternate order to reduce JIT/GC bias.
      for (let i = 0; i < 3; i++)
        for (const build of [baseline.buildSmoothSection, current.buildSmoothSection]) build(section, options);
      const times = [[], []] as number[][];
      for (let i = 0; i < 9; i++)
        for (const which of i % 2 ? [1, 0] : [0, 1]) {
          const start = performance.now();
          (which ? current : baseline).buildSmoothSection(section, options);
          times[which]!.push(performance.now() - start);
        }
      const area = Array.from({ length: 16 }, (_, facing) => {
        const yaw = (facing * Math.PI) / 8;
        const a = projectedMeshArea(before, yaw, 8 / VOXEL_TO_WORLD);
        const b = projectedMeshArea(after, yaw, 8 / VOXEL_TO_WORLD);
        return { facing: facing * 16, before: a, after: b, percent: 100 * (b / a - 1) };
      });
      console.log(
        JSON.stringify({
          shape,
          role: part.role,
          section: sectionIndex,
          name: section.name,
          options,
          before: auditSmoothMesh(before, options.supersample),
          after: auditSmoothMesh(after, options.supersample),
          tolerance1e6: {
            before: auditSmoothMesh(before, options.supersample, 1e-6),
            after: auditSmoothMesh(after, options.supersample, 1e-6),
          },
          beforeMs: median(times[0]!),
          afterMs: median(times[1]!),
          unchangedAttributes: Object.keys(before)
            .filter((k) => k !== 'indices')
            .every((k) => isDeepStrictEqual(before[k as keyof typeof before], after[k as keyof typeof after])),
          areaMaxAbsPercent: Math.max(...area.map((a) => Math.abs(a.percent))),
          area,
        }),
      );
    }
