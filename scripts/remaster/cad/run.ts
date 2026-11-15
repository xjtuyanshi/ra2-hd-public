/** One offline command: discover -> export -> propose -> refine -> render -> hash delivery manifest. */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname, join, relative, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';
import type { VoxelModel } from '../../../packages/ra2-assets/src/manifest';
import { decodePng } from '../../../packages/ra2-assets/src/node/png';
import { hardSurfaceSolid, type Vec3 } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
import { buildPart, type HardSurfaceSpec } from '../../../packages/hd-renderer/src/hardSurfaceMesh';
import { proposeHardSurface } from '../../../packages/hd-renderer/src/hardSurfacePropose';
import { refineHardSurface, type RefinementResult } from '../../../packages/hd-renderer/src/hardSurfaceRefine';
import {
  compareSilhouettes,
  partMeshes,
  rasterSilhouette,
  solidSurfaceMesh,
  type SilhouetteView,
} from '../../../packages/hd-renderer/src/hardSurfaceSilhouette';
import { buildStandInPartGeometry, voxelShape } from '../../../packages/hd-renderer/src/ra2/voxels';
import { voxelPaintInputFor } from '../../../packages/hd-renderer/src/ra2/voxelCleanModel';
import { analyseVoxelPaint } from '../../../packages/hd-renderer/src/voxelPaint';
import { ra2VoxelPaintPolicy, RA2_HOUSE_RAMP } from '../../../packages/hd-renderer/src/ra2/voxelPaintPolicy';
import { ra2HardSurfaceLook } from '../../../packages/hd-renderer/src/ra2/hardSurfaceLook';
import { decodeSection, discoverModel, externalOutputDirectory, sectionAffine, sha256, transformPoint } from './cache';
import { writeDeliveryManifest } from './manifest';

const args = process.argv.slice(2),
  flags = new Map<string, string>(),
  booleans = new Set<string>();
for (let i = 0; i < args.length; i++) {
  const k = args[i];
  if (['--no-render', '--export-only'].includes(k)) {
    booleans.add(k);
    continue;
  }
  if (
    ![
      '--id',
      '--cache-root',
      '--out',
      '--palette',
      '--house',
      '--evaluations',
      '--seconds',
      '--python',
      '--font',
    ].includes(k) ||
    !args[i + 1] ||
    args[i + 1].startsWith('--')
  )
    throw new Error(`Invalid argument ${k}`);
  if (flags.has(k)) throw new Error(`Duplicate argument ${k}`);
  flags.set(k, args[++i]);
}
for (const k of ['--id', '--cache-root', '--out', '--palette', '--house'])
  if (!flags.has(k))
    throw new Error(
      'Usage: run.ts --id ID --cache-root DIR --palette unit.png --house neutral|R,G,B --out DIR [--evaluations 32 --seconds 300 --python python3 --font FONT --no-render --export-only]',
    );
const numeric = (name: string, fallback: number) => {
  const n = Number(flags.get(name) ?? fallback);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`Invalid ${name}`);
  return n;
};
const evaluations = numeric('--evaluations', 32),
  seconds = numeric('--seconds', 300);
if (!Number.isInteger(evaluations)) throw new Error('Evaluation budget must be an integer');
const started = performance.now(),
  date = new Date().toISOString(),
  here = dirname(fileURLToPath(import.meta.url)),
  repo = resolve(here, '../../..');
const out = await externalOutputDirectory(flags.get('--out')!, repo);
const json = (value: unknown) =>
  JSON.stringify(
    value,
    (_, v: unknown) =>
      typeof v === 'number' && !Number.isFinite(v)
        ? String(v)
        : ArrayBuffer.isView(v)
          ? Array.from(v as unknown as ArrayLike<number>)
          : v,
    1,
  ) + '\n';
const save = async (name: string, value: unknown) => writeFile(join(out, name), json(value));
const log = async (message: string) => {
  const line = `${new Date().toISOString()} ${message}\n`;
  process.stdout.write(line);
  await writeFile(join(out, 'progress.txt'), line, { flag: 'a' });
};
const source = await discoverModel(flags.get('--cache-root')!, flags.get('--id')!),
  sourceBytes = await readFile(source),
  model = JSON.parse(sourceBytes.toString()) as VoxelModel;
for (const input of [source, resolve(flags.get('--palette')!)]) {
  const rel = relative(out, input);
  if (rel === '' || (!(rel === '..' || rel.startsWith('..' + sep)) && !isAbsolute(rel)))
    throw new Error('Output directory must not contain source inputs');
}
await log(
  `Automatic run start; model=${flags.get('--id')}; per-section compute budgets ${evaluations} evaluations / ${seconds}s; offline`,
);
const paletteBytes = await readFile(flags.get('--palette')!),
  paletteImage = decodePng(paletteBytes);
if (paletteImage.width * paletteImage.height !== 256 || paletteImage.channels < 3)
  throw new Error('Palette must contain exactly 256 RGB entries');
const palette = Array.from({ length: 256 }, (_, i) =>
  Array.from(paletteImage.data.slice(i * paletteImage.channels, i * paletteImage.channels + 3)),
);
const houseArg = flags.get('--house')!,
  house = houseArg === 'neutral' ? [0.6, 0.6, 0.6] : houseArg.split(',').map(Number);
if (house.length !== 3 || !house.every((v) => Number.isFinite(v) && v >= 0 && v <= 1))
  throw new Error('--house needs neutral or three sRGB values in [0,1]');
const look = ra2HardSurfaceLook(house as Vec3);
const colour = {
  parameter: houseArg,
  rgb: house,
  source: 'Explicit CLI parameter; neutral is [0.6,0.6,0.6], no faction inference',
  mask: 'Exact source palette indices 16..31; nearest source-surface lookup on CAD, never geometric stripes',
  ramp: RA2_HOUSE_RAMP,
};
const spec: HardSurfaceSpec = {
  units: 'section-local voxel lattice corner coordinates; per-section transforms in export.json',
  parts: {},
};
const exported: Record<string, unknown> = {},
  results: Record<string, RefinementResult> = {},
  proposals: Record<string, unknown> = {},
  timings: Record<string, number> = {};
const viewsByKey: Record<string, SilhouetteView[]> = {},
  solids: Record<string, ReturnType<typeof hardSurfaceSolid>> = {};
const categories = { body: 'chassis', turret: 'crown', barrel: 'tube' } as const,
  budgets = { body: 130, turret: 35, barrel: 3 };
// The camera matches the established offline research view, not an actual VM capture.
const facingTheta = (deg: number): number => {
  const d = (deg * Math.PI) / 180;
  return Math.atan2(Math.cos(d) - 2 * Math.sin(d), 2 * Math.sin(d) + Math.cos(d));
};
function project(p: Vec3, theta: number, scale: number, offset: [number, number]): [number, number] {
  const c = Math.cos(theta),
    s = Math.sin(theta),
    x = c * p[0] + s * p[2],
    z = -s * p[0] + c * p[2];
  return [30 * (x - z) * scale + offset[0], (15 * (x + z) - 36.742346 * p[1]) * scale + offset[1]];
}
for (const part of model.parts) {
  const sectionBudget = Math.floor(budgets[part.role] / part.sections.length);
  if (!sectionBudget) throw new Error('Section count exceeds part budget');
  for (let i = 0; i < part.sections.length; i++) {
    const t = performance.now(),
      section = part.sections[i],
      key = part.sections.length === 1 ? part.role : `${part.role}:${i}`;
    const raw = decodeSection(section),
      solid = hardSurfaceSolid(raw),
      pivot = part.role === 'body' ? 0 : (model.turretOffset ?? 0) / 256;
    const affine = sectionAffine(section, pivot),
      g = buildStandInPartGeometry([section], voxelShape),
      engine: Record<string, number[]> = {};
    for (const name of ['position', 'normal', 'vpos', 'pindex']) engine[name] = Array.from(g.getAttribute(name).array);
    engine.index = Array.from(g.getIndex()!.array);
    g.dispose();
    // Verify corner transformation against authoritative current-engine vertices before using it in metrics/images.
    let maxTransformError = 0;
    for (let j = 0; j < engine.vpos.length; j += 3) {
      const p = transformPoint(engine.vpos.slice(j, j + 3) as Vec3, affine);
      for (let a = 0; a < 3; a++)
        maxTransformError = Math.max(
          maxTransformError,
          Math.abs(p[a] - engine.position[j + a] - (a === 0 ? pivot : 0)),
        );
    }
    if (maxTransformError > 1e-5) throw new Error(`Engine/lattice transform mismatch: ${key}, ${maxTransformError}`);
    const input = voxelPaintInputFor(section),
      paint = analyseVoxelPaint(input, ra2VoxelPaintPolicy(palette.flat()));
    const rows = Array.from({ length: input.x.length }, (_, k) => [
      input.x[k],
      input.y[k],
      input.z[k],
      input.colour[k],
      ...paint.albedo.slice(k * 3, k * 3 + 3),
      paint.mask[k],
    ]);
    // Keep the exact remap mask independently verifiable, rather than trusting material classification.
    if (rows.some((r) => Boolean(r[7]) !== (r[3] >= 16 && r[3] <= 31)))
      throw new Error('Paint remap mask disagrees with source');
    exported[key] = {
      role: part.role,
      sectionIndex: i,
      section,
      raw,
      solid,
      affine,
      pivot,
      engine,
      paint: rows,
      maxTransformError,
    };
    solids[key] = solid;
    const views = Array.from({ length: 8 }, (_, j): SilhouetteView => ({
      name: String(j * 45),
      width: 900,
      height: 900,
      project: (p) => project(transformPoint(p, affine), facingTheta(j * 45), 8, [450, 550]),
    }));
    viewsByKey[key] = views;
    timings[`export:${key}`] = performance.now() - t;
    await log(`Exported ${key}: ${section.size.join('x')}, transform error ${maxTransformError}`);
    if (booleans.has('--export-only')) continue;
    const fitStart = performance.now();
    const proposed = proposeHardSurface(solid, { category: categories[part.role], budget: sectionBudget });
    proposals[key] = proposed;
    timings[`propose:${key}`] = performance.now() - fitStart;
    await log(
      `Proposed ${key}: ${proposed.regions.length} broad regions; mirror at ${proposed.symmetry.at}, IoU ${proposed.symmetry.iou.toFixed(5)}`,
    );
    // Checkpoint every formal evaluation, including rejected candidates; a timeout must leave a usable selection.
    const result = refineHardSurface(solid, proposed, views, {
      maxEvaluations: evaluations,
      maxMilliseconds: seconds * 1000,
      onEvaluation: (entry, selected) => {
        const line =
          JSON.stringify({ at: new Date().toISOString(), key, ...entry }, (_, v: unknown) =>
            typeof v === 'number' && !Number.isFinite(v) ? String(v) : v,
          ) + '\n';
        process.stdout.write(line);
        appendFileSync(join(out, 'evaluations.jsonl'), line);
        spec.parts[key] = selected;
        writeFileSync(join(out, 'checkpoint.json'), json(spec));
      },
    });
    results[key] = result;
    spec.parts[key] = result.part;
    timings[`fit:${key}`] = performance.now() - fitStart;
    await save('spec.json', spec);
    await save('refinement.json', results);
    await save('proposals.json', proposals);
    await log(
      `Selected ${key}: ${result.result.metrics.primitives} primitives, p99 ${result.result.metrics.dev_vox_to_cad.p99}/${result.result.metrics.dev_cad_to_vox.p99}; stop=${result.stop}`,
    );
  }
}
await save('export.json', {
  id: model.id,
  source,
  sourceSha256: sha256(sourceBytes),
  palette: resolve(flags.get('--palette')!),
  paletteSha256: sha256(paletteBytes),
  colours: colour,
  look,
  voxelShape,
  parts: exported,
  camera: {
    scale: 2.5,
    projection: [30, 15, 36.742346],
    facings: Array.from({ length: 8 }, (_, i) => facingTheta(i * 45)),
  },
});
await save('timing.json', {
  started: date,
  totalMs: performance.now() - started,
  phasesMs: timings,
  manualModelDecisions: [],
  note: 'Automatic execution durations; external tool development and visual review must be logged separately.',
});
let assembledPass = false;
if (!booleans.has('--export-only')) {
  const meshes: Record<string, unknown> = {};
  for (const [key, part] of Object.entries(spec.parts)) meshes[key] = buildPart(part);
  await save('cad_mesh.json', meshes);
  const silhouettes: Record<string, unknown> = {};
  for (const [key, result] of Object.entries(results)) silhouettes[key] = result.result.silhouettes;
  const assembled = Array.from({ length: 8 }, (_, f) => {
    const ref = new Uint8Array(900 * 900),
      cad = ref.slice();
    for (const key of Object.keys(results)) {
      const a = rasterSilhouette([solidSurfaceMesh(solids[key])], viewsByKey[key][f]),
        b = rasterSilhouette(partMeshes(spec.parts[key]), viewsByKey[key][f]);
      for (let k = 0; k < ref.length; k++) {
        ref[k] |= a[k];
        cad[k] |= b[k];
      }
    }
    return compareSilhouettes(ref, cad, String(f * 45));
  });
  silhouettes.assembled = assembled;
  assembledPass = assembled.every((v) => v.iou >= 0.97);
  await save('silhouette.json', silhouettes);
  await save('metrics.json', Object.fromEntries(Object.entries(results).map(([k, r]) => [k, r.result])));
  await writeFile(
    join(out, 'metrics.txt'),
    Object.entries(results)
      .map(([k, r]) => `${k}: ${json(r.result)}`)
      .join('\n') +
      '\nassembled: ' +
      json(silhouettes.assembled),
  );
  if (!booleans.has('--no-render')) {
    await log('Rendering boards and eight-facing table');
    const argv = [
      join(here, 'render.py'),
      '--input',
      out,
      '--output',
      out,
      ...(flags.has('--font') ? ['--font', flags.get('--font')!] : []),
    ];
    const rendered = spawnSync(flags.get('--python') ?? 'python3', argv, { stdio: 'inherit' });
    if (rendered.status !== 0) throw new Error(`Offline render failed: ${rendered.error ?? rendered.status}`);
  }
  const summary = Object.entries(results)
    .map(
      ([k, r]) =>
        `| ${k} | ${r.result.metrics.primitives} | ${r.result.metrics.dev_vox_to_cad.p99} | ${r.result.metrics.dev_cad_to_vox.p99} | ${Math.min(...r.result.silhouettes.map((v) => v.iou)).toFixed(6)} | ${r.stop} |`,
    )
    .join('\n');
  await writeFile(
    join(out, 'REPORT.md'),
    `# ${model.id} 自动 CAD 测量\n\n数值验收：**${assembledPass && Object.values(results).every((r) => r.result.pass) ? 'PASS' : 'FAIL'}**。视觉验收：**UNVERIFIED，需实际打开 1×、2× 板检查**。运行时验收：**UNVERIFIED，本工具仅离线运行**。\n\n| 部件 | 展开原语 | voxel→CAD p99 | CAD→voxel p99 | 最低八向 IoU | 停止原因 |\n| --- | ---: | ---: | ---: | ---: | --- |\n${summary}\n\n门槛：双向 p99 ≤ 0.85，各部件及整车每向 IoU ≥ 0.97；预算 body/turret/barrel = 130/35/3；没有 exact/cuboid-cover 回退。Infinity 表示正式采样未找到邻点，属于失败，不计作零误差。\n\n[1× 对比板](board_当前_新.png)、[2× 对比板](board_当前_新_2x.png)、[八向转台](turntable_8.png)、[全部指标](metrics.txt)、[逐次候选](evaluations.jsonl)、[工时](HOURS.md)、[交付清单](manifest.json)。\n\n当前图来自此 checkout 的 buildStandInPartGeometry，同相机与光照下离线重绘；新图使用正式 buildPart 网格。房屋色来自 --house ${houseArg}，RGB = ${house.join(', ')}，mask 来自源索引 16–31；中性色没有冒称为实机盟军默认色。几何分区、镜像轴、源涂装均自动推导；没有输入本车手工坐标。\n\n源缓存 SHA-256：${sha256(sourceBytes)}。自动阶段时长见 timing.json；工具开发与作者视觉审核应另记，不能计成零人工总工时。\n`,
  );
}
await save('timing.json', {
  started: date,
  finished: new Date().toISOString(),
  totalMs: performance.now() - started,
  phasesMs: timings,
  manualModelDecisions: [],
  note: 'Automatic elapsed; external tool development and visual review logged separately. Phases overlap only with external work.',
});
await writeFile(
  join(out, 'HOURS.md'),
  `# Automatic per-model timings\n\nStarted: ${date}. Finished: ${new Date().toISOString()}. Automatic total: ${((performance.now() - started) / 60000).toFixed(3)} minutes.\n\n| Phase | Automatic seconds |\n| --- | ---: |\n${Object.entries(
    timings,
  )
    .map(([k, v]) => `| ${k} | ${(v / 1000).toFixed(3)} |`)
    .join(
      '\n',
    )}\n\nNo model-specific coordinates or seed decisions were supplied to this run. Tool development, diagnosis, run selection and visual review are external hand-directed work and must be recorded separately; this file does not establish zero human labour. Elapsed includes serial computation, not active human typing.\n`,
);
await log('Automatic delivery complete; visual acceptance remains a separate gate');
const revision = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).stdout.trim();
await writeDeliveryManifest(out, {
  version: 1,
  generatedAt: new Date().toISOString(),
  model: model.id,
  sourceSha256: sha256(sourceBytes),
  revision,
  command: process.argv.slice(2),
  colour,
  metricPass: assembledPass && Object.values(results).length > 0 && Object.values(results).every((r) => r.result.pass),
  visualAcceptance: 'UNVERIFIED until a reviewer opens 1x and 2x boards',
  runtime: 'UNVERIFIED; offline only',
});
