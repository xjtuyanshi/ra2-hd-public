/** Remeasure source-compatible automatic deliveries and assemble the automatically preferred parts. */
import { readFile, writeFile, readdir, realpath } from 'node:fs/promises';
import { resolve, dirname, join, relative, isAbsolute, sep } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { buildPart, type HardSurfaceSpec, type PartSpec } from '../../../packages/hd-renderer/src/hardSurfaceMesh';
import type { SolidGrid } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
import { occupiedCells } from '../../../packages/hd-renderer/src/hardSurfacePropose';
import { tubeLanes } from '../../../packages/hd-renderer/src/hardSurfaceTube';
import {
  measureCandidate,
  compareCandidates,
  candidateRank,
} from '../../../packages/hd-renderer/src/hardSurfaceSelection';
import type { FitAcceptance } from '../../../packages/hd-renderer/src/hardSurfaceRefine';
import {
  rasterSilhouette,
  solidSurfaceMesh,
  partMeshes,
  compareSilhouettes,
  type SilhouetteView,
} from '../../../packages/hd-renderer/src/hardSurfaceSilhouette';
import { externalOutputDirectory, sha256, transformPoint } from './cache';
import { writeDeliveryManifest } from './manifest';

interface Exported {
  id: string;
  source: string;
  sourceSha256: string;
  palette: string;
  paletteSha256: string;
  colours: Record<string, unknown>;
  camera: { projection: number[]; facings: number[] };
  parts: Record<string, { role: 'body' | 'turret' | 'barrel'; affine: number[]; solid: SolidGrid }>;
}
interface Candidate {
  directory: string;
  exported: Exported;
  spec: HardSurfaceSpec;
  proposals: Record<string, { options: Record<string, unknown> }>;
  revision: string;
  automaticMs: number;
}
const args = process.argv.slice(2),
  inputs: string[] = [],
  flags = new Map<string, string>(),
  booleans = new Set<string>();
for (let i = 0; i < args.length; i++) {
  const key = args[i];
  if (key === '--no-render') {
    booleans.add(key);
    continue;
  }
  if (
    !['--candidate', '--out', '--font', '--cell', '--python'].includes(key) ||
    !args[i + 1] ||
    args[i + 1].startsWith('--')
  )
    throw new Error(`Invalid selection argument ${key}`);
  const value = args[++i];
  if (key === '--candidate') inputs.push(value);
  else {
    if (flags.has(key)) throw new Error(`Duplicate ${key}`);
    flags.set(key, value);
  }
}
if (!inputs.length || !flags.has('--out'))
  throw new Error('select.ts --candidate DELIVERY [--candidate DELIVERY] --out FRESH_DIRECTORY [--no-render]');
const started = performance.now(),
  date = new Date().toISOString(),
  here = dirname(fileURLToPath(import.meta.url)),
  repo = resolve(here, '../../..'),
  out = await externalOutputDirectory(flags.get('--out')!, repo);
if ((await readdir(out)).length) throw new Error('Candidate selection requires a fresh output directory');
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
const readJson = async (directory: string, name: string) => JSON.parse(await readFile(join(directory, name), 'utf8'));
const git = (...args: string[]) => {
  const r = spawnSync('git', args, { cwd: repo, encoding: 'utf8' });
  if (r.status !== 0) throw new Error(r.stderr);
  return r.stdout.trim();
};
const revision = git('rev-parse', 'HEAD'),
  sourceStatus = git('status', '--porcelain'),
  sourceNames = [
    'packages/hd-renderer/src/hardSurfaceMesh.ts',
    'packages/hd-renderer/src/hardSurfaceMetrics.ts',
    'packages/hd-renderer/src/hardSurfaceSilhouette.ts',
    'packages/hd-renderer/src/hardSurfaceSelection.ts',
    'scripts/remaster/cad/select.ts',
    'scripts/remaster/cad/render.py',
  ];
const hashes = async () =>
  Object.fromEntries(
    await Promise.all(sourceNames.map(async (name) => [name, sha256(await readFile(join(repo, name)))])),
  );
const sourceHashes = await hashes();
await save('source_snapshot.json', { revision, sourceStatus, hashes: sourceHashes, capturedAt: date });
const log = async (message: string) => {
  const line = `${new Date().toISOString()} ${message}\n`;
  process.stdout.write(line);
  await writeFile(join(out, 'progress.txt'), line, { flag: 'a' });
};
const candidates: Candidate[] = [];
for (const input of inputs) {
  const directory = await realpath(input),
    rel = relative(out, directory);
  if (rel === '' || (!(rel === '..' || rel.startsWith('..' + sep)) && !isAbsolute(rel)))
    throw new Error('Selection output cannot contain its input deliveries');
  if (candidates.some((c) => c.directory === directory)) throw new Error('Duplicate candidate directory');
  const manifest = await readJson(directory, 'manifest.json');
  for (const name of ['export.json', 'spec.json'])
    if (sha256(await readFile(join(directory, name))) !== manifest.files[name]?.sha256)
      throw new Error(`Candidate manifest hash mismatch: ${name}`);
  const exported = (await readJson(directory, 'export.json')) as Exported,
    spec = (await readJson(directory, 'spec.json')) as HardSurfaceSpec,
    timing = await readJson(directory, 'timing.json');
  if (!timing.finished) throw new Error('Incomplete automatic candidate');
  if (
    sha256(await readFile(exported.source)) !== exported.sourceSha256 ||
    sha256(await readFile(exported.palette)) !== exported.paletteSha256
  )
    throw new Error('Candidate source/palette changed');
  if (candidates.length) {
    const base = candidates[0].exported;
    if (
      exported.id !== base.id ||
      exported.sourceSha256 !== base.sourceSha256 ||
      exported.paletteSha256 !== base.paletteSha256 ||
      json(exported.camera) !== json(base.camera) ||
      json(exported.colours) !== json(base.colours) ||
      json(Object.keys(exported.parts)) !== json(Object.keys(base.parts))
    )
      throw new Error('Incompatible candidate source, camera, colour or section set');
    for (const key of Object.keys(base.parts))
      if (
        json(exported.parts[key].affine) !== json(base.parts[key].affine) ||
        json(exported.parts[key].solid) !== json(base.parts[key].solid)
      )
        throw new Error('Incompatible candidate lattice/affine');
  }
  if (json(Object.keys(spec.parts)) !== json(Object.keys(exported.parts)))
    throw new Error('Candidate is missing a source section');
  candidates.push({
    directory,
    exported,
    spec,
    proposals: await readJson(directory, 'proposals.json'),
    revision: manifest.revision,
    automaticMs: timing.totalMs,
  });
}
const base = candidates[0].exported,
  spec: HardSurfaceSpec = { units: candidates[0].spec.units, parts: {} },
  metrics: Record<string, FitAcceptance> = {},
  proposals: Record<string, unknown> = {},
  selection: Record<string, unknown> = {},
  evaluations: Record<string, unknown> = {},
  viewsByKey: Record<string, SilhouetteView[]> = {},
  phases: Record<string, number> = {};
for (const [key, part] of Object.entries(base.parts)) {
  const count = Object.values(base.parts).filter((p) => p.role === part.role).length,
    budget =
      part.role === 'barrel'
        ? 3 * tubeLanes(occupiedCells(part.solid)).length
        : Math.floor((part.role === 'body' ? 130 : 35) / count);
  // Replay the fixed version-1 formal camera, independently from the presentation zoom.
  const views = base.camera.facings.map((theta, i): SilhouetteView => ({
    name: String(i * 45),
    width: 900,
    height: 900,
    project: (p) => {
      const w = transformPoint(p, part.affine),
        c = Math.cos(theta),
        s = Math.sin(theta),
        x = c * w[0] + s * w[2],
        z = -s * w[0] + c * w[2];
      return [30 * (x - z) * 8 + 450, (15 * (x + z) - 36.742346 * w[1]) * 8 + 550];
    },
  }));
  viewsByKey[key] = views;
  const rows: Record<string, unknown>[] = [],
    partStart = performance.now();
  let best: { candidate: Candidate; value: FitAcceptance; part: PartSpec } | undefined;
  for (const candidate of candidates) {
    const fitted = candidate.spec.parts[key],
      expanded = buildPart(fitted);
    let reason = '';
    if (expanded.primitives.length > budget) reason = 'expanded primitive budget';
    if (part.role === 'barrel') {
      const groups = new Map<string, number>();
      for (const q of fitted.primitives) {
        if (q.type !== 'cyl') {
          reason = 'non-cylinder tube';
          break;
        }
        const g = `${q.axis}:${q.c}`;
        groups.set(g, (groups.get(g) ?? 0) + 1);
      }
      if ([...groups.values()].some((n) => n > 3) || groups.size !== budget / 3) reason = 'per-tube/coaxial budget';
    }
    if (reason) {
      rows.push({ directory: candidate.directory, rejected: reason });
      continue;
    }
    const value = measureCandidate(part.solid, fitted, views, budget);
    rows.push({
      directory: candidate.directory,
      revision: candidate.revision,
      rank: candidateRank(value),
      result: value,
    });
    if (!best || compareCandidates(value, best.value) < 0) best = { candidate, value, part: fitted };
    await log(
      `Measured ${key} from ${candidate.directory}; pass=${value.pass}; rank=${candidateRank(value).join('/')}`,
    );
  }
  if (!best) throw new Error(`No eligible candidate for ${key}`);
  spec.parts[key] = structuredClone(best.part);
  metrics[key] = best.value;
  evaluations[key] = rows;
  proposals[key] = {
    ...best.candidate.proposals[key],
    selectionOriginalBudget: best.candidate.proposals[key].options.budget,
    options: { ...best.candidate.proposals[key].options, budget },
  };
  selection[key] = {
    directory: best.candidate.directory,
    revision: best.candidate.revision,
    rank: candidateRank(best.value),
  };
  phases[`measure:${key}`] = performance.now() - partStart;
  await save('checkpoint.json', spec);
  await save('selection.json', selection);
  await save('selection_evaluations.json', evaluations);
}
const assemblyStart = performance.now(),
  silhouettes: Record<string, unknown> = Object.fromEntries(
    Object.entries(metrics).map(([key, value]) => [key, value.silhouettes]),
  );
const assembled = base.camera.facings.map((_, f) => {
  const source = new Uint8Array(900 * 900),
    cad = source.slice();
  for (const [key, part] of Object.entries(base.parts)) {
    const a = rasterSilhouette([solidSurfaceMesh(part.solid)], viewsByKey[key][f]),
      b = rasterSilhouette(partMeshes(spec.parts[key]), viewsByKey[key][f]);
    for (let k = 0; k < a.length; k++) {
      source[k] |= a[k];
      cad[k] |= b[k];
    }
  }
  return compareSilhouettes(source, cad, String(f * 45));
});
silhouettes.assembled = assembled;
phases.assembly = performance.now() - assemblyStart;
const numericPass = Object.values(metrics).every((v) => v.pass) && assembled.every((v) => v.iou >= 0.97);
await save('export.json', base);
await save('spec.json', spec);
await save('proposals.json', proposals);
await save('metrics.json', metrics);
await save('silhouette.json', silhouettes);
await save('cad_mesh.json', Object.fromEntries(Object.entries(spec.parts).map(([k, p]) => [k, buildPart(p)])));
await writeFile(join(out, 'metrics.txt'), json(metrics) + '\nassembled: ' + json(assembled));
if (!booleans.has('--no-render')) {
  const start = performance.now(),
    render = spawnSync(
      flags.get('--python') ?? 'python3',
      [
        join(here, 'render.py'),
        '--input',
        out,
        '--output',
        out,
        '--cell',
        flags.get('--cell') ?? '256',
        ...(flags.has('--font') ? ['--font', flags.get('--font')!] : []),
      ],
      { stdio: 'inherit' },
    );
  if (render.status !== 0) throw new Error(`Selection render failed: ${render.error ?? render.status}`);
  phases.render = performance.now() - start;
}
const upstream = candidates.reduce((s, c) => s + c.automaticMs, 0);
await save('timing.json', {
  started: date,
  finished: new Date().toISOString(),
  totalMs: performance.now() - started,
  candidateAutomaticMs: upstream,
  phasesMs: phases,
  manualModelDecisions: [],
  note: 'Selection elapsed and cumulative upstream trial durations are separate; concurrent wall windows are not additive throughput.',
});
await writeFile(
  join(out, 'HOURS.md'),
  `# Automatic candidate selection\n\nSelection started ${date}; completed ${new Date().toISOString()}. Selection wall minutes: ${((performance.now() - started) / 60000).toFixed(3)}. Cumulative upstream automatic trial minutes: ${(upstream / 60000).toFixed(3)} (overlapping windows, not end-to-end throughput).\n\n${candidates.map((c) => `- ${c.directory}: ${(c.automaticMs / 60000).toFixed(3)} minutes; revision ${c.revision}`).join('\n')}\n\nNo model coordinates or hand part choices were supplied. Candidate paths/ranking policy and author visual review are directed work and must be recorded separately; active human minutes UNVERIFIED.\n`,
);
await writeFile(
  join(out, 'REPORT.md'),
  `# ${base.id} 自动候选选优\n\n数值验收 **${numericPass ? 'PASS' : 'FAIL'}**；视觉 **UNVERIFIED**；实机/GPU **UNVERIFIED，离线流程**。\n\n所有候选来自同一源缓存、坐标/实心体、调色板和房屋色。每候选每部件用当前未改正式指标重新测量；固定优先级：完整部件PASS、失败门槛更少、最大归一短缺更小、原正式诊断loss更低。同分保留首个输入。部件选择后重新测量整车八向，不把部件通过当作整车通过。\n\n| 部件 | 原语 | 双向p99 | 最低IoU |\n| --- | ---: | --- | ---: |\n${Object.entries(
    metrics,
  )
    .map(
      ([k, v]) =>
        `| ${k} | ${v.metrics.primitives} | ${v.metrics.dev_vox_to_cad.p99}/${v.metrics.dev_cad_to_vox.p99} | ${Math.min(...v.silhouettes.map((s) => s.iou)).toFixed(6)} |`,
    )
    .join(
      '\n',
    )}\n\n[1×](board_当前_新.png)、[2×](board_当前_新_2x.png)、[八向](turntable_8.png)、[正式指标](metrics.txt)、[自动来源](selection.json)、[全部重测](selection_evaluations.json)、[工时](HOURS.md)。\n\n上排是当前stand-in几何的同相机/光照离线重绘，下排是所选正式CAD网格；不是实机截图，也没有自动视觉PASS。没有模型坐标或人工指定部件。p99≤0.85、每向IoU≥0.97、130/35/每炮管3预算均未放宽。\n`,
);
const finalHashes = await hashes();
await log(`Automatic candidate selection complete; metricPass=${numericPass}; visual remains separate`);
await writeDeliveryManifest(out, {
  kind: 'automatic-candidate-selection',
  model: base.id,
  revision,
  sourceStatusAtStart: sourceStatus,
  sourceFilesChangedDuringRun: sourceNames.filter((n) => sourceHashes[n] !== finalHashes[n]),
  sourceSha256: base.sourceSha256,
  metricPass: numericPass,
  visualAcceptance: 'UNVERIFIED until author opens boards',
  runtime: 'UNVERIFIED; offline only',
  candidates: candidates.map((c) => ({ directory: c.directory, revision: c.revision })),
  selection,
});
