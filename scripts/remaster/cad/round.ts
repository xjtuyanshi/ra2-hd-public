/** Source-supported curved replacements of an existing offline delivery; fixed-camera remeasurement throughout. */
import { access, readFile, writeFile, readdir } from 'node:fs/promises';
import { appendFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { coalescePart } from '../../../packages/hd-renderer/src/hardSurfaceCoalesce';
import { detectRoundRidges } from '../../../packages/hd-renderer/src/hardSurfaceRoundRidge';
import { buildPart, type PartSpec, type HardSurfaceSpec } from '../../../packages/hd-renderer/src/hardSurfaceMesh';
import {
  detectRoundFeatures,
  fitTaperedTube,
  fittedTubeAxis,
  detectRoundCaps,
  fitFilletedPipe,
  type RoundFeature,
} from '../../../packages/hd-renderer/src/hardSurfaceRoundFit';
import {
  patchRoundFeature,
  coplanarBoundaries,
  shiftCoplanarBoundary,
  upperPipeContinuation,
  patchUpperPipe,
} from '../../../packages/hd-renderer/src/hardSurfaceRoundPatch';
import { sweptCylinders, taperedCylinder } from '../../../packages/hd-renderer/src/hardSurfaceRound';
import { detectRoundShoulders, rankSourceFeatureSubsets } from '../../../packages/hd-renderer/src/hardSurfaceShoulder';
import { tubeLanes, tubeSegments, circularTubeGroups } from '../../../packages/hd-renderer/src/hardSurfaceTube';
import { occupiedCells, cylinderProposal } from '../../../packages/hd-renderer/src/hardSurfacePropose';
import { evaluateHardSurface, type HardSurfaceMetrics } from '../../../packages/hd-renderer/src/hardSurfaceMetrics';
import {
  compareSilhouettes,
  partMeshes,
  rasterSilhouette,
  solidSurfaceMesh,
  type SilhouetteView,
  type SilhouetteComparison,
} from '../../../packages/hd-renderer/src/hardSurfaceSilhouette';
import { type SolidGrid, type Vec3 } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
import { externalOutputDirectory, sha256, transformPoint } from './cache';
import { writeDeliveryManifest } from './manifest';

interface Exported {
  id: string;
  source: string;
  sourceSha256: string;
  palette: string;
  paletteSha256: string;
  colours: unknown;
  camera: { facings: number[]; scale: number };
  parts: Record<string, { role: 'body' | 'turret' | 'barrel'; solid: SolidGrid; affine: number[] }>;
}
const flags = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i];
  if (
    !['--input', '--out', '--seconds', '--font', '--cell', '--python', '--mode'].includes(key) ||
    !process.argv[i + 1] ||
    flags.has(key)
  )
    throw new Error(`Invalid ${key}`);
  flags.set(key, process.argv[i + 1]);
}
const mode = flags.get('--mode') ?? 'round';
if (!['round', 'polish', 'shoulder'].includes(mode)) throw new Error('Mode must be round, polish or shoulder');
if (!flags.has('--input') || !flags.has('--out'))
  throw new Error('round.ts --input AUTOMATIC_DELIVERY --out FRESH_OUTPUT [--seconds 1800]');
const here = dirname(fileURLToPath(import.meta.url)),
  repo = resolve(here, '../../..'),
  input = resolve(flags.get('--input')!);
const out = await externalOutputDirectory(flags.get('--out')!, repo);
if ((await readdir(out)).length) throw new Error('Round fit requires fresh output');
if (flags.has('--font')) await access(flags.get('--font')!);
const started = performance.now(),
  date = new Date().toISOString(),
  limit = Number(flags.get('--seconds') ?? 1800) * 1000;
if (!Number.isFinite(limit) || limit <= 0) throw new Error('Positive seconds required');
const json = (v: unknown, spacing = 1) =>
  JSON.stringify(
    v,
    (_, x: unknown) =>
      typeof x === 'number' && !Number.isFinite(x)
        ? String(x)
        : ArrayBuffer.isView(x)
          ? Array.from(x as unknown as ArrayLike<number>)
          : x,
    spacing,
  ) + '\n';
const save = async (name: string, value: unknown) => writeFile(join(out, name), json(value));
const readJson = async (dir: string, name: string) => JSON.parse(await readFile(join(dir, name), 'utf8'));
const manifest = await readJson(input, 'manifest.json');
for (const name of ['export.json', 'spec.json'])
  if (sha256(await readFile(join(input, name))) !== manifest.files[name]?.sha256)
    throw new Error(`Input hash mismatch: ${name}`);
const ex = (await readJson(input, 'export.json')) as Exported;
if (sha256(await readFile(ex.source)) !== ex.sourceSha256 || sha256(await readFile(ex.palette)) !== ex.paletteSha256)
  throw new Error('Live source/palette differs from frozen export');
let spec = (await readJson(input, 'spec.json')) as HardSurfaceSpec;
const git = (...args: string[]) => {
  const r = spawnSync('git', args, { encoding: 'utf8', cwd: repo });
  if (r.status !== 0) throw new Error(r.stderr);
  return r.stdout.trim();
};
const revision = git('rev-parse', 'HEAD'),
  sourceStatus = git('status', '--porcelain');
const sourceNames = [
  ...(await readdir(join(repo, 'packages/hd-renderer/src')))
    .filter((p) => p.startsWith('hardSurface') && p.endsWith('.ts'))
    .map((p) => 'packages/hd-renderer/src/' + p),
  ...(await readdir(here)).filter((p) => /\.(ts|py)$/.test(p)).map((p) => 'scripts/remaster/cad/' + p),
].sort();
const hashes = async () =>
  Object.fromEntries(await Promise.all(sourceNames.map(async (n) => [n, sha256(await readFile(join(repo, n)))])));
const sourceHashes = await hashes();
await save('source_snapshot.json', {
  revision,
  sourceStatus,
  hashes: sourceHashes,
  capturedAt: date,
  input,
  upstreamManifest: manifest,
});
await writeFile(join(out, 'export.json'), await readFile(join(input, 'export.json')));
// Zero records is a valid empty log; a missing file must not masquerade as that completed state.
await writeFile(join(out, 'evaluations.jsonl'), '');
const log = (s: string) => {
  const line = new Date().toISOString() + ' ' + s + '\n';
  process.stdout.write(line);
  appendFileSync(join(out, 'progress.txt'), line);
};
const views: Record<string, SilhouetteView[]> = {},
  reference: Record<string, Uint8Array[]> = {},
  metrics: Record<
    string,
    { metrics: HardSurfaceMetrics; silhouettes: SilhouetteComparison[]; pass: boolean; loss: number; finite: boolean }
  > = {};
for (const [key, p] of Object.entries(ex.parts)) {
  views[key] = ex.camera.facings.map((theta, i) => ({
    name: String(i * 45),
    width: 900,
    height: 900,
    project: (v: Vec3) => {
      const w = transformPoint(v, p.affine),
        c = Math.cos(theta),
        s = Math.sin(theta),
        x = c * w[0] + s * w[2],
        z = -s * w[0] + c * w[2];
      return [30 * (x - z) * 8 + 450, (15 * (x + z) - 36.742346 * w[1]) * 8 + 550];
    },
  }));
  reference[key] = views[key].map((v) => rasterSilhouette([solidSurfaceMesh(p.solid)], v));
}
const keys = Object.keys(ex.parts),
  assemblyReference = ex.camera.facings.map((_, f) =>
    reference[keys[0]][f].map((_, i) => (keys.some((k) => reference[k][f][i]) ? 1 : 0)),
  );
const partMasks: Record<string, Uint8Array[]> = {};
for (const key of keys) {
  const meshes = partMeshes(spec.parts[key]);
  partMasks[key] = views[key].map((v) => rasterSilhouette(meshes, v));
}
const assembled = (key?: string, masks?: Uint8Array[]) =>
  ex.camera.facings.map((_, f) => {
    const image = new Uint8Array(900 * 900);
    for (const k of keys) {
      const m = k === key ? masks![f] : partMasks[k][f];
      for (let i = 0; i < m.length; i++) image[i] |= m[i];
    }
    return compareSilhouettes(assemblyReference[f], image, String(f * 45));
  });
let currentAssembly = assembled();
const measurements = (key: string, p: PartSpec, masks: Uint8Array[]) => {
  const m = evaluateHardSurface(ex.parts[key].solid, p),
    silhouettes = masks.map((mask, i) => compareSilhouettes(reference[key][i], mask, String(i * 45)));
  const finite = [m.dev_vox_to_cad.p99, m.dev_cad_to_vox.p99, ...silhouettes.map((s) => s.iou)].every(Number.isFinite);
  return {
    metrics: m,
    silhouettes,
    finite,
    pass:
      finite &&
      Math.max(m.dev_vox_to_cad.p99, m.dev_cad_to_vox.p99) <= 1 &&
      silhouettes.every((s) => s.iou >= (ex.parts[key].role === 'barrel' ? 0.9 : 0.97)),
    loss: 0,
  };
};
const budget = (key: string) =>
  ex.parts[key].role === 'barrel'
    ? 3 * tubeLanes(occupiedCells(ex.parts[key].solid)).length
    : Math.floor(
        (ex.parts[key].role === 'body' ? 130 : 35) / keys.filter((k) => ex.parts[k].role === ex.parts[key].role).length,
      );
const phases: Record<string, number> = {},
  detections: Record<string, RoundFeature[]> = {},
  actions: unknown[] = [];
const bendFeatures: Record<string, unknown[]> = {};
const coalescing: Record<string, unknown> = {};
const ridgeCoalescing: Record<string, unknown[]> = {};
for (const key of keys) {
  const t = performance.now();
  metrics[key] = measurements(key, spec.parts[key], partMasks[key]);
  phases['baseline:' + key] = performance.now() - t;
}
await save('baseline_metrics.json', metrics);
await save('baseline_silhouette.json', { assembled: currentAssembly });
log(`${ex.id} baseline assembled min ${Math.min(...currentAssembly.map((s) => s.iou))}`);
const history = (value: unknown) =>
  appendFileSync(join(out, 'evaluations.jsonl'), json({ at: new Date().toISOString(), ...(value as object) }, 0));
const cost = (assembly: SilhouetteComparison[], m: HardSurfaceMetrics, role: string) => {
  const deficits = assembly.map((s) => Math.max(0, 0.97 - s.iou)),
    p99 = Math.max(m.dev_vox_to_cad.p99, m.dev_cad_to_vox.p99);
  return (
    100 * Math.max(...deficits) +
    10 * deficits.reduce((s, v) => s + v, 0) +
    0.1 * Math.max(0, (Number.isFinite(p99) ? p99 : 100) - 1) +
    0.001 * assembly.reduce((s, v) => s + 1 - v.iou, 0) +
    (role === 'barrel' ? 0.01 : 0) * m.primitives * 0.00001
  );
};
const consider = (key: string, p: PartSpec, action: string, allowEqualRound = false) => {
  if (performance.now() - started > limit) return false;
  try {
    if (
      (mode === 'polish' && action.startsWith('source ridge')) ||
      (mode === 'shoulder' && action.startsWith('source shoulder'))
    ) {
      const merged = coalescePart(p);
      p = merged.part;
      (ridgeCoalescing[key] ??= []).push({ action, merges: merged.merges });
      action += `; local convex unions=${merged.merges.length}`;
    }
    if ((p.carve?.length ?? 0) > 32) {
      history({ key, action, accepted: false, rejected: 'carve/schema budget' });
      return false;
    }
    const built = buildPart(p),
      count = built.primitives.length;
    if (ex.parts[key].role === 'barrel') {
      const groups = circularTubeGroups(p.primitives),
        lanes = budget(key) / 3;
      if (groups.size !== lanes || [...groups.values()].some((g) => g.length > 3)) {
        history({ key, action, accepted: false, rejected: 'per-tube coaxial chain budget' });
        return false;
      }
    }

    if (count > budget(key)) {
      history({ key, action, accepted: false, rejected: 'expanded budget', count, budget: budget(key) });
      return false;
    }
    const masks = views[key].map((v) => rasterSilhouette(built.meshes, v)),
      assembly = assembled(key, masks);
    const min = Math.min(...assembly.map((s) => s.iou)),
      previousMin = Math.min(...currentAssembly.map((s) => s.iou));
    if ((previousMin >= 0.97 && min < 0.97) || (previousMin < 0.97 && min < previousMin - 0.0005)) {
      history({ key, action, accepted: false, rejected: 'assembled floor preservation', min });
      return false;
    }
    if (action.startsWith('plate:') && min <= previousMin + 0.00001) {
      history({ key, action, accepted: false, rejected: 'no hard-floor improvement', min });
      return false;
    }
    const measured = measurements(key, p, masks),
      old = metrics[key],
      role = ex.parts[key].role;
    const oldLoss = cost(currentAssembly, old.metrics, role),
      loss = cost(assembly, measured.metrics, role);
    const minPart = Math.min(...measured.silhouettes.map((s) => s.iou)),
      oldPart = Math.min(...old.silhouettes.map((s) => s.iou));
    if (
      allowEqualRound &&
      (minPart < oldPart - 0.01 || measured.silhouettes.some((s, i) => s.iou < old.silhouettes[i].iou - 0.015))
    ) {
      history({
        key,
        action,
        accepted: false,
        rejected: 'source-part silhouette/proportion preservation',
        minPart,
        oldPart,
      });
      return false;
    }

    const oldMax = Math.max(old.metrics.dev_vox_to_cad.p99, old.metrics.dev_cad_to_vox.p99),
      newMax = Math.max(measured.metrics.dev_vox_to_cad.p99, measured.metrics.dev_cad_to_vox.p99);
    const special = action.startsWith('coalesce:')
      ? count < old.metrics.primitives &&
        min >= Math.min(0.97, previousMin) &&
        minPart >= oldPart - 0.001 &&
        newMax <= oldMax + 0.05
      : action.startsWith('prepare:')
        ? min > previousMin + 0.00001 && newMax <= Math.max(1, oldMax) + 0.05
        : undefined;
    const accepted =
      measured.finite &&
      (special !== undefined
        ? special
        : role === 'barrel'
          ? minPart > oldPart + 1e-6 && min >= Math.min(0.97, previousMin) - 1e-6
          : loss < oldLoss - 1e-7 ||
            (allowEqualRound &&
              p.primitives.filter((q) => q.round).length > spec.parts[key].primitives.filter((q) => q.round).length &&
              loss <= oldLoss + 0.0001 &&
              min >= Math.min(0.97, previousMin) &&
              Math.max(measured.metrics.dev_vox_to_cad.p99, measured.metrics.dev_cad_to_vox.p99) <=
                Math.max(old.metrics.dev_vox_to_cad.p99, old.metrics.dev_cad_to_vox.p99) + 0.05));
    measured.loss = loss;
    history({ key, action, accepted, assembledMin: min, minPart, loss, previousLoss: oldLoss, result: measured });
    if (accepted) {
      spec.parts[key] = structuredClone(p);
      partMasks[key] = masks;
      currentAssembly = assembly;
      metrics[key] = measured;
      actions.push({ key, action });
      log(
        `${key}: ${action}; min assembly=${min.toFixed(6)}, part=${minPart.toFixed(6)}, p99=${measured.metrics.dev_vox_to_cad.p99}/${measured.metrics.dev_cad_to_vox.p99}; primitives=${count}`,
      );
    }
    return accepted;
  } catch (error) {
    history({ key, action, accepted: false, error: String(error) });
    return false;
  }
};
for (const key of keys) {
  const t = performance.now(),
    solid = ex.parts[key].solid;
  if (mode === 'shoulder') {
    if (ex.parts[key].role !== 'barrel') {
      const merged = coalescePart(spec.parts[key]);
      coalescing[key] = {
        merges: merged.merges,
        before: buildPart(spec.parts[key]).primitives.length,
        proposed: buildPart(merged.part).primitives.length,
      };
      if (merged.merges.length) consider(key, merged.part, `coalesce: ${merged.merges.length} exact convex unions`);
      const features = detectRoundShoulders(solid);
      detections[key] = features;
      const base = structuredClone(spec.parts[key]);
      let jointAccepted = false;
      for (const proposal of rankSourceFeatureSubsets(features)) {
        if (performance.now() - started > limit) break;
        let candidate = structuredClone(base);
        for (const i of proposal.indices) candidate = patchRoundFeature(candidate, features[i], `shoulder_subset_${i}`);
        if (
          consider(
            key,
            candidate,
            `source shoulder subset ${proposal.indices.join(',')}; supported source cells=${proposal.coveredCells}`,
            true,
          )
        ) {
          jointAccepted = true;
          break;
        }
      }
      if (!jointAccepted) {
        // Related shoulders may free each other's old clipping fragments; only the complete source-supported bundle is evaluated.
        const groups = new Map<string, number[]>();
        features.forEach((f, i) => {
          const signature = f.axis + ':' + f.points.map((p) => p['xyz'.indexOf(f.axis)]).join(',');
          const group = groups.get(signature) ?? [];
          group.push(i);
          groups.set(signature, group);
        });
        for (const indices of groups.values())
          if (indices.length > 1) {
            let candidate = structuredClone(spec.parts[key]);
            for (const i of indices) candidate = patchRoundFeature(candidate, features[i], `shoulder_bundle_${i}`);
            consider(key, candidate, `source shoulder bundle ${indices.join(',')}`, true);
          }
        for (const [i, f] of features.entries())
          consider(
            key,
            patchRoundFeature(spec.parts[key], f, `shoulder_${i}`),
            `source shoulder ${i}; section-height-error=${f.profileError}; ${f.evidence.slices} slices`,
            true,
          );
      }
    }
  } else if (mode === 'polish') {
    if (ex.parts[key].role !== 'barrel') {
      const merged = coalescePart(spec.parts[key]);
      coalescing[key] = {
        merges: merged.merges,
        planeTolerance: 1e-7,
        before: buildPart(spec.parts[key]).primitives.length,
        proposed: buildPart(merged.part).primitives.length,
      };
      if (merged.merges.length) consider(key, merged.part, `coalesce: ${merged.merges.length} exact convex unions`);
      const marginDeadline = started + limit * 0.4;
      for (
        let pass = 0;
        pass < 2 && Math.min(...currentAssembly.map((s) => s.iou)) < 0.971 && performance.now() < marginDeadline;
        pass++
      ) {
        let improved = false;
        for (const group of coplanarBoundaries(spec.parts[key])) {
          if (Math.min(...currentAssembly.map((s) => s.iou)) >= 0.971 || performance.now() > marginDeadline) break;
          for (const delta of [-0.25, 0.25, -0.5, 0.5])
            improved =
              consider(key, shiftCoplanarBoundary(spec.parts[key], group, delta), `prepare:${group.key}/${delta}`) ||
              improved;
        }
        if (!improved) break;
      }
      const ridges = detectRoundRidges(solid);
      detections[key] = ridges;
      for (const [i, ridge] of ridges.entries())
        consider(
          key,
          patchUpperPipe(spec.parts[key], ridge, `ridge_${i}`, 'z', ridge.floorOffset, ridge.floorSlope),
          `source ridge ${i}; raw-profile-height-error=${ridge.maximumHeightError}; ${ridge.evidence.slices} slices`,
          true,
        );
    }
  } else if (ex.parts[key].role === 'barrel') {
    const cells = occupiedCells(solid),
      lanes = tubeLanes(cells);
    const source = lanes.flatMap((lane, i) => {
      const cylinder = cylinderProposal(lane, 'x');
      if (cylinder.type !== 'cyl') throw new Error('Expected tube axis');
      return tubeSegments(lane, 'x', 3).map((segment, j) => {
        const q = cylinderProposal(segment, 'x');
        if (q.type !== 'cyl') throw new Error('Expected tube');
        q.c = [...cylinder.c];
        q.name = `barrel_${i}_${j}`;
        q.chamfer = 0.025;
        q.mirror = false;
        return { q, cells: segment, taper: fitTaperedTube(segment, 'x', cylinder.c) };
      });
    });
    consider(key, { primitives: source.map((s) => s.q) }, 'source-profile circular chain');
    const elevated = lanes.flatMap((lane, i) => {
      const axis = fittedTubeAxis(lane, 'x');
      if (axis.residual > 0.6) return [];
      const point = (x: number) =>
        axis.start.map((v, k) => v + ((x - axis.start[0]) * axis.direction[k]) / axis.direction[0]) as Vec3;
      return tubeSegments(lane, 'x', 3).map((segment, j) => {
        const q = cylinderProposal(segment, 'x');
        if (q.type !== 'cyl') throw new Error('Expected source tube');
        const radius = q.r * Math.sqrt(axis.direction[0]);
        return {
          ...taperedCylinder(point(q.range[0]), point(q.range[1]), [radius, radius]),
          name: `barrel_${i}_${j}`,
          mirror: false,
        };
      });
    });
    if (elevated.length === source.length)
      consider(key, { primitives: elevated }, 'source-supported shared elevated axis');

    // Whole-lane centre updates preserve coaxiality. No transverse boxes, clips, or fabricated muzzle rings.
    for (let pass = 0; pass < 3; pass++) {
      let improved = false;
      const centres = [
        ...new Set(
          spec.parts[key].primitives
            .filter((q) => q.type === 'cyl')
            .map((q) => (q.type === 'cyl' ? q.c.join(',') : '')),
        ),
      ];
      for (const centre of centres)
        for (const dimension of [0, 1])
          for (const delta of [-0.2, 0.2, -0.1, 0.1]) {
            const p = structuredClone(spec.parts[key]);
            for (const q of p.primitives) if (q.type === 'cyl' && q.c.join(',') === centre) q.c[dimension] += delta;
            improved = consider(key, p, `coaxial centre ${centre}/${dimension}/${delta}`) || improved;
          }
      for (let i = 0; i < spec.parts[key].primitives.length; i++)
        for (const delta of [-0.3, 0.3, -0.15, 0.15, -0.075, 0.075]) {
          const p = structuredClone(spec.parts[key]),
            q = p.primitives[i];
          if (q.type === 'cyl') q.r = Math.max(0.2, q.r + delta);
          else if (q.round?.type === 'taper') {
            const shape = q.round;
            p.primitives[i] = {
              ...q,
              ...taperedCylinder(
                shape.start,
                shape.end,
                shape.radii.map((r) => Math.max(0.2, r + delta)) as [number, number],
              ),
            };
          } else continue;
          improved = consider(key, p, `circular radius ${i}/${delta}`) || improved;
        }
      if (!improved) break;
    }
    for (let i = 0; i < source.length; i++)
      if (source[i].taper) {
        const p = structuredClone(spec.parts[key]);
        if (p.primitives.length !== source.length) continue;
        p.primitives[i] = { ...source[i].taper!, name: source[i].q.name, mirror: false };
        consider(key, p, `source-coherent taper ${i}`);
      }
  } else {
    const features = detectRoundFeatures(solid);
    detections[key] = [...features];
    log(`${key}: ${features.length} cross-section-supported round features`);
    // Every operation starts at the best observed state and must survive unchanged formal evaluation.
    const selectedUpperCells: Set<string>[] = [];
    for (const [i, f] of features.entries()) {
      if (selectedUpperCells.some((set) => f.cells.filter((p) => set.has(p.join(','))).length > f.cells.length * 0.5)) {
        history({
          key,
          action: `feature ${i}`,
          accepted: false,
          rejected: 'already covered by accepted source upper pipe',
        });
        continue;
      }
      const upper = upperPipeContinuation(solid, f);
      if (upper) detections[key].push(upper);
      const tryUpper = () => {
        if (!upper) return false;
        const accepted = consider(
          key,
          patchUpperPipe(spec.parts[key], upper, `upper_${i}`),
          `source-supported upper pipe continuation ${i}; ${upper.evidence.slices} slices`,
          true,
        );
        if (accepted) selectedUpperCells.push(new Set(upper.cells.map((p) => p.join(','))));
        return accepted;
      };
      // Prefer a source-backed longer visible track over rounding only its short seed bend.
      const preferUpper = upper && upper.evidence.slices > f.evidence.slices * 1.1;
      if (preferUpper && tryUpper()) continue;
      const bend = fitFilletedPipe(solid, f.points, f.radius);
      if (bend) {
        (bendFeatures[key] ??= []).push({ sourceFeature: i, ...bend });
        if (
          consider(
            key,
            patchRoundFeature(spec.parts[key], f, `bend_${i}`, bend.primitives),
            `source-supported torus elbows ${i}; supports=${bend.joints.map((j) => j.support.toFixed(3)).join(',')}`,
            true,
          )
        ) {
          selectedUpperCells.push(new Set(f.cells.map((p) => p.join(','))));
          continue;
        }
      }
      if (!preferUpper && tryUpper()) continue;
      const original = structuredClone(spec.parts[key]);
      let best = structuredClone(original),
        bestMetrics = metrics[key],
        bestMasks = partMasks[key],
        bestAssembly = currentAssembly;
      for (const factor of [1, 0.9, 1.1, 1.2]) {
        const p = patchRoundFeature(original, f, `round_${i}`, sweptCylinders(f.points, f.radius * factor));
        if (consider(key, p, `source round feature ${i}; radius factor ${factor}`, true)) {
          best = structuredClone(spec.parts[key]);
          bestMetrics = metrics[key];
          bestMasks = partMasks[key];
          bestAssembly = currentAssembly;
        }
      }
      spec.parts[key] = best;
      metrics[key] = bestMetrics;
      partMasks[key] = bestMasks;
      currentAssembly = bestAssembly;
    }
    for (const [i, cap] of detectRoundCaps(solid).entries()) {
      const f: RoundFeature = {
        axis: cap.axis,
        points: [],
        radius: 0,
        cells: cap.cells,
        bounds: cap.bounds,
        primitives: [cap.primitive],
        evidence: {
          slices: cap.sectionAreas.length,
          exposedArc: 0,
          radialResidual: 0,
          sourceSupport: 0,
          ambiguousSmallCrossSection: false,
        },
      };
      consider(
        key,
        patchRoundFeature(spec.parts[key], f, `cap_${i}`),
        `source cap profile ${i}; base=${cap.base}; areas=${cap.sectionAreas.join(',')}`,
        true,
      );
    }
  }
  if (ex.parts[key].role !== 'barrel') {
    // Move coplanar boundaries together, including opposite inequalities at a shared BSP seam.
    // This keeps adjoining plates flush while exposing the same source-supported broad surfaces.
    for (let pass = 0; pass < 2 && Math.min(...currentAssembly.map((s) => s.iou)) < 0.97; pass++) {
      const groups = coplanarBoundaries(spec.parts[key]);
      let improved = false;
      for (const group of groups) {
        if (Math.min(...currentAssembly.map((s) => s.iou)) >= 0.97 || performance.now() - started > limit) break;
        for (const delta of [-0.25, 0.25, -0.5, 0.5]) {
          const p = shiftCoplanarBoundary(spec.parts[key], group, delta);
          improved = consider(key, p, `plate:${group.key}/${delta}; pass ${pass}`) || improved;
        }
      }
      if (!improved) break;
    }
  }
  phases['round-fit:' + key] = performance.now() - t;
  await save('checkpoint.json', spec);
  await save('features.json', detections);
}
await save('spec.json', spec);
await save('metrics.json', metrics);
await save('actions.json', actions);
await save('bend_features.json', bendFeatures);
await save('coalescing.json', { initial: coalescing, ridgeCandidates: ridgeCoalescing });
await save('silhouette.json', {
  ...Object.fromEntries(keys.map((k) => [k, metrics[k].silhouettes])),
  assembled: currentAssembly,
});
await save('cad_mesh.json', Object.fromEntries(Object.entries(spec.parts).map(([k, p]) => [k, buildPart(p)])));
await save(
  'proposals.json',
  Object.fromEntries(keys.map((k) => [k, { features: detections[k] ?? [], options: { budget: budget(k) } }])),
);
await writeFile(join(out, 'metrics.txt'), json(metrics) + 'assembled: ' + json(currentAssembly));
const argv = [join(here, 'render.py'), '--input', out, '--output', out, '--cell', flags.get('--cell') ?? '256'];
if (flags.has('--font')) argv.push('--font', flags.get('--font')!);
const renderStart = performance.now(),
  rendered = spawnSync(flags.get('--python') ?? 'python3', argv, { stdio: 'inherit' });
if (rendered.status !== 0) throw new Error(`Render failed: ${rendered.error ?? rendered.status}`);
phases.render = performance.now() - renderStart;
const minAssembly = Math.min(...currentAssembly.map((s) => s.iou)),
  barrels = keys.filter((k) => ex.parts[k].role === 'barrel'),
  barrelApplicable = barrels.length > 0,
  barrelTarget = barrels.every((k) => metrics[k].silhouettes.every((s) => s.iou >= 0.9));
await save('timing.json', {
  started: date,
  finished: new Date().toISOString(),
  totalMs: performance.now() - started,
  phasesMs: phases,
  runMode: mode,
  stopReason: performance.now() - started > limit ? 'time-budget' : 'search-complete',
  manualModelDecisions: [],
  note: 'Automatic wall clock including source detection, formal evaluation, and boards; prior baseline time excluded.',
});
await writeFile(
  join(out, 'HOURS.md'),
  `# Automatic round reconstruction\n\nStarted ${date}; automatic ${((performance.now() - started) / 60000).toFixed(3)} minutes. Baseline ${input} was an earlier automatic delivery and is not counted as newly reconstructed.\n\n${Object.entries(
    phases,
  )
    .map(([k, v]) => `${k}: ${(v / 60000).toFixed(3)} min`)
    .join(
      '\n\n',
    )}\n\nNo model coordinates or hand-picked parts. Shared development, diagnosis and actual author review must be logged separately. Human active minutes UNVERIFIED.\n`,
);
await writeFile(
  join(out, 'REPORT.md'),
  `# ${ex.id} 圆件离线复测\n\n整车硬门槛：**${minAssembly >= 0.97 ? 'PASS' : 'FAIL'}**（最低 ${minAssembly.toFixed(6)}）。炮管0.9目标：${!barrelApplicable ? '不适用' : barrelTarget ? '达到' : 'FAIL'}。最终视觉验收 **UNVERIFIED，需打开1×、2×、八向板**。\n\n| 部件 | 展开原语 | 最低八向IoU | p99 voxel→CAD / CAD→voxel |\n| --- | ---: | ---: | --- |\n${keys.map((k) => `| ${k} | ${metrics[k].metrics.primitives} | ${Math.min(...metrics[k].silhouettes.map((s) => s.iou)).toFixed(6)} | ${metrics[k].metrics.dev_vox_to_cad.p99}/${metrics[k].metrics.dev_cad_to_vox.p99} |`).join('\n')}\n\n[1×](board_当前_新.png)、[2×](board_当前_新_2x.png)、[八向](turntable_8.png)、[检测证据](features.json)、[候选日志](evaluations.jsonl)。涂装不变，未运行VM。原语包括曲面扫掠的全部凸片，没有隐藏细分预算或体素盒覆盖。\n`,
);
const finalHashes = await hashes();
log(
  `Complete: assembled min ${minAssembly.toFixed(6)}; barrel target ${barrelApplicable ? barrelTarget : 'not applicable'}; review pending`,
);
await writeDeliveryManifest(out, {
  kind: 'automatic-round-reconstruction',
  runMode: mode,
  model: ex.id,
  revision,
  sourceStatusAtStart: sourceStatus,
  sourceFilesChangedDuringRun: sourceNames.filter((n) => sourceHashes[n] !== finalHashes[n]),
  sourceSha256: ex.sourceSha256,
  baseline: input,
  gate: 'revised-2026-11-15',
  assembledHardPass: minAssembly >= 0.97,
  barrelApplicable,
  barrelTarget,
  metricPass: minAssembly >= 0.97,
  visualAcceptance: 'UNVERIFIED',
  runtime: 'UNVERIFIED; offline only',
});
