import { buildPart, type PartSpec, type Primitive } from './hardSurfaceMesh';
import {
  evaluateHardSurface,
  cadSurfaceSamples,
  voxelSurfaceSamples,
  nearestDistances,
  rayVisible,
  type HardSurfaceMetrics,
} from './hardSurfaceMetrics';
import {
  cellBounds,
  splitRegionCandidates,
  fitRegion,
  mirrorRegionPairs,
  reflectPrimitive,
  type RegionModel,
  type RegionProposal,
} from './hardSurfacePropose';
import {
  compareSilhouettes,
  partMeshes,
  rasterSilhouette,
  solidSurfaceMesh,
  type SilhouetteComparison,
  type SilhouetteView,
} from './hardSurfaceSilhouette';
import type { SolidGrid } from './hardSurfaceSolid';
import { proposeRecesses } from './hardSurfaceRecess';

export interface RefinementOptions {
  maxEvaluations: number;
  maxMilliseconds: number;
  neighborhood?: 'strict' | 'broad';
  p99?: number;
  iou?: number;
  onEvaluation?: (entry: RefinementEntry, part: PartSpec) => void;
}
export interface FitAcceptance {
  metrics: HardSurfaceMetrics;
  silhouettes: SilhouetteComparison[];
  pass: boolean;
  loss: number;
  finite: boolean;
}
export interface RefinementEntry {
  evaluation: number;
  elapsedMs: number;
  action: string;
  accepted: boolean;
  result?: FitAcceptance;
  error?: string;
}
export interface RefinementResult {
  part: PartSpec;
  result: FitAcceptance;
  history: RefinementEntry[];
  stop: 'pass' | 'primitive-budget' | 'evaluation-budget' | 'time-budget' | 'no-improving-region';
}
function shifted(q: Primitive, delta: number): Primitive {
  const out = structuredClone(q);
  // Axis-aligned partition faces stay fixed: shrinking them creates internal cracks and artificial surfaces.
  if (out.type === 'poly')
    for (const p of out.planes)
      if (p.slice(0, 3).filter((v) => v !== 0).length > 1) p[3] += delta * Math.hypot(p[0], p[1], p[2]);
  if (out.type === 'box') {
    out.min = out.min.map((v) => v - delta) as typeof out.min;
    out.max = out.max.map((v) => v + delta) as typeof out.max;
  }
  if (out.type === 'cyl') out.r = Math.max(0.2, out.r + delta);
  return out;
}
/** Formal bidirectional surface gates and every facing participate in selection; no exact-cover fallback. */
export function refineHardSurface(
  solid: SolidGrid,
  model: RegionModel,
  views: SilhouetteView[],
  options: RefinementOptions,
): RefinementResult {
  if (!views.length || options.maxEvaluations < 1 || options.maxMilliseconds <= 0)
    throw new Error('Positive refinement budgets and views are required');
  const started = performance.now(),
    history: RefinementEntry[] = [],
    p99 = options.p99 ?? 0.85,
    iou = options.iou ?? 0.97;
  const reference = views.map((v) => rasterSilhouette([solidSurfaceMesh(solid)], v));
  const evaluate = (part: PartSpec): FitAcceptance => {
    const meshes = partMeshes(part),
      silhouettes = views.map((v, i) => compareSilhouettes(reference[i], rasterSilhouette(meshes, v), v.name));
    const metrics = evaluateHardSurface(solid, part);
    const distances = [metrics.dev_vox_to_cad.p99, metrics.dev_cad_to_vox.p99];
    const finite = [...distances, metrics.iou, ...silhouettes.map((s) => s.iou)].every(Number.isFinite);
    // An unmatched reference sample is a failed gate, not an exception that destroys the initial checkpoint.
    // Keep Infinity in the metric and use a finite diagnostic penalty to search out of a poor initial hull.
    const surface = finite
      ? Math.max(...distances) / p99
      : 100 + 10 * (1 - metrics.iou) + distances.filter(Number.isFinite).reduce((s, v) => s + v, 0);
    const shape = (1 - Math.min(...silhouettes.map((s) => s.iou))) / (1 - iou);
    return {
      metrics,
      silhouettes,
      finite,
      pass: finite && surface <= 1 && shape <= 1 && metrics.primitives <= model.options.budget,
      loss:
        Math.max(surface, shape) +
        0.15 * (surface + shape) +
        0.05 * silhouettes.reduce((s, v) => s + 1 - v.iou, 0) +
        0.1 * (1 - metrics.iou) +
        0.00005 * metrics.primitives,
    };
  };
  let part = structuredClone(model.part),
    regions = structuredClone(model.regions),
    result = evaluate(part);
  const record = (action: string, accepted: boolean, value?: FitAcceptance, error?: string) => {
    const entry: RefinementEntry = {
      evaluation: history.length,
      elapsedMs: performance.now() - started,
      action,
      accepted,
      ...(value ? { result: value } : {}),
      ...(error ? { error } : {}),
    };
    history.push(entry);
    options.onEvaluation?.(entry, part);
  };
  record('initial proposal', true, result);
  const whole = cellBounds(regions.flatMap((r) => r.cells));
  const voxelSamples = voxelSurfaceSamples(solid),
    visibleSamples = voxelSamples.filter((s) => rayVisible(solid, s));
  let residualPart: PartSpec | undefined,
    surfaceWeights: number[] = [];
  const tried = new Set<string>();
  const splitCandidates = new Map<string, RegionProposal[][]>();
  const reserveParameter = (key: string, state: unknown): boolean => {
    // The same delta is a new candidate after a parameter change. Coaxial groups include every segment's radius.
    const scoped = `${key}@${JSON.stringify(state)}`;
    if (tried.has(scoped)) return false;
    tried.add(scoped);
    return true;
  };
  let stop: RefinementResult['stop'] = 'no-improving-region';
  while (!result.pass) {
    if (history.length >= options.maxEvaluations) {
      stop = 'evaluation-budget';
      break;
    }
    if (performance.now() - started >= options.maxMilliseconds) {
      stop = 'time-budget';
      break;
    }
    // Back-project the worst facing's actual silhouette residual to broad lattice-region screen bounds.
    const worst = result.silhouettes.reduce((a, b) => (a.iou < b.iou ? a : b)).name;
    const wi = views.findIndex((v) => v.name === worst),
      wv = views[wi],
      mask = rasterSilhouette(partMeshes(part), wv);
    if (residualPart !== part) {
      const built = buildPart(part),
        samples = cadSurfaceSamples(built.primitives, built.meshes);
      const cadDistances = nearestDistances(
        samples.map((s) => s.point),
        voxelSamples.map((s) => s.point),
      );
      const voxelDistances = nearestDistances(
        visibleSamples.map((s) => s.point),
        samples.map((s) => s.point),
      );
      surfaceWeights = regions.map(() => 0);
      samples.forEach((s, i) => {
        if (cadDistances[i] <= p99) return;
        const name = built.primitives[s.primitive].name?.split('#')[0];
        const owner = regions.findIndex((r) => r.primitive.name === name);
        if (owner >= 0) surfaceWeights[owner]++;
      });
      // Normalize the two sample populations so dense CAD triangles do not hide missing source regions.
      const voxelWeight = samples.length / Math.max(1, visibleSamples.length);
      visibleSamples.forEach((s, i) => {
        if (voxelDistances[i] <= p99) return;
        regions.forEach((r, j) => {
          if (s.point.every((v, a) => v >= r.bounds[0][a] - 0.5 && v <= r.bounds[1][a] + 0.5))
            surfaceWeights[j] += voxelWeight;
        });
      });
      residualPart = part;
    }
    const ranked = regions
      .map((r, i) => {
        const corners = [];
        for (const x of [r.bounds[0][0], r.bounds[1][0]])
          for (const y of [r.bounds[0][1], r.bounds[1][1]])
            for (const z of [r.bounds[0][2], r.bounds[1][2]]) corners.push(wv.project([x, y, z]));
        const lo = corners.reduce((p, q) => [Math.min(p[0], q[0]), Math.min(p[1], q[1])]),
          hi = corners.reduce((p, q) => [Math.max(p[0], q[0]), Math.max(p[1], q[1])]);
        let residual = 0;
        for (let y = Math.max(0, Math.floor(lo[1])); y <= Math.min(wv.height - 1, Math.ceil(hi[1])); y++)
          for (let x = Math.max(0, Math.floor(lo[0])); x <= Math.min(wv.width - 1, Math.ceil(hi[0])); x++)
            if (reference[wi][y * wv.width + x] !== mask[y * wv.width + x]) residual++;
        const surfaceFail = result.metrics.dev_vox_to_cad.p99 > p99 || result.metrics.dev_cad_to_vox.p99 > p99;
        return { r, i, residual: residual + (surfaceFail ? surfaceWeights[i] : 0) };
      })
      .sort((a, b) => b.residual - a.residual || b.r.error - a.r.error || b.r.cells.length - a.r.cells.length);
    const mirrorPairs =
      model.options.category === 'crown' ? mirrorRegionPairs(regions, model.symmetry) : new Map<number, number>();
    let candidate: PartSpec | undefined,
      action = '',
      nextRegions = regions;
    if (model.options.category !== 'tube' && (part.carve?.length ?? 0) < 8) {
      for (const recess of proposeRecesses(solid, buildPart(part).primitives)) {
        const key = `recess:${recess.id}`;
        if (tried.has(key)) continue;
        tried.add(key);
        candidate = { ...structuredClone(part), carve: [...(part.carve ?? []), recess.carve] };
        action = `${key}; ${recess.cells} residual cells, source overlap ${recess.sourceOverlap}, enclosed axes ${recess.enclosedAxes.join(',')}, source-face trim axes ${recess.trimAxes.join(',')}`;
        break;
      }
    }
    if (!candidate && model.options.category === 'tube') {
      const groups = new Map<string, number[]>();
      part.primitives.forEach((q, i) => {
        if (q.type !== 'cyl') return;
        const key = `${q.axis}:${q.c.join(',')}`;
        const g = groups.get(key);
        if (g) g.push(i);
        else groups.set(key, [i]);
      });
      outer: for (const [group, indices] of groups)
        for (const dimension of [0, 1])
          for (const delta of [-0.15, 0.15, -0.075, 0.075]) {
            const key = `axis:${group}:${dimension}:${delta}`;
            if (
              !reserveParameter(
                key,
                indices.map((i) => part.primitives[i]),
              )
            )
              continue;
            candidate = structuredClone(part);
            nextRegions = structuredClone(regions);
            for (const i of indices) {
              const q = candidate.primitives[i];
              if (q.type === 'cyl') q.c[dimension] += delta;
              nextRegions[i].primitive = q;
            }
            action = `${key}; shared by ${indices.length} coaxial segments`;
            break outer;
          }
    }
    if (!candidate)
      for (const { r, i } of ranked) {
        if (part.primitives.length >= model.options.budget) continue;
        if (model.options.category === 'tube' && model.options.primitivesPerTube && r.primitive.type === 'cyl') {
          const q = r.primitive;
          if (
            part.primitives.filter((p) => p.type === 'cyl' && p.axis === q.axis && p.c.every((v, a) => v === q.c[a]))
              .length >= model.options.primitivesPerTube
          )
            continue;
        }
        const state = r.id + (r.primitive.type === 'cyl' ? JSON.stringify(r.primitive.c) : '');
        if (!splitCandidates.has(state))
          splitCandidates.set(
            state,
            splitRegionCandidates(r, whole, model.options, options.neighborhood !== 'strict').slice(
              0,
              options.neighborhood === 'strict' ? 1 : 4,
            ),
          );
        const choices = splitCandidates.get(state)!;
        const alternative = choices.findIndex((_, j) => !tried.has(`split:${state}:${j}`));
        if (alternative < 0) continue;
        const key = `split:${state}:${alternative}`;
        tried.add(key);
        const children = choices[alternative];
        if (children) {
          const partner = mirrorPairs.get(i);
          if (partner !== undefined) {
            if (part.primitives.length + 2 > model.options.budget) continue;
            const axis = 'xyz'.indexOf(model.symmetry.axis);
            const reflected = children.map((child, j) => {
              const cells = child.cells.map(
                (p) => p.map((v, a) => (a === axis ? 2 * model.symmetry.at - v - 1 : v)) as typeof p,
              );
              const region = fitRegion(cells, regions[partner].id + (j ? 'b' : 'a'), whole, model.options);
              region.primitive = { ...reflectPrimitive(child.primitive, model.symmetry), name: region.primitive.name };
              return region;
            });
            nextRegions = regions.filter((_, j) => j !== i && j !== partner).concat(children, reflected);
          } else nextRegions = [...regions.slice(0, i), ...children, ...regions.slice(i + 1)];
          candidate = { ...part, primitives: nextRegions.map((r) => r.primitive) };
          action = `${key}; worst facing ${worst}`;
          break;
        }
      }
    if (!candidate && !model.options.tunePlanes)
      for (const { r, i } of ranked) {
        const key = `tune:${r.id}`;
        if (r.primitive.type !== 'poly' || !reserveParameter(key, r.primitive)) continue;
        const fitted = fitRegion(r.cells, r.id, whole, { ...model.options, tunePlanes: true });
        candidate = structuredClone(part);
        candidate.primitives[i] = fitted.primitive;
        nextRegions = structuredClone(regions);
        nextRegions[i] = fitted;
        action = `${key}; surface and facing ${worst}`;
        break;
      }
    if (!candidate)
      for (const { r, i } of ranked) {
        const q = part.primitives[i];
        if (options.neighborhood === 'strict') break;
        if (q.type !== 'poly') continue;
        for (let plane = 0; plane < q.planes.length; plane++) {
          const p = q.planes[plane];
          if (p.slice(0, 3).filter((v) => v !== 0).length < 2) continue;
          for (const delta of [-0.15, 0.15, -0.075, 0.075]) {
            const key = `plane:${r.id}:${plane}:${delta}`;
            if (!reserveParameter(key, q)) continue;
            candidate = structuredClone(part);
            const changed = candidate.primitives[i];
            if (changed.type === 'poly') changed.planes[plane][3] += delta * Math.hypot(p[0], p[1], p[2]);
            nextRegions = structuredClone(regions);
            nextRegions[i].primitive = changed;
            action = `${key}; independent snapped plate, facing ${worst}`;
            break;
          }
          if (candidate) break;
        }
        if (candidate) break;
      }
    if (!candidate)
      for (const { r, i } of ranked) {
        for (const delta of [-0.15, 0.15, -0.075, 0.075]) {
          const key = `offset:${r.id}:${delta}`;
          if (!reserveParameter(key, part.primitives[i])) continue;
          candidate = structuredClone(part);
          candidate.primitives[i] = shifted(candidate.primitives[i], delta);
          nextRegions = structuredClone(regions);
          nextRegions[i].primitive = candidate.primitives[i];
          action = `${key}; worst facing ${worst}`;
          break;
        }
        if (candidate) break;
      }
    if (!candidate) {
      stop = part.primitives.length >= model.options.budget ? 'primitive-budget' : 'no-improving-region';
      break;
    }
    try {
      // A local edit to an exactly supported bilateral region moves its reflected plate too.
      // Do this before formal evaluation; never silently symmetrize an accepted result afterward.
      if (candidate.primitives.length === part.primitives.length && mirrorPairs.size) {
        for (const [i, j] of mirrorPairs) {
          if (i >= j) continue;
          const changedI = JSON.stringify(candidate.primitives[i]) !== JSON.stringify(part.primitives[i]),
            changedJ = JSON.stringify(candidate.primitives[j]) !== JSON.stringify(part.primitives[j]);
          if (changedI === changedJ) continue;
          const from = changedI ? i : j,
            to = changedI ? j : i;
          candidate.primitives[to] = {
            ...reflectPrimitive(candidate.primitives[from], model.symmetry),
            name: part.primitives[to].name,
          };
          nextRegions = structuredClone(nextRegions);
          nextRegions[to].primitive = candidate.primitives[to];
          action += `; source-exact mirror ${regions[to].id}`;
        }
      }
      if (buildPart(candidate).primitives.length > model.options.budget) {
        record(action, false, undefined, 'Expanded primitive budget exceeded');
        continue;
      }
      const trial = evaluate(candidate);
      // Once a surface gate closes, a silhouette improvement cannot reopen it.
      const preserveSurface = result.metrics.dev_vox_to_cad.p99 <= p99 && result.metrics.dev_cad_to_vox.p99 <= p99;
      const preserveSilhouette = result.silhouettes.every((s) => s.iou >= iou);
      const accepted =
        trial.loss < result.loss - 1e-7 &&
        (!preserveSurface || (trial.metrics.dev_vox_to_cad.p99 <= p99 && trial.metrics.dev_cad_to_vox.p99 <= p99)) &&
        (!preserveSilhouette || trial.silhouettes.every((s) => s.iou >= iou));
      if (accepted) {
        part = candidate;
        regions = nextRegions;
        result = trial;
      }
      record(action, accepted, trial);
    } catch (error) {
      record(action, false, undefined, String(error));
    }
  }
  return { part, result, history, stop: result.pass ? 'pass' : stop };
}
