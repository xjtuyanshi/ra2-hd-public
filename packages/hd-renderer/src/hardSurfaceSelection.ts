/** Compare source-compatible automatic candidates without changing the formal metrics. */
import { evaluateHardSurface } from './hardSurfaceMetrics';
import {
  compareSilhouettes,
  rasterSilhouette,
  partMeshes,
  solidSurfaceMesh,
  type SilhouetteView,
} from './hardSurfaceSilhouette';
import type { PartSpec } from './hardSurfaceMesh';
import type { SolidGrid } from './hardSurfaceSolid';
import type { FitAcceptance } from './hardSurfaceRefine';

export function measureCandidate(
  solid: SolidGrid,
  part: PartSpec,
  views: SilhouetteView[],
  budget: number,
): FitAcceptance {
  const metrics = evaluateHardSurface(solid, part),
    meshes = partMeshes(part),
    source = solidSurfaceMesh(solid),
    silhouettes = views.map((v) =>
      compareSilhouettes(rasterSilhouette([source], v), rasterSilhouette(meshes, v), v.name),
    ),
    distances = [metrics.dev_vox_to_cad.p99, metrics.dev_cad_to_vox.p99],
    finite = [...distances, metrics.iou, ...silhouettes.map((s) => s.iou)].every(Number.isFinite),
    surface = finite
      ? Math.max(...distances) / 0.85
      : 100 + 10 * (1 - metrics.iou) + distances.filter(Number.isFinite).reduce((s, v) => s + v, 0),
    shape = (1 - Math.min(...silhouettes.map((s) => s.iou))) / (1 - 0.97);
  if (!views.length) throw new Error('Candidate measurement requires silhouette views');
  return {
    metrics,
    silhouettes,
    finite,
    pass: finite && surface <= 1 && shape <= 1 && metrics.primitives <= budget,
    loss:
      Math.max(surface, shape) +
      0.15 * (surface + shape) +
      0.05 * silhouettes.reduce((s, v) => s + 1 - v.iou, 0) +
      0.1 * (1 - metrics.iou) +
      0.00005 * metrics.primitives,
  };
}
/** Full pass, fewer failed independent gates, smaller worst normalized shortfall, then formal diagnostic loss. */
export function candidateRank(value: FitAcceptance): number[] {
  const surfaces = [value.metrics.dev_vox_to_cad.p99, value.metrics.dev_cad_to_vox.p99];
  return [
    value.pass ? 0 : 1,
    surfaces.filter((v) => v > 0.85).length + value.silhouettes.filter((v) => v.iou < 0.97).length,
    Math.max(...surfaces.map((v) => v / 0.85), ...value.silhouettes.map((v) => (1 - v.iou) / (1 - 0.97))),
    value.loss,
  ];
}
export function compareCandidates(a: FitAcceptance, b: FitAcceptance): number {
  const ar = candidateRank(a),
    br = candidateRank(b);
  for (let i = 0; i < ar.length; i++) if (ar[i] !== br[i]) return ar[i] < br[i] ? -1 : 1;
  return 0;
}
