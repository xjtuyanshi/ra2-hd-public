import type { LibraryAnimLayer } from '../../../ra2-assets/src/library';
import { GROUND_DEPTH } from './cardDepth';

/** Factory delivery layers and construction-yard production layers identify the production-art contract. */
export function isProductionBuilding(layers: readonly Pick<LibraryAnimLayer, 'key'>[]): boolean {
  return layers.some((layer) => layer.key === 'DeployingAnim' || layer.key === 'ProductionAnim');
}

/** Native negative ZAdjust pulls a layer forward. Doors retain their explicit native ground plane. */
export function productionLayerPush(layer: Pick<LibraryAnimLayer, 'key' | 'zAdjust'>, strength: number): number {
  if (layer.key === 'DoorAnim' || layer.key === 'UnderDoorAnim' || layer.key === 'UnderRoofDoorAnim') return 0;
  const weight = Math.max(0, Math.min(1, Number.isFinite(strength) ? strength : 0));
  return weight && layer.zAdjust ? -layer.zAdjust * GROUND_DEPTH * weight : 0;
}
