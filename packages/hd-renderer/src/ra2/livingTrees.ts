import type { MapObject } from '../../../ra2-assets/src/manifest';

/**
 * Cache kind 'tree' means TerrainType, including rocks and posts. Only the stock tree families are safe to bend;
 * unknown/modded terrain stays rigid until its class can be supplied explicitly. No art/height heuristic.
 */
export function isLivingTree(object: Pick<MapObject, 'kind' | 'type'>): boolean {
  return object.kind === 'tree' && /^(?:TREE\d{2}|TIBTRE\d{2})$/i.test(object.type);
}
