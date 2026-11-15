import type { ShaderNode } from '../tsl';
import type { MapObject } from '../../../ra2-assets/src/manifest';

/**
 * Cache kind 'tree' means TerrainType, including rocks and posts. Only the stock tree families are safe to bend;
 * unknown/modded terrain stays rigid until its class can be supplied explicitly. No art/height heuristic.
 */
export function isLivingTree(object: Pick<MapObject, 'kind' | 'type'>): boolean {
  return object.kind === 'tree' && /^(?:TREE\d{2}|TIBTRE\d{2})$/i.test(object.type);
}

/** Only ordinary upright foliage may borrow the lawn grade. None also covers stock TIBTRE ore trees. */
export function treeToneEligible(
  wind: boolean,
  flat: boolean,
  exempt: boolean,
  none: boolean,
  converter: boolean,
  paint: boolean,
): boolean {
  return wind && !flat && !exempt && !none && !converter && !paint;
}

/** Instance-uniform shader twin of treeToneEligible; no per-pixel art classification or texture reads. */
export function treeToneEligibleNode(
  wind: ShaderNode,
  flat: ShaderNode,
  exempt: ShaderNode,
  none: ShaderNode,
  converter: ShaderNode,
  paint: ShaderNode,
): ShaderNode {
  return wind
    .greaterThan(0.5)
    .and(flat.lessThan(0.5))
    .and(exempt.lessThan(0.5))
    .and(none.lessThan(0.5))
    .and(converter.lessThan(0.5))
    .and(paint.lessThan(0.5));
}
