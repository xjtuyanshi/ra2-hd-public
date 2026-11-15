import { dot, float, mix, smoothstep, vec2 } from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Compact convex kernel suppresses one-texel grain while keeping broad texture and strong colour edges. */
export function calmFineGrain(
  source: ShaderNode,
  tap: (offset: ShaderNode) => ShaderNode,
  share: ShaderNode,
): ShaderNode {
  let result = source;
  for (const [x, y] of [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ]) {
    const neighbour = tap(vec2(x!, y!));
    const delta = neighbour.sub(source);
    const weight = float(1)
      .sub(smoothstep(0.0025, 0.025, dot(delta, delta)))
      .mul(0.09);
    result = result.add(delta.mul(weight));
  }
  return mix(source, result, share);
}
