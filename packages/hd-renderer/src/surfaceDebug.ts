import { floor, mod, vec4 } from 'three/tsl';
import type { ShaderNode } from './tsl';

/** Diagnostic display of caller-supplied category bits. No colour classification or extra target. */
export function surfaceCategoryDisplay(codes: ShaderNode): ShaderNode {
  const category: ShaderNode = mod(floor(codes.z.mul(256).add(0.5)), 8);
  return vec4(category.div(8), category.div(8), category.div(8), 1);
}
