import { cos, sin, vec3 } from 'three/tsl';
import type { ShaderNode } from '../tsl';
import type { AmbientContact } from '../ambientContact';

/** Engine facing in radians, north = -z. The slight up component models a rounded body, not a camera-facing card. */
export function facingReceiverNormal(facing: number): [number, number, number] {
  const length = Math.hypot(1, 0.25);
  return [Math.sin(facing) / length, 0.25 / length, -Math.cos(facing) / length];
}
export function facingReceiverNormalNode(facing: ShaderNode): ShaderNode {
  return vec3(sin(facing), 0.25, cos(facing).negate()).normalize();
}
/** SHP bodies have no world-space mesh footprint; derive a bounded ground footprint from their actual art width. */
export function spriteContactFootprint(
  x: number,
  z: number,
  height: number,
  widthPixels: number,
  infantry: boolean,
  yaw: number,
): AmbientContact {
  const width = Math.max(infantry ? 0.16 : 0.3, Math.min(infantry ? 0.35 : 1.5, widthPixels / 60));
  return { x, z, height, width, depth: infantry ? width : width * 1.3, yaw };
}
