import { mix, positionWorld, vec3, float } from 'three/tsl';
import { cloudToneNode } from './cloudTone';
import { objectGroundAnchor } from './objectGroundAnchor';
import type { AtmosphereNodes } from './atmosphere';
import type { ShaderNode } from './tsl';

/** Surface policy controls the cloud field's own tint independently of its darkness. */
export function surfaceCloudFactor(nodes: AtmosphereNodes, coverage: ShaderNode): ShaderNode {
  const colour = nodes.alive.colourFromCoverage(coverage);
  const neutral = vec3(float(1).sub(coverage.mul(nodes.alive.cloudStrength)));
  return nodes.cloudTone.strength.greaterThan(0).select(mix(neutral, colour, nodes.cloudTone.cloudTintShare), colour);
}

/** Reuses one foot coverage and factor across all lighting terms of a material. Legacy sampling stays unchanged. */
export function objectCloudTone(nodes: AtmosphereNodes) {
  const coverage = nodes.alive
    .coverageAt(mix(positionWorld.xz, objectGroundAnchor.xz, nodes.cloudTone.strength.greaterThan(0).select(1, 0)))
    .toVar('aliveCloudCoverage');
  const shadow = surfaceCloudFactor(nodes, coverage).toVar('aliveCloudFactor');
  return {
    coverage,
    shadow,
    shade: (colour: ShaderNode) => nodes.alive.shadeAt(colour, shadow),
    grade: (colour: ShaderNode) =>
      cloudToneNode(colour, colour, nodes.alive.gradeCoverage(coverage), nodes.cloudTone, nodes.night, true),
  };
}
