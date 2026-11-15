import { mix, positionWorld, vec3, float } from 'three/tsl';
import { linearCloudFactor } from './livingWorldNodes';
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

/** Objects retain paint brightness and borrow only their configured share of ground cloud attenuation. */
export function objectCloudFactor(
  nodes: AtmosphereNodes,
  coverage: ShaderNode,
  ground: ShaderNode = surfaceCloudFactor(nodes, coverage),
): ShaderNode {
  return nodes.cloudTone.strength
    .greaterThan(0)
    .select(mix(vec3(1), ground, nodes.cloudTone.objectShare).max(nodes.cloudTone.objectFloor), ground);
}

/** Reuse an already computed surface factor when an instance opts into more ground light. */
export function surfaceFollowingCloudFactor(
  object: ShaderNode,
  ground: ShaderNode,
  share: ShaderNode,
  enabled: ShaderNode,
): ShaderNode {
  return enabled.select(mix(object, ground, share), object);
}

/** Reuses one foot coverage and factor across all lighting terms of a material. Legacy sampling stays unchanged. */
export function objectCloudTone(nodes: AtmosphereNodes) {
  const coverage = nodes.alive
    .coverageAt(mix(positionWorld.xz, objectGroundAnchor.xz, nodes.cloudTone.strength.greaterThan(0).select(1, 0)))
    .toVar('aliveCloudCoverage');
  const shadow = objectCloudFactor(nodes, coverage).toVar('aliveCloudFactor');
  const linearShadow = linearCloudFactor(shadow).toVar();
  return {
    coverage,
    shadow,
    linearShadow,
    shade: (colour: ShaderNode) => nodes.alive.shadeAt(colour, linearShadow),
    grade: (colour: ShaderNode) =>
      cloudToneNode(
        colour,
        colour,
        nodes.alive.gradeCoverage(coverage),
        nodes.cloudTone,
        nodes.night,
        true,
        float(0),
        positionWorld.y.sub(objectGroundAnchor.y).max(0),
      ),
  };
}
