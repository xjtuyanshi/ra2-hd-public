import { Fn, If, float, mix, positionWorld, sRGBTransferEOTF, sRGBTransferOETF, vec3 } from 'three/tsl';
import { ALIVE_CLOUD_TINT } from './livingWorld';
import type { LegacyLivingWorldNodes } from './livingWorldLegacy';
import type { ShaderNode } from './tsl';

// A separate layout name avoids colliding with the modern linear-light shader in the same material.
const shade = Fn(([source, factor]: ShaderNode[]) => {
  const result = source.toVar();
  If(factor.x.notEqual(1).or(factor.y.notEqual(1)).or(factor.z.notEqual(1)), () => {
    result.assign(sRGBTransferEOTF((sRGBTransferOETF(source.max(0)) as ShaderNode).mul(factor)));
  });
  return result;
}).setLayout({
  name: 'referenceCloudShade',
  type: 'vec3',
  inputs: [
    { name: 'source', type: 'vec3' },
    { name: 'factor', type: 'vec3' },
  ],
});

/** Preserve the reference analytic field and palette-domain attenuation beside a modern cached receiver. */
export function legacyCloudReference(source: ShaderNode, clouds: LegacyLivingWorldNodes): ShaderNode {
  const grey = clouds.shadowAt(positionWorld.xz);
  const cover = float(1).sub(grey).div(clouds.cloudDarkening).clamp();
  const factor = mix(vec3(1), vec3(...ALIVE_CLOUD_TINT), cover).mul(grey);
  return shade(source, factor);
}
