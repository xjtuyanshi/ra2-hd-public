import { Mesh, MeshBasicNodeMaterial, MultiplyBlending, PlaneGeometry } from 'three/webgpu';
import { abs, float, max, smoothstep, uv } from 'three/tsl';
import type { AtmosphereNodes } from './atmosphere';
import { MIRROR_SKIP_LAYER } from './world';

const resources = new WeakMap<AtmosphereNodes, { geometry: PlaneGeometry; material: MeshBasicNodeMaterial }>();
/** Context-owned geometry/material; each caller owns only a footprint mesh and its placement. */
export function softFootprint(nodes: AtmosphereNodes): Mesh {
  let shared = resources.get(nodes);
  if (!shared) {
    const geometry = nodes.own(new PlaneGeometry(2, 2).rotateX(-Math.PI / 2));
    const material = nodes.own(
      new MeshBasicNodeMaterial({
        transparent: true,
        depthWrite: false,
        blending: MultiplyBlending,
        premultipliedAlpha: true,
      }),
    );
    const at = abs(uv().sub(0.5).mul(2));
    material.colorNode = nodes.cloudTone.shadeTint.mul(0.025);
    material.opacityNode = float(1)
      .sub(smoothstep(0.7, 1, max(at.x, at.y)))
      .mul(nodes.cloudTone.contact)
      .mul(nodes.cloudTone.strength)
      .mul(0.6)
      .min(0.75);
    material.alphaTest = 0.001;
    material.name = 'soft-footprint-contact';
    shared = { geometry, material };
    resources.set(nodes, shared);
  }
  const mesh = new Mesh(shared.geometry, shared.material);
  mesh.name = 'soft-footprint-contact';
  mesh.layers.set(MIRROR_SKIP_LAYER);
  mesh.renderOrder = 4.48;
  return mesh;
}
