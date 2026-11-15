import {
  ClampToEdgeWrapping,
  HalfFloatType,
  LinearFilter,
  NodeMaterial,
  QuadMesh,
  RedFormat,
  RenderTarget,
  type WebGPURenderer,
} from 'three/webgpu';
import { uv, vec2, vec4 } from 'three/tsl';
import type { LivingWorldNodes } from './livingWorldNodes';

/** A world-space coverage cache, independent of camera, surface and number of scene passes. */
export class CloudCoverageMap {
  readonly target: RenderTarget;
  readonly material = new NodeMaterial();
  private readonly quad: QuadMesh;
  private signature = '';
  private disposed = false;

  constructor(
    width: number,
    height: number,
    private readonly nodes: LivingWorldNodes,
  ) {
    this.target = new RenderTarget(Math.max(1, Math.ceil(width * 4)), Math.max(1, Math.ceil(height * 4)), {
      format: RedFormat,
      type: HalfFloatType,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      wrapS: ClampToEdgeWrapping,
      wrapT: ClampToEdgeWrapping,
      depthBuffer: false,
      generateMipmaps: false,
    });
    // RedFormat + HalfFloatType: three picks r16float (WebGPU) / R16F (WebGL2); a WebGL internalFormat string
    // here would be passed verbatim to WebGPU and rejected.
    this.target.texture.name = 'CloudCoverageMap';
    this.material.depthTest = this.material.depthWrite = this.material.toneMapped = false;
    // Render-target TextureNode sampling and QuadMesh UV share the same top-origin convention.
    this.material.fragmentNode = vec4(nodes.analyticCoverageAt(uv().mul(vec2(width, height))), 0, 0, 1);
    this.quad = new QuadMesh(this.material);
    nodes.coverageTexture.value = this.target.texture;
    nodes.coverageBounds.value.set(0, 0, width, height);
    nodes.coverageCached.value = 1;
  }

  /** Called before depth/reflection/main draws. Disabled clouds submit no cache pass. */
  update(renderer: WebGPURenderer): boolean {
    if (this.disposed || this.nodes.coverageEnabled.value <= 0) return false;
    const signature = JSON.stringify([
      this.nodes.clock.value,
      ...this.nodes.windDirection.value.toArray(),
      ...this.nodes.cloudLattice.value.toArray(),
      ...this.nodes.cloudForm.value.toArray(),
    ]);
    if (signature === this.signature) return false;
    const previous = renderer.getRenderTarget(),
      face = renderer.getActiveCubeFace(),
      level = renderer.getActiveMipmapLevel(),
      mrt = renderer.getMRT();
    try {
      renderer.setMRT(null);
      renderer.setRenderTarget(this.target);
      this.quad.render(renderer);
      this.signature = signature;
    } finally {
      renderer.setRenderTarget(previous, face, level);
      renderer.setMRT(mrt);
    }
    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.nodes.coverageTexture.value === this.target.texture) this.nodes.coverageCached.value = 0;
    this.target.dispose();
    this.material.dispose();
  }
}
