import {
  ClampToEdgeWrapping,
  HalfFloatType,
  LinearFilter,
  NodeMaterial,
  QuadMesh,
  RenderTarget,
  RGBAFormat,
  type Texture,
  type WebGPURenderer,
} from 'three/webgpu';
import { Fn, If, dot, float, mix, step, texture, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import {
  CloudToneUniforms,
  cloudGreennessNode,
  cloudToneEndpointGainNode,
  cloudToneMappedNode,
  cloudToneMeanNode,
} from './cloudTone';
import type { ShaderNode } from './tsl';

export interface CloudToneMapOptions {
  texture: Texture;
  width: number;
  height: number;
  /** Block sizes in source texture texels, including the source scale. */
  blockWidth: number;
  contextWidth: number;
  uniforms: CloudToneUniforms;
  night: { value: number };
  /** Optional surface-space light applied to means, never to individual source pixels. */
  transformMean?(mean: ShaderNode, at: ShaderNode): ShaderNode;
  sourceVersion?(): number;
}

/** Cached low-frequency gain; its terrain owner updates before all scene/depth/mirror passes and disposes it. */
export class CloudToneMap {
  readonly target: RenderTarget;
  readonly material: NodeMaterial;
  private readonly quad: QuadMesh;
  private readonly night = uniform(0);
  private signature = '';
  private lastNight = -Infinity;
  private dirty = true;
  private disposed = false;
  renders = 0;

  constructor(private readonly options: CloudToneMapOptions) {
    const { width, height, blockWidth, contextWidth, texture: source, uniforms } = options;
    const makeTarget = () =>
      new RenderTarget(2 * Math.ceil(width / blockWidth), Math.ceil(height / blockWidth), {
        format: RGBAFormat,
        type: HalfFloatType,
        minFilter: LinearFilter,
        magFilter: LinearFilter,
        wrapS: ClampToEdgeWrapping,
        wrapT: ClampToEdgeWrapping,
        depthBuffer: false,
        generateMipmaps: false,
      });
    this.target = makeTarget();
    this.target.texture.name = 'CloudToneMap.endpoints';
    const texel = vec2(1 / width, 1 / height),
      bounds = vec4(0.5 / width, 0.5 / height, 1 - 0.5 / width, 1 - 0.5 / height);
    this.material = new NodeMaterial();
    this.material.depthTest = this.material.depthWrite = false;
    this.material.toneMapped = false;
    // Side-by-side endpoints share one binding; each half has exactly the former target's texel centres.
    const shade = step(0.5, uv().x);
    const at = vec2(uv().x.mul(2).sub(shade), uv().y);
    const transform = (colour: ShaderNode) => options.transformMean?.(colour, at) ?? colour;
    const mean = transform(cloudToneMeanNode(source, at, texel, blockWidth, bounds).toVar()).toVar();
    const context = transform(cloudToneMeanNode(source, at, texel, contextWidth, bounds).toVar()).toVar();
    const alpha = Fn(() => {
      const value = dot(mean, vec3(0.2126, 0.7152, 0.0722)).toVar();
      If(shade.greaterThan(0), () => {
        value.assign(0);
        If(uniforms.snow.lessThanEqual(0), () => {
          value.assign(cloudGreennessNode(mean, uniforms));
        });
      });
      return value;
    })();
    this.material.fragmentNode = vec4(cloudToneEndpointGainNode(mean, context, shade, uniforms, this.night), alpha);
    this.quad = new QuadMesh(this.material);
  }

  invalidate(): void {
    this.dirty = true;
  }

  /** Direct uniform edits are detected too; static views never redraw the targets every frame. */
  update(renderer: WebGPURenderer): boolean {
    if (this.disposed || this.options.uniforms.strength.value <= 0) return false;
    const signature = JSON.stringify([
      this.options.sourceVersion?.(),
      ...Object.values(this.options.uniforms).map((u) => (typeof u.value === 'number' ? u.value : u.value.toArray())),
    ]);
    const night = this.options.night.value;
    if (!this.dirty && signature === this.signature && Math.abs(night - this.lastNight) <= 0.005) return false;
    this.night.value = night;
    const target = renderer.getRenderTarget(),
      face = renderer.getActiveCubeFace(),
      level = renderer.getActiveMipmapLevel(),
      mrt = renderer.getMRT();
    try {
      renderer.setMRT(null);
      renderer.setRenderTarget(this.target);
      this.quad.render(renderer);
      this.signature = signature;
      this.lastNight = night;
      this.dirty = false;
      this.renders++;
    } finally {
      renderer.setRenderTarget(target, face, level);
      renderer.setMRT(mrt);
    }
    return true;
  }

  grade(
    source: ShaderNode,
    at: ShaderNode,
    shadeAmount: ShaderNode,
    night: ShaderNode,
    wet: ShaderNode = float(0),
  ): ShaderNode {
    // Clamp within each half before remapping so filtering never leaks across the endpoint seam.
    const halfWidth = this.target.width / 2;
    const point = at.clamp(
      vec2(0.5 / halfWidth, 0.5 / this.target.height),
      vec2(1 - 0.5 / halfWidth, 1 - 0.5 / this.target.height),
    );
    const lit = texture(this.target.texture, point.mul(vec2(0.5, 1))),
      shade = texture(this.target.texture, point.mul(vec2(0.5, 1)).add(vec2(0.5, 0)));
    return cloudToneMappedNode(
      source,
      mix(lit.rgb, shade.rgb, shadeAmount),
      lit.a,
      shade.a,
      shadeAmount,
      this.options.uniforms,
      night,
      wet,
    );
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.target.dispose();
    this.material.dispose();
  }
}
