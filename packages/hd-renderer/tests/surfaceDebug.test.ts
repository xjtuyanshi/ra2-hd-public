import { describe, expect, it } from 'vitest';
import { DataTexture, MeshBasicNodeMaterial, OrthographicCamera, QuadMesh, Scene, WebGPURenderer } from 'three/webgpu';
import { texture } from 'three/tsl';
import { surfaceCategoryDisplay } from '../src/surfaceDebug';
import { AtmosphereNodes } from '../src/atmosphere';

describe('opt-in surface diagnostics', () => {
  it('leaves ordinary worlds outside diagnostic category packing', () => {
    const nodes = new AtmosphereNodes(false);
    expect(nodes.surfaceClasses).toBe(false);
    expect(nodes.alive.modern).toBe(false);
    nodes.dispose();
  });
  it.each([false, true])('builds discrete category output without texture interpolation (WebGL=%s)', (forceWebGL) => {
    const image = new DataTexture(new Uint8Array(16), 2, 2);
    const material = new MeshBasicNodeMaterial();
    material.fragmentNode = surfaceCategoryDisplay(texture(image));
    const quad = new QuadMesh(material);
    const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement, forceWebGL });
    renderer.hasFeature = () => false;
    const builder = renderer.backend.createNodeBuilder(quad, renderer);
    builder.camera = new OrthographicCamera();
    builder.scene = new Scene();
    builder.build();
    expect(builder.fragmentShader).toContain('floor(');
    expect(builder.fragmentShader).not.toMatch(/\b(?:NaN|undefined)\b/);
    image.dispose();
    material.dispose();
  });
});
