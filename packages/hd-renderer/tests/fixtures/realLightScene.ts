import { DataTexture, Mesh, OrthographicCamera, Scene, WebGPURenderer, type RenderTarget } from 'three/webgpu';
import { AtmosphereNodes } from '../../src/atmosphere';
import { buildRa2Terrain } from '../../src/ra2/terrain';
import { SpriteLayer } from '../../src/ra2/sprites';
import { MapLightGrade } from '../../src/ra2/mapLighting';
import { buildVoxelGeometry } from '../../src/ra2/voxels';
import { buildStandInVoxelParts } from '../../src/ra2/voxelSmoothGeometry';
import { voxelPbrMaterialFor } from '../../src/ra2/voxelPbr';
import type { Ra2Cache } from '../../src/ra2/cache';
import { syntheticModel, syntheticSection } from './syntheticVoxels';

export function shaderBytes(mesh: Mesh, target?: RenderTarget): string {
  const renderer = new WebGPURenderer({ canvas: { width: 1, height: 1 } as HTMLCanvasElement });
  renderer.hasFeature = () => false;
  renderer.hasCompatibility = () => false;
  if (target) {
    renderer.setRenderTarget(target);
    renderer.setMRT((mesh.material as { mrtNode: never }).mrtNode);
  }
  const backend = renderer.backend as unknown as {
    capabilities: { getUniformBufferLimit(): number };
    createNodeBuilder(
      m: Mesh,
      r: WebGPURenderer,
    ): {
      camera: OrthographicCamera;
      scene: Scene;
      build(): void;
      vertexShader: string;
      fragmentShader: string;
      getAttributesArray(): {
        name: string;
        node?: { attribute?: { array: ArrayLike<number>; data?: { array: ArrayLike<number> } } };
      }[];
    };
  };
  backend.capabilities.getUniformBufferLimit = () => 65536;
  const b = backend.createNodeBuilder(mesh, renderer);
  b.camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  b.scene = new Scene();
  b.build();
  return b.vertexShader + b.fragmentShader;
}
export function receiverScene(nodes = new AtmosphereNodes()) {
  const tex = () => new DataTexture(new Uint8Array(256 * 4 * 4), 256, 4);
  const atlas = tex();
  const grade = new MapLightGrade(
    { width: 2, height: 2, z: new Uint8Array(4), exists: new Uint8Array(4).fill(1) },
    { ambient: 1, level: 0, ground: 0, red: 1, green: 1, blue: 1 },
  );
  const source = {
    nodes,
    manifest: {
      grid: { width: 2, height: 2 },
      theater: 'urban',
      sprites: {},
      spriteAtlas: { width: 256, height: 4 },
      terrainAtlas: { width: 256, height: 4 },
      tiles: [{ rect: [0, 0, 60, 30], anchor: [30, 0], x: 0, y: 0, rampType: 0, terrainType: 0, file: 'synthetic' }],
    },
    exists: new Uint8Array(4).fill(1),
    height: new Uint8Array(4),
    tile: new Uint16Array(4),
    spriteAtlas: atlas,
    terrainAtlas: atlas,
    paletteAtlas: atlas,
    waterMask: atlas,
    paletteRow: new Map(),
    mapLight: grade,
  } as unknown as Ra2Cache;
  const sprite = new SpriteLayer(source, [], 64, undefined, { color: tex(), remap: tex() }, tex(), { texture: tex() });
  const terrain = buildRa2Terrain(source);
  const palette = new Uint8Array(768);
  const [part] = buildVoxelGeometry(syntheticModel([syntheticSection(2)]), palette, nodes);
  const [standIn] = buildStandInVoxelParts(syntheticModel([syntheticSection(2)]));
  const meshes = [
    sprite.mesh,
    new Mesh(part!.geometry, part!.material),
    new Mesh(standIn!.geometry, voxelPbrMaterialFor(palette, nodes)),
  ];
  terrain.mesh.traverse((o) => {
    if (o instanceof Mesh)
      for (const mat of Array.isArray(o.material) ? o.material : [o.material]) meshes.push(new Mesh(o.geometry, mat));
  });
  return {
    meshes,
    source,
    dispose() {
      for (const m of meshes) {
        m.geometry.dispose();
        for (const mat of Array.isArray(m.material) ? m.material : [m.material]) mat.dispose();
      }
      grade.dispose();
      atlas.dispose();
      nodes.dispose();
    },
  };
}
