import type { VoxelModel, VoxelSection } from '../../../ra2-assets/src/manifest';
import { sectionNormalsMode } from '../../../ra2-assets/src/voxelNormals';
import { lookupNormal, normalTableFor } from '../../../ra2-formats/src/vxl/vxlNormals';
import { attachVoxelPaint, mergeCleanVertexPaint, type CleanVertexPaint } from '../voxelCleanPaint';
import { refineCleanVertexPaint } from '../voxelCleanPaintRefine';
import { analyseVoxelPaint, type VoxelPaintInput, type VoxelPaintResult } from '../voxelPaint';
import { DEFAULT_VOXEL_SHAPE, type VoxelShapeMesh, type VoxelShapeOptions } from '../voxelShapeMesh';
import { relaxVoxelShapeMesh, smoothVoxelShapeNormals, type VoxelShapeRelaxResult } from '../voxelShapeRelax';
import { ra2VoxelPaintPolicy } from './voxelPaintPolicy';
import { buildVoxelShapeSection, mergeShapeMeshes, type VoxelShapePart } from './voxelShapeModel';

/**
 * RA2/YR "clean voxel" vehicle: the shape-locked mesh of every part with the de-dithered paint joined per vertex.
 * Same frame and turret pivot as buildVoxelShapeModel; paint uses the unit palette policy (house ramp 16-31 masked).
 */
export interface VoxelCleanPart extends VoxelShapePart {
  paint: CleanVertexPaint;
  /** Original buffers remain available for comparisons and backwards-compatible offline exports. */
  originalMesh: VoxelShapeMesh;
  originalPaint: CleanVertexPaint;
  relaxation?: VoxelShapeRelaxResult;
  /** Per-section paint analyses (regions and seam features), for reports. */
  analyses: VoxelPaintResult[];
}

export interface VoxelCleanModelOptions {
  /** Round-two geometry and paint refinement; false preserves the original mesh and paint path. */
  refine?: boolean;
}

function decode(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** The paint analysis input of one section, in the section's own cell-record order (the order the mesher indexes). */
export function voxelPaintInputFor(section: VoxelSection): VoxelPaintInput {
  const raw = decode(section.voxels);
  const n = Math.floor(raw.length / 5);
  const table = normalTableFor(sectionNormalsMode(section, raw));
  const input = {
    size: section.size,
    x: new Uint8Array(n),
    y: new Uint8Array(n),
    z: new Uint8Array(n),
    colour: new Uint8Array(n),
    normal: new Float32Array(n * 3),
  };
  for (let i = 0; i < n; i++) {
    input.x[i] = raw[i * 5]!;
    input.y[i] = raw[i * 5 + 1]!;
    input.z[i] = raw[i * 5 + 2]!;
    input.colour[i] = raw[i * 5 + 3]!;
    input.normal.set(lookupNormal(table, raw[i * 5 + 4]!), i * 3);
  }
  return input;
}

/** `palette`: the 256-colour unit palette as 768 8-bit sRGB bytes. */
export function buildVoxelCleanModel(
  model: VoxelModel,
  palette: ArrayLike<number>,
  options: Readonly<VoxelShapeOptions> = DEFAULT_VOXEL_SHAPE,
  cleanOptions: VoxelCleanModelOptions = {},
): VoxelCleanPart[] {
  const policy = ra2VoxelPaintPolicy(palette);
  const turretOffset = (model.turretOffset ?? 0) / 256;
  return model.parts.map((part) => {
    const meshes: VoxelShapeMesh[] = [];
    const paints: CleanVertexPaint[] = [];
    const analyses: VoxelPaintResult[] = [];
    for (const section of part.sections) {
      const mesh = buildVoxelShapeSection(section, options);
      const analysis = analyseVoxelPaint(voxelPaintInputFor(section), policy);
      meshes.push(mesh);
      analyses.push(analysis);
      paints.push(attachVoxelPaint(mesh, analysis));
    }
    const originalMesh = mergeShapeMeshes(meshes);
    const originalPaint = mergeCleanVertexPaint(paints);
    let mesh = originalMesh;
    let paint = originalPaint;
    let relaxation: VoxelShapeRelaxResult | undefined;
    if (cleanOptions.refine !== false) {
      const smallVertices = Uint8Array.from(originalPaint.mask, (mask) => (mask || part.role === 'barrel' ? 1 : 0));
      relaxation = relaxVoxelShapeMesh(originalMesh, { smallVertices });
      const originalNormals = smoothVoxelShapeNormals(originalMesh);
      const relaxedMesh = { ...originalMesh, positions: relaxation.positions, normals: relaxation.normals };
      const relaxedNormals = smoothVoxelShapeNormals(relaxedMesh);
      // Broad panels retain their original shading; stairs and small round parts follow the relaxed surface.
      for (let v = 0; v < relaxation.zone.length; v++)
        if (!relaxation.zone[v]) {
          for (let k = 0; k < 3; k++) relaxedNormals[v * 3 + k] = originalNormals[v * 3 + k]!;
        }
      mesh = { ...relaxedMesh, normals: relaxedNormals };
      paint = refineCleanVertexPaint(originalPaint, originalMesh);
    }
    return {
      role: part.role,
      turretOffset,
      mesh,
      paint,
      originalMesh,
      originalPaint,
      relaxation,
      analyses,
    };
  });
}
