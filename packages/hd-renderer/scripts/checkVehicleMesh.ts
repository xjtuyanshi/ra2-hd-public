import { readFile } from 'node:fs/promises';
import type { VoxelModel } from '../../ra2-assets/src/manifest';
import { vehiclePartGeometry } from '../src/ra2/vehicleMeshGeometry';
import { buildVehicleSection } from '../src/ra2/vehicleMesh';
import { buildOutlinePartGeometry } from '../src/ra2/voxels';
import { compareVehicleSilhouettes, shapeMesh, type ShapeMesh } from '../src/ra2/vehicleMeshShape';

// Read only the project's extracted VXL/HVA JSON. Never write derived game data into the checkout.
if (!process.argv[2])
  throw new Error('Usage: pnpm exec tsx packages/hd-renderer/scripts/checkVehicleMesh.ts <cached-voxel.json> [...]');
let failed = false;
for (const file of process.argv.slice(2)) {
  const model = JSON.parse(await readFile(file, 'utf8')) as VoxelModel;
  const start = performance.now();
  const originals = model.parts.map((p) => buildOutlinePartGeometry(p.sections, { cornerRelax: 0, normalBlend: 0 }));
  const refined: ShapeMesh[] = [];
  const names: string[] = [];
  const originalIdentity = model.parts.map((part) => ({ role: part.role, names: part.sections.map((s) => s.name) }));
  const refinedIdentity: typeof originalIdentity = [];
  for (const part of model.parts) {
    const sectionMeshes = part.sections.map((section) => buildVehicleSection(section));
    const geometry = vehiclePartGeometry(sectionMeshes, part.sections);
    refinedIdentity.push({ role: part.role, names: geometry.userData.sectionNames });
    geometry.dispose();
    for (const [index, section] of part.sections.entries()) {
      const mesh = sectionMeshes[index]!;
      if (section.voxels.length && !mesh.indices.length) throw new Error(`Lost section ${part.role}:${section.name}`);
      if (part.role !== 'body')
        for (let i = 0; i < mesh.position.length; i += 3) mesh.position[i]! += (model.turretOffset ?? 0) / 256;
      refined.push(mesh);
      names.push(`${part.role}:${section.name}`);
    }
  }
  const partIdentityEqual = JSON.stringify(originalIdentity) === JSON.stringify(refinedIdentity);
  if (!partIdentityEqual) throw new Error('Vehicle part/section identity changed');
  const base = originals.map((g, part) => {
    const mesh = shapeMesh(g);
    if (model.parts[part]!.role !== 'body')
      for (let i = 0; i < mesh.position.length; i += 3) mesh.position[i]! += (model.turretOffset ?? 0) / 256;
    return mesh;
  });
  console.log(
    `${model.id}: parts=${model.parts.length}; sections=${names.join(', ')}; buildMs=${(performance.now() - start).toFixed(1)}`,
  );
  const rows = Array.from({ length: 16 }, (_, i) => compareVehicleSilhouettes(base, refined, i * 16));
  console.table(rows);
  const outward = Math.max(...rows.map((r) => r.outward));
  const inward = Math.max(...rows.map((r) => r.inward));
  const pass = Math.max(outward, inward) <= 0.5;
  failed ||= !pass;
  console.log(
    JSON.stringify({
      model: model.id,
      partCount: model.parts.length,
      refinedPartCount: refinedIdentity.length,
      partIdentityEqual,
      sectionNames: names,
      outward,
      inward,
      minIoU: Math.min(...rows.map((r) => r.iou)),
      minEngineIoU: Math.min(...rows.map((r) => r.engineIoU)),
      precision: 0.125,
      target: 0.5,
      pass,
    }),
  );
  originals.forEach((g) => g.dispose());
}
if (failed) process.exitCode = 1;
