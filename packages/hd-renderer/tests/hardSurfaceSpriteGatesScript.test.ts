import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, rmSync, symlinkSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import { encodePng, type PngChannels } from '../../ra2-assets/src/node/png';
const script = resolve('packages/hd-renderer/scripts/hardSurfaceSpriteGates.ts');
function invoke(...args: string[]) {
  return execFileSync(process.execPath, ['--import', 'tsx', script, ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}
describe('sprite gate CLI, synthetic PNG buffers only', () => {
  it('decodes float depth, measures native coverage/colour, and guards every output path', () => {
    const dir = mkdtempSync(join(tmpdir(), 'sprite-gates-'));
    try {
      const w = 8,
        h = 8,
        ss = 4;
      const png = (name: string, width: number, height: number, channels: PngChannels, data: Uint8Array) =>
        writeFileSync(join(dir, name + '.png'), encodePng({ width, height, channels, data }));
      const alpha = (x: number, y: number) => x >= 1 && x < 7 && y >= 1 && y < 7;
      const original = Uint8Array.from({ length: w * h * 4 }, (_, i) =>
        i % 4 === 3 ? +alpha(Math.floor(i / 4) % w, Math.floor(i / 4 / w)) * 255 : 100,
      );
      png('original', w, h, 4, original);
      png(
        'indices',
        w,
        h,
        1,
        Uint8Array.from({ length: w * h }, (_, i) => (alpha(i % w, Math.floor(i / w)) ? 17 : 0)),
      );
      png(
        'coverage',
        w * ss,
        h * ss,
        1,
        Uint8Array.from({ length: w * h * ss * ss }, (_, i) =>
          alpha(Math.floor((i % (w * ss)) / ss), Math.floor(i / (w * ss * ss))) ? 255 : 0,
        ),
      );
      png(
        'normal',
        w * ss,
        h * ss,
        3,
        Uint8Array.from({ length: w * h * ss * ss * 3 }, (_, i) => (i % 3 === 2 ? 255 : 128)),
      );
      for (const [name, value] of [
        ['nx', 0],
        ['ny', 0],
        ['nz', 1],
      ] as const) {
        const component = new Float32Array(w * h * ss * ss).fill(value);
        png(name, w * ss, h * ss, 4, new Uint8Array(component.buffer));
      }
      const depths = new Float32Array(w * h * ss * ss).fill(2);
      png('depth', w * ss, h * ss, 4, new Uint8Array(depths.buffer));
      png(
        'faces',
        w,
        h,
        3,
        Uint8Array.from({ length: w * h * 3 }, (_, i) => (i % 3 === 0 ? 1 : 0)),
      );
      png('colour', w, h, 3, new Uint8Array(w * h * 3).fill(100));
      const manifest = {
        original: 'original.png',
        indices: 'indices.png',
        coverage: 'coverage.png',
        normals: 'normal.png',
        normalX: 'nx.png',
        normalY: 'ny.png',
        normalZ: 'nz.png',
        depth: 'depth.png',
        faceIds: 'faces.png',
        colour: 'colour.png',
        supersampling: ss,
        camera: { right: [1, 0, 0], up: [0, 1, 0], towards: [0, 0, 1], origin: [0, 0], pixelsPerWorld: 10 },
        normalAngleDegrees: 20,
        depthJumpWorld: 0.1,
        houseIndexRange: [16, 31],
        faceLabels: { '1': { part: 'synthetic', deltaE76: 999 } },
      };
      const input = join(dir, 'buffers.json'),
        output = join(dir, 'gates.json');
      writeFileSync(input, JSON.stringify(manifest));
      invoke(input, output);
      const report = JSON.parse(readFileSync(output, 'utf8'));
      expect(report.G1).toMatchObject({ iou: 1, pass: true, intersection: 36 });
      expect(report.G2).toMatchObject({ precision: null, total: 0, pass: false });
      expect(report.G4).toMatchObject({ originalHousePixels: 36, coveredHousePixels: 36 });
      expect(report.G4.house).toMatchObject({ faces: 1, mean: 0 });
      expect(report.G4.geometricFaceSummary).toMatchObject({ faces: 1, mean: 0 });
      expect(report.nullControl.controls).toHaveLength(8);
      expect(report.nullControl.pass).toBe(false);
      expect(report.definitions.normalPrecision).toBe('float32-LE components');
      writeFileSync(
        input,
        JSON.stringify({
          ...manifest,
          frozenSource: {
            edgeMaskSha256: report.sourceStructure.edgeMaskSha256,
            optionsSha256: report.sourceStructure.optionsSha256,
            normalSha256: 'wrong',
          },
        }),
      );
      expect(() => invoke(input, output)).toThrow('Frozen source structure changed');

      writeFileSync(
        input,
        JSON.stringify({ ...manifest, frozenSource: { edgeMaskSha256: 'wrong', optionsSha256: 'wrong' } }),
      );
      expect(() => invoke(input, output)).toThrow('Frozen source structure changed');
      writeFileSync(input, JSON.stringify(manifest));
      expect(existsSync(join(dir, 'rendered_interior.png'))).toBe(true);
      expect(() => invoke(input, input)).toThrow('overwrite an input');
      const source = readFileSync(script, 'utf8');
      symlinkSync(script, join(dir, 'bad.json'));
      expect(() => invoke(input, join(dir, 'bad.json'))).toThrow('outside the repository');
      expect(readFileSync(script, 'utf8')).toBe(source);
      rmSync(join(dir, 'original_edges.png'));
      symlinkSync(join(dir, 'original.png'), join(dir, 'original_edges.png'));
      expect(() => invoke(input, output)).toThrow('overwrite an input');
      writeFileSync(input, JSON.stringify({ ...manifest, supersampling: 2 }));
      expect(() => invoke(input, output)).toThrow('at least 4x4');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
