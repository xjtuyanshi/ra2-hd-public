import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { HvaFile, Palette, Vfs, VxlFile, lookupNormal, normalTableFor } from '../../ra2-formats/src/index';
import { analyseVoxelPaint, regionLocalVariance, type VoxelPaintInput, type VoxelPaintResult } from '../src/voxelPaint';
import { RA2_HOUSE_RAMP, ra2VoxelPaintPolicy } from '../src/ra2/voxelPaintPolicy';

/**
 * Debug views and evidence for the voxel paint analysis on supplied RA2/YR models.
 * Reads the player's own game directory read-only and writes JSON/PNG to an explicit output directory outside the
 * checkout; nothing game-derived belongs in Git.
 *
 *   pnpm exec tsx packages/hd-renderer/scripts/voxelPaintDebug.ts <game-dir> <out-dir> [ID=body,turret,barrel ...]
 */
const [gameDir, outDir, ...specs] = process.argv.slice(2);
if (!gameDir || !outDir) throw new Error('Usage: voxelPaintDebug.ts <game-dir> <out-dir> [ID=body,turret,barrel ...]');
const models = (
  specs.length
    ? specs
    : ['HTNK=htnk,htnktur,htnkbarl', 'GTNK=gtnk,gtnktur,gtnkbarl', 'MTNK=mtnk,mtnktur,mtnkbarl', 'SREF=sref,sreftur']
).map((s) => {
  const [id, files] = s.split('=');
  return { id, files: files.split(',') };
});
mkdirSync(outDir, { recursive: true });

const vfs = new Vfs();
for (const n of ['ra2.mix', 'ra2md.mix', 'expandmd01.mix'])
  vfs.mountData(new Uint8Array(readFileSync(join(gameDir, n))));
for (const n of ['local.mix', 'localmd.mix', 'cache.mix', 'cachemd.mix']) {
  const owner = (vfs as unknown as { mounts: { read(n: string): unknown }[] }).mounts.find((m) => m.read(n));
  if (owner) vfs.mountNested(owner as never, n);
}
const need = (name: string) => {
  const f = vfs.openFile(name);
  if (!f) throw new Error(`missing ${name}`);
  return f;
};
const palette = Palette.parse(need('unittem.pal')).rgb;
const policy = ra2VoxelPaintPolicy(palette);
const art = Buffer.from(need('artmd.ini')).toString('latin1');
const turretOffsetOf = (id: string) => {
  const start = art.search(new RegExp(`^\\[${id}\\]`, 'm'));
  if (start < 0) return 0;
  const body = art.slice(start + 1, art.indexOf('\n[', start + 1));
  const m = /^TurretOffset=(-?\d+)/m.exec(body);
  return m ? Number(m[1]) : 0;
};

// ---------- tiny PNG writer ----------
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(b: Buffer): number {
  let c = 0xffffffff;
  for (const v of b) c = CRC[(c ^ v) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function png(w: number, h: number, rgb: Uint8Array): Buffer {
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    Buffer.from(rgb.buffer, rgb.byteOffset + y * w * 3, w * 3).copy(raw, y * (w * 3 + 1) + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// ---------- per-part analysis ----------
interface PartData {
  file: string;
  input: VoxelPaintInput;
  result: VoxelPaintResult;
  world: Float64Array; // 3 per voxel
  worldNormal: Float64Array; // original stored normal, world space
  smoothNormal: Float64Array; // smoothed, world space
  voxelSize: number;
}

function smoothNormals(input: VoxelPaintInput, surface: Uint8Array, radius: number): Float64Array {
  const n = input.colour.length;
  const [sx, sy, sz] = input.size;
  const cell = new Int32Array(sx * sy * sz).fill(-1);
  for (let i = 0; i < n; i++) cell[(input.z[i] * sy + input.y[i]) * sx + input.x[i]] = i;
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    let a = 0;
    let b = 0;
    let c = 0;
    for (let dz = -radius; dz <= radius; dz++)
      for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++) {
          const X = input.x[i] + dx;
          const Y = input.y[i] + dy;
          const Z = input.z[i] + dz;
          if (X < 0 || Y < 0 || Z < 0 || X >= sx || Y >= sy || Z >= sz) continue;
          const j = cell[(Z * sy + Y) * sx + X];
          if (j < 0 || !surface[j]) continue;
          const w = 1 / (1 + dx * dx + dy * dy + dz * dz);
          const nx = input.normal[j * 3];
          const ny = input.normal[j * 3 + 1];
          const nz = input.normal[j * 3 + 2];
          const l = Math.hypot(nx, ny, nz) || 1;
          a += (w * nx) / l;
          b += (w * ny) / l;
          c += (w * nz) / l;
        }
    const l = Math.hypot(a, b, c) || 1;
    out[i * 3] = a / l;
    out[i * 3 + 1] = b / l;
    out[i * 3 + 2] = c / l;
  }
  return out;
}

function loadPart(file: string, offsetVoxels: number): PartData[] {
  const vxl = VxlFile.parse(need(`${file}.vxl`));
  const hva = HvaFile.parse(need(`${file}.hva`));
  return vxl.sections.map((s, si) => {
    const table = normalTableFor(s.normalsMode);
    const n = s.voxels.length;
    const input = {
      size: [s.cx, s.cy, s.cz] as [number, number, number],
      x: new Uint8Array(n),
      y: new Uint8Array(n),
      z: new Uint8Array(n),
      colour: new Uint8Array(n),
      normal: new Float32Array(n * 3),
    };
    s.voxels.forEach((v, i) => {
      input.x[i] = v.x;
      input.y[i] = v.y;
      input.z[i] = v.z;
      input.colour[i] = v.color;
      input.normal.set(lookupNormal(table, v.normal), i * 3);
    });
    const result = analyseVoxelPaint(input, policy);
    const m = hva.matrix(si, 0);
    const size = [0, 1, 2].map((k) => (s.maxBounds[k] - s.minBounds[k]) / [s.cx, s.cy, s.cz][k]);
    const world = new Float64Array(n * 3);
    const worldNormal = new Float64Array(n * 3);
    const smooth = smoothNormals(input, result.surface, 2);
    const smoothNormal = new Float64Array(n * 3);
    const rot = (src: ArrayLike<number>, dst: Float64Array, i: number) => {
      for (let r = 0; r < 3; r++)
        dst[i * 3 + r] = m[r * 4] * src[i * 3] + m[r * 4 + 1] * src[i * 3 + 1] + m[r * 4 + 2] * src[i * 3 + 2];
    };
    for (let i = 0; i < n; i++) {
      const p = [
        s.minBounds[0] + (input.x[i] + 0.5) * size[0],
        s.minBounds[1] + (input.y[i] + 0.5) * size[1],
        s.minBounds[2] + (input.z[i] + 0.5) * size[2],
      ];
      for (let r = 0; r < 3; r++)
        world[i * 3 + r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3] * s.scale;
      world[i * 3] += offsetVoxels;
      rot(input.normal, worldNormal, i);
      rot(smooth, smoothNormal, i);
    }
    return { file, input, result, world, worldNormal, smoothNormal, voxelSize: Math.max(...size) };
  });
}

// ---------- splat renderer ----------
const AZ = (-135 * Math.PI) / 180; // camera from the front-left quarter, like a unit facing south-west
const EL = (30 * Math.PI) / 180;
const VIEW = [Math.cos(EL) * Math.cos(AZ), Math.cos(EL) * Math.sin(AZ), Math.sin(EL)];
const RIGHT = [-Math.sin(AZ), Math.cos(AZ), 0];
const UP = [
  VIEW[1] * RIGHT[2] - VIEW[2] * RIGHT[1],
  VIEW[2] * RIGHT[0] - VIEW[0] * RIGHT[2],
  VIEW[0] * RIGHT[1] - VIEW[1] * RIGHT[0],
];
if (UP[2] < 0) for (let k = 0; k < 3; k++) UP[k] = -UP[k];
const LIGHT = (() => {
  const l = [
    VIEW[0] * 0.6 + UP[0] * 0.9 - RIGHT[0] * 0.5,
    VIEW[1] * 0.6 + UP[1] * 0.9 - RIGHT[1] * 0.5,
    VIEW[2] * 0.6 + UP[2] * 0.9 - RIGHT[2] * 0.5,
  ];
  const len = Math.hypot(l[0], l[1], l[2]);
  return l.map((v) => v / len);
})();
const PX = 7; // pixels per voxel unit

type Shader = (part: PartData, i: number) => [number, number, number];

function render(parts: PartData[], shade: Shader, id?: (part: PartData, i: number) => number) {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const p of parts)
    for (let i = 0; i < p.input.colour.length; i++) {
      const w = [p.world[i * 3], p.world[i * 3 + 1], p.world[i * 3 + 2]];
      const u = w[0] * RIGHT[0] + w[1] * RIGHT[1] + w[2] * RIGHT[2];
      const v = w[0] * UP[0] + w[1] * UP[1] + w[2] * UP[2];
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  const pad = 2;
  const W = Math.ceil((maxU - minU + 2 * pad) * PX);
  const H = Math.ceil((maxV - minV + 2 * pad) * PX);
  const img = new Uint8Array(W * H * 3).fill(232);
  const depth = new Float64Array(W * H).fill(-Infinity);
  const ids = new Int32Array(W * H).fill(-1);
  for (const p of parts) {
    const r = Math.ceil(p.voxelSize * PX * 0.75);
    for (let i = 0; i < p.input.colour.length; i++) {
      if (!p.result.surface[i]) continue;
      const w = [p.world[i * 3], p.world[i * 3 + 1], p.world[i * 3 + 2]];
      const u = (w[0] * RIGHT[0] + w[1] * RIGHT[1] + w[2] * RIGHT[2] - minU + pad) * PX;
      const v = (maxV - (w[0] * UP[0] + w[1] * UP[1] + w[2] * UP[2]) + pad) * PX;
      const d = w[0] * VIEW[0] + w[1] * VIEW[1] + w[2] * VIEW[2];
      const c = shade(p, i);
      const k = id ? id(p, i) : 0;
      for (let y = Math.round(v - r); y < Math.round(v + r); y++)
        for (let x = Math.round(u - r); x < Math.round(u + r); x++) {
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const q = y * W + x;
          if (d <= depth[q]) continue;
          depth[q] = d;
          ids[q] = k;
          img[q * 3] = c[0];
          img[q * 3 + 1] = c[1];
          img[q * 3 + 2] = c[2];
        }
    }
  }
  return { W, H, img, ids, depth };
}

const lambert = (nx: number, ny: number, nz: number) =>
  0.38 + 0.8 * Math.max(0, nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]);
const clamp8 = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
const hashColour = (k: number): [number, number, number] => {
  let h = (k * 2654435761) >>> 0;
  const c: number[] = [];
  for (let i = 0; i < 3; i++) {
    h = (h ^ (h >>> 13)) * 1274126177;
    c.push(60 + ((h >>> 8) % 180));
  }
  return c as [number, number, number];
};
const MATERIAL_RGB: Record<string, [number, number, number]> = {
  painted: [90, 160, 90],
  'bare-metal': [80, 120, 200],
  rubber: [60, 50, 50],
  glass: [120, 220, 240],
  light: [250, 220, 60],
};

/** Mean absolute 4-neighbour Laplacian of luminance over pixels whose 3x3 block is one region id (interior). */
function speckleEnergy(a: { W: number; H: number; img: Uint8Array }, ids: Int32Array): number {
  const { W, H, img } = a;
  const lum = (q: number) => 0.2126 * img[q * 3] + 0.7152 * img[q * 3 + 1] + 0.0722 * img[q * 3 + 2];
  let sum = 0;
  let cnt = 0;
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const q = y * W + x;
      const k = ids[q];
      if (k < 0) continue;
      let inside = true;
      for (let dy = -1; dy <= 1 && inside; dy++)
        for (let dx = -1; dx <= 1; dx++) if (ids[q + dy * W + dx] !== k) inside = false;
      if (!inside) continue;
      sum += Math.abs(4 * lum(q) - lum(q - 1) - lum(q + 1) - lum(q - W) - lum(q + W));
      cnt++;
    }
  return cnt ? sum / cnt : 0;
}

// ---------- run ----------
const summary: unknown[] = [];
for (const { id, files } of models) {
  const offset = (turretOffsetOf(id) / 256) * (60 / Math.SQRT2);
  const parts = files.flatMap((f, k) => loadPart(f, k === 0 ? 0 : offset));
  const partIndex = new Map(parts.map((p, k) => [p, k]));
  // Region id unique across parts; seam voxels get a distinct id so the speckle metric skips them.
  const regionKey = (p: PartData, i: number) =>
    p.result.seamFeature[i] >= 0 ? -2 - partIndex.get(p)! : partIndex.get(p)! * 100000 + p.result.region[i];
  const pal = (c: number): [number, number, number] => [palette[c * 3], palette[c * 3 + 1], palette[c * 3 + 2]];
  const views: Record<string, Shader> = {
    original_lit: (p, i) => {
      const l = lambert(p.worldNormal[i * 3], p.worldNormal[i * 3 + 1], p.worldNormal[i * 3 + 2]);
      return pal(p.input.colour[i]).map((v) => clamp8(v * l)) as [number, number, number];
    },
    // Same smoothed lighting as clean_lit but the original per-voxel palette paint: isolates what the paint change does.
    original_paint_smooth_lit: (p, i) => {
      const l = lambert(p.smoothNormal[i * 3], p.smoothNormal[i * 3 + 1], p.smoothNormal[i * 3 + 2]);
      return pal(p.input.colour[i]).map((v) => clamp8(v * l)) as [number, number, number];
    },
    clean_lit: (p, i) => {
      const l = lambert(p.smoothNormal[i * 3], p.smoothNormal[i * 3 + 1], p.smoothNormal[i * 3 + 2]);
      const s = 1 - 0.65 * p.result.seam[i];
      return [0, 1, 2].map((k) => clamp8(p.result.albedo[i * 3 + k] * l * s)) as [number, number, number];
    },
    original_palette: (p, i) => pal(p.input.colour[i]),
    albedo: (p, i) => [p.result.albedo[i * 3], p.result.albedo[i * 3 + 1], p.result.albedo[i * 3 + 2]],
    regions: (p, i) => hashColour(partIndex.get(p)! * 977 + p.result.region[i]),
    seams: (p, i) => {
      const f = p.result.seamFeature[i];
      if (f < 0) return [200, 200, 200];
      const kind = p.result.seams[f].kind;
      const s = 0.4 + 0.6 * p.result.seam[i];
      return kind === 'line' ? [clamp8(255 * s), 20, 20] : [20, 40, clamp8(255 * s)];
    },
    house_mask: (p, i) => (p.result.mask[i] ? [220, 30, 30] : [190, 190, 190]),
    material: (p, i) => MATERIAL_RGB[p.result.regions[p.result.region[i]].material],
  };
  const images: Record<string, string> = {};
  let energy: Record<string, number> = {};
  for (const [name, shader] of Object.entries(views)) {
    const r = render(parts, shader, regionKey);
    const file = join(outDir, `${id}_${name}.png`);
    writeFileSync(file, png(r.W, r.H, r.img));
    images[name] = file;
    if (name !== 'regions' && name !== 'seams' && name !== 'house_mask' && name !== 'material')
      energy = { ...energy, [name]: +speckleEnergy(r, r.ids).toFixed(3) };
  }
  const partReports = parts.map((p) => {
    const before = regionLocalVariance(
      p.input,
      p.result,
      (() => {
        const rgb = new Uint8Array(p.input.colour.length * 3);
        for (let i = 0; i < p.input.colour.length; i++) rgb.set(pal(p.input.colour[i]), i * 3);
        return rgb;
      })(),
    );
    const after = regionLocalVariance(p.input, p.result, p.result.albedo);
    let houseIn = 0;
    let houseOut = 0;
    let houseMismatch = 0;
    for (let i = 0; i < p.input.colour.length; i++) {
      const c = p.input.colour[i];
      const h = c >= RA2_HOUSE_RAMP[0] && c <= RA2_HOUSE_RAMP[1] ? 1 : 0;
      houseIn += h;
      houseOut += p.result.mask[i];
      if (h !== p.result.mask[i]) houseMismatch++;
    }
    const surfaceCount = p.result.surface.reduce((a, b) => a + b, 0);
    const distinctBefore = new Set(Array.from(p.input.colour)).size;
    return {
      file: p.file,
      voxels: p.input.colour.length,
      surfaceVoxels: surfaceCount,
      paletteEntries: distinctBefore,
      regions: p.result.regions.length,
      variance: { before, after },
      house: { paletteVoxels: houseIn, maskVoxels: houseOut, mismatches: houseMismatch },
      seams: {
        lines: p.result.seams.filter((s) => s.kind === 'line').length,
        dots: p.result.seams.filter((s) => s.kind === 'dot').length,
        voxels: p.result.seamFeature.reduce((a, v) => a + (v >= 0 ? 1 : 0), 0),
        list: p.result.seams,
      },
      regionList: p.result.regions,
    };
  });
  const report = {
    id,
    files,
    turretOffsetLeptons: turretOffsetOf(id),
    imageSpeckleEnergy: energy,
    images,
    parts: partReports,
  };
  writeFileSync(join(outDir, `${id}.json`), JSON.stringify(report, null, 1));
  summary.push({
    id,
    imageSpeckleEnergy: energy,
    parts: partReports.map((r) => ({
      file: r.file,
      voxels: r.voxels,
      surface: r.surfaceVoxels,
      paletteEntries: r.paletteEntries,
      regions: r.regions,
      deltaEBefore: +r.variance.before.meanDeltaE.toFixed(3),
      steppedBefore: +r.variance.before.steppedFraction.toFixed(3),
      deltaEAfter: +r.variance.after.meanDeltaE.toFixed(3),
      pairs: r.variance.before.pairs,
      house: r.house,
      seamLines: r.seams.lines,
      seamDots: r.seams.dots,
      seamVoxels: r.seams.voxels,
    })),
  });
}
writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 1));
console.log(JSON.stringify(summary, null, 1));
