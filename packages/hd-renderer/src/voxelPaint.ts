/**
 * Voxel paint de-dithering: turns per-voxel palette colours into a clean material description.
 *
 * Indexed voxel art bakes shading steps and dither into neighbouring palette entries of one ramp. Drawn per voxel
 * at a few pixels each, that turns into speckle. This module segments the surface into paint regions (connected
 * voxels of one colour family, similar lightness and compatible normals), gives each region one flat base albedo
 * (a trimmed mean of its own voxels, never a per-voxel shade), keeps masked families (for example remappable player
 * colour) exactly as the palette marks them, and keeps the dark lines the artist drew on purpose (panel seams,
 * hatch outlines, grooves, bolt holes) as a separate seam map instead of averaging them away.
 *
 * Generic mechanism only: the palette, its colour families, which families are masks and which material each palette
 * entry belongs to are injected by a game policy. The result is deterministic for a given input and policy.
 */

export type PaintMaterial = 'painted' | 'bare-metal' | 'rubber' | 'glass' | 'light';

export const PAINT_MATERIALS: readonly PaintMaterial[] = ['painted', 'bare-metal', 'rubber', 'glass', 'light'];

export interface PaintFamily {
  /** Stable name used in debug output. */
  readonly name: string;
  /**
   * A masked family (for example player colour) is exact per voxel: its voxels are never merged into another family's
   * region and no other family's voxel is merged into its regions.
   */
  readonly masked?: boolean;
}

export interface VoxelPaintPolicy {
  /** 256 x RGB, 8-bit sRGB. */
  readonly palette: ArrayLike<number>;
  readonly families: readonly PaintFamily[];
  /** Colour family of a palette index; -1 for an entry outside every family (it becomes a family of its own). */
  familyOf(index: number): number;
  /** Material of a palette index; a region takes the voxel-weighted majority of its members. */
  materialOf(index: number): PaintMaterial;
}

export interface VoxelPaintOptions {
  /** Neighbours join one facet only when their normals agree at least this much (cosine). */
  readonly normalCos: number;
  /** Neighbours join one facet only when their CIE L* differ by at most this (shading steps and dither pairs). */
  readonly joinDelta: number;
  /** Adjacent facets of one family whose median L* differ by at most this merge into one part region. */
  readonly mergeDelta: number;
  /** A line-shaped region darker than its host by at least this L* is a drawn seam. */
  readonly seamDelta: number;
  /** A dot (shorter than lineMinLength) must be at least this much darker than its host to stay as a detail dot. */
  readonly dotDelta: number;
  /** Regions up to this many voxels that are neither seams nor dots are speckle and join their host. */
  readonly speckleSize: number;
  /** Minimum extent, in voxels, of a seam line; shorter dark features are dots. */
  readonly lineMinLength: number;
  /** A seam region has at least this fraction of voxels on its boundary (it is thin, not a filled patch). */
  readonly lineBoundary: number;
  /**
   * A seam region has at most this many voxels per voxel of length (straight lines ~1, hatch outlines and branching panel-line networks a few); a larger
   * dark part such as a whole track face is a part of its own, not a line drawn on its neighbour.
   */
  readonly lineMaxWidth: number;
  /** Fraction trimmed from each end (by L*) before averaging a region's albedo. */
  readonly trim: number;
  /** L* darkness that maps to full seam strength 1. */
  readonly seamFull: number;
}

export const DEFAULT_VOXEL_PAINT: Readonly<VoxelPaintOptions> = Object.freeze({
  normalCos: 0.35,
  joinDelta: 10,
  mergeDelta: 7,
  seamDelta: 10,
  dotDelta: 18,
  speckleSize: 3,
  lineMinLength: 3,
  lineBoundary: 0.9,
  lineMaxWidth: 12,
  trim: 0.15,
  seamFull: 40,
});

export interface VoxelPaintInput {
  /** Grid size (x, y, z). */
  readonly size: readonly [number, number, number];
  /** Per-voxel grid coordinates and palette indices, all of equal length. */
  readonly x: ArrayLike<number>;
  readonly y: ArrayLike<number>;
  readonly z: ArrayLike<number>;
  readonly colour: ArrayLike<number>;
  /** 3 floats per voxel: the voxel's stored surface normal (need not be unit length). */
  readonly normal: ArrayLike<number>;
}

export interface PaintRegion {
  readonly id: number;
  /** Policy family index, -1 for an entry outside every family. */
  readonly family: number;
  readonly familyName: string;
  readonly material: PaintMaterial;
  readonly masked: boolean;
  /** Member voxels including interior ones and the seam voxels drawn on it. */
  readonly voxels: number;
  readonly surfaceVoxels: number;
  /** Flat base albedo, 8-bit sRGB. */
  readonly albedo: readonly [number, number, number];
  /** CIE L* of the albedo. */
  readonly lightness: number;
  /** Palette entries that collapsed into this region's albedo (index -> surface voxel count, seams excluded). */
  readonly indices: Readonly<Record<number, number>>;
}

export type SeamKind = 'line' | 'dot';

export interface SeamFeature {
  readonly id: number;
  readonly kind: SeamKind;
  /** The region the feature is drawn on. */
  readonly region: number;
  readonly voxels: number;
  /** Largest bounding-box extent in voxels. */
  readonly length: number;
  /** Axis of the largest extent. */
  readonly axis: 'x' | 'y' | 'z';
  /** Mean L* darkness below the host region's median. */
  readonly darkness: number;
  readonly bboxMin: readonly [number, number, number];
  readonly bboxMax: readonly [number, number, number];
  readonly indices: Readonly<Record<number, number>>;
}

export interface VoxelPaintResult {
  /** 1 for voxels with at least one empty 6-neighbour. */
  readonly surface: Uint8Array;
  /** Region id per voxel (interior voxels take the nearest surface voxel's region). */
  readonly region: Int32Array;
  /** Seam feature id per voxel, -1 when the voxel is not on a seam. */
  readonly seamFeature: Int32Array;
  /** Seam strength 0..1 per voxel (how much darker the original drew it than its host region). */
  readonly seam: Float32Array;
  /** 1 where the voxel's own palette entry is in a masked family; identical to the input by construction. */
  readonly mask: Uint8Array;
  /** Flat albedo per voxel (RGB, 8-bit sRGB) = its region albedo. */
  readonly albedo: Uint8Array;
  readonly regions: readonly PaintRegion[];
  readonly seams: readonly SeamFeature[];
}

// ---------- colour helpers ----------

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(v: number): number {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(c * 255)));
}

/** CIE L*a*b* (D65) of an 8-bit sRGB colour. */
export function srgbToLab(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  const X = (0.4124 * lr + 0.3576 * lg + 0.1805 * lb) / 0.95047;
  const Y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
  const Z = (0.0193 * lr + 0.1192 * lg + 0.9505 * lb) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);
  const fx = f(X);
  const fy = f(Y);
  const fz = f(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// ---------- union-find ----------

class Dsu {
  readonly parent: Int32Array;
  constructor(n: number) {
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(a: number): number {
    const p = this.parent;
    while (p[a] !== a) {
      p[a] = p[p[a]];
      a = p[a];
    }
    return a;
  }
  /** Deterministic: the smaller root survives. */
  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;
    if (ra < rb) this.parent[rb] = ra;
    else this.parent[ra] = rb;
  }
}

const NEIGHBOURS_26: readonly (readonly [number, number, number])[] = (() => {
  const out: [number, number, number][] = [];
  for (let dz = -1; dz <= 1; dz++)
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx || dy || dz) out.push([dx, dy, dz]);
  return out;
})();

function median(values: number[]): number {
  if (!values.length) return 0;
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Occupancy grid shared by the analysis and its metrics. */
export class VoxelGrid {
  readonly cell: Int32Array;
  readonly sx: number;
  readonly sy: number;
  readonly sz: number;
  constructor(readonly input: VoxelPaintInput) {
    [this.sx, this.sy, this.sz] = input.size;
    this.cell = new Int32Array(this.sx * this.sy * this.sz).fill(-1);
    for (let i = 0; i < input.x.length; i++) this.cell[this.key(input.x[i], input.y[i], input.z[i])] = i;
  }
  key(x: number, y: number, z: number): number {
    return (z * this.sy + y) * this.sx + x;
  }
  at(x: number, y: number, z: number): number {
    if (x < 0 || y < 0 || z < 0 || x >= this.sx || y >= this.sy || z >= this.sz) return -1;
    return this.cell[this.key(x, y, z)];
  }
  /** Calls fn for each occupied 26-neighbour of voxel i, in a fixed order. */
  forNeighbours(i: number, fn: (j: number) => void): void {
    const { x, y, z } = this.input;
    for (const [dx, dy, dz] of NEIGHBOURS_26) {
      const j = this.at(x[i] + dx, y[i] + dy, z[i] + dz);
      if (j >= 0) fn(j);
    }
  }
  surfaceFlags(): Uint8Array {
    const { x, y, z } = this.input;
    const n = x.length;
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const X = x[i];
      const Y = y[i];
      const Z = z[i];
      if (
        this.at(X + 1, Y, Z) < 0 ||
        this.at(X - 1, Y, Z) < 0 ||
        this.at(X, Y + 1, Z) < 0 ||
        this.at(X, Y - 1, Z) < 0 ||
        this.at(X, Y, Z + 1) < 0 ||
        this.at(X, Y, Z - 1) < 0
      )
        out[i] = 1;
    }
    return out;
  }
}

interface WorkRegion {
  /** Surface voxels only. */
  members: number[];
  family: number;
  masked: boolean;
  /** Median L*. */
  lightness: number;
}

/** Segments a voxel model's paint; see the module comment. */
export function analyseVoxelPaint(
  input: VoxelPaintInput,
  policy: VoxelPaintPolicy,
  options: Partial<VoxelPaintOptions> = {},
): VoxelPaintResult {
  const o: VoxelPaintOptions = { ...DEFAULT_VOXEL_PAINT, ...options };
  const n = input.colour.length;
  const grid = new VoxelGrid(input);
  const surface = grid.surfaceFlags();
  const pal = policy.palette;
  const labL = new Float64Array(256);
  for (let c = 0; c < 256; c++) labL[c] = srgbToLab(pal[c * 3], pal[c * 3 + 1], pal[c * 3 + 2])[0];
  const familyCount = policy.families.length;
  const familyOfVoxel = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const f = policy.familyOf(input.colour[i]);
    familyOfVoxel[i] = f >= 0 ? f : familyCount + input.colour[i];
  }
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const f = familyOfVoxel[i];
    mask[i] = f < familyCount && policy.families[f].masked ? 1 : 0;
  }
  const L = (i: number) => labL[input.colour[i]];
  const unit = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = input.normal[i * 3];
    const b = input.normal[i * 3 + 1];
    const c = input.normal[i * 3 + 2];
    const len = Math.hypot(a, b, c) || 1;
    unit[i * 3] = a / len;
    unit[i * 3 + 1] = b / len;
    unit[i * 3 + 2] = c / len;
  }
  const dot = (i: number, j: number) =>
    unit[i * 3] * unit[j * 3] + unit[i * 3 + 1] * unit[j * 3 + 1] + unit[i * 3 + 2] * unit[j * 3 + 2];

  // 1. Facets: surface voxels of one family with compatible normals, grown breadth-first from each unclaimed voxel
  // (in voxel order) while a candidate stays within joinDelta L* of the facet's running mean. Comparing with the
  // running mean, not just the neighbour, stops a long shading ramp from chaining into one facet.
  const dsu = new Dsu(n);
  {
    const claimed = new Uint8Array(n);
    for (let seed = 0; seed < n; seed++) {
      if (!surface[seed] || claimed[seed]) continue;
      claimed[seed] = 1;
      let sum = L(seed);
      let count = 1;
      const queue = [seed];
      for (let q = 0; q < queue.length; q++) {
        const i = queue[q];
        grid.forNeighbours(i, (j) => {
          if (claimed[j] || !surface[j]) return;
          if (familyOfVoxel[j] !== familyOfVoxel[seed]) return;
          if (Math.abs(L(j) - sum / count) > o.joinDelta) return;
          if (dot(i, j) < o.normalCos) return;
          claimed[j] = 1;
          dsu.union(seed, j);
          sum += L(j);
          count++;
          queue.push(j);
        });
      }
    }
  }

  const buildRegions = (): Map<number, WorkRegion> => {
    const map = new Map<number, WorkRegion>();
    for (let i = 0; i < n; i++) {
      if (!surface[i]) continue;
      const r = dsu.find(i);
      let reg = map.get(r);
      if (!reg) {
        reg = { members: [], family: familyOfVoxel[i], masked: mask[i] === 1, lightness: 0 };
        map.set(r, reg);
      }
      reg.members.push(i);
    }
    for (const reg of map.values()) reg.lightness = median(reg.members.map(L));
    return map;
  };
  /** Contact counts between one region and its neighbour regions (surface 26-neighbour pairs). */
  const contacts = (root: number, reg: WorkRegion): Map<number, number> => {
    const c = new Map<number, number>();
    for (const i of reg.members)
      grid.forNeighbours(i, (j) => {
        if (!surface[j]) return;
        const r = dsu.find(j);
        if (r !== root) c.set(r, (c.get(r) ?? 0) + 1);
      });
    return c;
  };

  // 2. Part merge: adjacent facets of one family with nearly equal paint become one part (facets split by a bend).
  // Greedy from the largest facet, against the part's running voxel-weighted lightness, so it cannot drift either.
  {
    const regions = buildRegions();
    const facetOf = new Int32Array(n).fill(-1);
    for (const [root, reg] of regions) for (const i of reg.members) facetOf[i] = root;
    const facetContacts = (root: number): number[] => {
      const seen = new Set<number>();
      for (const i of regions.get(root)!.members)
        grid.forNeighbours(i, (j) => {
          const f = facetOf[j];
          if (f >= 0 && f !== root) seen.add(f);
        });
      return [...seen].sort((a, b) => a - b);
    };
    const order = [...regions.keys()].sort(
      (a, b) => regions.get(b)!.members.length - regions.get(a)!.members.length || a - b,
    );
    const done = new Set<number>();
    for (const start of order) {
      if (done.has(start)) continue;
      done.add(start);
      const first = regions.get(start)!;
      let weight = first.members.length;
      let sum = first.lightness * weight;
      const queue = [start];
      for (let q = 0; q < queue.length; q++) {
        for (const facet of facetContacts(queue[q])) {
          if (done.has(facet)) continue;
          const r2 = regions.get(facet)!;
          if (r2.family !== first.family) continue;
          if (Math.abs(r2.lightness - sum / weight) > o.mergeDelta) continue;
          done.add(facet);
          dsu.union(start, facet);
          weight += r2.members.length;
          sum += r2.lightness * r2.members.length;
          queue.push(facet);
        }
      }
    }
  }

  // 3. Small and thin regions, smallest first: drawn seam lines, detail dots, speckle, or kept detail parts.
  const seamOf = new Int32Array(n).fill(-1);
  const seamDark = new Float32Array(n);
  const seamFeatures: Omit<SeamFeature, 'region'>[] = [];
  {
    const regions = buildRegions();
    const roots = [...regions.keys()].sort(
      (a, b) => regions.get(a)!.members.length - regions.get(b)!.members.length || a - b,
    );
    const absorbed = new Set<number>();
    for (const root of roots) {
      if (absorbed.has(root) || dsu.find(root) !== root) continue;
      const reg = regions.get(root)!;
      const c = contacts(root, reg);
      // Host: the most-touching neighbour region at least as large as this one (ties: lower root).
      let host = -1;
      let best = 0;
      for (const [r, k] of [...c.entries()].sort((a, b) => a[0] - b[0])) {
        const r2 = regions.get(r);
        if (!r2 || absorbed.has(r) || r2.members.length < reg.members.length) continue;
        if (k > best) {
          best = k;
          host = r;
        }
      }
      if (host < 0) continue;
      const hostReg = regions.get(host)!;
      if (hostReg.masked !== reg.masked) continue; // masks stay exact
      const darkness = hostReg.lightness - reg.lightness;
      const mn = [Infinity, Infinity, Infinity];
      const mx = [-Infinity, -Infinity, -Infinity];
      let boundary = 0;
      for (const i of reg.members) {
        const p = [input.x[i], input.y[i], input.z[i]];
        for (let k = 0; k < 3; k++) {
          mn[k] = Math.min(mn[k], p[k]);
          mx[k] = Math.max(mx[k], p[k]);
        }
        let edge = false;
        // Only same-facing neighbours count, so the far side of a thin plate does not make a face look like a line.
        grid.forNeighbours(i, (j) => {
          if (surface[j] && dot(i, j) >= o.normalCos && dsu.find(j) !== root) edge = true;
        });
        if (edge) boundary++;
      }
      const ext = mx.map((v, k) => v - mn[k] + 1);
      const length = Math.max(...ext);
      const thin = boundary / reg.members.length >= o.lineBoundary && reg.members.length <= length * o.lineMaxWidth;
      const isLine = thin && length >= o.lineMinLength && darkness >= o.seamDelta;
      const isDot = !isLine && length < o.lineMinLength && darkness >= o.dotDelta;
      const isSpeckle = !isLine && !isDot && reg.members.length <= o.speckleSize && Math.abs(darkness) < o.dotDelta;
      if (isLine || isDot) {
        const id = seamFeatures.length;
        const indices: Record<number, number> = {};
        let dsum = 0;
        for (const i of reg.members) {
          seamOf[i] = id;
          const d = Math.max(0, hostReg.lightness - L(i));
          seamDark[i] = d;
          dsum += d;
          indices[input.colour[i]] = (indices[input.colour[i]] ?? 0) + 1;
        }
        seamFeatures.push({
          id,
          kind: isLine ? 'line' : 'dot',
          voxels: reg.members.length,
          length,
          axis: (['x', 'y', 'z'] as const)[ext.indexOf(length)],
          darkness: dsum / reg.members.length,
          bboxMin: mn as [number, number, number],
          bboxMax: mx as [number, number, number],
          indices,
        });
      }
      if (isLine || isDot || isSpeckle) {
        // The host keeps its own root only if it is the smaller index; track the absorbed one either way.
        dsu.union(host, root);
        const survivor = dsu.find(root);
        const gone = survivor === root ? host : root;
        absorbed.add(gone);
        if (survivor !== host) regions.set(survivor, { ...hostReg, members: hostReg.members.concat(reg.members) });
        else hostReg.members.push(...reg.members);
      }
    }
  }

  // 4. Final regions, deterministic ids in order of first surface voxel.
  const rootId = new Map<number, number>();
  const finalMembers: number[][] = [];
  for (let i = 0; i < n; i++) {
    if (!surface[i]) continue;
    const r = dsu.find(i);
    let id = rootId.get(r);
    if (id === undefined) {
      id = finalMembers.length;
      rootId.set(r, id);
      finalMembers.push([]);
    }
    finalMembers[id].push(i);
  }
  const region = new Int32Array(n).fill(-1);
  for (let id = 0; id < finalMembers.length; id++) for (const i of finalMembers[id]) region[i] = id;
  // Interior voxels take the region of the nearest surface voxel (breadth-first over occupied neighbours).
  let frontier: number[] = [];
  for (let i = 0; i < n; i++) if (surface[i]) frontier.push(i);
  while (frontier.length) {
    const next: number[] = [];
    for (const i of frontier)
      grid.forNeighbours(i, (j) => {
        if (region[j] < 0) {
          region[j] = region[i];
          next.push(j);
        }
      });
    frontier = next;
  }

  // 5. Albedo: trimmed mean in linear light of each region's non-seam surface voxels.
  const regions: PaintRegion[] = [];
  const counts = new Int32Array(finalMembers.length);
  for (let i = 0; i < n; i++) if (region[i] >= 0) counts[region[i]]++;
  for (let id = 0; id < finalMembers.length; id++) {
    const all = finalMembers[id];
    const plain = all.filter((i) => seamOf[i] < 0);
    const base = plain.length ? plain : all;
    const sorted = base.slice().sort((a, b) => L(a) - L(b) || a - b);
    const cut = Math.floor(sorted.length * o.trim);
    const kept = sorted.length - 2 * cut > 0 ? sorted.slice(cut, sorted.length - cut) : sorted;
    let r = 0;
    let g = 0;
    let b = 0;
    for (const i of kept) {
      const c = input.colour[i];
      r += srgbToLinear(pal[c * 3]);
      g += srgbToLinear(pal[c * 3 + 1]);
      b += srgbToLinear(pal[c * 3 + 2]);
    }
    const indices: Record<number, number> = {};
    const votes = new Map<PaintMaterial, number>();
    for (const i of base) {
      const c = input.colour[i];
      indices[c] = (indices[c] ?? 0) + 1;
      const m = policy.materialOf(c);
      votes.set(m, (votes.get(m) ?? 0) + 1);
    }
    let material: PaintMaterial = 'painted';
    let top = -1;
    for (const m of PAINT_MATERIALS) {
      const v = votes.get(m) ?? 0;
      if (v > top) {
        top = v;
        material = m;
      }
    }
    const rgb: [number, number, number] = [
      linearToSrgb(r / kept.length),
      linearToSrgb(g / kept.length),
      linearToSrgb(b / kept.length),
    ];
    const fam = familyOfVoxel[all[0]];
    regions.push({
      id,
      family: fam < familyCount ? fam : -1,
      familyName: fam < familyCount ? policy.families[fam].name : `index${fam - familyCount}`,
      material,
      masked: mask[all[0]] === 1,
      voxels: counts[id],
      surfaceVoxels: all.length,
      albedo: rgb,
      lightness: srgbToLab(rgb[0], rgb[1], rgb[2])[0],
      indices,
    });
  }
  const albedo = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const id = region[i];
    if (id < 0) continue;
    albedo.set(regions[id].albedo, i * 3);
  }
  const seam = new Float32Array(n);
  const featureRegion = new Int32Array(seamFeatures.length).fill(-1);
  for (let i = 0; i < n; i++) {
    const f = seamOf[i];
    if (f < 0) continue;
    seam[i] = Math.min(1, seamDark[i] / o.seamFull);
    if (featureRegion[f] < 0) featureRegion[f] = region[i];
  }
  const seams: SeamFeature[] = seamFeatures.map((f) => ({ ...f, region: featureRegion[f.id] }));
  return { surface, region, seamFeature: seamOf, seam, mask, albedo, regions, seams };
}

export interface PaintVarianceStats {
  /** Neighbouring surface voxel pairs counted (both non-seam, same region). */
  readonly pairs: number;
  /** Mean CIE76 colour difference between such neighbours. */
  readonly meanDeltaE: number;
  /** Fraction of such pairs whose colours differ by more than 2 ΔE (a visible step). */
  readonly steppedFraction: number;
}

/**
 * Local colour variation inside regions: compares each surface voxel with its 26 surface neighbours of the same
 * region (seam voxels excluded, so drawn design lines do not count as noise). `rgb` is 3 bytes per voxel.
 */
export function regionLocalVariance(
  input: VoxelPaintInput,
  result: VoxelPaintResult,
  rgb: ArrayLike<number>,
): PaintVarianceStats {
  const grid = new VoxelGrid(input);
  const n = input.colour.length;
  const lab: [number, number, number][] = [];
  for (let i = 0; i < n; i++) lab.push(srgbToLab(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]));
  let pairs = 0;
  let sum = 0;
  let stepped = 0;
  for (let i = 0; i < n; i++) {
    if (!result.surface[i] || result.seamFeature[i] >= 0) continue;
    grid.forNeighbours(i, (j) => {
      if (j <= i || !result.surface[j] || result.seamFeature[j] >= 0) return;
      if (result.region[j] !== result.region[i]) return;
      const d = Math.hypot(lab[i][0] - lab[j][0], lab[i][1] - lab[j][1], lab[i][2] - lab[j][2]);
      pairs++;
      sum += d;
      if (d > 2) stepped++;
    });
  }
  return { pairs, meanDeltaE: pairs ? sum / pairs : 0, steppedFraction: pairs ? stepped / pairs : 0 };
}
