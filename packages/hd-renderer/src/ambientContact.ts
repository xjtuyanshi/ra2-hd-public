import { DataTexture, LinearFilter, RedFormat, UnsignedByteType } from 'three/webgpu';
import { float, texture } from 'three/tsl';
import type { ShaderNode } from './tsl';
import type { RealLightNodes } from './realLight';
import { NIGHT_ART_DEFAULTS } from './nightArt';

/** Ground-space receiver contract. Game modules supply poses/footprints; no identity, asset or timing policy here. */
export interface AmbientContact {
  x: number;
  z: number;
  width: number;
  depth: number;
  yaw?: number;
  band?: boolean;
  height?: number;
}
export function ambientContactAt(p: AmbientContact, x: number, z: number): number {
  const c = Math.cos(p.yaw ?? 0),
    s = Math.sin(p.yaw ?? 0);
  const dx = x - p.x,
    dz = z - p.z;
  const qx = Math.abs(c * dx + s * dz) - p.width / 2;
  const qz = Math.abs(-s * dx + c * dz) - p.depth / 2;
  const distance = Math.hypot(Math.max(0, qx), Math.max(0, qz));
  const edge = p.band ? NIGHT_ART_DEFAULTS.wallSoftness : NIGHT_ART_DEFAULTS.unitSoftness;
  if (p.band && Math.max(qx, qz) < -edge) return 0;
  const t = Math.min(1, distance / edge);
  return 1 - t * t * (3 - 2 * t);
}

/** Quantised semantic hash; polling an unchanged footprint does not allocate JSON strings. */
export function contactHash(contacts: readonly AmbientContact[]): number {
  let hash = 2166136261;
  for (const p of contacts)
    for (const v of [p.x, p.z, p.width, p.depth, p.yaw ?? 0, p.band ? 1 : 0, p.height ?? -999])
      hash = Math.imul(hash ^ Math.round(v * 64), 16777619);
  return Math.imul(hash ^ contacts.length, 16777619) >>> 0;
}
type Rect = { x0: number; z0: number; x1: number; z1: number };
/** Small, incrementally rasterised max-union mask. Dirty tiles include both old and new footprints. */
export class AmbientContactField {
  readonly texture: DataTexture;
  readonly groups = new Map<object, readonly AmbientContact[]>();
  private readonly keys = new Map<object, number>();
  private readonly dirty = new Set<number>();
  private lastFrame: number | undefined;
  readonly density: number;
  private readonly tile = 16;
  private readonly columns: number;
  rasterTexels = 0;
  updates = 0;
  constructor(
    readonly width: number,
    readonly depth: number,
  ) {
    this.density = Math.min(4, 1024 / Math.max(width, depth));
    const w = Math.min(1024, Math.ceil(width * this.density)),
      h = Math.min(1024, Math.ceil(depth * this.density));
    this.columns = Math.ceil(w / this.tile);
    this.texture = new DataTexture(new Uint8Array(w * h), w, h, RedFormat, UnsignedByteType);
    this.texture.minFilter = this.texture.magFilter = LinearFilter;
    this.texture.needsUpdate = true;
    this.invalidate();
  }
  private bounds(p: AmbientContact): Rect {
    const reach =
      Math.hypot(p.width, p.depth) / 2 + Math.max(NIGHT_ART_DEFAULTS.unitSoftness, NIGHT_ART_DEFAULTS.wallSoftness);
    const { width: w, height: h } = this.texture.image;
    return {
      x0: Math.max(0, Math.floor(((p.x - reach) * w) / this.width)),
      z0: Math.max(0, Math.floor(((p.z - reach) * h) / this.depth)),
      x1: Math.min(w, Math.ceil(((p.x + reach) * w) / this.width)),
      z1: Math.min(h, Math.ceil(((p.z + reach) * h) / this.depth)),
    };
  }
  private mark(contacts: readonly AmbientContact[]): void {
    for (const p of contacts) {
      const r = this.bounds(p);
      for (let z = Math.floor(r.z0 / this.tile); z < Math.ceil(r.z1 / this.tile); z++)
        for (let x = Math.floor(r.x0 / this.tile); x < Math.ceil(r.x1 / this.tile); x++)
          this.dirty.add(z * this.columns + x);
    }
  }
  set(owner: object, contacts: readonly AmbientContact[]): void {
    const key = contactHash(contacts);
    if (this.keys.get(owner) === key) return;
    this.mark(this.groups.get(owner) ?? []);
    // Snapshot caller-owned mutable poses; the old footprint remains available on the next update.
    const copy = contacts.map((p) => ({ ...p }));
    this.keys.set(owner, key);
    this.groups.set(owner, copy);
    this.mark(copy);
  }
  invalidate(): void {
    const { height: h } = this.texture.image;
    for (let i = 0; i < this.columns * Math.ceil(h / this.tile); i++) this.dirty.add(i);
  }
  delete(owner: object): void {
    this.mark(this.groups.get(owner) ?? []);
    this.keys.delete(owner);
    this.groups.delete(owner);
  }
  update(groundAt: (x: number, z: number) => number, logicFrame?: number): boolean {
    this.rasterTexels = 0;
    if (!this.dirty.size || (logicFrame !== undefined && this.lastFrame === logicFrame)) return false;
    this.lastFrame = logicFrame;
    const { data, width: w, height: h } = this.texture.image as { data: Uint8Array; width: number; height: number };
    const sx = w / this.width,
      sz = h / this.depth;
    for (const id of this.dirty) {
      const x0 = (id % this.columns) * this.tile,
        z0 = Math.floor(id / this.columns) * this.tile;
      const x1 = Math.min(w, x0 + this.tile),
        z1 = Math.min(h, z0 + this.tile);
      this.rasterTexels += (x1 - x0) * (z1 - z0);
      for (let z = z0; z < z1; z++) data.fill(0, z * w + x0, z * w + x1);
      for (const contacts of this.groups.values())
        for (const p of contacts) {
          const r = this.bounds(p);
          for (let j = Math.max(z0, r.z0); j < Math.min(z1, r.z1); j++)
            for (let i = Math.max(x0, r.x0); i < Math.min(x1, r.x1); i++) {
              const x = (i + 0.5) / sx,
                z = (j + 0.5) / sz;
              if (p.height !== undefined && Math.abs(groundAt(x, z) - p.height) > 0.35) continue;
              const at = j * w + i;
              data[at] = Math.max(data[at]!, Math.round(255 * ambientContactAt(p, x, z)));
            }
        }
    }
    this.dirty.clear();
    this.updates++;
    this.texture.needsUpdate = true;
    return true;
  }
  dispose(): void {
    this.groups.clear();
    this.keys.clear();
    this.dirty.clear();
    this.texture.dispose();
  }
}

/** Scene multiplier only. The physical local-light term never samples this mask. */
export function ambientContactNode(nodes: RealLightNodes | undefined, xz: ShaderNode): ShaderNode {
  if (!nodes) return float(1);
  return float(1).sub(
    texture(nodes.ambientContact, xz.div(nodes.size))
      .r.mul(nodes.ambientOcclusion.clamp())
      .mul(nodes.enabled)
      .mul(nodes.nightWeight),
  );
}
