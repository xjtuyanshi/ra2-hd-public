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

/** Renderer-owned mask, shared by contact decals and contact twins. Max union prevents duplicate part/page darkening. */
export class AmbientContactField {
  readonly texture: DataTexture;
  readonly groups = new Map<object, readonly AmbientContact[]>();
  private dirty = true;
  private keys = new Map<object, string>();
  readonly density: number;
  constructor(
    readonly width: number,
    readonly depth: number,
  ) {
    this.density = Math.min(8, 2048 / Math.max(width, depth));
    const w = Math.ceil(width * this.density),
      h = Math.ceil(depth * this.density);
    this.texture = new DataTexture(new Uint8Array(w * h), w, h, RedFormat, UnsignedByteType);
    this.texture.minFilter = this.texture.magFilter = LinearFilter;
    this.texture.needsUpdate = true;
  }
  set(owner: object, contacts: readonly AmbientContact[]): void {
    const key = JSON.stringify(contacts);
    if (this.keys.get(owner) === key) return;
    this.keys.set(owner, key);
    this.groups.set(owner, contacts);
    this.dirty = true;
  }
  invalidate(): void {
    this.dirty = true;
  }
  delete(owner: object): void {
    this.keys.delete(owner);
    if (this.groups.delete(owner)) this.dirty = true;
  }
  update(groundAt: (x: number, z: number) => number): void {
    if (!this.dirty) return;
    this.dirty = false;
    const { data, width: w, height: h } = this.texture.image as { data: Uint8Array; width: number; height: number };
    data.fill(0);
    const sx = w / this.width,
      sz = h / this.depth;
    for (const contacts of this.groups.values())
      for (const p of contacts) {
        const reach = Math.hypot(p.width, p.depth) / 2 + NIGHT_ART_DEFAULTS.unitSoftness;
        for (let j = Math.max(0, Math.floor((p.z - reach) * sz)); j < Math.min(h, Math.ceil((p.z + reach) * sz)); j++)
          for (
            let i = Math.max(0, Math.floor((p.x - reach) * sx));
            i < Math.min(w, Math.ceil((p.x + reach) * sx));
            i++
          ) {
            const x = (i + 0.5) / sx,
              z = (j + 0.5) / sz;
            // Raised bridges and cliff edges must not project an AO pool onto a lower, unrelated surface.
            if (p.height !== undefined && Math.abs(groundAt(x, z) - p.height) > 0.35) continue;
            const at = j * w + i;
            data[at] = Math.max(data[at]!, Math.round(255 * ambientContactAt(p, x, z)));
          }
      }
    this.texture.needsUpdate = true;
  }
  dispose(): void {
    this.groups.clear();
    this.keys.clear();
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
