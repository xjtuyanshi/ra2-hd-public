import { describe, expect, it } from 'vitest';
import { Texture, Vector3 } from 'three/webgpu';
import { PAINT_MATERIAL, PAINT_STYLES, paintPreset, parsePaintStyle } from '../src/ra2/paintLook';
import { CONTACT_SHADE_MARGIN, CONTACT_SHADE_REACH } from '../src/ra2/spriteLook';
import { SpritePool } from '../src/live/spritePool';
import { SpriteLayer } from '../src/ra2/sprites';
import { painterObject } from '../src/ra2/painterOrder';
import type { Ra2Cache } from '../src/ra2/cache';
import type { DirtyAttribute } from '../src/dirtyAttribute';

describe('house paint style', () => {
  it.each(PAINT_STYLES)('parses %s', (style) => expect(parsePaintStyle(style)).toBe(style));
  it.each([null, undefined, '', 'MATERIAL', 'flat'])('keeps classic for %s', (value) =>
    expect(parsePaintStyle(value)).toBe('classic'),
  );
  it('defaults to classic', () => expect(paintPreset.style).toBe('classic'));
  it('keeps the material within the ramp (no neon, grime and wear below the ramp colour)', () => {
    expect(PAINT_MATERIAL.saturation).toBeLessThan(1);
    expect(PAINT_MATERIAL.grime).toBeLessThanOrEqual(0.5);
    expect(PAINT_MATERIAL.wear).toBeLessThanOrEqual(0.5);
    expect(PAINT_MATERIAL.contrastMin).toBeLessThan(1);
    expect(PAINT_MATERIAL.contrastMax).toBeGreaterThan(1);
  });
  it('lets warm houses (Gold, Orange) keep their colour: full saturation, lighter grime and wear', () => {
    expect(PAINT_MATERIAL.warmFrom).toBeLessThan(PAINT_MATERIAL.warmTo);
    expect(PAINT_MATERIAL.warmGrime).toBeLessThan(1);
    expect(PAINT_MATERIAL.warmWear).toBeLessThan(1);
    expect(PAINT_MATERIAL.warmDirt).toBeLessThan(1);
  });
});

describe('live building contact shade', () => {
  it('fades to zero inside the decal quad (no straight edge at the quad sides)', () =>
    // The quad's east/west sides are margin / 60 * sqrt(2) cells from the foundation.
    expect((CONTACT_SHADE_MARGIN / 60) * Math.SQRT2).toBeGreaterThanOrEqual(CONTACT_SHADE_REACH));

  const cache = {
    paletteAtlas: { image: { height: 1 } },
    paletteRow: new Map([['library:unit', 0]]),
    manifest: { spriteAtlas: { width: 64, height: 64 } },
    spriteAtlas: new Texture(),
  } as unknown as Ra2Cache;
  const frame = { rect: [0, 0, 300, 165] as [number, number, number, number], offset: [-120, -30] as [number, number] };

  it('writes the contact code with the foundation, and a reused pool slot drops the kind', () => {
    const pool = new SpritePool();
    pool.begin();
    const painter = painterObject(2, 100);
    painter.foundation = [5, 3];
    painter.x = 40;
    painter.z = 50;
    pool.painter = painter;
    const contact = pool.take(new Vector3(42.5, 0, 51.5), frame, 'library:unit', 0, false, true);
    contact.kind = 'contact';
    contact.painter = painter;
    const layer = new SpriteLayer(cache, [contact]);
    const flags = (layer as unknown as { flags: DirtyAttribute }).flags;
    // Contact code 3 (a flat decal), then foundation [5, 3] and the draw point's offset (dx - dz, -dx - dz).
    expect(Array.from(flags.array.slice(0, 8))).toEqual([3, 1, 0, 0, 5, 3, 1, -4]);
    layer.mesh.geometry.dispose();
    layer.mesh.material.dispose();
    for (const mesh of layer.groundPasses) mesh.material.dispose();
    pool.begin();
    expect(pool.take(new Vector3(0, 0, 0), frame, 'library:unit', 0, false, true).kind).toBeUndefined();
  });
});
