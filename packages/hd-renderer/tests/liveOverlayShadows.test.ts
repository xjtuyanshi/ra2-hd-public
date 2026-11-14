import { describe, expect, it } from 'vitest';
import type { TypeLibrary } from '../../ra2-assets/src/library';
import type { MapObject } from '../../ra2-assets/src/manifest';
import { LiveView } from '../src/live/liveView';
import { buildOverlayInstances } from '../src/live/overlays';
import { SpritePool } from '../src/live/spritePool';
import type { SpriteInstance } from '../src/ra2/sprites';

const frame = { rect: [0, 0, 8, 8] as [number, number, number, number], offset: [0, 0] as [number, number] };
const sprite = {
  frames: [frame, frame, frame, frame],
  shadows: [frame, frame, frame, frame],
  palette: 'unit',
  remap: false,
};

describe('map overlay shadow shape', () => {
  it('keeps rock and rubble art shadows while library and stock high decks still follow the sun', () => {
    const overlay = { sprite: 'art', palette: 'unit', remap: false, flat: false, frames: 4, drawOffset: [0, 0] };
    const library = {
      overlays: {
        SROCK01: overlay,
        CUSTOMDECK: { ...overlay, highBridge: true },
        BRIDGE1: overlay,
        RUBBLE: { ...overlay, rubble: true },
      },
      sprites: { art: sprite },
      types: { RUIN: { kind: 'building', sprite: 'art' } },
    } as unknown as TypeLibrary;
    const result = buildOverlayInstances(
      {
        originX: 0,
        originY: 0,
        width: 4,
        height: 1,
        index: new Int16Array([0, 1, 2, 3]),
        data: new Uint8Array(4),
        types: ['SROCK01', 'CUSTOMDECK', 'BRIDGE1', 'RUBBLE'],
        rubble: [{ slot: 3, type: 'RUIN' }],
        revision: 1,
      },
      library,
      { width: 4, height: 1, levels: new Uint8Array(4), exists: new Uint8Array([1, 1, 1, 1]) },
      () => 'unit',
    );
    expect([...result.missingTypes]).toEqual([]);
    const instances = result.pages.get(0)!;
    const shadows = instances.filter((s) => s.shadow);
    expect(shadows).toHaveLength(4);
    expect(shadows.map((s) => s.terrainShadow)).toEqual([true, undefined, undefined, true]);
    expect(instances.filter((s) => !s.shadow).every((s) => s.terrainShadow === undefined)).toBe(true);
  });

  it('tags legacy overlay shadows and clears the marker when their pooled slot becomes a bridge or smudge', () => {
    const objects: MapObject[] = [{ kind: 'overlay', type: 'TROCK01', sprite: 'art', x: 0, y: 0, z: 0 }];
    const instances: SpriteInstance[] = [];
    const view = Object.assign(Object.create(LiveView.prototype), {
      cache: { manifest: { sprites: { art: sprite } } },
      cellPool: new SpritePool(),
      cellObjects: new Map([[0, objects]]),
      cellInstances: instances,
      overlayInstances: { active: false },
      smudgeInstances: { active: false },
      cellLayer: { setInstances: () => {} },
    }) as { flushCells(): void };
    view.flushCells();
    const shadow = instances[0]!;
    expect(shadow.shadow).toBe(true);
    expect(shadow.terrainShadow).toBe(true);
    expect(instances[1]!.terrainShadow).toBeUndefined();
    objects[0] = { ...objects[0]!, type: 'BRIDGE1' };
    view.flushCells();
    expect(instances[0]).toBe(shadow);
    expect(shadow.terrainShadow).toBeUndefined();
    objects[0] = { ...objects[0]!, kind: 'smudge', type: 'SCORCH01' };
    view.flushCells();
    expect(instances[0]).toBe(shadow);
    expect(shadow.terrainShadow).toBeUndefined();
  });
});
