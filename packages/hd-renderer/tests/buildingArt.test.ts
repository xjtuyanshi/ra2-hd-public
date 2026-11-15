import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildingArtPages } from '../src/ra2/buildingArt';
import { LiveLibrary } from '../src/live/library';
import { applyBlendTone } from '../src/ra2/blendTone';
import { lush2Defaults } from '../src/ra2/lush2Tone';
import type { Ra2Cache } from '../src/ra2/cache';
import type { HdSettings } from '../src/HdRenderer';

afterEach(() => vi.unstubAllGlobals());
const page = { file: 'source.png', remap: 'paint.png', width: 64, height: 32, scale: 2 };

describe('opt-in colour-only building atlas', () => {
  it('retains coordinates, paint, scale and unmodified pages', () => {
    const original = [page, null, page];
    const changed = buildingArtPages(original, { version: 1, files: ['clean.png', null, null] });
    expect(changed).toEqual([{ ...page, file: 'clean.png' }, null, page]);
    expect(changed[2]).toBe(page);
    expect(original[0]).toEqual(page);
  });
  it.each([
    null,
    {},
    { version: 1, files: [] },
    { version: 1, files: ['../x.png'] },
    { version: 1, files: ['/x.png'] },
  ])('rejects invalid variants %j', (value) => {
    expect(() => buildingArtPages([page], value)).toThrow();
  });
  it('requests no alternate art by default and rejects a missing explicit variant', async () => {
    const fetcher = vi.fn(async (url: string) =>
      url.endsWith('/library.json')
        ? Response.json({ palettes: {}, spriteAtlasesHd: [page] })
        : new Response(null, { status: 404 }),
    );
    vi.stubGlobal('fetch', fetcher);
    const library = await LiveLibrary.load('/library', {} as Ra2Cache);
    expect(library!.library.spriteAtlasesHd).toEqual([page]);
    expect(fetcher.mock.calls.some(([url]) => url.includes('buildings-clean'))).toBe(false);
    await expect(LiveLibrary.load('/library', {} as Ra2Cache, true)).rejects.toThrow('buildings-clean.json: 404');
    fetcher.mockImplementation(async (url: string) =>
      url.endsWith('/library.json')
        ? Response.json({ palettes: {}, spriteAtlasesHd: [page] })
        : url.endsWith('/buildings-clean.json')
          ? Response.json({ version: 1, files: ['clean.png'] })
          : new Response(null, { status: 404 }),
    );
    expect((await LiveLibrary.load('/library', {} as Ra2Cache, true))!.library.spriteAtlasesHd).toEqual([
      { ...page, file: 'clean.png' },
    ]);
  });
  it('keeps moving clouds, source clump contrast and theatre-safe shared object shade', () => {
    const settings = {} as HdSettings;
    const alive = applyBlendTone(settings, 'temperate', 'tuned');
    expect(alive.clouds).toBe(true);
    expect(alive.speed).toBeGreaterThan(0);
    expect(settings.cloudTone!.detail).toBe(1);
    expect(settings.cloudTone!.objectFloor).toBe(0.9);
    expect(settings.tone).toBeNull();
    const snow = {} as HdSettings;
    applyBlendTone(snow, 'snow', 'ground');
    expect(snow.cloudTone!.snowLift).toBe(lush2Defaults('snow').snowLift);
    expect(snow.cloudTone!.gold).toBe(0);
  });
});
