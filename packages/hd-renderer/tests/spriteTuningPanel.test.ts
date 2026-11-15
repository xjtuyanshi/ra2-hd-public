import { afterEach, describe, expect, it, vi } from 'vitest';
import { tuningModel, mountTuningPanel } from '../src/ui/tuningPanel';
import { RECOMMENDED, SPRITE_INTEGRATION_SCHEMA } from '../src/ra2/spriteIntegrationTuning';

afterEach(() => vi.unstubAllGlobals());
describe('generic tuning model', () => {
  it('bounds finite values, survives inaccessible/corrupt storage, resets and keeps edits across A/B', () => {
    let values = { ...RECOMMENDED };
    const ab = vi.fn();
    const options = {
      schema: SPRITE_INTEGRATION_SCHEMA,
      recommended: RECOMMENDED,
      storageKey: 'test',
      get: () => values,
      set: (next: typeof values) => {
        values = next;
      },
      ab,
    };
    const model = tuningModel(options, () => {
      throw new Error('blocked');
    });
    model.set({ edgeSigma: 5, paint: -1, edgeSoft: Infinity });
    expect(values.edgeSigma).toBe(1.4);
    expect(values.paint).toBe(0);
    expect(values.edgeSoft).toBe(RECOMMENDED.edgeSoft);
    model.ab(false);
    model.set({ soften: 0.9 });
    model.ab(true);
    expect(values.soften).toBe(0.9);
    model.reset();
    expect(model.get()).toEqual(RECOMMENDED);
    const saved = new Map<string, string>();
    const storage = () => ({
      getItem: (k: string) => saved.get(k) ?? '{bad',
      setItem: (k: string, v: string) => {
        saved.set(k, v);
      },
    });
    const persist = tuningModel(options, storage);
    persist.set({ deHalo: 0.4 });
    const restored = tuningModel(options, storage);
    expect(restored.get().deHalo).toBe(0.4);
  });
});

/** A DOM contract double, not browser acceptance. It exercises listeners/ownership without opening a browser. */
class ElementDouble extends EventTarget {
  children: ElementDouble[] = [];
  style = {};
  hidden = false;
  textContent = '';
  type = '';
  value = '';
  min = '';
  max = '';
  step = '';
  parent: ElementDouble | null = null;
  constructor(readonly tag = 'div') {
    super();
  }
  append(...items: ElementDouble[]) {
    for (const item of items) {
      item.parent = this;
      this.children.push(item);
    }
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this);
    this.parent = null;
  }
  blur() {}
  setAttribute() {}
  closest() {
    return ['input', 'textarea', 'select'].includes(this.tag) ? this : null;
  }
}
it('owns the DOM and capture hotkey, ignores inputs, supports hide and releases the listener', () => {
  const win = new EventTarget(),
    host = new ElementDouble(),
    ab = vi.fn();
  const add = vi.spyOn(win, 'addEventListener'),
    remove = vi.spyOn(win, 'removeEventListener');
  vi.stubGlobal('window', win);
  vi.stubGlobal('document', { createElement: (tag: string) => new ElementDouble(tag) });
  vi.stubGlobal('localStorage', {
    getItem: () => null,
    setItem: () => {
      throw new Error('denied');
    },
  });
  const panel = mountTuningPanel(host as unknown as HTMLElement, {
    schema: SPRITE_INTEGRATION_SCHEMA,
    recommended: { ...RECOMMENDED },
    storageKey: 'test',
    get: () => ({ ...RECOMMENDED }),
    set: vi.fn(),
    ab,
  });
  expect(host.children).toHaveLength(1);
  const root = host.children[0]!;
  expect(root.style).toMatchObject({ maxHeight: 'calc(100vh - 24px)', overflowY: 'auto', boxSizing: 'border-box' });
  const rows = root.children.at(-1)!.children;
  expect(rows).toHaveLength(SPRITE_INTEGRATION_SCHEMA.length);
  for (const [i, field] of SPRITE_INTEGRATION_SCHEMA.entries()) {
    const row = rows[i]!;
    expect((row.style as { cssText: string }).cssText).toContain('height:34px');
    const input = row.children[1]!;
    expect([input.min, input.max, input.step]).toEqual([String(field.min), String(field.max), String(field.step)]);
    input.value = String(field.max);
    input.dispatchEvent(new Event('input'));
    expect(panel.get()[field.key]).toBe(field.max);
  }
  root.children[1]!.dispatchEvent(new Event('click'));
  expect(panel.get()).toEqual({ ...RECOMMENDED });

  expect(add).toHaveBeenCalledWith('keydown', expect.any(Function), { capture: true });
  const key = (code: string, shiftKey = false, input = false) => {
    const e = new Event('keydown', { cancelable: true });
    Object.assign(e, { code, shiftKey, repeat: false });
    if (input) Object.defineProperty(e, 'target', { value: new ElementDouble('input') });
    win.dispatchEvent(e);
    return e;
  };
  const initial = ab.mock.calls.length;
  expect(key('KeyX').defaultPrevented).toBe(false);
  expect(key('Backquote', false, true).defaultPrevented).toBe(false);
  expect(ab).toHaveBeenCalledTimes(initial);
  expect(key('Backquote').defaultPrevented).toBe(true);
  expect(ab).toHaveBeenLastCalledWith(false);
  expect(host.children[0]!.children[0]!.textContent).toBe('当前');
  key('Backquote');
  expect(ab).toHaveBeenLastCalledWith(true);
  key('Backquote', true);
  expect(host.children[0]!.hidden).toBe(true);
  panel.dispose();
  expect(host.children).toHaveLength(0);
  expect(remove).toHaveBeenCalledWith('keydown', expect.any(Function), { capture: true });
  const calls = ab.mock.calls.length;
  key('Backquote');
  expect(ab).toHaveBeenCalledTimes(calls);
});

it('Backslash and bottom-right sprite panel coexist with the default Backquote panel', () => {
  const win = new EventTarget(),
    host = new ElementDouble();
  vi.stubGlobal('window', win);
  vi.stubGlobal('document', { createElement: (tag: string) => new ElementDouble(tag) });
  vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => {} });
  const oldAB = vi.fn(),
    spriteAB = vi.fn();
  const options = {
    schema: SPRITE_INTEGRATION_SCHEMA,
    recommended: RECOMMENDED,
    storageKey: 'a',
    get: () => ({ ...RECOMMENDED }),
    set: vi.fn(),
  };
  const a = mountTuningPanel(host as unknown as HTMLElement, { ...options, ab: oldAB });
  const b = mountTuningPanel(host as unknown as HTMLElement, {
    ...options,
    ab: spriteAB,
    hotkey: 'Backslash',
    corner: 'bottom-right',
    title: '精灵融合',
    names: ['新', '当前'],
  });
  expect(host.children[0]!.style).toMatchObject({ top: '12px' });
  expect(host.children[1]!.style).toMatchObject({ bottom: '12px' });
  const key = (code: string, shiftKey = false) => {
    const e = new Event('keydown', { cancelable: true });
    Object.assign(e, { code, shiftKey });
    win.dispatchEvent(e);
  };
  const oldCalls = oldAB.mock.calls.length;
  key('Backslash');
  expect(spriteAB).toHaveBeenLastCalledWith(false);
  expect(oldAB).toHaveBeenCalledTimes(oldCalls);
  key('Backslash', true);
  expect(host.children[1]!.hidden).toBe(true);
  expect(host.children[0]!.hidden).toBe(false);
  key('Backquote');
  expect(oldAB).toHaveBeenLastCalledWith(false);
  a.dispose();
  b.dispose();
  expect(host.children).toHaveLength(0);
});

it('the sprite controller keeps uniform identity and edited values across A/B, and disposes its listener', async () => {
  const { SpriteIntegration } = await import('../src/spriteIntegration');
  const { mountSpriteIntegrationTuning, spriteIntegrationAPI } = await import('../src/ra2/spriteIntegrationTuning');
  vi.stubGlobal('window', new EventTarget());
  vi.stubGlobal('document', { createElement: (tag: string) => new ElementDouble(tag) });
  vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => {} });
  const group = new SpriteIntegration();
  group.set(RECOMMENDED);
  const node = group.uniforms.soften,
    host = new ElementDouble();
  const panel = mountSpriteIntegrationTuning(host as unknown as HTMLElement, group);
  panel.ab(false);
  panel.set({ soften: 0.8 });
  expect(group.get()).toEqual({ ...RECOMMENDED, on: 0, soften: 0.8 });
  panel.ab(true);
  expect(group.get().soften).toBe(0.8);
  expect(group.get().on).toBe(1);
  panel.set({ on: 0 });
  panel.ab(false);
  panel.ab(true);
  expect(group.get().on).toBe(0);
  expect(group.uniforms.soften).toBe(node);
  const api = spriteIntegrationAPI(group);
  api.set({ edgeSigma: -1, relight: Infinity });
  expect(api.get().edgeSigma).toBe(0.5);
  expect(api.get().relight).toBe(RECOMMENDED.relight);
  api.ab(true);
  expect(api.get().on).toBe(1);
  panel.dispose();
  expect(host.children).toHaveLength(0);
});

it('loads v2 recommendations instead of v1 and restores every new control', async () => {
  const { SpriteIntegration, INTEGRATION_KEYS } = await import('../src/spriteIntegration');
  const { mountSpriteIntegrationTuning } = await import('../src/ra2/spriteIntegrationTuning');
  vi.stubGlobal('window', new EventTarget());
  vi.stubGlobal('document', { createElement: (tag: string) => new ElementDouble(tag) });
  const read = vi.fn(() => null);
  vi.stubGlobal('localStorage', { getItem: read, setItem: () => {} });
  const group = new SpriteIntegration();
  group.set(RECOMMENDED);
  const host = new ElementDouble();
  const panel = mountSpriteIntegrationTuning(host as unknown as HTMLElement, group);
  expect(read).toHaveBeenCalledWith('hd-sprite-v2');
  expect(SPRITE_INTEGRATION_SCHEMA.map((f) => f.key).sort()).toEqual([...INTEGRATION_KEYS].sort());
  panel.set({ objSoft: 0, vCut: 0, treeDetail: 0 });
  const find = (node: ElementDouble): ElementDouble | undefined =>
    node.textContent === '恢复推荐值' ? node : node.children.map(find).find(Boolean);
  find(host)!.dispatchEvent(new Event('click'));
  expect(group.get()).toEqual(RECOMMENDED);
  panel.dispose();
});
