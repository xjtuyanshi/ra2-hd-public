import { afterEach, describe, expect, it, vi } from 'vitest';
import { tuningModel, mountTuningPanel } from '../src/ui/tuningPanel';
import { lush2Defaults, lush2Mapping, LUSH2_SCHEMA } from '../src/ra2/lush2Tone';
import { ALIVE_DEFAULTS } from '../src/livingWorld';

afterEach(() => vi.unstubAllGlobals());
describe('generic tuning model', () => {
  it('survives throwing getters, writes and corrupt records, bounds values and resets recommendations', () => {
    const recommended = lush2Defaults('snow');
    let current = { ...recommended };
    const ab = vi.fn();
    const storage = () => {
      throw new Error('storage blocked');
    };
    const model = tuningModel(
      {
        schema: LUSH2_SCHEMA,
        recommended,
        storageKey: 'snow',
        get: () => current,
        set: (v) => {
          current = v;
        },
        ab,
      },
      storage,
    );
    model.set({ strength: 1.4, patchScale: Infinity, grassLight: -1, grassChroma: 5, softness: -1, shadeDepth: 1 });
    expect(current.grassChroma).toBe(1.6);
    expect(current.softness).toBe(0.2);
    expect(current.shadeDepth).toBe(0.5);
    expect(current.strength).toBe(1.4);
    expect(current.patchScale).toBe(recommended.patchScale);
    expect(current.grassLight).toBe(0.7);
    model.reset();
    expect(model.get()).toEqual(recommended);
    const write = vi.fn(() => {
      throw new Error('quota');
    });
    const broken = tuningModel(
      {
        schema: LUSH2_SCHEMA,
        recommended,
        storageKey: 'snow',
        get: () => current,
        set: (v) => {
          current = v;
        },
        ab,
      },
      () => ({ getItem: () => '{corrupt', setItem: write }),
    );
    expect(() => broken.set({ contact: 0 })).not.toThrow();
    expect(write).toHaveBeenCalled();
  });
  it('persists per theatre and restores no-tone + exact URL alive across repeated A/B and edits while current', () => {
    const stored = new Map<string, string>();
    const storage = () => ({
      getItem: (k: string) => stored.get(k) ?? null,
      setItem: (k: string, v: string) => {
        stored.set(k, v);
      },
    });
    let values = lush2Defaults('temperate');
    const urlAlive = { ...ALIVE_DEFAULTS, clouds: true, preset: 'deep' as const, cloud: 0.4, speed: 0 };
    let settings: unknown;
    let alive: unknown;
    const model = tuningModel(
      {
        schema: LUSH2_SCHEMA,
        recommended: values,
        storageKey: 'temperate',
        get: () => values,
        set: (v) => {
          values = v;
        },
        ab: (on) => {
          const mapped = lush2Mapping(values, 'temperate');
          settings = { groundTone: null, tone: null, cloudTone: on ? mapped.tone : null };
          alive = on ? mapped.alive : structuredClone(urlAlive);
        },
      },
      storage,
    );
    model.set({ gold: 0.7 });
    const tuned = structuredClone(settings);
    for (let i = 0; i < 3; i++) {
      model.ab(false);
      expect(settings).toEqual({ groundTone: null, tone: null, cloudTone: null });
      expect(alive).toEqual(urlAlive);
      model.ab(true);
      expect(settings).toEqual(tuned);
    }
    model.ab(false);
    model.set({ contact: 0 });
    expect(alive).toEqual(urlAlive);
    expect(settings).toEqual({ groundTone: null, tone: null, cloudTone: null });
    expect(stored.has('snow')).toBe(false);
    const restored = tuningModel(
      {
        schema: LUSH2_SCHEMA,
        recommended: lush2Defaults('temperate'),
        storageKey: 'temperate',
        get: () => lush2Defaults('temperate'),
        set: vi.fn(),
        ab: vi.fn(),
      },
      storage,
    );
    expect(restored.get().gold).toBe(0.7);
    expect(restored.get().contact).toBe(0);
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
  isContentEditable = false;
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
  blur = vi.fn();
  setAttribute() {}
  closest() {
    return this.isContentEditable || ['input', 'textarea', 'select'].includes(this.tag) ? this : null;
  }
}
it('owns the DOM and capture hotkey, ignores inputs, supports hide and releases the listener', () => {
  const win = new EventTarget(),
    host = new ElementDouble(),
    ab = vi.fn();
  const add = vi.spyOn(win, 'addEventListener'),
    remove = vi.spyOn(win, 'removeEventListener');
  vi.stubGlobal('window', win);
  const doc = { createElement: (tag: string) => new ElementDouble(tag), activeElement: null as ElementDouble | null };
  vi.stubGlobal('document', doc);
  vi.stubGlobal('localStorage', {
    getItem: () => null,
    setItem: () => {
      throw new Error('denied');
    },
  });
  let nativeTyping = false;
  const returnFocus = vi.fn();
  const panel = mountTuningPanel(host as unknown as HTMLElement, {
    textEntryActive: () => nativeTyping,
    returnFocus,
    schema: LUSH2_SCHEMA,
    recommended: lush2Defaults('temperate'),
    storageKey: 'test',
    get: () => lush2Defaults('temperate'),
    set: vi.fn(),
    ab,
  });
  expect(host.children).toHaveLength(1);
  const root = host.children[0]!;
  expect(root.children.at(-1)!.hidden).toBe(true);
  expect(root.style).toMatchObject({
    left: '12px',
    top: '12px',
    width: '260px',
    maxHeight: 'calc(100vh - 24px)',
    overflowY: 'auto',
    boxSizing: 'border-box',
  });
  const rows = root.children.at(-1)!.children;
  expect(rows).toHaveLength(LUSH2_SCHEMA.length);
  for (const [i, field] of LUSH2_SCHEMA.entries()) {
    const row = rows[i]!;
    expect((row.style as { cssText: string }).cssText).toContain('height:34px');
    const input = row.children[1]!;
    expect([input.min, input.max, input.step]).toEqual([String(field.min), String(field.max), String(field.step)]);
    input.value = String(field.max);
    input.dispatchEvent(new Event('input'));
    expect(panel.get()[field.key]).toBe(field.max);
    for (const event of ['change', 'pointerup']) {
      const before = returnFocus.mock.calls.length;
      input.dispatchEvent(new Event(event));
      expect(input.blur).toHaveBeenCalled();
      expect(returnFocus).toHaveBeenCalledTimes(before + 1);
      expect(input.blur.mock.invocationCallOrder.at(-1)).toBeLessThan(returnFocus.mock.invocationCallOrder.at(-1)!);
    }
  }
  root.children[1]!.dispatchEvent(new Event('click'));
  expect(panel.get()).toEqual(lush2Defaults('temperate'));
  for (const button of root.children.slice(1, 4)) {
    const before = returnFocus.mock.calls.length;
    button.dispatchEvent(new Event('click'));
    expect(button.blur).toHaveBeenCalled();
    expect(returnFocus).toHaveBeenCalledTimes(before + 1);
  }
  panel.ab(true);

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
  for (const tag of ['input', 'textarea', 'select']) {
    doc.activeElement = new ElementDouble(tag);
    expect(key('Backquote').defaultPrevented).toBe(false);
  }
  doc.activeElement = new ElementDouble();
  doc.activeElement.isContentEditable = true;
  expect(key('Backquote').defaultPrevented).toBe(false);
  doc.activeElement = null;
  nativeTyping = true;
  const callsWhileTyping = ab.mock.calls.length;
  for (const shift of [false, true]) expect(key('Backquote', shift).defaultPrevented).toBe(false);
  expect(ab).toHaveBeenCalledTimes(callsWhileTyping);
  expect(root.hidden).toBe(false);
  nativeTyping = false;
  const composing = Object.assign(new Event('keydown', { cancelable: true }), { code: 'Backquote', isComposing: true });
  win.dispatchEvent(composing);
  expect(composing.defaultPrevented).toBe(false);
  const game = vi.fn();
  win.addEventListener('keydown', game, { capture: true });
  expect(key('Backquote').defaultPrevented).toBe(true);
  expect(game).not.toHaveBeenCalled();
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

it('the actual RA2 controller restores URL clouds and no-tone settings, preserving edits made during current', async () => {
  const { mountLush2Tuning } = await import('../src/ra2/lush2Tuning');
  const { AtmosphereNodes } = await import('../src/atmosphere');
  const nodes = new AtmosphereNodes();
  const host = new ElementDouble();
  const win = new EventTarget();
  vi.stubGlobal('window', win);
  vi.stubGlobal('document', { createElement: (tag: string) => new ElementDouble(tag) });
  const readStorage = vi.fn((key: string) =>
    key.startsWith('hd-lush2-v1:') || key.startsWith('hd-lush2-v4:')
      ? JSON.stringify({ ...lush2Defaults('temperate'), grassChroma: 1.2, shadeChroma: 1.25 })
      : null,
  );
  vi.stubGlobal('localStorage', { getItem: readStorage, setItem: () => {} });
  const url = { ...ALIVE_DEFAULTS, clouds: true, preset: 'deep' as const, cloud: 0.3, speed: 0.2 };
  nodes.alive.set(url);
  nodes.alive.update(20, 0.6, 0, 0, false);
  const baseline = {
    clock: nodes.alive.clock.value,
    strength: nodes.alive.cloudStrength.value,
    lattice: nodes.alive.cloudLattice.value.toArray(),
    form: nodes.alive.cloudForm.value.toArray(),
    body: nodes.alive.waterBody.value,
    sky: nodes.alive.waterSky.value,
  };
  const settings = { groundTone: null, tone: null, cloudTone: null };
  let actualAlive: unknown;
  const hd = {
    settings,
    setAlive: (change: typeof url) => {
      actualAlive = nodes.alive.set(change);
    },
    applySettings: () => {
      nodes.cloudTone.set(settings.cloudTone);
      nodes.alive.update(20, 0.6, 0, 0, false);
    },
  };
  let nativeTyping = true;
  const panel = mountLush2Tuning(
    host as unknown as HTMLElement,
    hd as unknown as import('../src/HdRenderer').HdRenderer,
    'temperate',
    url,
    undefined,
    () => nativeTyping,
  );
  try {
    expect(readStorage).toHaveBeenCalledWith('hd-lush2-v7:temperate');
    expect(panel.get()).toEqual(lush2Defaults('temperate'));
    const backquote = () => {
      const event = Object.assign(new Event('keydown', { cancelable: true }), { code: 'Backquote' });
      win.dispatchEvent(event);
      return event;
    };
    expect(backquote().defaultPrevented).toBe(false);
    expect(settings.cloudTone).not.toBeNull();
    nativeTyping = false;
    expect(backquote().defaultPrevented).toBe(true);
    expect(settings.cloudTone).toBeNull();
    for (let i = 0; i < 3; i++) {
      panel.ab(false);
      expect(settings).toEqual({ groundTone: null, tone: null, cloudTone: null });
      expect(actualAlive).toEqual(url);
      expect({
        clock: nodes.alive.clock.value,
        strength: nodes.alive.cloudStrength.value,
        lattice: nodes.alive.cloudLattice.value.toArray(),
        form: nodes.alive.cloudForm.value.toArray(),
        body: nodes.alive.waterBody.value,
        sky: nodes.alive.waterSky.value,
      }).toEqual(baseline);
      panel.set({ gold: 0.6 });
      expect(settings.cloudTone).toBeNull();
      panel.ab(true);
      expect(nodes.cloudTone.gold.value).toBe(0.6);
      expect(nodes.alive.windAmplitude.value).toBe(0);
    }
  } finally {
    panel.dispose();
    nodes.dispose();
  }
});
