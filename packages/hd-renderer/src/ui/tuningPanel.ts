export interface TuningField<K extends string = string> {
  key: K;
  label: string;
  min: number;
  max: number;
  step: number;
}
export interface TuningOptions<K extends string> {
  schema: readonly TuningField<K>[];
  recommended: Record<K, number>;
  storageKey: string;
  get(): Record<K, number>;
  set(values: Record<K, number>): void;
  ab(enabled: boolean): void;
  names?: readonly [string, string];
  hotkey?: string;
  corner?: 'top-right' | 'bottom-right';
  title?: string;
  /** Host-provided text-entry state, observed or inferred from forwarded input. */
  textEntryActive?(): boolean;
  /** Return keyboard ownership to the host after a completed control action. */
  returnFocus?(): void;
}
/** Storage is optional, including environments where merely accessing localStorage throws. */
export function tuningModel<K extends string>(
  options: TuningOptions<K>,
  storage: () => Pick<Storage, 'getItem' | 'setItem'> = () => localStorage,
) {
  let values = { ...options.get() };
  let enabled = true;
  const clean = (partial: Partial<Record<K, number>>) => {
    for (const f of options.schema) {
      const value = partial[f.key];
      if (typeof value === 'number' && Number.isFinite(value)) values[f.key] = Math.min(f.max, Math.max(f.min, value));
    }
  };
  try {
    clean(JSON.parse(storage().getItem(options.storageKey) ?? '{}'));
  } catch {
    /* Private mode and corrupt records keep the caller's defaults. */
  }
  const apply = () => {
    options.set({ ...values });
    options.ab(enabled);
  };
  apply();
  return {
    get: () => ({ ...values }),
    isNew: () => enabled,
    set(partial: Partial<Record<K, number>>) {
      clean(partial);
      try {
        storage().setItem(options.storageKey, JSON.stringify(values));
      } catch {
        /* Tuning still works without persistence. */
      }
      apply();
    },
    ab(next: boolean) {
      enabled = next;
      options.ab(next);
    },
    reset() {
      this.set(options.recommended);
    },
  };
}
/** Plain DOM development overlay. The returned owner removes every DOM node and listener it creates. */
export function mountTuningPanel<K extends string>(host: HTMLElement, options: TuningOptions<K>) {
  const model = tuningModel(options);
  const root = document.createElement('section');
  root.setAttribute('aria-label', options.title ?? '画面调节');
  Object.assign(root.style, {
    position: 'absolute',
    ...(options.corner === 'bottom-right'
      ? { bottom: '12px', right: '12px' }
      : options.corner === 'top-right'
        ? { top: '12px', right: '12px' }
        : { top: '12px', left: '12px' }),
    zIndex: '10000',
    width: '260px',
    maxHeight: 'calc(100vh - 24px)',
    overflowY: 'auto',
    boxSizing: 'border-box',
    background: '#101a22ee',
    color: '#fff',
    padding: '12px',
    borderRadius: '8px',
    font: '13px sans-serif',
    pointerEvents: 'auto',
  });
  const removers: (() => void)[] = [];
  const listen = (target: EventTarget, key: string, listener: EventListener, capture = false) => {
    const options = { capture };
    target.addEventListener(key, listener, options);
    removers.push(() => target.removeEventListener(key, listener, options));
  };
  const state = document.createElement('strong');
  state.style.cssText = 'display:block;font-size:25px;margin-bottom:8px';
  const body = document.createElement('div');
  body.hidden = true;
  const release = (control: HTMLElement) => {
    control.blur();
    options.returnFocus?.();
  };
  const inputs = new Map<K, { input: HTMLInputElement; output: HTMLOutputElement }>();
  const refresh = () => {
    state.textContent = (options.names ?? ['润r3', '当前'])[model.isNew() ? 0 : 1];
    const values = model.get();
    for (const [key, { input, output }] of inputs) {
      input.value = String(values[key]);
      output.textContent = String(values[key]);
    }
  };
  const button = (label: string, action: () => void) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.style.margin = '2px';
    listen(b, 'click', () => {
      action();
      refresh();
      release(b);
    });
    return b;
  };
  root.append(
    state,
    button('恢复推荐值', () => model.reset()),
    button('新/旧 A/B', () => model.ab(!model.isNew())),
    button('展开/折叠', () => {
      body.hidden = !body.hidden;
    }),
    body,
  );
  for (const field of options.schema) {
    const row = document.createElement('label');
    row.style.cssText =
      'display:grid;grid-template-columns:1fr auto;grid-template-rows:16px 18px;height:34px;align-items:center';
    row.textContent = field.label + ' ';
    const output = document.createElement('output');
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(field.min);
    input.max = String(field.max);
    input.step = String(field.step);
    input.style.cssText = 'grid-column:1 / -1;width:100%;height:16px;margin:0';
    listen(input, 'input', () => {
      model.set({ [field.key]: Number(input.value) } as Partial<Record<K, number>>);
      refresh();
    });
    listen(input, 'change', () => release(input));
    listen(input, 'pointerup', () => release(input));
    inputs.set(field.key, { input, output });
    row.append(output, input);
    body.append(row);
  }
  listen(
    window,
    'keydown',
    (event) => {
      const e = event as KeyboardEvent;
      const target = e.target as HTMLElement | null;
      const focused = document.activeElement as HTMLElement | null;
      if (
        e.isComposing ||
        options.textEntryActive?.() ||
        target?.isContentEditable ||
        focused?.isContentEditable ||
        target?.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"])') ||
        focused?.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"])')
      )
        return;
      if (e.code !== (options.hotkey ?? 'Backquote') || e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (e.repeat) return;
      if (e.shiftKey) root.hidden = !root.hidden;
      else model.ab(!model.isNew());
      refresh();
    },
    true,
  );
  host.append(root);
  refresh();
  return {
    get: model.get,
    set(partial: Partial<Record<K, number>>) {
      model.set(partial);
      refresh();
    },
    ab(enabled: boolean) {
      model.ab(enabled);
      refresh();
    },
    dispose() {
      for (const remove of removers) remove();
      root.remove();
    },
  };
}
