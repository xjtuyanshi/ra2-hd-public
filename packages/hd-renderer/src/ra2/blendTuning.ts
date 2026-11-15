import type { HdRenderer } from '../HdRenderer';
import type { AliveOptions } from '../livingWorld';
import { mountTuningPanel } from '../ui/tuningPanel';
import { LUSH2_SCHEMA } from './lush2Tone';
import { applyBlendTone, blendDefaults, parseBlendPreset, type BlendPreset } from './blendTone';

/** Session-owned overlay: presets and sliders share the exact runtime recipe, including its light tints. */
export function mountBlendTuning(
  host: HTMLElement,
  hd: HdRenderer,
  theater: string,
  currentAlive: AliveOptions,
  params: URLSearchParams,
  returnFocus?: () => void,
  textEntryActive?: () => boolean,
) {
  let preset = parseBlendPreset(params.get('hd-blend-preset'));
  let values = blendDefaults(theater, preset);
  let enabled = true;
  const matchingPreset = () =>
    (['dramatic', 'tuned'] as const).find((p) =>
      Object.entries(blendDefaults(theater, p)).every(([key, value]) => values[key as keyof typeof values] === value),
    ) ?? 'custom';
  let select: HTMLSelectElement | undefined;
  const panel = mountTuningPanel(host, {
    schema: LUSH2_SCHEMA,
    recommended: blendDefaults(theater),
    get storageKey() {
      return `hd-blend-C2-v2:${theater}:${preset}`;
    },
    names: ['融合', '当前'],
    returnFocus,
    textEntryActive,
    get: () => ({ ...values }),
    set: (next) => {
      values = { ...next };
      if (select) select.value = matchingPreset();
    },
    ab: (next) => {
      enabled = next;
      if (enabled) hd.setAlive(applyBlendTone(hd.settings, theater, preset, values));
      else {
        hd.settings.cloudTone = null;
        hd.settings.groundTone = null;
        hd.settings.tone = null;
        hd.setAlive(currentAlive);
      }
      hd.applySettings();
    },
  });
  const root = host.querySelector('section[aria-label="画面调节"]')!;
  select = document.createElement('select');
  select.setAttribute('aria-label', '融合预设');
  for (const [value, text] of [
    ['dramatic', 'dramatic · 强化'],
    ['tuned', 'tuned · 柔和'],
    ['custom', 'custom · 自定义'],
  ]) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    select.append(option);
  }
  select.value = matchingPreset();
  const change = () => {
    if (select!.value === 'custom') return;
    preset = select!.value as BlendPreset;
    panel.set(blendDefaults(theater, preset));
    select!.value = preset;
    select!.blur();
    returnFocus?.();
  };
  const resetPreset = () => {
    preset = 'dramatic';
  };
  const recommendedButton = root.querySelector('button')!;
  recommendedButton.addEventListener('click', resetPreset, true);
  select.addEventListener('change', change);
  root.prepend(select);
  let removePolish = () => {};
  if (params.get('hd-bld') === '1') {
    const row = document.createElement('label');
    row.textContent = '建筑细噪 [buildingPolish] ';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1';
    input.step = '.01';
    input.value = String(hd.surfacePolishAmount.value);
    input.style.width = '100%';
    const output = document.createElement('output');
    output.textContent = input.value;
    const update = () => {
      hd.surfacePolishAmount.value = Number(input.value);
      output.textContent = input.value;
    };
    const release = () => {
      input.blur();
      returnFocus?.();
    };
    const reset = () => {
      input.value = '1';
      update();
    };
    input.addEventListener('input', update);
    input.addEventListener('change', release);
    input.addEventListener('pointerup', release);
    const resetButton = root.querySelector('button')!;
    resetButton.addEventListener('click', reset, true);
    row.append(output, input);
    root.querySelector('div')!.append(row);
    removePolish = () => {
      input.removeEventListener('input', update);
      input.removeEventListener('change', release);
      input.removeEventListener('pointerup', release);
      resetButton.removeEventListener('click', reset, true);
    };
  }
  return {
    ...panel,
    dispose() {
      recommendedButton.removeEventListener('click', resetPreset, true);
      removePolish();
      select!.removeEventListener('change', change);
      panel.dispose();
    },
  };
}
