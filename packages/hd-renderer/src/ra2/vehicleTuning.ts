import type { LiveView } from '../live/liveView';
import { mountTuningPanel, type TuningField } from '../ui/tuningPanel';
import { voxelPbrFor, type VoxelPbrKnob } from './voxelPbr';

// The original voxel colour ramps supply every material boundary. No painted atlas or new parts are introduced.
export const VEHICLE_TUNED = {
  factoryDepth: 1,
  brightness: 1,
  shadeSmooth: 0.6,
  aoStrength: 0.55,
  sunGain: 0.85,
  fill: 0.7,
  specular: 0.35,
  envStrength: 0.25,
  houseSat: 1,
  houseGain: 1,
  wear: 0.12,
  cavity: 0.25,
  edgeLight: 0.04,
};
export const VEHICLE_DRAMATIC = { ...VEHICLE_TUNED, shadeSmooth: 0.25, aoStrength: 0.85, cavity: 0.5, specular: 0.7 };
type Key = keyof typeof VEHICLE_TUNED;
export const VEHICLE_SCHEMA: readonly TuningField<Key>[] = [
  ['factoryDepth', '工厂动画层深度', 0, 1],
  ['brightness', '亮度', 0.5, 1.5],
  ['shadeSmooth', '原色阶平滑', 0, 1],
  ['aoStrength', '接缝遮蔽', 0, 1],
  ['sunGain', '方向光', 0, 1.5],
  ['fill', '天空补光', 0, 1.5],
  ['specular', '金属反光', 0, 1],
  ['envStrength', '环境反射', 0, 1],
  ['houseSat', '阵营色浓度', 0.5, 1.5],
  ['houseGain', '阵营色亮度', 0.5, 1.5],
  ['wear', '原曲率磨损', 0, 0.3],
  ['cavity', '原接缝深度', 0, 0.6],
  ['edgeLight', '边缘反光', 0, 0.15],
].map(([key, label, min, max]) => ({
  key: key as Key,
  label: String(label),
  min: Number(min),
  max: Number(max),
  step: 0.01,
}));

export function parseVehicleParams(params: URLSearchParams) {
  if (params.get('hd-veh') !== '1') return null;
  const values = { ...(params.get('hd-veh-preset') === 'dramatic' ? VEHICLE_DRAMATIC : VEHICLE_TUNED) };
  for (const field of VEHICLE_SCHEMA) {
    const raw = params.get(`hd-veh-${field.key}`);
    if (raw === null || raw.trim() === '') continue;
    const value = Number(raw);
    if (Number.isFinite(value)) values[field.key] = Math.max(field.min, Math.min(field.max, value));
  }
  return values;
}

export function mountVehicleTuning(
  host: HTMLElement,
  view: LiveView,
  params: URLSearchParams,
  returnFocus: () => void,
  textEntryActive: () => boolean,
) {
  const state = voxelPbrFor(view.hd.nodes);
  // Procedural tread bands/dust/streaks invent patterns absent from the source. Disable those for this review.
  for (const key of ['tread', 'dust', 'streaks', 'grime', 'microNoise'] as const) state[key].value = 0;
  let values = parseVehicleParams(params) ?? { ...VEHICLE_TUNED };
  const panel = mountTuningPanel(host, {
    schema: VEHICLE_SCHEMA,
    recommended: VEHICLE_TUNED,
    storageKey: 'hd-veh-v1',
    title: '高清车辆',
    names: ['新车辆', '当前'],
    hotkey: 'Backquote',
    corner: 'top-right',
    returnFocus,
    textEntryActive,
    get: () => values,
    set: (next) => {
      values = next;
      for (const [key, value] of Object.entries(next))
        if (key !== 'factoryDepth') state[key as VoxelPbrKnob].value = value;
    },
    ab: (enabled) => {
      view.setVehicleRefinement(enabled);
      view.setProductionDepth(enabled ? values.factoryDepth : 0);
    },
  });
  // An explicit URL review takes precedence over persisted panel edits.
  if ([...params.keys()].some((key) => key.startsWith('hd-veh-'))) panel.set(parseVehicleParams(params)!);
  const root = host.querySelector('section[aria-label="高清车辆"]')!;
  const preset = document.createElement('select');
  preset.setAttribute('aria-label', '车辆预设');
  for (const [value, label] of [
    ['tuned', '微调'],
    ['dramatic', '强化'],
  ] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    preset.append(option);
  }
  preset.value = params.get('hd-veh-preset') === 'dramatic' ? 'dramatic' : 'tuned';
  const change = () => {
    panel.set(preset.value === 'dramatic' ? VEHICLE_DRAMATIC : VEHICLE_TUNED);
    preset.blur();
    returnFocus();
  };
  preset.addEventListener('change', change);
  root.insertBefore(preset, root.firstChild);
  return {
    ...panel,
    dispose() {
      preset.removeEventListener('change', change);
      panel.dispose();
    },
  };
}
