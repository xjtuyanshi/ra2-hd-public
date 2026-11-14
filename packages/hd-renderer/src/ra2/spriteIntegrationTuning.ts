import {
  spriteIntegrationFor,
  type IntegrationKey,
  type IntegrationValues,
  type SpriteIntegration,
} from '../spriteIntegration';
import { mountTuningPanel, type TuningField } from '../ui/tuningPanel';

export const RECOMMENDED: IntegrationValues = {
  on: 1,
  edgeSigma: 0.95,
  edgeSoft: 1,
  thinKeep: 0.56,
  paint: 1,
  deHalo: 0.7,
  soften: 0.35,
  highlightRoll: 0.75,
  relight: 0.8,
  specular: 0.45,
  normalSmooth: 1,
  rimFade: 1,
  shadowSoft: 1.5,
  objSharp: 1,
  objClarity: 0.8,
  objSoft: 0.25,
  objContrast: 0.12,
  outlineFeather: 1.2,
  vSharpen: 1,
  vShine: 0.7,
  vCut: 0.5,
  treeDetail: 1,
};
export const SPRITE_INTEGRATION_SCHEMA: readonly TuningField<IntegrationKey>[] = [
  { key: 'on', label: '开关 [on]', min: 0, max: 1, step: 1 },
  { key: 'edgeSigma', label: '轮廓平滑 [edgeSigma]', min: 0.5, max: 1.4, step: 0.01 },
  { key: 'edgeSoft', label: '边缘柔和(设备像素) [edgeSoft]', min: 0.5, max: 3, step: 0.01 },
  { key: 'thinKeep', label: '细节保护 [thinKeep]', min: 0.4, max: 0.75, step: 0.01 },
  ...(
    [
      ['paint', '房屋色平滑'],
      ['deHalo', '去晕'],
      ['soften', '柔化'],
      ['highlightRoll', '高光收敛'],
      ['relight', '重光照'],
      ['specular', '反光'],
      ['normalSmooth', '法线平滑'],
      ['rimFade', '轮廓渐隐'],
      ['objSharp', '对象去锐化'],
      ['objClarity', '对象去清晰度'],
      ['objSoft', '对象柔化'],
      ['objContrast', '对象对比收敛'],
      ['vSharpen', '坦克去锐化'],
      ['vShine', '坦克去假反光'],
      ['treeDetail', '树冠细节保护'],
    ] as const
  ).map(([key, label]) => ({ key, label: `${label} [${key}]`, min: 0, max: 1, step: 0.01 })),
  { key: 'outlineFeather', label: '轮廓羽化(设备像素) [outlineFeather]', min: 0, max: 3, step: 0.01 },
  { key: 'vCut', label: '坦克轮廓裁切 [vCut]', min: 0, max: 0.9, step: 0.01 },
  { key: 'shadowSoft', label: '投影柔边 [shadowSoft]', min: 0, max: 3, step: 0.01 },
];
export function configureSpriteIntegration(owner: object, enabled: boolean): void {
  const group = spriteIntegrationFor(owner);
  group.compiled = enabled;
  if (enabled) group.set(RECOMMENDED);
}
/** Public controller works without a DOM panel (production and untuned sandbox). */
export function spriteIntegrationAPI(group: SpriteIntegration) {
  return {
    get: () => group.get(),
    set: (partial: Partial<IntegrationValues>) => {
      const next: Partial<IntegrationValues> = {};
      for (const f of SPRITE_INTEGRATION_SCHEMA) {
        const value = partial[f.key];
        if (typeof value === 'number' && Number.isFinite(value)) next[f.key] = Math.min(f.max, Math.max(f.min, value));
      }
      group.set(next);
    },
    ab: (on: boolean) => group.ab(on),
  };
}
export function mountSpriteIntegrationTuning(host: HTMLElement, group: SpriteIntegration) {
  // Keep A/B state separate from saved parameters, including edits made while viewing the current picture.
  let values = group.get();
  let enabled = true;
  const panel = mountTuningPanel(host, {
    schema: SPRITE_INTEGRATION_SCHEMA,
    recommended: RECOMMENDED,
    storageKey: 'hd-sprite-v2',
    names: ['新', '当前'],
    hotkey: 'Backslash',
    corner: 'bottom-right',
    title: '精灵融合',
    get: () => values,
    set: (next) => {
      values = next;
      group.set({ ...next, on: enabled ? next.on : 0 });
    },
    ab: (on) => {
      enabled = on;
      group.set({ on: on ? values.on : 0 });
    },
  });
  return { ...panel, get: () => group.get() };
}
