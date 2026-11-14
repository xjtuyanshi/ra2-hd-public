import { describe, expect, it } from 'vitest';
import { controlsCoverPicture } from '../../src/ui/pages/game/controlsPlacement';

// The folded handle's corner: 10 CSS px in from the top-left of the window.
const handle = { left: 10, top: 10, right: 36, bottom: 36 };

describe('控制面板是否盖住游戏画面', () => {
  it('游戏铺满窗口时（机主默认视野 1728×1030 内容区）展开面板会盖住画面', () => {
    expect(controlsCoverPicture(handle, 176, { left: 0, top: 0, right: 1728, bottom: 1030 })).toBe(true);
    expect(controlsCoverPicture(handle, 144, { left: 0, top: 0, right: 390, bottom: 844 })).toBe(true);
  });

  it('画面左侧留白够宽时面板在画面旁边，不算盖住', () => {
    // 4:3 picture centred in a 1440×900 window: 270 px of margin on each side.
    expect(controlsCoverPicture(handle, 176, { left: 270, top: 0, right: 1170, bottom: 900 })).toBe(false);
    // Exactly touching is beside, one pixel less is over the picture.
    expect(controlsCoverPicture(handle, 176, { left: 186, top: 0, right: 1000, bottom: 900 })).toBe(false);
    expect(controlsCoverPicture(handle, 176, { left: 185, top: 0, right: 1000, bottom: 900 })).toBe(true);
  });

  it('画面尚未布局或宽度未知时不收起', () => {
    expect(controlsCoverPicture(handle, 176, { left: 0, top: 0, right: 0, bottom: 0 })).toBe(false);
    expect(controlsCoverPicture(handle, Number.NaN, { left: 0, top: 0, right: 800, bottom: 600 })).toBe(false);
    expect(controlsCoverPicture(handle, 176, { left: 0, top: 0, right: 800, bottom: 5 })).toBe(false);
  });
});
