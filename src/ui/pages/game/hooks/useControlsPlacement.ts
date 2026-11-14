import { useEffect, useRef, useState } from 'react';
import { controlsCoverPicture } from '../controlsPlacement';
import { setControlsCollapsed } from '../gameInput';

/**
 * Keep the control panel off the running game picture. While the game runs and the expanded panel would lie over the
 * picture (a window the game fills, a phone), the panel folds to its small handle when that starts and whenever the
 * player presses on the game; the handle opens it on demand. A picture with room to its left keeps the panel open
 * beside it, and a panel this hook folded opens again once a resized window leaves that room. Only layout changes
 * and presses on the game run this; neither frame state nor pointer motion passes through React.
 */
export function useControlsPlacement(
  panel: HTMLElement | null,
  canvas: HTMLCanvasElement | null,
  running: boolean,
  collapsed: boolean,
): void {
  const [covers, setCovers] = useState(false);
  // Set when this hook folded the panel, so a wider window can restore it; a player's own fold stays folded.
  const autoFolded = useRef(false);
  useEffect(() => {
    if (!canvas || !panel) return;
    const measure = () => {
      const width = parseFloat(getComputedStyle(panel).getPropertyValue('--vm-controls-width'));
      setCovers(controlsCoverPicture(panel.getBoundingClientRect(), width, canvas.getBoundingClientRect()));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(canvas);
    const stage = canvas.closest('#stage');
    if (stage) observer.observe(stage);
    window.addEventListener('resize', measure);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [canvas, panel]);
  useEffect(() => {
    if (running && covers) {
      autoFolded.current = true;
      setControlsCollapsed(true);
    } else if (!covers && autoFolded.current) {
      autoFolded.current = false;
      setControlsCollapsed(false);
    }
  }, [running, covers]);
  useEffect(() => {
    if (!canvas || !running || !covers || collapsed) return;
    // Capture phase and passive: the game's own pointerdown handler still receives the press unchanged.
    const fold = () => {
      autoFolded.current = true;
      setControlsCollapsed(true);
    };
    canvas.addEventListener('pointerdown', fold, { capture: true, passive: true });
    return () => canvas.removeEventListener('pointerdown', fold, { capture: true });
  }, [canvas, running, covers, collapsed]);
}
