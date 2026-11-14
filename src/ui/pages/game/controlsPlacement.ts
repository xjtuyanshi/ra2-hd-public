/** A screen rectangle in CSS pixels (the subset of DOMRect the placement rule reads). */
export interface PlacementRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/**
 * Whether the expanded control panel, a fixed column `panelWidth` CSS px wide whose top-left corner is the folded
 * panel's, would lie over the game picture. A picture letterboxed far enough to the right leaves the panel beside it,
 * the only case in which the expanded panel may stay open during play without hiding part of the game.
 */
export function controlsCoverPicture(panel: PlacementRect, panelWidth: number, picture: PlacementRect): boolean {
  if (!(panelWidth > 0) || picture.right <= picture.left || picture.bottom <= picture.top) return false;
  return picture.left < panel.left + panelWidth && picture.right > panel.left && picture.bottom > panel.top;
}
