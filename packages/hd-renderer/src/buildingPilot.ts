/** Explicit development gate shared by the live bridge and FX bench. */
export function parseBuildingPilot(value: string | null | undefined): boolean {
  return value === 'pilot';
}

/** CPU twin of the building-only shader coverage selection. */
export function buildingPaintCoverage(hdMask: number, indexedMask: number, building: boolean, pilot = false): number {
  return pilot && building ? hdMask : Math.max(hdMask, indexedMask);
}
