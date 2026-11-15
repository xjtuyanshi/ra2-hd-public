import { DEFAULT_PAINTED_STEEL_MATERIALS, type HardSurfaceLookParameters, type LookVector } from '../hardSurfaceLook';

/** Chroma/brightness bounds are independent of the explicitly selected player colour. */
export const RA2_CAD_HOUSE_LIMITS = { peak: 0.72, trough: 0.17 } as const;

/** Vehicle look policy for offline reconstruction; geometry and fitted specs remain external. */
export const RA2_HARD_SURFACE_LOOK: HardSurfaceLookParameters = {
  cream: [0.64, 0.615, 0.565],
  houseTone: [0.6, 0.6, 0.6],
  hatchColour: [0.36, 0.38, 0.43],
  barrelSteel: [0.42, 0.44, 0.47],
  wear: 0.3,
  wallOcclusion: 0.45,
  wallGrime: 0.32,
  lines: { width: 0.42, aoWidth: 1 },
  seams: { minLength: 2, maxGap: 1.6, minOverlap: 0.5 },
  paintCodes: { auto: 0, house: 1, steel: 2, flat: 3, wheel: 4, drum: 5, hatch: 6 },
  materials: DEFAULT_PAINTED_STEEL_MATERIALS,
};

/** A player colour is an explicit input, never inferred from the unit's faction or original remap RGB. */
export function ra2HardSurfaceLook(houseTone: LookVector): HardSurfaceLookParameters {
  if (!houseTone.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) throw new Error('Invalid house colour');
  return { ...structuredClone(RA2_HARD_SURFACE_LOOK), houseTone: [...houseTone] };
}
