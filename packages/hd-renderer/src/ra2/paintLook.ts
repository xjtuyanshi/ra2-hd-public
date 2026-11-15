/**
 * House paint on HD building pages (opt-in material decode, `hd-paint=material` in game, `paint=material` in the FX
 * bench). Classic remains the default until the owner approves the material look.
 *
 * Classic: inside the paint mask the HD texel's encoded grey becomes a remap index and the owner's 16-step ramp colour
 * replaces the redraw entirely (sprites.ts). Only luminance survives, so worn edges, grime and the sheen of painted
 * metal read as a flat, saturated sticker, and the bake's binary mask leaves its DarkBlue design colour on the texels
 * just outside the mask.
 *
 * Material keeps the ramp's luminance (the redraw's shading) and adds the redraw's own local structure as material:
 * texels darker than their paint neighbourhood take a desaturated warm grime, lighter ones a desaturated worn metal,
 * the brightest encoded steps a neutral sheen, and the whole paint is capped below the ramp's full saturation (warm
 * houses such as Gold keep theirs). Bluish texels on original paint texels but outside the HD mask (the bake's
 * DarkBlue fringe) are rotated to the owner's hue at their own luminance and chroma. The paint takes the same night desaturation as the building.
 */
import { clamp, dot, exp2, float, max, min, mix, normalize, smoothstep, vec3 } from 'three/tsl';
import type { ShaderNode } from '../tsl';

export const PAINT_STYLES = ['classic', 'material'] as const;
export type PaintStyle = (typeof PAINT_STYLES)[number];

export function parsePaintStyle(value: string | null | undefined): PaintStyle {
  return PAINT_STYLES.find((style) => style === value) ?? 'classic';
}

/** Chosen before the world is built (LiveView.create), like waterPreset; read when sprite layers are made. */
export const paintPreset: { style: PaintStyle; buildingPilot: boolean } = { style: 'classic', buildingPilot: false };

/**
 * Tuning of the material decode (linear colour; `d` is the texel's remap index minus its paint neighbourhood's mean,
 * positive when darker). Measured on the temperate GA/NA/YA bodies against every multiplayer house colour (see the
 * remaster base_fix notes): grime and wear stay below about half, so the ramp's hue still dominates the panel.
 */
export const PAINT_MATERIAL = {
  /** Saturation kept from the ramp colour (1 = the engine's ramp). */
  saturation: 0.86,
  /** |d| (remap steps) where grime / wear starts and where it is full. */
  detailFrom: 0.6,
  detailTo: 3,
  grime: 0.5,
  wear: 0.22,
  /** Encoded steps from the brightest where the neutral sheen fades in, and its strength. */
  sheenFrom: 0.6,
  sheen: 0.12,
  /** Grime tint (times the texel's luminance) and worn-metal brightness. */
  dirt: [0.95, 0.88, 0.78] as const,
  metal: 1.1,
  /** Extra luminance contrast per remap step of d (exp2), clamped. */
  contrast: 0.15,
  contrastMin: 0.7,
  contrastMax: 1.4,
  /** HD texels from the centre of the four neighbourhood taps (bilinear, so each tap averages 2x2). */
  tapOffset: 1.5,
  /**
   * Warm houses (Gold, Orange, yellow-ish): how far `warmth` (0..1, min(r, g) - b over max(r, g, b) of the ramp
   * colour, smoothstepped over warmFrom..warmTo) keeps them. At full warmth the saturation goes back to 1, the grime and
   * wear mixes are scaled by these factors and the grime is a neutral darkening (a warm-brown grime on yellow reads as
   * olive, and desaturation turns Gold into cream; red, blue, green and purple are not warm and keep the values above).
   */
  warmFrom: 0.15,
  warmTo: 0.45,
  warmGrime: 0.5,
  warmWear: 0.5,
  warmDirt: 0.88,
} as const;

/**
 * Chroma direction (linear Rec.709, luminance removed, unit length) of DarkBlue, the house every building redraw was
 * composed with (ra2-assets/python/building_redraw.py comp2x): its fringe outside the mask has this hue.
 */
const DESIGN_HOUSE_CHROMA = vec3(-0.2778, -0.0144, 0.9605);
const REC709 = vec3(0.2126, 0.7152, 0.0722);

/**
 * Material house colour: `house` is the ramp colour at the texel's own index, `detail` the index minus its paint
 * neighbourhood's mean (0 where the index is not the HD page's own), `encoded` the index's step from the brightest.
 */
export function materialPaint(house: ShaderNode, detail: ShaderNode, encoded: ShaderNode): ShaderNode {
  const p = PAINT_MATERIAL;
  const y: ShaderNode = dot(house, REC709).toVar('paintLuma');
  const top: ShaderNode = max(max(house.x, house.y), house.z);
  const warmth: ShaderNode = smoothstep(
    p.warmFrom,
    p.warmTo,
    min(house.x, house.y).sub(house.z).div(max(top, 1e-4)),
  ).toVar('paintWarmth');
  const base: ShaderNode = mix(vec3(y), house, mix(float(p.saturation), float(1), warmth));
  const grime: ShaderNode = smoothstep(p.detailFrom, p.detailTo, detail)
    .mul(p.grime)
    .mul(mix(float(1), float(p.warmGrime), warmth));
  const wear: ShaderNode = max(
    smoothstep(p.detailFrom, p.detailTo, detail.negate()).mul(p.wear),
    float(1)
      .sub(smoothstep(0, p.sheenFrom, encoded))
      .mul(p.sheen),
  ).mul(mix(float(1), float(p.warmWear), warmth));
  // Warm houses darken their own colour instead of taking the warm-brown grime.
  const dirt: ShaderNode = mix(vec3(...p.dirt).mul(y), base.mul(p.warmDirt), warmth);
  const dirty: ShaderNode = mix(base, dirt, grime);
  const worn: ShaderNode = mix(dirty, vec3(y.mul(p.metal)), wear);
  return worn.mul(clamp(exp2(detail.mul(-p.contrast)), p.contrastMin, p.contrastMax));
}

/**
 * The bake's DarkBlue fringe next to the paint, recoloured: a bluish texel (`rgb`, linear) takes the owner's hue
 * (`house`, any ramp colour) at its own luminance and chroma, by `near` (0..1). sprites.ts gives `near` only to HD
 * texels outside the HD mask whose own original 1x texel is a remap index (16..31): Allied lavender chrome beside the
 * paint (GAREFN's domes on the red bar, GAWEAP's tower plate) is bluish too but is not house paint in the original, so
 * it must keep its colour. The art farm's clean pass moves the fringe outside the 1x paint to the original's chroma.
 */
export function recolourDesignFringe(rgb: ShaderNode, house: ShaderNode, near: ShaderNode): ShaderNode {
  const y: ShaderNode = dot(rgb, REC709);
  const chroma: ShaderNode = rgb.sub(y).toVar('fringeChroma');
  const size: ShaderNode = chroma.length();
  const bluish: ShaderNode = smoothstep(0.35, 0.75, dot(chroma, DESIGN_HOUSE_CHROMA).div(max(size, 1e-5))).mul(
    smoothstep(0.004, 0.02, size),
  );
  const houseChroma: ShaderNode = house.sub(dot(house, REC709));
  const turned: ShaderNode = vec3(y).add(
    normalize(houseChroma.add(vec3(1e-6))).mul(size.mul(PAINT_MATERIAL.saturation)),
  );
  return mix(rgb, max(turned, vec3(0)), bluish.mul(near));
}
