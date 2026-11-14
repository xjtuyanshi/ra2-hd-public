import { Fn, If, float, luminance, max, mix, vec3 } from 'three/tsl';
import type { ShaderNode } from './tsl';
import type { RealLightNodes, Rgb } from './realLight';

/**
 * Night integration defaults, in linear art space. Live uniforms may override these without rebuilding materials.
 * Sandbox calibration (SOV07TMD night, owner resolution, 2026-11-14): with base 0.24 the shoulder took 17 L* off the
 * GIs' 95th percentile and 34 L* off the projected Grizzlies (whose redraw colour carries the page exposure and
 * gain, about five times the sprite scale), flattening their texture. Base 0.38 on sprite art and a 0.21 luminance
 * scale on projected redraws bring both to about -7..-10 L* at p95 with the body mean within about 2 L*.
 */
export const NIGHT_ART_DEFAULTS = {
  rolloff: 0.8,
  base: 0.38,
  projectedScale: 0.21,
  shoulder: 0.12,
  wrap: 0.35,
  ao: 0.4,
  unitSoftness: 0.4,
  wallSoftness: 0.3,
} as const;

export function nightHighlightScale(
  y: number,
  base: number = NIGHT_ART_DEFAULTS.base,
  amount: number = NIGHT_ART_DEFAULTS.rolloff,
): number {
  const excess = Math.max(0, y - Math.max(0, base));
  return (
    1 -
    (Math.min(1, Math.max(0, amount)) * excess * excess) / ((excess + NIGHT_ART_DEFAULTS.shoulder) * Math.max(y, 1e-8))
  );
}
export function nightArtColour(
  scene: Rgb,
  albedo: Rgb,
  ambient: Rgb,
  rim: number,
  night: number,
  amount: number = NIGHT_ART_DEFAULTS.rolloff,
  wrap: number = NIGHT_ART_DEFAULTS.wrap,
): Rgb {
  if (night === 0) return scene;
  const y = albedo[0] * 0.2126 + albedo[1] * 0.7152 + albedo[2] * 0.0722;
  const scale = 1 + (nightHighlightScale(y, NIGHT_ART_DEFAULTS.base, amount) - 1) * night;
  return scene.map((v, i) => {
    const graded = v * scale;
    return (
      graded +
      (Math.min(graded, ambient[i]!) - graded) * Math.min(1, Math.max(0, rim)) * Math.min(1, Math.max(0, wrap)) * night
    );
  }) as unknown as Rgb;
}
export const alphaRim = (neighbours: readonly number[]): number =>
  1 - Math.min(1, ...neighbours.map((a) => Math.max(0, a)));
export const nightSharpen = (sharpen: number, night: number): number => sharpen * (1 - night);
export const nightArtWeight = (nodes: RealLightNodes, night: ShaderNode): ShaderNode =>
  nodes.enabled.mul(night.clamp());

/** Only scene radiance enters this function. Local irradiance and emissive panes are added by the caller afterwards. */
export function nightArtNode(
  scene: ShaderNode,
  art: ShaderNode,
  ambient: ShaderNode,
  rim: ShaderNode,
  nodes: RealLightNodes,
  night: ShaderNode,
  artScale?: ShaderNode,
): ShaderNode {
  return Fn(() => {
    const result = scene.toVar();
    const weight = nightArtWeight(nodes, night);
    // Exact identity at night=0, including day highlights: no roundoff from a redundant grade or colour mix.
    If(weight.greaterThan(0), () => {
      const y = artScale ? luminance(art).mul(artScale) : luminance(art);
      const excess = max(y.sub(nodes.artBase.max(0)), 0);
      const scale = float(1).sub(
        nodes.artRolloff
          .clamp()
          .mul(excess.mul(excess))
          .div(excess.add(NIGHT_ART_DEFAULTS.shoulder).mul(max(y, 1e-8)))
          .mul(weight),
      );
      const graded = result.mul(scale);
      result.assign(mix(graded, ambient.min(graded), rim.clamp().mul(nodes.artWrap.clamp()).mul(weight)));
    });
    return result;
  })();
}

/** Shared ambient target: the pixel's own albedo under the receiver's scene light, never a sampled bright neighbour. */
export function nightAmbient(art: ShaderNode, sceneLight: ShaderNode, moonGrey: ShaderNode): ShaderNode {
  return vec3(luminance(art)).mul(moonGrey).mul(sceneLight).mul(0.5);
}
