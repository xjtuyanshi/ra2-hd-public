/** Frozen f2542b5 pixel recipe for same-build Sunday A/B regressions. */
import { cloudObjectSnow, cloudToneGain, cloudGreenness, type CloudToneParams } from '../../src/cloudTone';
const W = [0.2126, 0.7152, 0.0722] as const;
const blend = (a: number, b: number, t: number) => a + (b - a) * t;
const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const luma = (rgb: readonly number[]) => rgb.reduce((s, x, i) => s + x * W[i]!, 0);
export function r3CloudToneRgb(
  rgb: readonly [number, number, number],
  shade: number,
  p: CloudToneParams,
  night = 0,
  mean: readonly [number, number, number] = rgb,
  object = false,
  wet = 0,
  height = 0,
  context: readonly [number, number, number] = mean,
): [number, number, number] {
  const amount = p.strength * blend(1, p.nightKeep, night);
  if (amount === 0) return [...rgb];
  const y = luma(rgb),
    my = Math.max(luma(mean), 1e-6),
    lit = 1 - shade;
  if (object) {
    const src = cloudObjectSnow(rgb, p, night);
    const tinted = src.map(
      (v, i) =>
        v *
        blend(1, p.shadeTint[i]!, shade) *
        blend(1, p.moonTint[i]!, night) *
        blend(1, p.grassSunTint[i]! ** 0.25, p.bounce * (1 - smooth(0, p.bounceHeight, height)) * (1 - night)),
    );
    const sy = luma(src);
    const shoulder = 1 - (p.objectShoulder * Math.max(0, sy - p.objectHighlight)) / Math.max(sy, 1e-6);
    return tinted.map((v, i) =>
      Math.max(
        0,
        blend(
          rgb[i]!,
          ((v * sy) / Math.max(luma(tinted), 1e-6)) *
            Math.max(p.objectFloor, blend(1, p.grassLight, p.objectShare)) *
            shoulder,
          amount,
        ),
      ),
    ) as [number, number, number];
  }
  if (p.snow > 0) {
    const gain = cloudToneGain(mean, context, shade, p, night);
    return rgb.map((v, i) => Math.max(0, v * blend(1, gain[i]!, amount))) as [number, number, number];
  }
  const gain = cloudToneGain(mean, context, shade, p, night);
  const detailed = rgb.map((v, i) => v * gain[i]!);
  const materialGreen = cloudGreenness(mean, p);
  if (p.detail !== 1) {
    const contrast = (Math.max(y, 1e-6) / my) ** ((p.detail - 1) * (1 - night));
    detailed.forEach((v, i) => {
      detailed[i] = v * contrast;
    });
  }
  const wetGate = smooth(0, 0.3, wet);
  const wetShare = p.wet * wet * (1 - night);
  const tintY = Math.max(luma(p.wetTint), 1e-6);
  detailed.forEach((v, i) => {
    detailed[i] = blend(v, rgb[i]!, wetGate) * blend(1, (p.wetTint[i]! * (1 - p.wetDepth)) / tintY, wetShare);
  });
  const wetY = luma(detailed);
  detailed.forEach((v, i) => {
    detailed[i] = wetY + (v - wetY) * blend(1, p.wetChroma, wetShare);
  });
  const gold =
    p.gold *
    materialGreen *
    (1 - wetGate) *
    lit ** 2 *
    (1 - night) *
    smooth(p.highlightGate[2], p.highlightGate[3], y / my);
  const warm = detailed.map((v, i) => v * p.goldTint[i]!);
  const warmY = Math.max(luma(warm), 1e-6),
    outY = luma(detailed);
  return detailed.map((v, i) =>
    Math.max(0, blend(rgb[i]!, blend(v, ((warm[i]! * outY) / warmY) * p.sunGain, gold), amount)),
  ) as [number, number, number];
}
