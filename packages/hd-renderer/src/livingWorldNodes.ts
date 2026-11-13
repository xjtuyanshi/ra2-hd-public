import { Vector2 } from 'three/webgpu';
import {
  Fn,
  If,
  dot,
  float,
  floor,
  fract,
  mix,
  positionWorld,
  sin,
  cos,
  smoothstep,
  step,
  sRGBTransferEOTF,
  sRGBTransferOETF,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import {
  ALIVE,
  ALIVE_CLOUD_TINT,
  ALIVE_DEFAULTS,
  aliveDayWeight,
  resolveAlive,
  type AliveChange,
  type AliveOptions,
} from './livingWorld';
import type { ShaderNode } from './tsl';

const hash = Fn(([p]: ShaderNode[]) => {
  const q = fract(vec3(p.x, p.y, p.x).mul(0.1031)).toVar();
  q.addAssign(dot(q, q.yzx.add(33.33)));
  return fract(q.x.add(q.y).mul(q.z));
});

const noise = Fn(([p]: ShaderNode[]) => {
  const i = floor(p);
  const f: ShaderNode = fract(p);
  const u = f
    .mul(f)
    .mul(f)
    .mul(f.mul(f.mul(6).sub(15)).add(10));
  const gradient = (offset: ShaderNode): ShaderNode => {
    const h = hash(i.add(offset));
    const direction = step(0.5, vec2(h, fract(h.mul(2))))
      .mul(2)
      .sub(1)
      .mul(Math.SQRT1_2);
    return dot(direction, f.sub(offset));
  };
  return mix(
    mix(gradient(vec2(0, 0)), gradient(vec2(1, 0)), u.x),
    mix(gradient(vec2(0, 1)), gradient(vec2(1, 1)), u.x),
    u.y,
  );
});

// A real shader function owns its neutral initializer on EVERY invocation. An inline Fn shared between select/If
// branches lets TSL cache its result variable in the first branch: unlit ore then reads the other branch's zero.
// Pass all state explicitly so neither the result nor renderer-local uniforms can leak across branches/renderers.
const cloudShadow = Fn(([xz, clock, direction, strength]: ShaderNode[]) => {
  const result = float(1).toVar('aliveCloudNeutral');
  If(strength.greaterThan(0), () => {
    const drift = direction.mul(clock.mul(ALIVE.drift));
    const morph = vec2(sin(clock.mul(0.07)), cos(clock.mul(0.09))).mul(ALIVE.cloudMorph / ALIVE.cloudScale);
    const advected = xz.sub(drift);
    const stretched = vec2(
      dot(advected, direction).div(ALIVE.cloudStretch),
      dot(advected, vec2(direction.y.negate(), direction.x)),
    );
    const p = stretched.div(ALIVE.cloudScale);
    const n = noise(p.add(morph))
      .mul(ALIVE.cloudFineWeight)
      .add(
        noise(stretched.div(ALIVE.cloudLargeScale).add(vec2(17.2, 9.4)).add(vec2(morph.y.negate(), morph.x))).mul(
          1 - ALIVE.cloudFineWeight,
        ),
      );
    result.assign(float(1).sub(smoothstep(ALIVE.cloudEdgeLow, ALIVE.cloudEdgeHigh, n).mul(strength)));
  });
  return result;
}).setLayout({
  name: 'aliveCloudShadow',
  type: 'float',
  inputs: [
    { name: 'xz', type: 'vec2' },
    { name: 'clock', type: 'float' },
    { name: 'direction', type: 'vec2' },
    { name: 'strength', type: 'float' },
  ],
});

const cloudColour = Fn(([grey]: ShaderNode[]) => {
  const cover = float(1).sub(grey).div(ALIVE.cloudDarkening).clamp();
  return mix(vec3(1), vec3(...ALIVE_CLOUD_TINT), cover).mul(grey);
}).setLayout({ name: 'aliveCloudColour', type: 'vec3', inputs: [{ name: 'grey', type: 'float' }] });

/** Palette-domain attenuation: keeps the displayed shadow depth independent of the linear-light transfer curve. */
export const shadeCloudColour = Fn(([source, factor]: ShaderNode[]) => {
  const result = source.toVar('aliveCloudLit');
  If(factor.x.notEqual(1).or(factor.y.notEqual(1)).or(factor.z.notEqual(1)), () => {
    result.assign(sRGBTransferEOTF((sRGBTransferOETF(source.max(0)) as ShaderNode).mul(factor)));
  });
  return result;
}).setLayout({
  name: 'aliveCloudShade',
  type: 'vec3',
  inputs: [
    { name: 'source', type: 'vec3' },
    { name: 'factor', type: 'vec3' },
  ],
});

/** Uniform-only switches: the graph exists from its first warm frame, even with every part off. */
export class LivingWorldNodes {
  private options: AliveOptions = { ...ALIVE_DEFAULTS };
  readonly clock = uniform(0).setName('aliveClock');
  readonly windDirection = uniform(new Vector2(Math.cos(0.6), Math.sin(0.6))).setName('aliveWindDirection');
  readonly cloudStrength = uniform(0).setName('aliveCloudStrength');
  readonly windAmplitude = uniform(0).setName('aliveWindAmplitude');
  readonly debugCloud = uniform(0).setName('aliveDebugCloud');
  readonly shadowAt = (xz: ShaderNode): ShaderNode =>
    cloudShadow(xz, this.clock, this.windDirection, this.cloudStrength);
  // One VarNode per material builder shares all optical terms. Branching consumers (sprites) append it at
  // their colour root, before sibling branches. Keep the neutral initializer in the named cloudShadow Fn.
  readonly shadow: ShaderNode = cloudColour(this.shadowAt(positionWorld.xz)).toVar('aliveCloudFactor');

  set(change: AliveChange): Readonly<AliveOptions> {
    this.options = resolveAlive(change, this.options);
    return { ...this.options };
  }

  update(seconds: number, direction: number, night: number, overcast: number, airless: boolean): void {
    this.clock.value = seconds * this.options.speed;
    this.windDirection.value.set(Math.cos(direction), Math.sin(direction));
    this.cloudStrength.value = this.options.clouds
      ? this.options.cloud * ALIVE.cloudDarkening * aliveDayWeight(night, overcast, airless)
      : 0;
    this.debugCloud.value = this.options.debugCloud && this.options.clouds ? 1 : 0;
    this.windAmplitude.value = this.options.wind && !airless ? this.options.sway : 0;
  }

  copyValuesFrom(source: LivingWorldNodes): void {
    this.clock.value = source.clock.value;
    this.windDirection.value.copy(source.windDirection.value);
    this.cloudStrength.value = source.cloudStrength.value;
    this.windAmplitude.value = source.windAmplitude.value;
    this.debugCloud.value = source.debugCloud.value;
  }

  /** The same field and palette attenuation for direct light, painted fill and reflected scene light. */
  shade(colour: ShaderNode): ShaderNode {
    return shadeCloudColour(colour, this.shadow);
  }

  /** Weight both darkening and cool tint; clear/off remains exactly one, including on water. */
  shadowWith(weight: ShaderNode | number): ShaderNode {
    return mix(vec3(1), this.shadow, weight);
  }

  shadeWith(colour: ShaderNode, weight: ShaderNode | number): ShaderNode {
    return shadeCloudColour(colour, this.shadowWith(weight));
  }

  /** Raw encoded grey in the existing display pass; bypass scene grade/exposure without a new target. */
  debugView(display: ShaderNode, xz: ShaderNode, visible: ShaderNode): ShaderNode {
    return Fn(([colour, point, visibility]: ShaderNode[]) => {
      const result = colour.toVar();
      If(this.debugCloud.greaterThan(0.5), () => {
        result.assign(vec4(vec3(this.shadowAt(point)).mul(visibility), 1));
      });
      return result;
    })(display, xz, visible);
  }

  gust(xz: ShaderNode): ShaderNode {
    const wave = dot(xz, this.windDirection)
      .sub(this.clock.mul(ALIVE.drift))
      .mul((Math.PI * 2) / ALIVE.gustLength);
    return sin(wave).mul(0.5).add(0.5).pow(4);
  }

  /** Preserve the input expression exactly while off. The caller supplies a grass-only eligibility mask. */
  grass(colour: ShaderNode, mask: ShaderNode): ShaderNode {
    return Fn(([source, eligible]: ShaderNode[]) => {
      const result = source.toVar();
      If(this.windAmplitude.greaterThan(0), () => {
        If(eligible.greaterThan(0), () => {
          const ripple = this.gust(positionWorld.xz).mul(2).sub(1).mul(0.05).mul(this.windAmplitude).mul(eligible);
          result.mulAssign(vec3(float(1).add(ripple), float(1).add(ripple.mul(1.02)), float(1).add(ripple.mul(0.8))));
        });
      });
      return result;
    })(colour, mask);
  }

  /** GPU-only, instance-constant phase; displacement is in the camera plane and leaves depth policy untouched. */
  sway(anchor: ShaderNode, height: ShaderNode): ShaderNode {
    const seed = hash(anchor.xz);
    const period = seed.sub(0.5).mul(0.8).add(ALIVE.period);
    const phase = this.clock
      .mul(Math.PI * 2)
      .div(period)
      .add(seed.mul(Math.PI * 2));
    const gust = this.gust(anchor.xz);
    return sin(phase)
      .mul(this.windAmplitude)
      .mul(gust.mul(ALIVE.gustPixels - ALIVE.crownPixels).add(ALIVE.crownPixels))
      .mul(height.clamp().pow(ALIVE.heightPower));
  }
}
