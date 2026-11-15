import { Texture, Vector2, Vector4 } from 'three/webgpu';
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
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import {
  ALIVE,
  ALIVE_CLOUD_SHAPES,
  ALIVE_CLOUD_TINT,
  ALIVE_DEFAULTS,
  aliveDayWeight,
  resolveAlive,
  type AliveChange,
  type AliveOptions,
  type CloudShape,
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
// The shape arrives as uniforms (lattice inverses, threshold, fine weight, inverse stretch) so presets switch at run
// time without a recompile; the soft preset evaluates the same expression as the former constants.
export const analyticCloudCoverage = Fn(([xz, clock, direction, strength, lattice, form]: ShaderNode[]) => {
  const result = float(0).toVar('aliveCloudNeutral');
  If(strength.greaterThan(0), () => {
    const drift = direction.mul(clock.mul(ALIVE.drift));
    const morph = vec2(sin(clock.mul(0.07)), cos(clock.mul(0.09))).mul(lattice.x.mul(ALIVE.cloudMorph));
    const advected = xz.sub(drift);
    const along: ShaderNode = dot(advected, direction);
    const stretched = vec2(along.mul(form.y), dot(advected, vec2(direction.y.negate(), direction.x)));
    const p = stretched.mul(lattice.x);
    const n = noise(p.add(morph))
      .mul(form.x)
      .add(
        noise(stretched.mul(lattice.y).add(vec2(17.2, 9.4)).add(vec2(morph.y.negate(), morph.x))).mul(
          float(1).sub(form.x),
        ),
      );
    result.assign(smoothstep(lattice.z, lattice.w, n));
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
    { name: 'lattice', type: 'vec4' },
    { name: 'form', type: 'vec2' },
  ],
});

const cloudColour = Fn(([grey, darkening]: ShaderNode[]) => {
  const cover = float(1).sub(grey).div(darkening).clamp();
  return mix(vec3(1), vec3(...ALIVE_CLOUD_TINT), cover).mul(grey);
}).setLayout({
  name: 'aliveCloudColour',
  type: 'vec3',
  inputs: [
    { name: 'grey', type: 'float' },
    { name: 'darkening', type: 'float' },
  ],
});

const latticeOf = (shape: Readonly<CloudShape>, target = new Vector4()) =>
  target.set(1 / shape.scale, 1 / shape.largeScale, shape.edgeLow, shape.edgeHigh);
const formOf = (shape: Readonly<CloudShape>, target = new Vector2()) => target.set(shape.fineWeight, 1 / shape.stretch);

/** Convert once beside the shared cloud factor, then reuse across all lighting terms. */
const linearCloudFactorNode = Fn(([factor]: ShaderNode[]) => {
  const result = vec3(1).toVar('aliveCloudLinearFactor');
  If(factor.x.notEqual(1).or(factor.y.notEqual(1)).or(factor.z.notEqual(1)), () => {
    result.assign(factor.pow(2.2));
  });
  return result;
});

// A factor can feed both an emissive art term and a physical lighting finish. Share its conversion as well.
const linearFactors = new WeakMap<ShaderNode, ShaderNode>();
export function linearCloudFactor(factor: ShaderNode): ShaderNode {
  let linear = linearFactors.get(factor);
  if (!linear) {
    linear = linearCloudFactorNode(factor).toVar();
    linearFactors.set(factor, linear);
  }
  return linear;
}

/** Power-law palette attenuation. The input factor is already linear, including its exact off identity. */
export const shadeCloudColour = Fn(([source, factor]: ShaderNode[]) => {
  const result = source.toVar('aliveCloudLit');
  If(factor.x.notEqual(1).or(factor.y.notEqual(1)).or(factor.z.notEqual(1)), () => {
    result.assign(source.mul(factor));
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
  readonly cloudAmount = uniform(0).setName('aliveCloudAmount');
  readonly coverageEnabled = uniform(0).setName('aliveCoverageEnabled');
  readonly cloudStrength = uniform(0).setName('aliveCloudStrength');
  readonly windAmplitude = uniform(0).setName('aliveWindAmplitude');
  readonly debugCloud = uniform(0).setName('aliveDebugCloud');
  readonly cloudLattice = uniform(latticeOf(ALIVE_CLOUD_SHAPES.soft)).setName('aliveCloudLattice');
  readonly cloudForm = uniform(formOf(ALIVE_CLOUD_SHAPES.soft)).setName('aliveCloudForm');
  readonly cloudDarkening = uniform(ALIVE_CLOUD_SHAPES.soft.darkening).setName('aliveCloudDarkening');
  /** Preset water shares of the land shadow: body/seabed and sky reflection (glints always take all of it). */
  readonly waterBody = uniform(ALIVE_CLOUD_SHAPES.soft.waterBodyWeight).setName('aliveWaterBody');
  readonly waterSky = uniform(ALIVE_CLOUD_SHAPES.soft.waterSkyWeight).setName('aliveWaterSky');
  /** A renderer binds its owned world-space target before the first scene draw. */
  private readonly emptyCoverage = new Texture();
  readonly coverageTexture = texture(this.emptyCoverage);
  readonly coverageBounds = uniform(new Vector4(0, 0, 1, 1));
  readonly coverageCached = uniform(0);
  readonly analyticCoverageAt = (xz: ShaderNode): ShaderNode =>
    analyticCloudCoverage(xz, this.clock, this.windDirection, this.coverageEnabled, this.cloudLattice, this.cloudForm);
  private readonly cachedCoverage = Fn(([xz, strength, bounds, cached]: ShaderNode[]) => {
    const result = float(0).toVar('aliveCloudNeutral');
    If(strength.greaterThan(0).and(cached.greaterThan(0)), () => {
      result.assign(this.coverageTexture.sample(xz.sub(bounds.xy).div(bounds.zw)).r);
    });
    return result;
  }).setLayout({
    name: 'aliveCachedCoverage',
    type: 'float',
    inputs: [
      { name: 'xz', type: 'vec2' },
      { name: 'strength', type: 'float' },
      { name: 'bounds', type: 'vec4' },
      { name: 'cached', type: 'float' },
    ],
  });
  readonly coverageAt = (xz: ShaderNode): ShaderNode =>
    this.cachedCoverage(xz, this.coverageEnabled, this.coverageBounds, this.coverageCached);
  /** Raw coverage before darkening/strength; one shared VarNode per material. */
  readonly coverage: ShaderNode = this.coverageAt(positionWorld.xz).toVar('aliveCloudCoverage');
  readonly shadowAt = (xz: ShaderNode): ShaderNode => float(1).sub(this.coverageAt(xz).mul(this.cloudStrength));
  readonly shadow: ShaderNode = this.colourFromCoverage(this.coverage).toVar('aliveCloudFactor');

  readonly linearShadow = linearCloudFactor(this.shadow).toVar('aliveLinearShadow');
  private readonly weightedLinearShadows = new Map<ShaderNode | number, ShaderNode>();

  colourFromCoverage(coverage: ShaderNode): ShaderNode {
    return cloudColour(float(1).sub(coverage.mul(this.cloudStrength)), this.cloudDarkening.max(1e-6));
  }
  /** Use the same temporal fade as physical cloud darkening, including overcast and night. */
  gradeCoverage(coverage: ShaderNode): ShaderNode {
    return coverage.mul(this.cloudAmount).clamp();
  }
  shadeAt(colour: ShaderNode, factor: ShaderNode): ShaderNode {
    return shadeCloudColour(colour, factor);
  }

  set(change: AliveChange): Readonly<AliveOptions> {
    this.options = resolveAlive(change, this.options);
    return { ...this.options };
  }

  update(seconds: number, direction: number, night: number, overcast: number, airless: boolean): void {
    this.clock.value = seconds * this.options.speed;
    this.windDirection.value.set(Math.cos(direction), Math.sin(direction));
    const shape =
      this.options.preset === 'lush'
        ? (this.options.shape ?? ALIVE_CLOUD_SHAPES.lush)
        : ALIVE_CLOUD_SHAPES[this.options.preset];
    this.coverageEnabled.value = this.options.clouds && !airless ? 1 : 0;
    latticeOf(shape, this.cloudLattice.value);
    formOf(shape, this.cloudForm.value);
    this.cloudDarkening.value = shape.darkening;
    this.waterBody.value = shape.waterBodyWeight;
    this.waterSky.value = shape.waterSkyWeight;
    this.cloudAmount.value = this.options.clouds ? this.options.cloud * aliveDayWeight(night, overcast, airless) : 0;
    this.cloudStrength.value = this.options.clouds
      ? this.options.cloud * shape.darkening * aliveDayWeight(night, overcast, airless)
      : 0;
    this.debugCloud.value = this.options.debugCloud && this.options.clouds ? 1 : 0;
    this.windAmplitude.value = this.options.wind && !airless ? this.options.sway : 0;
  }

  copyValuesFrom(source: LivingWorldNodes): void {
    this.clock.value = source.clock.value;
    this.coverageEnabled.value = source.coverageEnabled.value;
    this.cloudAmount.value = source.cloudAmount.value;
    this.windDirection.value.copy(source.windDirection.value);
    this.cloudStrength.value = source.cloudStrength.value;
    this.cloudLattice.value.copy(source.cloudLattice.value);
    this.cloudForm.value.copy(source.cloudForm.value);
    this.cloudDarkening.value = source.cloudDarkening.value;
    this.waterBody.value = source.waterBody.value;
    this.waterSky.value = source.waterSky.value;
    this.windAmplitude.value = source.windAmplitude.value;
    this.debugCloud.value = source.debugCloud.value;
  }

  /** The same field and palette attenuation for direct light, painted fill and reflected scene light. */
  shade(colour: ShaderNode): ShaderNode {
    return shadeCloudColour(colour, this.linearShadow);
  }

  /** Weight both darkening and cool tint; clear/off remains exactly one, including on water. */
  shadowWith(weight: ShaderNode | number): ShaderNode {
    return mix(vec3(1), this.shadow, weight);
  }

  shadeWith(colour: ShaderNode, weight: ShaderNode | number): ShaderNode {
    let factor = this.weightedLinearShadows.get(weight);
    if (!factor) {
      factor = linearCloudFactor(this.shadowWith(weight)).toVar();
      this.weightedLinearShadows.set(weight, factor);
    }
    return shadeCloudColour(colour, factor);
  }

  dispose(): void {
    this.emptyCoverage.dispose();
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
