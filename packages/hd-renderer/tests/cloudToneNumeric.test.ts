import { expect, it, vi } from 'vitest';
import {
  cloudToneGain,
  cloudToneGainNode,
  cloudGreenness,
  cloudGreennessNode,
  CloudToneUniforms,
  cloudToneNode,
  cloudToneRgb,
} from '../src/cloudTone';
import type { CloudToneParams } from '../src/cloudTone';
import { lush2Defaults, lush2Mapping } from '../src/ra2/lush2Tone';
import { objectCloudFactor } from '../src/objectCloudTone';
import { analyticCloudCoverage, linearCloudFactor, shadeCloudColour } from '../src/livingWorldNodes';
import { ALIVE_CLOUD_SHAPES, cloudCoverage } from '../src/livingWorld';
import type { AtmosphereNodes } from '../src/atmosphere';

/** Numeric TSL operation double: evaluates the production GPU recipe without a browser/device.
 * Real r186 WGSL generation and texture bindings are checked separately in cloudToneMap.test.ts. */
const { N, n, vector } = vi.hoisted(() => {
  type Input = number | number[] | N | { toArray(): number[] };
  class N {
    constructor(public value: Input) {}
    data(): number[] {
      return this.value instanceof N
        ? this.value.data()
        : typeof this.value === 'number'
          ? [this.value]
          : Array.isArray(this.value)
            ? this.value
            : this.value.toArray();
    }
    op(other: Input, fn: (a: number, b: number) => number): N {
      const a = this.data(),
        b = n(other).data();
      return new N(
        Array.from({ length: Math.max(a.length, b.length) }, (_, i) => fn(a[i % a.length]!, b[i % b.length]!)),
      );
    }
    add(v: Input) {
      return this.op(v, (a, b) => a + b);
    }
    sub(v: Input) {
      return this.op(v, (a, b) => a - b);
    }
    mul(v: Input) {
      return this.op(v, (a, b) => a * b);
    }
    div(v: Input) {
      return this.op(v, (a, b) => a / b);
    }
    pow(v: Input) {
      return this.op(v, (a, b) => a ** b);
    }
    max(v: Input) {
      return this.op(v, Math.max);
    }
    min(v: Input) {
      return this.op(v, Math.min);
    }
    mod(v: Input) {
      return this.op(v, (a, b) => ((a % b) + b) % b);
    }
    clamp() {
      return this.max(0).min(1);
    }
    negate() {
      return this.mul(-1);
    }
    and(v: Input) {
      return this.op(v, (a, b) => Number(Boolean(a) && Boolean(b)));
    }
    addAssign(v: Input) {
      return this.assign(this.add(v));
    }
    sqrt() {
      return new N(this.data().map(Math.sqrt));
    }
    abs() {
      return new N(this.data().map(Math.abs));
    }
    equal(v: Input) {
      return this.op(v, (a, b) => Number(a === b));
    }
    notEqual(v: Input) {
      return this.op(v, (a, b) => Number(a !== b));
    }
    greaterThan(v: Input) {
      return this.op(v, (a, b) => Number(a > b));
    }
    lessThanEqual(v: Input) {
      return this.op(v, (a, b) => Number(a <= b));
    }
    or(v: Input) {
      return this.op(v, (a, b) => Number(Boolean(a) || Boolean(b)));
    }
    select(a: Input, b: Input) {
      return this.data()[0] ? n(a) : n(b);
    }
    setName() {
      return this;
    }
    onObjectUpdate() {
      return this;
    }
    toStack() {
      return this;
    }
    toVar() {
      return new N([...this.data()]);
    }
    assign(v: Input) {
      this.value = [...n(v).data()];
      return this;
    }
    mulAssign(v: Input) {
      return this.assign(this.mul(v));
    }
    get x() {
      return new N(this.data()[0]!);
    }
    get r() {
      return this.x;
    }
    get y() {
      return new N(this.data()[1]!);
    }
    get g() {
      return this.y;
    }
    get z() {
      return new N(this.data()[2]!);
    }
    get b() {
      return this.z;
    }
    get yzx() {
      const v = this.data();
      return new N([v[1]!, v[2]!, v[0]!]);
    }
    get w() {
      return new N(this.data()[3]!);
    }
  }
  const n = (v: Input) => (v instanceof N ? v : new N(v));
  const vector = (...v: Input[]) => new N(v.flatMap((x) => n(x).data()));
  return { N, n, vector };
});
type N = InstanceType<typeof N>;
type Input = ConstructorParameters<typeof N>[0];
vi.mock('three/tsl', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three/tsl')>();
  return {
    ...actual,
    uniform: (v: Input) => new N(v),
    float: (v: Input) => n(v),
    vec2: (...args: Input[]) =>
      args.length === 1 && n(args[0]!).data().length === 1 ? vector(args[0]!, args[0]!) : vector(...args),
    vec3: (...args: Input[]) =>
      args.length === 1 && n(args[0]!).data().length === 1 ? vector(args[0]!, args[0]!, args[0]!) : vector(...args),
    vec4: (...args: Input[]) =>
      args.length === 1 && n(args[0]!).data().length === 1
        ? vector(args[0]!, args[0]!, args[0]!, args[0]!)
        : vector(...args),
    Fn: (fn: (args: N[]) => N) => {
      const call = (...args: N[]) => fn(args);
      call.setLayout = () => call;
      return call;
    },
    floor: (v: Input) => new N(n(v).data().map(Math.floor)),
    fract: (v: Input) =>
      new N(
        n(v)
          .data()
          .map((x) => x - Math.floor(x)),
      ),
    step: (edge: Input, v: Input) => n(v).op(edge, (x, e) => Number(x >= e)),
    If: (condition: N, then: () => void) => {
      const yes = Boolean(condition.data()[0]);
      if (yes) then();
      return {
        Else: (otherwise: () => void) => {
          if (!yes) otherwise();
        },
      };
    },
    cos: (v: Input) => new N(n(v).data().map(Math.cos)),
    sin: (v: Input) => new N(n(v).data().map(Math.sin)),
    dot: (a: Input, b: Input) =>
      new N(
        n(a)
          .mul(b)
          .data()
          .reduce((x, y) => x + y, 0),
      ),
    mix: (a: Input, b: Input, t: Input) => n(a).add(n(b).sub(a).mul(t)),
    smoothstep: (a: Input, b: Input, v: Input) => {
      const t = n(v).sub(a).div(n(b).sub(a)).max(0).min(1);
      return t.mul(t).mul(n(3).sub(t.mul(2)));
    },
    sRGBTransferOETF: (v: Input) =>
      new N(
        n(v)
          .data()
          .map((x) => (x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055)),
      ),
  };
});

it('production TSL mean gain and CPU reference agree for day/night, shade, material/context and slider extremes', () => {
  const means = [
    [0.3, 0.4, 0.08],
    [0.5, 0.3, 0.12],
    [0, 0, 0],
    [0.3, 0.3, 0.3],
    [0.02, 0.5, 0.01],
  ] as const;
  const base = lush2Mapping(lush2Defaults('temperate'), 'temperate').tone;
  const p = new CloudToneUniforms();
  for (const grassChroma of [0.8, 1.08, 1.6])
    for (const grassLight of [0.7, 0.95, 1.15]) {
      const params: CloudToneParams = { ...base, grassChroma, grassLight };
      p.set(params);
      for (const mean of means)
        for (const context of means.slice(0, 2))
          for (const night of [0, 0.4, 1])
            for (const shade of [0, 0.35, 1]) {
              const actual = cloudToneGainNode(
                vector(...mean),
                vector(...context),
                n(shade),
                p,
                n(night),
              ) as unknown as N;
              actual
                .data()
                .forEach((v, i) => expect(v).toBeCloseTo(cloudToneGain(mean, context, shade, params, night)[i]!, 9));
              const green = cloudGreennessNode(vector(...mean), p) as unknown as N;
              expect(green.data()[0]).toBeCloseTo(cloudGreenness(mean, params), 12);
            }
    }
});

it('production TSL pixel finish, snow and object recipes agree with the CPU twin', () => {
  for (const theater of ['temperate', 'snow'])
    for (const detail of [0.4, 1, 1.2]) {
      const params = { ...lush2Mapping(lush2Defaults(theater), theater).tone, detail };
      const p = new CloudToneUniforms();
      p.set(params);
      for (const rgb of [
        [0, 0, 0],
        [0.2, 0.3, 0.05],
        [0.7, 0.8, 0.75],
        [1, 1, 1],
      ] as const)
        for (const object of [false, true])
          for (const shade of [0, 0.5, 1])
            for (const night of [0, 0.5, 1])
              for (const wet of [0, 0.5, 1]) {
                const mean = [0.3, 0.4, 0.08] as const;
                const actual = cloudToneNode(
                  vector(...rgb),
                  vector(...mean),
                  n(shade),
                  p,
                  n(night),
                  object,
                  n(wet),
                  n(0.2),
                ) as unknown as N;
                actual
                  .data()
                  .forEach((v, i) =>
                    expect(v).toBeCloseTo(cloudToneRgb(rgb, shade, params, night, mean, object, wet, 0.2)[i]!, 9),
                  );
              }
    }
});

it('lifts warm snow-block rocks without adding blue and neutralises lavender only by its cool cast', () => {
  const base = lush2Mapping(lush2Defaults('snow'), 'snow').tone;
  const warm = [0.2, 0.16, 0.12] as const;
  const lavender = [0.3, 0.26, 0.36] as const;
  for (const snowLift of [1, 1.25, 2])
    for (const snowBlue of [0, 0.5, 1]) {
      const params = { ...base, snowLift, snowBlue };
      for (const shade of [0, 1]) {
        const out = cloudToneRgb(warm, shade, params);
        const tintRatio =
          (1 + (params.snowShadeTint[2] - 1) * snowBlue * shade) /
          (1 + (params.snowShadeTint[0] - 1) * snowBlue * shade);
        expect(out[2] / out[0]).toBeLessThanOrEqual((warm[2] / warm[0]) * tintRatio + 1e-6);
        if (shade === 0) expect(out[2] / out[0]).toBeCloseTo(warm[2] / warm[0], 12);
      }
      const out = cloudToneRgb(lavender, 0, params);
      expect(out[2] / out[0]).toBeLessThan(lavender[2] / lavender[0]);
      expect(out[2] / out[0]).toBeGreaterThanOrEqual(1);
    }
  // The chromatic gain itself is luma-one, separate from the scalar lift.
  const gain = cloudToneGain(lavender, lavender, 0, { ...base, snowLift: 1 });
  expect(gain.reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i]!, 0)).toBeCloseTo(1, 12);
});

it('keeps CPU and production TSL snow gains/pixels equivalent across cool-cast and lift extremes', () => {
  const base = lush2Mapping(lush2Defaults('snow'), 'snow').tone;
  const means = [
    [0.2, 0.16, 0.12],
    [0.3, 0.26, 0.36],
    [0.3, 0.3, 0.31], // Partial display-space cool-cast weight.
    [0.3, 0.3, 0.3],
    [0, 0, 0],
    [1, 1, 1],
  ] as const;
  for (const snowLift of [1, 1.25, 2])
    for (const snowBlue of [0, 1])
      for (const strength of [0, 0.6, 1, 1.5]) {
        const params = { ...base, snowLift, snowBlue, strength };
        const p = new CloudToneUniforms();
        p.set(params);
        for (const mean of means)
          for (const shade of [0, 0.4, 1])
            for (const night of [0, 0.5, 1]) {
              const gain = cloudToneGainNode(vector(...mean), vector(...mean), n(shade), p, n(night)) as unknown as N;
              gain
                .data()
                .forEach((v, i) => expect(v).toBeCloseTo(cloudToneGain(mean, mean, shade, params, night)[i]!, 9));
              for (const rgb of [mean, [0.2, 0.16, 0.12]] as const) {
                const out = cloudToneNode(vector(...rgb), vector(...mean), n(shade), p, n(night)) as unknown as N;
                out
                  .data()
                  .forEach((v, i) => expect(v).toBeCloseTo(cloudToneRgb(rgb, shade, params, night, mean)[i]!, 9));
              }
            }
      }
});

it('objects borrow only their configured cloud share per channel, preserve legacy off, and enforce their floor', () => {
  const p = new CloudToneUniforms();
  p.set(lush2Mapping(lush2Defaults('temperate'), 'temperate').tone);
  for (const strength of [0, 1])
    for (const share of [0, 0.5, 1])
      for (const floor of [0.5, 0.8, 1]) {
        p.strength.value = strength;
        p.objectShare.value = share;
        p.objectFloor.value = floor;
        const ground = [0.4, 0.6, 0.8];
        const nodes = {
          cloudTone: p,
          alive: { colourFromCoverage: () => vector(...ground), cloudStrength: n(0.5) },
        } as unknown as AtmosphereNodes;
        // cloudTintShare=1 isolates a deliberately chromatic ground factor.
        p.cloudTintShare.value = 1;
        const actual = objectCloudFactor(nodes, n(1)) as unknown as N;
        actual
          .data()
          .forEach((v, i) =>
            expect(v).toBeCloseTo(strength ? Math.max(floor, 1 + (ground[i]! - 1) * share) : ground[i]!, 12),
          );
      }
});

it('uses the unchanged analytic field at world-cache texel centres for every preset, drift and morph', () => {
  // Evaluate the production coverage-pass TSL numerically at the exact QuadMesh raster centres.
  const width = 120,
    height = 143,
    nx = width * 4,
    nz = height * 4;
  for (const shape of Object.values(ALIVE_CLOUD_SHAPES))
    for (const clock of [0, 3.5, 17, 82])
      for (const angle of [-1.2, 0, 0.6])
        for (let iz = 0; iz < nz; iz += 37)
          for (let ix = 0; ix < nx; ix += 31) {
            const x = ((ix + 0.5) / nx) * width,
              z = ((iz + 0.5) / nz) * height;
            const actual = analyticCloudCoverage(
              vector(x, z),
              n(clock),
              vector(Math.cos(angle), Math.sin(angle)),
              n(1),
              vector(1 / shape.scale, 1 / shape.largeScale, shape.edgeLow, shape.edgeHigh),
              vector(shape.fineWeight, 1 / shape.stretch),
            ) as unknown as N;
            expect(actual.data()[0]).toBeCloseTo(cloudCoverage(x, z, clock, angle, 1, shape), 8);
          }
});

it('bounds the power-law approximation against exact sRGB attenuation in mid-tones', () => {
  const eotf = (x: number) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);
  const oetf = (x: number) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
  let maxLinear = 0,
    maxDisplay = 0;
  for (let source = 0.05; source <= 0.80001; source += 0.005)
    for (let factor = 0.5; factor <= 1.00001; factor += 0.005) {
      const linear = linearCloudFactor(vector(factor, factor, factor));
      const actual = (shadeCloudColour(vector(source, source, source), linear) as unknown as N).data()[0]!;
      const exact = eotf(oetf(source) * factor);
      maxLinear = Math.max(maxLinear, Math.abs(actual - exact));
      maxDisplay = Math.max(maxDisplay, Math.abs(oetf(actual) - oetf(exact)));
    }
  expect(maxLinear).toBeLessThan(0.009);
  expect(maxDisplay).toBeLessThan(0.019);
  expect((shadeCloudColour(vector(-0.1, 0, 2), linearCloudFactor(vector(1, 1, 1))) as unknown as N).data()).toEqual([
    -0.1, 0, 2,
  ]);
});

it('snow objects: cool bright snow is neutralised and lifted like the ground, warm or dark paint keeps its ratios', () => {
  const params = lush2Mapping(lush2Defaults('snow'), 'snow').tone;
  const p = new CloudToneUniforms();
  p.set(params);
  const lavenderSnow = [0.26, 0.24, 0.48] as const; // display ~ (140,137,185): snow on a tree under mission light
  const warmPaint = [0.3, 0.06, 0.04] as const;
  const darkNeedle = [0.02, 0.02, 0.035] as const;
  const snow = cloudToneRgb(lavenderSnow, 0, params, 0, lavenderSnow, true);
  expect(snow[2] / snow[0]).toBeLessThan(1.15);
  expect(snow[1]).toBeGreaterThan(lavenderSnow[1] * 1.5);
  // Compare against the same object step without the snow lift/neutralisation (bounce/tint still apply).
  const noSnow = { ...params, snow: 0 };
  const ratio = (v: readonly number[]) => v[2]! / v[0]!;
  const paint = cloudToneRgb(warmPaint, 0, params, 0, warmPaint, true);
  expect(ratio(paint)).toBeCloseTo(ratio(cloudToneRgb(warmPaint, 0, noSnow, 0, warmPaint, true)), 9);
  const needle = cloudToneRgb(darkNeedle, 0, params, 0, darkNeedle, true);
  expect(ratio(needle)).toBeCloseTo(ratio(cloudToneRgb(darkNeedle, 0, noSnow, 0, darkNeedle, true)), 2);
  // Temperate objects are untouched by the snow step.
  const temperate = lush2Mapping(lush2Defaults('temperate'), 'temperate').tone;
  const q = new CloudToneUniforms();
  q.set(temperate);
  for (const rgb of [lavenderSnow, warmPaint, darkNeedle])
    for (const [pp, uu] of [
      [params, p],
      [temperate, q],
    ] as const)
      for (const night of [0, 0.5])
        (cloudToneNode(vector(...rgb), vector(...rgb), n(0.3), uu, n(night), true) as unknown as N)
          .data()
          .forEach((v, i) => expect(v).toBeCloseTo(cloudToneRgb(rgb, 0.3, pp, night, rgb, true)[i]!, 9));
});
