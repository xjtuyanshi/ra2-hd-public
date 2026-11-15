import { describe, expect, it } from 'vitest';
import { AtmosphereNodes } from '../src/atmosphere';
import { EffectsLayer } from '../src/effects';
import { LIGHT_LOOK_CURRENT, LIGHT_LOOK_NEW, createLightLook, lightLook } from '../src/lightLook';
import { realCardAlpha } from '../src/live/effectSprites';
import { shaderBytes } from './fixtures/realLightScene';

/** Fragment main() of a compiled material. */
function fragmentMain(code: string): string {
  const at = code.lastIndexOf('fn main(');
  return code.slice(at);
}

/** Text of the brace block that starts at the first '{' at or after `from`, and the index just after it. */
function block(code: string, from: number): [string, number] {
  const open = code.indexOf('{', from);
  let depth = 0;
  for (let i = open; i < code.length; i++) {
    if (code[i] === '{') depth++;
    else if (code[i] === '}' && --depth === 0) return [code.slice(open + 1, i), i + 1];
  }
  throw new Error('unbalanced');
}

const assigned = (text: string) => new Set([...text.matchAll(/\b(nodeVar\d+)\s*=[^=]/g)].map((m) => m[1]!));
const used = (text: string) => new Set([...text.matchAll(/\b(nodeVar\d+)\b/g)].map((m) => m[1]!));

describe('round 8: real smoke', () => {
  it('keeps the presets: cur and round 7 draw no round-8 smoke, new does', () => {
    expect(LIGHT_LOOK_CURRENT.smokeReal).toBe(0);
    expect(LIGHT_LOOK_NEW.smokeReal).toBe(1);
    const look = createLightLook();
    expect(lightLook(look, 'cur').smokeReal).toBe(0);
    expect(lightLook(look, { smokeReal: 1, smokeScale: 9 }).smokeScale).toBe(4); // Wide parameters clamp at 4.
    expect(lightLook(look, 'new').smokeHaze).toBe(LIGHT_LOOK_NEW.smokeHaze);
  });

  it('bounds the original smoke cards: no texel beyond its own art, mean coverage within 10 % of the original', () => {
    // Coverage 0 stays 0 (the outline only erodes inward), and over the field's range the mean stays within 10 %.
    let before = 0,
      after = 0;
    for (let c = 0; c <= 20; c++)
      for (let f = 0; f <= 40; f++) {
        const coverage = c / 20;
        // The shared field: weighted sum of the noise texture's fBm channels (each sd ~0.1), so ~0.5 +- 2 sd of 0.055.
        const field = 0.39 + (0.22 * f) / 40;
        const a = realCardAlpha(coverage, field, LIGHT_LOOK_NEW);
        expect(a).toBeGreaterThanOrEqual(0);
        expect(a).toBeLessThanOrEqual(1);
        if (coverage === 0) expect(a).toBe(0);
        before += coverage > 0.5 ? 1 : 0; // The crisp original mask (smoothstep 0.35..0.65 of coverage).
        after += a;
      }
    expect(Math.abs(after / before - 1)).toBeLessThan(0.1);
  });

  it.each(['billow', 'column', 'smoke'] as const)(
    'assigns every input the %s legacy branch reads before the round-8 branch',
    (pool) => {
      // TSL caches a node into a variable where it is first built. Built first inside the round-8 If, the legacy Else
      // read an unassigned age (0): every legacy puff vanished on the first live frames ('cur' showed no smoke).
      const nodes = new AtmosphereNodes({ realLight: true });
      const effects = new EffectsLayer(nodes) as unknown as Record<string, { mesh: import('three/webgpu').Mesh }>;
      const main = fragmentMain(shaderBytes(effects[pool]!.mesh));
      const branch = main.search(/if \( \( \( object\.nodeUniform\d+ \* object\.nodeUniform\d+ \) > 0\.5 \) \)/);
      expect(branch).toBeGreaterThan(0);
      const before = assigned(main.slice(0, branch));
      const [real, end] = block(main, branch);
      const elseAt = main.indexOf('else', end);
      expect(elseAt - end).toBeLessThan(8);
      const [legacy] = block(main, elseAt);
      const onlyReal = [...assigned(real)].filter((v) => !before.has(v) && !assigned(legacy).has(v));
      expect([...used(legacy)].filter((v) => onlyReal.includes(v))).toEqual([]);
      nodes.dispose();
    },
  );
});
