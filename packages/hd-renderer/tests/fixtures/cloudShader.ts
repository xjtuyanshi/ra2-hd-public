import { expect } from 'vitest';

/** Balance nested blocks; stopping at the first closing brace misses sibling-branch regressions. */
function blockAt(shader: string, open: number) {
  expect(shader[open], 'shader block opening brace').toBe('{');
  let depth = 1;
  let end = open + 1;
  for (; depth && end < shader.length; end++) {
    if (shader[end] === '{') depth++;
    if (shader[end] === '}') depth--;
  }
  expect(depth, 'balanced shader block').toBe(0);
  return shader.slice(open + 1, end - 1);
}

function depthAt(body: string, index: number) {
  const prefix = body.slice(0, index);
  return (prefix.match(/\{/g)?.length ?? 0) - (prefix.match(/\}/g)?.length ?? 0);
}

const fetchPattern = /\b(?:textureSample\w*|textureLoad|textureLod|textureGrad|texture|texelFetch)\s*\(\s*(\w+)/g;

/** WGSL layout functions cannot capture texture bindings from main's resource scope. */
export function assertNoTextureLayoutFunctions(shader: string, label = 'shader') {
  for (const declaration of shader.matchAll(/\bfn\s+(\w+)\s*\([^]*?\)[^{]*\{/g)) {
    if (declaration[1] === 'main') continue;
    const body = blockAt(shader, declaration.index! + declaration[0].length - 1);
    expect(body, `${label}: ${declaration[1]} must not sample/load textures`).not.toMatch(
      /\btexture(?:Sample\w*|Load)\s*\(/,
    );
  }
}

/** Audit the inline cache fetch, independent of backend-generated binding numbers. */
/**
 * branchLocal: a view whose coverage is used only inside one uniform branch (the raw cloud debug display) may initialise
 * and fetch inside that branch, provided both sit at the same depth; every material receiver must use main's top level.
 */
export function assertCachedCloudReceiver(
  shader: string,
  label = 'cloud receiver',
  cachedUniform?: string,
  branchLocal = false,
) {
  expect(shader, label).not.toContain('aliveCloudShadow');
  expect(shader, label).not.toContain('aliveCachedCoverage');
  const declaration = shader.match(/\b(?:fn|void) main\s*\([^]*?\)[^{]*\{/);
  expect(declaration, `${label}: main`).not.toBeNull();
  const main = blockAt(shader, declaration!.index! + declaration![0].length - 1);
  const initializers = [...main.matchAll(/\baliveCloudNeutral\s*=\s*0\.0\s*;/g)];
  expect(initializers, `${label}: one fresh zero initializer`).toHaveLength(1);
  const neutral = initializers[0]!.index!;
  if (!branchLocal) expect(depthAt(main, neutral), `${label}: neutral must initialize at main's top level`).toBe(0);

  const gates = [...main.matchAll(/\bif\s*\(([^{};]*)\)\s*\{/g)].filter((gate) =>
    gate[1]!.includes('aliveCoverageEnabled'),
  );
  expect(gates, `${label}: one coverage gate`).toHaveLength(1);
  const gate = gates[0]!;
  const condition = gate[1]!.replace(/[()\s]/g, '');
  const flags = condition.match(/^(?:object\.)?aliveCoverageEnabled>0\.0&&(?:object\.)?(\w+)>0\.0$/);
  expect(flags, `${label}: coverage enabled AND cached uniform`).not.toBeNull();
  const cached = flags![1]!;
  expect(cached, `${label}: cached is a separate flag`).not.toBe('aliveCoverageEnabled');
  if (cachedUniform) expect(cached, `${label}: actual coverageCached binding`).toBe(cachedUniform);
  // Both backends declare scalar uniforms; do not accept a local varying as the cached flag.
  expect(shader, `${label}: cached flag declaration`).toMatch(
    new RegExp(`(?:\\b${cached}\\s*:\\s*f32|\\bfloat\\s+${cached}\\s*;)`),
  );
  expect(gate.index, `${label}: zero precedes gate`).toBeGreaterThan(neutral);
  expect(depthAt(main, gate.index!), `${label}: gate at the initializer's level`).toBe(
    branchLocal ? depthAt(main, neutral) : 0,
  );
  const branch = blockAt(main, gate.index! + gate[0].length - 1);
  const fetches = [...branch.matchAll(fetchPattern)];
  expect(fetches, `${label}: one guarded coverage fetch`).toHaveLength(1);
  const binding = fetches[0]![1];
  expect(
    [...shader.matchAll(fetchPattern)].filter((fetch) => fetch[1] === binding),
    `${label}: coverage texture fetched once across the entire material`,
  ).toHaveLength(1);
  expect(branch, `${label}: fetch writes neutral coverage`).toMatch(/\baliveCloudNeutral\s*=\s*(?!0\.0)[^;]+;/);
  expect(main.match(/\baliveCloudNeutral\s*=/g), `${label}: only zero and fetched assignments`).toHaveLength(2);
  return { main, branch, binding };
}
