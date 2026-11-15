import type { Axis, Vec3 } from './hardSurfaceSolid';

/** Separate parallel tubes in the transverse plane without bridging source-empty space. */
function transverseComponents(cells: readonly Vec3[], axis: Axis): Vec3[][] {
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    key = (p: Vec3) => `${p[uv[0]]},${p[uv[1]]}`,
    keys = new Set(cells.map(key)),
    groups: Vec3[][] = [];
  while (keys.size) {
    const first = keys.values().next().value!,
      queue = [first],
      component = new Set([first]);
    keys.delete(first);
    for (let i = 0; i < queue.length; i++) {
      const [u, v] = queue[i].split(',').map(Number);
      for (const [du, dv] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const k = `${u + du},${v + dv}`;
        if (keys.delete(k)) {
          component.add(k);
          queue.push(k);
        }
      }
    }
    groups.push(cells.filter((p) => component.has(key(p))));
  }
  return groups;
}

export function tubeLanes(cells: readonly Vec3[], axis: Axis = 'x'): Vec3[][] {
  const union = transverseComponents(cells, axis);
  if (union.length !== 1) return union;
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a),
    axial = [...new Set(cells.map((p) => p[a]))].sort((a, b) => a - b);
  if (axial.length < 6) return union;
  const slices = axial.map((x) =>
      transverseComponents(
        cells.filter((p) => p[a] === x),
        axis,
      ),
    ),
    counts = new Map<number, number>();
  for (const slice of slices) counts.set(slice.length, (counts.get(slice.length) ?? 0) + 1);
  const modal = [...counts].sort((p, q) => q[1] - p[1] || p[0] - q[0])[0];
  // A shared short root must not join two long tubes. Require a majority of independently
  // disconnected axial slices; isolated holes or a split muzzle alone cannot create a tube.
  if (modal[0] < 2 || modal[1] <= axial.length / 2) return union;
  const centres = slices
    .filter((s) => s.length === modal[0])
    .map((s) =>
      s
        .map((g) => uv.map((u) => g.reduce((sum, p) => sum + p[u] + 0.5, 0) / g.length))
        .sort((p, q) => p[0] - q[0] || p[1] - q[1]),
    );
  const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)];
  const axes = Array.from({ length: modal[0] }, (_, i) => [0, 1].map((k) => median(centres.map((s) => s[i][k]))));
  const groups = axes.map(() => [] as Vec3[]);
  for (const p of cells) {
    const distances = axes.map((c) => uv.reduce((sum, u, k) => sum + (p[u] + 0.5 - c[k]) ** 2, 0));
    groups[distances.indexOf(Math.min(...distances))].push(p);
  }
  return groups;
}

/**
 * Dynamic programming over source cross-section changes, not arbitrary quarter-length cuts.
 * The budget applies independently to each lane. Tiny end changes may share a segment; no
 * transverse partitions or voxel boxes are created. The caller fits one shared axis.
 */
export function tubeSegments(cells: readonly Vec3[], axis: Axis, budget: number): Vec3[][] {
  if (!Number.isSafeInteger(budget) || budget < 1) throw new Error('Positive tube segment budget required');
  const a = 'xyz'.indexOf(axis),
    uv = [0, 1, 2].filter((k) => k !== a);
  const slices = new Map<number, Vec3[]>();
  for (const p of cells) {
    const row = slices.get(p[a]);
    if (row) row.push(p);
    else slices.set(p[a], [p]);
  }
  if (!slices.size) throw new Error('Cannot segment an empty tube');
  const axial = [...slices.keys()].sort((a, b) => a - b),
    runs: Vec3[][] = [];
  for (let i = 0; i < axial.length; i++) {
    if (!i || axial[i] !== axial[i - 1] + 1) runs.push([]);
    runs[runs.length - 1].push(...slices.get(axial[i])!);
  }
  if (runs.length > budget) throw new Error('Disconnected axial runs exceed tube segment budget');
  if (runs.length > 1)
    return runs.flatMap((g, i) =>
      tubeSegments(g, axis, Math.floor(budget / runs.length) + (i < budget % runs.length ? 1 : 0)),
    );
  const lo = Math.min(...slices.keys()),
    hi = Math.max(...slices.keys()) + 1;
  const signature = (x: number) =>
    (slices.get(x) ?? [])
      .map((p) => `${p[uv[0]]},${p[uv[1]]}`)
      .sort()
      .join(';');
  const cuts = [lo];
  for (let x = lo + 1; x < hi; x++) if (signature(x) !== signature(x - 1)) cuts.push(x);
  cuts.push(hi);
  // Cross-section area and both spans distinguish sleeves from equal-area eccentric noise.
  const profile = Array.from({ length: hi - lo }, (_, i) => {
    const row = slices.get(lo + i) ?? [];
    return [
      row.length,
      ...uv.map((u) => (row.length ? Math.max(...row.map((p) => p[u])) - Math.min(...row.map((p) => p[u])) + 1 : 0)),
    ];
  });
  const cost = (l: number, h: number) => {
    const rows = profile.slice(l - lo, h - lo),
      mean = [0, 1, 2].map((k) => rows.reduce((s, p) => s + p[k], 0) / rows.length);
    return rows.reduce((s, p) => s + p.reduce((t, v, k) => t + (v - mean[k]) ** 2 / Math.max(1, mean[k]), 0), 0);
  };
  const dp = Array.from({ length: budget + 1 }, () => new Array<number>(cuts.length).fill(Infinity)),
    prev = dp.map(() => new Array<number>(cuts.length).fill(-1));
  dp[0][0] = 0;
  for (let n = 1; n <= budget; n++)
    for (let j = 1; j < cuts.length; j++)
      for (let i = 0; i < j; i++) {
        // Interior sleeves need axial extent. A source-supported terminal muzzle/cap may
        // occupy one slice; it is still a coaxial circular segment, never a voxel box.
        if (cuts[j] - cuts[i] < 2 && hi - lo > 1 && cuts[i] !== lo && cuts[j] !== hi) continue;
        const candidate = dp[n - 1][i] + cost(cuts[i], cuts[j]) + 0.15;
        if (candidate < dp[n][j]) {
          dp[n][j] = candidate;
          prev[n][j] = i;
        }
      }
  let n = 1;
  for (let k = 2; k <= budget; k++) if (dp[k][cuts.length - 1] < dp[n][cuts.length - 1]) n = k;
  const ranges: [number, number][] = [];
  for (let j = cuts.length - 1; j > 0; n--) {
    const i = prev[n][j];
    if (i < 0) return [[...cells]];
    ranges.unshift([cuts[i], cuts[j]]);
    j = i;
  }
  return ranges.map(([l, h]) => cells.filter((p) => p[a] >= l && p[a] < h));
}
