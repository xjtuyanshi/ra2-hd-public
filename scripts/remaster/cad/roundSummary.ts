/** Revised gate: assembly is hard, p99 is advisory, and actual visual review is mandatory. */
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPart, type HardSurfaceSpec } from '../../../packages/hd-renderer/src/hardSurfaceMesh';
import type { HardSurfaceMetrics } from '../../../packages/hd-renderer/src/hardSurfaceMetrics';
import type { SilhouetteComparison } from '../../../packages/hd-renderer/src/hardSurfaceSilhouette';
import { occupiedCells } from '../../../packages/hd-renderer/src/hardSurfacePropose';
import { tubeLanes, circularTubeGroups } from '../../../packages/hd-renderer/src/hardSurfaceTube';
import type { SolidGrid } from '../../../packages/hd-renderer/src/hardSurfaceSolid';
interface Review {
  pass: boolean;
  reason: string;
  agentDirectedMinutes: number;
  sourceNothingAdded: boolean;
  proportionsPreserved: boolean;
  scalesOpened: string[];
}
const optional = async <T>(p: string): Promise<T | undefined> => {
  try {
    return JSON.parse(await readFile(p, 'utf8')) as T;
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return;
    throw e;
  }
};
export async function summarizeRoundBatch(out: string): Promise<void> {
  const status = await optional<{
    startedAt: string;
    finishedAt?: string;
    ids: string[];
    units: { id: string; exitCode: number; automaticMinutes: number }[];
  }>(join(out, 'round_batch_status.json'));
  if (!status) throw new Error('Missing batch process evidence');
  const rows: string[] = [],
    details: Record<string, unknown> = {};
  let passes = 0,
    hardPasses = 0;
  for (const id of status.ids) {
    const dir = join(out, 'units', id),
      process = status.units.find((u) => u.id === id),
      m = await optional<Record<string, { metrics: HardSurfaceMetrics; silhouettes: SilhouetteComparison[] }>>(
        join(dir, 'metrics.json'),
      ),
      silhouette = await optional<{ assembled: SilhouetteComparison[] }>(join(dir, 'silhouette.json')),
      review = await optional<Review>(join(dir, 'visual_review.json')),
      spec = await optional<HardSurfaceSpec>(join(dir, 'spec.json')),
      ex = await optional<{ parts: Record<string, { role: string; solid: SolidGrid }> }>(join(dir, 'export.json'));
    const assembledMin = silhouette?.assembled.length === 8 ? Math.min(...silhouette.assembled.map((s) => s.iou)) : NaN;
    const parts = m ? Object.entries(m) : [],
      actual: Record<string, number> = {},
      budgets: Record<string, number> = {};
    let budgetPass = !!spec && !!ex;
    if (spec && ex)
      for (const [key, part] of Object.entries(spec.parts)) {
        actual[key] = buildPart(part).primitives.length;
        const role = ex.parts[key].role;
        budgets[key] =
          role === 'barrel'
            ? 3 * tubeLanes(occupiedCells(ex.parts[key].solid)).length
            : Math.floor((role === 'body' ? 130 : 35) / Object.values(ex.parts).filter((p) => p.role === role).length);
        if (actual[key] > budgets[key]) budgetPass = false;
        if (role === 'barrel') {
          try {
            const groups = circularTubeGroups(part.primitives);
            if (groups.size !== budgets[key] / 3 || [...groups.values()].some((g) => g.length > 3)) budgetPass = false;
          } catch {
            budgetPass = false;
          }
        }
      }
    const completeMetrics =
      !!spec &&
      !!ex &&
      Object.keys(ex.parts).length > 0 &&
      Object.keys(spec.parts).length === Object.keys(ex.parts).length &&
      parts.length === Object.keys(ex.parts).length &&
      parts.every(
        ([key, v]) =>
          actual[key] === v.metrics.primitives &&
          v.silhouettes.length === 8 &&
          v.silhouettes.every((s) => Number.isFinite(s.iou) && s.iou >= 0 && s.iou <= 1),
      );
    const hard =
      process?.exitCode === 0 && Number.isFinite(assembledMin) && assembledMin >= 0.97 && budgetPass && completeMetrics;
    if (hard) hardPasses++;
    const p99guide =
      parts.length > 0 &&
      parts.every(([, v]) =>
        [v.metrics.dev_vox_to_cad.p99, v.metrics.dev_cad_to_vox.p99].every((x) => Number.isFinite(x) && x <= 1),
      );
    const barrels = parts.filter(([k]) => ex?.parts[k].role === 'barrel'),
      barrelTarget = barrels.every(([, v]) => v.silhouettes.length === 8 && v.silhouettes.every((s) => s.iou >= 0.9));
    const visual =
      review?.pass === true &&
      review.sourceNothingAdded === true &&
      review.proportionsPreserved === true &&
      ['1x', '2x', 'turntable_8'].every((s) => review.scalesOpened.includes(s));
    const full = hard && visual;
    if (full) passes++;
    const result = full ? 'PASS' : !hard || review?.pass === false ? 'FAIL' : 'UNVERIFIED';
    const fmt = (fn: (k: string, v: { metrics: HardSurfaceMetrics; silhouettes: SilhouetteComparison[] }) => string) =>
      parts.map(([k, v]) => fn(k, v)).join('<br>') || 'MISSING';
    const partIoU = fmt((k, v) => `${k}: ${Math.min(...v.silhouettes.map((s) => s.iou)).toFixed(6)}`),
      p99 = fmt((k, v) => `${k}: ${v.metrics.dev_vox_to_cad.p99}/${v.metrics.dev_cad_to_vox.p99}`);
    rows.push(
      `| [${id}](units/${id}/REPORT.md) | ${Number.isFinite(assembledMin) ? assembledMin.toFixed(6) : 'MISSING'} ${hard ? 'PASS' : 'FAIL'} | ${partIoU} | ${p99}${p99guide ? '' : ' (guide未达)'} | ${Object.entries(
        actual,
      )
        .map(([k, n]) => `${k}: ${n}/${budgets[k]}`)
        .join(
          '<br>',
        )} | ${barrels.length ? (barrelTarget ? '达到' : '未达') : '不适用'} | ${process?.automaticMinutes.toFixed(3) ?? 'MISSING'} | ${result} | ${review?.reason ?? 'UNVERIFIED：尚未实际审阅'} |`,
    );
    details[id] = {
      process: process ?? 'MISSING',
      assembledMin,
      budgetPass,
      completeMetrics,
      actual,
      budgets,
      assembledHardPass: hard,
      p99guide,
      barrelTarget,
      barrelApplicable: barrels.length > 0,
      review: review ?? 'UNVERIFIED',
      fullPass: full,
      result,
    };
  }
  const generatedAt = new Date().toISOString();
  await writeFile(
    join(out, 'round_acceptance.json'),
    JSON.stringify(
      { generatedAt, gate: 'REVISED 2026-11-15', hardPasses, fullPasses: passes, units: details },
      null,
      1,
    ) + '\n',
  );
  await writeFile(
    join(out, 'SUMMARY.md'),
    `# 十车圆件 CAD 复测\n\n整车硬门槛 **${hardPasses}/${status.ids.length}**；含作者视觉检查的 REVISED gate **${passes}/${status.ids.length} PASS**。生成 ${generatedAt}。\n\n硬门槛为整车全部8向 IoU≥0.97，且保留展开预算 body≤130、turret≤35、每根检测炮管≤3；无新增部件、比例与无阶梯检查仍须逐车目视通过。p99≤1.0是参考，不擅自升级为硬门槛。炮管每部件IoU≥0.9为单列目标，未达不隐藏，也不冒充达到。\n\n| 车型 | 整车最低IoU（HARD） | 各部件最低IoU | 双向p99（guide≤1.0） | 展开原语/预算 | 炮管0.9目标 | 本轮自动分钟 | REVISED | 本人1:1观察 |\n| --- | --- | --- | --- | --- | --- | ---: | --- | --- |\n${rows.join('\n')}\n\n[十车总览1×](overview_10.png)、[中文报告](REPORT.md)、[共享工时](HOURS.md)、[逐时进展](progress.txt)、[进程状态](round_batch_status.json)、[完整判定](round_acceptance.json)。各车目录含当前/新1×、严格最近邻2×与八向转台。1×沿用研究视距2.5，都是离线同相机/光照图，不是VM或GPU实机截图。\n\n自动分钟是本轮检测、源支持候选搜索、原始正式指标重测和出图的子进程墙钟；重用的上一轮自动车体不是本轮从零重建，原始来源及其时间另存每车source_snapshot.json / HOURS.md。并行时间相互重叠，不能相加当吞吐或人工时间。共享工具开发、诊断及作者审阅另记，active human minutes UNVERIFIED。\n\n涂装/磨损层冻结。用户最终审美认可、VM、GPU、实机交互、性能与长局均 UNVERIFIED；本轮没有启动VM。\n`,
  );
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('roundSummary.ts BATCH_DIRECTORY');
  await summarizeRoundBatch(resolve(process.argv[2]));
}
