/** Rebuild an offline batch summary from actual process, metric and author-review evidence. */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { FitAcceptance } from '../../../packages/hd-renderer/src/hardSurfaceRefine';
import { externalOutputDirectory } from './cache';

export interface BatchStatus {
  startedAt: string;
  finishedAt?: string;
  preGate: boolean;
  configuration: Record<string, unknown>;
  units: { id: string; startedAt: string; finishedAt: string; exitCode: number; automaticMinutes: number }[];
}
async function optionalJson<T>(path: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(path, 'utf8')) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw error;
  }
}
export async function summarizeBatch(out: string): Promise<void> {
  const status = JSON.parse(await readFile(join(out, 'batch_status.json'), 'utf8')) as BatchStatus;
  const rows: string[] = [],
    details: Record<string, unknown> = {};
  let passes = 0;
  for (const unit of status.units) {
    const dir = join(out, 'units', unit.id),
      metrics = await optionalJson<Record<string, FitAcceptance>>(join(dir, 'metrics.json')),
      manifest = await optionalJson<{ metricPass: boolean }>(join(dir, 'manifest.json')),
      review = await optionalJson<{ pass: boolean; agentDirectedMinutes: number; reason: string }>(
        join(dir, 'visual_review.json'),
      );
    const numericPass = unit.exitCode === 0 && manifest?.metricPass === true,
      fullPass = numericPass && review?.pass === true;
    if (fullPass) passes++;
    const parts = metrics ? Object.entries(metrics) : [],
      format = (fn: (key: string, value: FitAcceptance) => string) =>
        parts.map(([k, v]) => fn(k, v)).join('<br>') || 'MISSING',
      primitives = format((k, v) => `${k}: ${v.metrics.primitives}`),
      iou = format((k, v) => `${k}: ${Math.min(...v.silhouettes.map((s) => s.iou)).toFixed(6)}`),
      p99 = format((k, v) => `${k}: ${v.metrics.dev_vox_to_cad.p99}/${v.metrics.dev_cad_to_vox.p99}`),
      result = fullPass ? 'PASS' : !numericPass || review?.pass === false ? 'FAIL' : 'UNVERIFIED';
    const hand = review ? `${review.agentDirectedMinutes.toFixed(3)} (agent review); human UNVERIFIED` : 'UNVERIFIED';
    rows.push(
      `| [${unit.id}](units/${unit.id}/REPORT.md) | ${primitives} | ${iou} | ${p99} | ${result} | ${unit.automaticMinutes.toFixed(3)} | ${hand} |`,
    );
    details[unit.id] = { ...unit, numericPass, visualReview: review ?? 'UNVERIFIED', fullPass, result };
  }
  await writeFile(
    join(out, 'batch_acceptance.json'),
    JSON.stringify(
      { generatedAt: new Date().toISOString(), preGate: status.preGate, fullPasses: passes, units: details },
      null,
      1,
    ) + '\n',
  );
  await writeFile(
    join(out, 'SUMMARY.md'),
    `# 十车 CAD 批量结果\n\n批次：**${status.preGate ? 'pre-gate' : 'gated'}**；完整验收通过 **${passes}/${status.units.length}**。开始 ${status.startedAt}；结束 ${status.finishedAt ?? '运行中'}。\n\n正式门槛未改：每部件双向 p99 ≤0.85；每部件和整车八向 IoU ≥0.97；body ≤130、turret ≤35、每根检测炮管 ≤3 原语。p99 列顺序为 voxel→CAD / CAD→voxel。原车型没有的部件不列出，不作假通过。FAIL 包含数值、视觉或进程交付失败；未审阅不是 PASS。\n\n| 车型 | 展开原语 | 每部件最低 IoU | 每部件 p99 | 完整验收 | 自动分钟 | 手工指导分钟 |\n| --- | --- | --- | --- | --- | ---: | --- |\n${rows.join('\n')}\n\n每车目录含 board_当前_新 的 1×/2×、turntable_8、全部逐向指标、自动工时、候选日志和源快照。1× 是研究视距 2.5 的离线图，不是实机截图。\n\n手工列记录作者代理实际打开图板的墙钟窗口；active human minutes 均 UNVERIFIED。模型坐标/种子手工编辑为零，不能等同于零总人工。共享工具开发、诊断、配置、汇总在 [HOURS.md](HOURS.md) 单列；没有用失败模型证明合格模型 <10 分钟/辆。并行任务共享主机，自动分钟是观察到的单任务墙钟，不是受控吞吐基准。\n\n用户审美认可、GPU/运行时、交互、性能和实战均 UNVERIFIED；本批次没有启动游戏 VM。\n`,
  );
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('summary.ts BATCH_DIRECTORY');
  const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
  await summarizeBatch(await externalOutputDirectory(process.argv[2], repo));
}
