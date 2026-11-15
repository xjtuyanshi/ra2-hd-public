/** Bounded concurrent offline curved-reconstruction batch; child failures remain visible. */
import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { writeFile, readFile, access } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { externalOutputDirectory } from './cache';
import { summarizeRoundBatch } from './roundSummary';
const flags = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) {
  const k = process.argv[i];
  if (
    ![
      '--ids',
      '--input-root',
      '--out',
      '--concurrency',
      '--seconds',
      '--font',
      '--cell',
      '--python',
      '--mode',
    ].includes(k) ||
    !process.argv[i + 1] ||
    flags.has(k)
  )
    throw new Error(`Invalid ${k}`);
  flags.set(k, process.argv[i + 1]);
}
for (const k of ['--ids', '--input-root', '--out']) if (!flags.has(k)) throw new Error(`Required ${k}`);
const ids = flags.get('--ids')!.split(','),
  concurrency = Number(flags.get('--concurrency') ?? 4),
  here = dirname(fileURLToPath(import.meta.url));
if (
  ids.some((id) => !/^\w+$/.test(id)) ||
  new Set(ids).size !== ids.length ||
  !Number.isInteger(concurrency) ||
  concurrency < 1 ||
  concurrency > 8
)
  throw new Error('Invalid IDs/concurrency');
const out = await externalOutputDirectory(flags.get('--out')!, resolve(here, '../../..'));
try {
  await access(join(out, 'round_batch_status.json'));
  throw new Error('Fresh batch status required');
} catch (e) {
  if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e;
}
const startedAt = new Date().toISOString(),
  units: {
    id: string;
    startedAt: string;
    finishedAt: string;
    exitCode: number;
    signal: string | null;
    automaticMinutes: number;
  }[] = [];
let writing = Promise.resolve();
const persist = (finishedAt?: string) => {
  const snapshot =
    JSON.stringify({ startedAt, finishedAt, configuration: Object.fromEntries(flags), units, ids }, null, 1) + '\n';
  writing = writing.then(() => writeFile(join(out, 'round_batch_status.json'), snapshot));
  return writing;
};
await persist();
let index = 0;
await Promise.all(
  Array.from({ length: Math.min(concurrency, ids.length) }, async () => {
    while (index < ids.length) {
      const id = ids[index++],
        started = performance.now(),
        at = new Date().toISOString(),
        log = createWriteStream(join(out, `round_${id}.log`));
      const args = [
        'exec',
        'tsx',
        join(here, 'round.ts'),
        '--input',
        join(resolve(flags.get('--input-root')!), id),
        '--out',
        join(out, 'units', id),
      ];
      for (const k of ['--seconds', '--font', '--cell', '--python', '--mode'])
        if (flags.has(k)) args.push(k, flags.get(k)!);
      const result = await new Promise<{ exitCode: number; signal: string | null }>((resolveResult) => {
        const child = spawn('pnpm', args, { cwd: resolve(here, '../../..'), stdio: ['ignore', 'pipe', 'pipe'] });
        child.stdout.pipe(log, { end: false });
        child.stderr.pipe(log, { end: false });
        child.on('error', (error) => {
          log.write(String(error) + '\n');
          resolveResult({ exitCode: 1, signal: null });
        });
        child.on('close', (code, signal) => resolveResult({ exitCode: code ?? 1, signal }));
      });
      await new Promise<void>((r) => log.end(r));
      const unit = {
        id,
        startedAt: at,
        finishedAt: new Date().toISOString(),
        automaticMinutes: (performance.now() - started) / 60000,
        ...result,
      };
      units.push(unit);
      await persist();
      console.log(
        `${new Date().toISOString()} ${id}: exit=${result.exitCode}; ${unit.automaticMinutes.toFixed(3)} min`,
      );
    }
  }),
);
await persist(new Date().toISOString());
await summarizeRoundBatch(out);
const failed = units.filter((u) => u.exitCode !== 0);
// Read the persisted status before reporting a completed batch, rather than treating spawn success as acceptance.
console.log(await readFile(join(out, 'round_batch_status.json'), 'utf8'));
if (failed.length) process.exitCode = 1;
