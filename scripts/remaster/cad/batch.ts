/** Bounded concurrent offline runs; a failed vehicle never removes the remaining vehicles from the batch. */
import { appendFile, writeFile, rename } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { spawn } from 'node:child_process';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { externalOutputDirectory } from './cache';
import { summarizeBatch, type BatchStatus } from './summary';

const flags = new Map<string, string>(),
  args = process.argv.slice(2);
let preGate = false;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--pre-gate') {
    preGate = true;
    continue;
  }
  if (
    ![
      '--out',
      '--ids',
      '--cache-root',
      '--palette',
      '--house',
      '--concurrency',
      '--evaluations',
      '--seconds',
      '--font',
      '--cell',
      '--python',
    ].includes(args[i]) ||
    !args[i + 1] ||
    args[i + 1].startsWith('--') ||
    flags.has(args[i])
  )
    throw new Error(`Invalid batch argument ${args[i]}`);
  flags.set(args[i], args[++i]);
}
for (const k of ['--out', '--ids', '--cache-root', '--palette', '--house'])
  if (!flags.has(k)) throw new Error(`Required ${k}`);
// Gated batches require external visual acceptance. This command records its caller's choice,
// rather than fabricating an automatic visual PASS from numerical scores.
const ids = flags.get('--ids')!.split(','),
  concurrency = Number(flags.get('--concurrency') ?? 4);
if (!ids.length || new Set(ids).size !== ids.length || ids.some((id) => !/^[A-Za-z0-9_-]+$/.test(id)))
  throw new Error('Invalid or duplicate model IDs');
if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 16)
  throw new Error('Concurrency must be 1..16');
const here = dirname(fileURLToPath(import.meta.url)),
  repo = resolve(here, '../../..'),
  out = await externalOutputDirectory(flags.get('--out')!, repo),
  status: BatchStatus = {
    startedAt: new Date().toISOString(),
    preGate,
    configuration: Object.fromEntries(flags),
    units: [],
  };
const log = async (message: string) => {
  const line = `${new Date().toISOString()} ${message}\n`;
  process.stdout.write(line);
  await appendFile(join(out, 'progress.txt'), line);
};
await log(`Batch start ${preGate ? 'pre-gate' : 'gated'}; ${ids.join(',')}; concurrency=${concurrency}`);
let statusWrite = Promise.resolve();
const saveStatus = () => {
  statusWrite = statusWrite.then(async () => {
    await writeFile(join(out, '.batch_status.tmp'), JSON.stringify(status, null, 1) + '\n');
    await rename(join(out, '.batch_status.tmp'), join(out, 'batch_status.json'));
  });
  return statusWrite;
};
await saveStatus();
let cursor = 0;
await Promise.all(
  Array.from({ length: Math.min(concurrency, ids.length) }, async () => {
    while (cursor < ids.length) {
      const id = ids[cursor++],
        startedAt = new Date().toISOString(),
        start = performance.now(),
        output = createWriteStream(join(out, `batch_${id}.log`)),
        argv = ['--import', 'tsx', join(here, 'run.ts'), '--id', id, '--out', join(out, 'units', id)];
      for (const [key, value] of flags) if (!['--out', '--ids', '--concurrency'].includes(key)) argv.push(key, value);
      await log(`Start ${id}`);
      const exitCode = await new Promise<number>((done) => {
        const child = spawn(process.execPath, argv, { cwd: repo, stdio: ['ignore', 'pipe', 'pipe'] });
        child.stdout.pipe(output, { end: false });
        child.stderr.pipe(output, { end: false });
        child.once('error', (error) => output.write(`${error}\n`));
        child.once('close', (code) => {
          output.end();
          done(code ?? -1);
        });
      });
      status.units.push({
        id,
        startedAt,
        finishedAt: new Date().toISOString(),
        exitCode,
        automaticMinutes: (performance.now() - start) / 60000,
      });
      status.units.sort((a, b) => ids.indexOf(a.id) - ids.indexOf(b.id));
      await saveStatus();
      await log(`Finished ${id}; exit=${exitCode}`);
    }
  }),
);
status.finishedAt = new Date().toISOString();
await saveStatus();
await summarizeBatch(out);
await log('Batch complete; numerical/visual/process results remain separate');
if (status.units.some((u) => u.exitCode !== 0)) process.exitCode = 1;
