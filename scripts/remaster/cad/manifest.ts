/** Hash an assembled offline delivery without changing source resources. */
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256 } from './cache';

export async function writeDeliveryManifest(directory: string, metadata: Record<string, unknown>): Promise<void> {
  const files: Record<string, { sha256: string; bytes: number }> = {};
  for (const name of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (name.isSymbolicLink()) throw new Error('Delivery must not contain symbolic links');
    if (!name.isFile() || !/\.(png|json|jsonl|txt|md)$/.test(name.name) || name.name === 'manifest.json') continue;
    const bytes = await readFile(join(directory, name.name));
    files[name.name] = { sha256: sha256(bytes), bytes: bytes.length };
  }
  await writeFile(
    join(directory, 'manifest.json'),
    JSON.stringify({ ...metadata, version: 1, generatedAt: new Date().toISOString(), files }, null, 1) + '\n',
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const dir = process.argv[2];
  if (!dir || process.argv.length !== 3) throw new Error('Usage: manifest.ts <delivery-directory>');
  const metadata = JSON.parse(await readFile(join(dir, 'manifest.json'), 'utf8')) as Record<string, unknown>;
  delete metadata.files;
  await writeDeliveryManifest(dir, metadata);
}
