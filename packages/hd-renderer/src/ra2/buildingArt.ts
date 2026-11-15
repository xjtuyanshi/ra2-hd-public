import type { HdAtlasRef } from './cache';

/** A page variant can replace colour only; layout, remap and all non-building texels stay source-owned. */
export function buildingArtPages(original: (HdAtlasRef | null)[], sidecar: unknown): (HdAtlasRef | null)[] {
  const candidate = sidecar as { version?: unknown; files?: unknown } | null;
  if (candidate?.version !== 1 || !Array.isArray(candidate.files) || candidate.files.length !== original.length)
    throw new Error('Invalid building-art sidecar');
  const files = candidate.files;
  return original.map((page, index) => {
    const file = files[index];
    if (file === null) return page;
    if (!page || typeof file !== 'string' || !/^[\w@.-]+\.png$/.test(file) || file.includes('..'))
      throw new Error(`Invalid building-art page ${index}`);
    return { ...page, file };
  });
}
