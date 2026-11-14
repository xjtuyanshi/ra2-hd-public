import type { HdCacheManifest, TileEntry } from '../../../ra2-assets/src/manifest';

/** Clear/rough tiles may contain lawns; their sampled green art supplies the final mask, never land type alone. */
export function canHaveLivingGrass(theater: HdCacheManifest['theater'], tile: Pick<TileEntry, 'terrainType'>): boolean {
  return (
    (theater === 'temperate' || theater === 'urban' || theater === 'newurban') &&
    (tile.terrainType === 13 || tile.terrainType === 14)
  );
}
