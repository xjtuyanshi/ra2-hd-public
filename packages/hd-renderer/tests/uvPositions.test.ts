import { BufferAttribute, BufferGeometry, FloatType, NearestFilter } from 'three/webgpu';
import { expect, it } from 'vitest';
import { uvPositionTexture } from '../src/uvPositions';

it('recovers barycentric world positions at UV texel centres and chooses the caller-visible overlapping layer', () => {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    'position',
    new BufferAttribute(
      new Float32Array([0, 1, 0, 4, 1, 0, 0, 1, 4, 0, 3, 0, 4, 3, 0, 0, 3, 4, 0, 2, 0, 4, 2, 0, 0, 2, 4]),
      3,
    ),
  );
  geometry.setAttribute(
    'artUv',
    new BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), 2),
  );
  geometry.setIndex([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  const map = uvPositionTexture(geometry, 'artUv', 4, 4, [0, 1, 0]);
  try {
    expect(map.type).toBe(FloatType);
    expect(map.minFilter).toBe(NearestFilter);
    const data = map.image.data;
    expect(Array.from(data.slice(0, 4))).toEqual([0.5, 3, 0.5, 1]);
    expect(Array.from(data.slice(4, 8))).toEqual([1.5, 3, 0.5, 1]);
    expect(Array.from(data.slice(-4))).toEqual([0, 0, 0, 0]);
  } finally {
    map.dispose();
    geometry.dispose();
  }
});
