import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import vehicle_pages


def fake_bake(root: Path, frames: int = 16) -> None:
    """A flat 'hull' block: house paint in the left third, steel elsewhere, normals facing up."""
    meta = {'zoom': 15, 'looks': {'rem': {'frames': []}}}
    for look in ('rem', 'exy'):
        (root / look).mkdir(parents=True)
    for k in range(frames):
        name = f'body_{k:03d}'
        black = np.zeros((90, 120, 3), np.uint8)
        white = np.full((90, 120, 3), 245, np.uint8)
        box = (slice(30, 60), slice(30, 90))
        black[box] = white[box] = (137, 137, 113)
        paint = black.copy()
        paint[30:60, 30:50] = (0, 200, 0)
        normal = np.zeros((90, 120, 4), np.uint8)
        normal[box] = (128, 255, 128, 255)
        debug = np.zeros((90, 120, 3), np.uint8)
        debug[box] = (70, 128, 128)
        debug[30:60, 30:50, 0] = 21
        for suffix, image in (('k', black), ('w', white), ('p', paint)):
            Image.fromarray(image).save(root / 'rem' / f'{name}_{suffix}.png')
        Image.fromarray(normal).save(root / 'rem' / f'{name}_n.png')
        Image.fromarray(debug).save(root / 'exy' / f'{name}_k.png')
        meta['looks']['rem']['frames'].append({'k': k, 'facing': k * 256 // frames, 'part': 'body',
                                               'pivot': [60.0, 45.0]})
    (root / 'meta.json').write_text(json.dumps(meta))


def palette_png(path: Path) -> Path:
    palette = np.zeros((1, 256, 3), np.uint8)
    palette[0, 16:32] = [(255 - 15 * i, 0, 0) for i in range(16)]
    palette[0, 70] = (137, 137, 113)
    Image.fromarray(palette).save(path)
    return path


class VehiclePagesTest(unittest.TestCase):
    def test_build_keeps_the_mesh_silhouette_and_marks_baked_shading(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            fake_bake(tmp_path / 'bake')
            manifest = vehicle_pages.build('TEST', tmp_path / 'bake', palette_png(tmp_path / 'unit.png'), tmp_path / 'out',
                                           page_zoom=5, review=tmp_path / 'review')
            assert manifest['shading'] == 'baked'
            assert manifest['frames'] == 16 and manifest['facingStep'] == 16
            assert np.allclose(manifest['jacobian'], vehicle_pages.BAKE_JACOBIAN * 1.25)
            shape = json.loads((tmp_path / 'review' / 'TEST_shape.json').read_text())
            assert min(s['iou_mesh'] for s in shape) == 1.0
            frame = manifest['parts']['body']['frames'][0]
            # The 60x30 bake box shrinks with the page: bake zoom 15 -> page zoom 5.
            x, y, w, h = frame['rect']
            assert 18 <= w <= 22 and 8 <= h <= 12
            paint = np.asarray(Image.open(tmp_path / 'out' / 'TEST_body_paint.png'))
            colour = np.asarray(Image.open(tmp_path / 'out' / 'TEST_body.png'))
            assert paint[..., 0].max() == 255  # house area found
            assert colour[..., 3].max() == 255

    def test_coverage_normalises_tone_mapped_backgrounds(self):
        black = np.zeros((10, 10, 3), np.float32)
        white = np.full((10, 10, 3), 245, np.float32)
        white[5:, 5:] = black[5:, 5:] = 100
        alpha = vehicle_pages.coverage(black, white)
        assert alpha[0, 0] == 0 and alpha[9, 9] == 1

    def test_view_direction_points_up(self):
        assert vehicle_pages.view_direction()[1] > 0


if __name__ == '__main__':
    unittest.main()
