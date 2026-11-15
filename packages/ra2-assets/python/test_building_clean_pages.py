import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from building_clean_pages import run


class BuildingCleanPagesTest(unittest.TestCase):
    def test_overlay_install_preserves_source_and_nonbuilding_art(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, farm = root / 'source', root / 'farm'
            library = source / '_library' / 'temperate'
            library.mkdir(parents=True)
            farm.mkdir()
            (farm / '_library').symlink_to(source / '_library', target_is_directory=True)
            y, x = np.mgrid[:24, :48]
            rgba = np.full((24, 48, 4), 255, np.uint8)
            rgba[:, :, :3] = np.repeat((100 + (x + y) % 2 * 20)[:, :, None], 3, axis=2)
            Image.fromarray(rgba).save(library / 'colour.png')
            Image.fromarray(np.zeros((24, 48), np.uint8)).save(library / 'paint.png')
            data = {'palettes': {}, 'spriteAtlasesHd': [
                {'file': 'colour.png', 'remap': 'paint.png', 'width': 48, 'height': 24, 'scale': 1}],
                'types': {'BASE': {'kind': 'building', 'sprite': 'building:BASE', 'building': {
                    'layers': [{'sprite': 'anim:BASE'}]}}, 'ORE': {'kind': 'overlay', 'sprite': 'ore:ORE'}},
                'sprites': {
                    'building:BASE': {'frames': [{'rect': [0, 0, 24, 24]}, {'rect': [0, 0, 0, 0]}]},
                    'anim:BASE': {'frames': [{'rect': [0, 0, 24, 24]}]},
                    'ore:ORE': {'frames': [{'rect': [24, 0, 24, 24]}]}}}
            (library / 'library.json').write_text(json.dumps(data))
            source_bytes = (library / 'colour.png').read_bytes()
            report = run(source, farm, 'temperate', root / 'report.json')
            self.assertEqual((library / 'colour.png').read_bytes(), source_bytes)
            installed = farm / '_library' / 'temperate'
            sidecar = json.loads((installed / 'buildings-clean.json').read_text())
            output = np.array(Image.open(installed / sidecar['files'][0]))
            np.testing.assert_array_equal(output[:, 24:], rgba[:, 24:])
            np.testing.assert_array_equal(output[:, :, 3], rgba[:, :, 3])
            self.assertFalse(np.array_equal(output[:, :24], rgba[:, :24]))
            self.assertEqual(report['pages'][0]['rectangles'], 1)
            self.assertFalse((library / 'buildings-clean.json').exists())
            with self.assertRaises(ValueError):
                run(source, source, 'temperate', root / 'bad.json')


if __name__ == '__main__':
    unittest.main()
