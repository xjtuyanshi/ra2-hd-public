import unittest
import cv2
import numpy as np
from shape_locked_denoise import denoise, metrics


class ShapeLockedDenoiseTest(unittest.TestCase):
    def test_noise_weathering_and_coverage(self):
        y, x = np.mgrid[:128, :160]
        broad = 100 + 28 * np.sin(x / 17) + 15 * np.sin(y / 21)
        rgb = np.repeat((broad + 12 * ((x + y) % 2 * 2 - 1))[:, :, None], 3, axis=2)
        source = np.zeros((128, 160, 4), np.uint8)
        source[:, :, :3] = rgb.astype(np.uint8)
        source[4:-4, 4:-4, 3] = 255
        remap = np.zeros(source.shape[:2], np.uint8)
        remap[:, 80:] = 255
        result = denoise(source, remap)
        m = metrics(source, result, remap)
        self.assertTrue(m['alpha_exact'])
        self.assertTrue(m['boundary_rgb_exact'])
        self.assertLess(m['after']['hp_rms'], m['before']['hp_rms'] * .5)
        for before, after in zip(m['before']['variance'], m['after']['variance']):
            self.assertGreaterEqual(after / before, .8)
        np.testing.assert_array_equal(result, denoise(source, remap))
        np.testing.assert_array_equal(source[:, 79:82], result[:, 79:82])

    def test_dramatic_reduces_flat_speckle_against_tuned_and_keeps_coarse_wear(self):
        y, x = np.mgrid[:160, :192]
        source = np.full((160, 192, 4), 255, np.uint8)
        broad = 125 + 30 * np.sin(x / 18) + 12 * np.sin(y / 27)
        source[:, :, :3] = np.repeat((broad + 18 * ((x + y) % 2 * 2 - 1))[:, :, None], 3, axis=2)
        source[:3, :, 3] = 0
        remap = np.zeros((160, 192), np.uint8)
        remap[:, 96:] = 255
        tuned = denoise(source, remap, metal_lift=.035)
        dramatic = denoise(source, remap, dramatic=True)
        old, new = metrics(source, tuned, remap), metrics(source, dramatic, remap)
        self.assertLess(new['after']['safe_flat_hp_rms'], old['after']['safe_flat_hp_rms'] * .25)
        self.assertTrue(new['alpha_exact'])
        self.assertTrue(new['boundary_rgb_exact'])
        for before, after in zip(new['before']['variance'], new['after']['variance']):
            self.assertGreaterEqual(after / before, .8)
        np.testing.assert_array_equal(dramatic, denoise(source, remap, dramatic=True))

    def test_constant_and_tiny_pages(self):
        for size in [1, 2, 3, 16]:
            source = np.full((size, size, 4), 255, np.uint8)
            source[:, :, :3] = [42, 99, 103]
            np.testing.assert_array_equal(denoise(source, np.zeros((size, size), np.uint8)), source)

    def test_optional_metal_lift_never_reencodes_paint_or_coverage(self):
        source = np.full((32, 64, 4), 255, np.uint8)
        source[:, :, :3] = 80
        remap = np.zeros((32, 64), np.uint8)
        remap[:, 32:] = 255
        after = denoise(source, remap, metal_lift=.035)
        self.assertGreater(after[16, 16, 0], source[16, 16, 0])
        np.testing.assert_array_equal(after[:, 32:], source[:, 32:])
        np.testing.assert_array_equal(after[:, :, 3], source[:, :, 3])


if __name__ == '__main__':
    unittest.main()
