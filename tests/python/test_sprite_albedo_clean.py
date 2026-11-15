"""Asset-free contracts for offline albedo cleanup and body-derived animation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[2] / 'packages/ra2-assets/python'))
from sprite_albedo_clean import clean, fractional_mask, transfer_layer, matching_context


def test_alpha_and_broad_edge_survive_dedither():
    original = np.full((20, 24, 3), 70, np.uint8); original[:, 12:] = 170
    hd = np.repeat(np.repeat(original, 2, 0), 2, 1).astype(float)
    checker = (np.indices(hd.shape[:2]).sum(0) % 2 * 2 - 1) * 12
    hd += checker[..., None]
    alpha = np.arange(40*48, dtype=np.uint16).reshape(40, 48).astype(np.uint8)
    alpha[5:-5, 5:-5] = 255
    rgba = np.dstack([hd.astype(np.uint8), alpha])
    out = clean(rgba, original)
    assert np.array_equal(out[..., 3], alpha)
    assert out[10:30, 5:15, :3].std() < rgba[10:30, 5:15, :3].std() / 2
    assert out[20, 30, 0] - out[20, 15, 0] > 80


def test_fractional_mask_has_exact_index_policy_and_coverage():
    idx = np.array([[0, 16, 31, 32], [15, 16, 31, 255]], np.uint8)
    m = fractional_mask(idx, 16, 31)
    assert m.shape == (4, 8)
    assert ((m > 0) & (m < 255)).any()
    assert m[:, 0].max() == 0 and m[:, -1].max() == 0


def test_transfer_copies_static_body_preserves_alpha_and_emissive():
    idx = np.full((6, 8), 40, np.uint8)
    body = np.full((12, 16, 4), 120, np.uint8); body[..., 3] = 255
    layer = np.full_like(body, 119); layer[..., 3] = np.arange(192).reshape(12, 16)
    pal = np.full((256, 3), 90, np.uint8)
    unlit = np.zeros(idx.shape, bool); unlit[2, 3] = True
    out, seam = transfer_layer(idx, layer, (2, 4), idx, body, (2, 4), pal, unlit)
    assert np.array_equal(out[..., 3], layer[..., 3])
    assert np.array_equal(out[4:6, 6:8, :3], layer[4:6, 6:8, :3])
    assert seam == 0

def test_cleanup_preserves_native_means_and_local_damage():
    idx = np.full((10,12),40,np.uint8);palette=np.full((256,3),90,np.uint8)
    current=np.full((20,24,4),100,np.uint8);current[...,3]=255
    current[6:14,8:16,:3]=35
    current[::2,::2,:3]+=8;current[1::2,1::2,:3]-=8
    protect=np.zeros(idx.shape,bool);protect[4,5]=True
    out=clean(current,palette[idx],protected=protect)
    means=lambda a:a[...,:3].astype(int).reshape(10,2,12,2,3).sum((1,3))
    assert np.array_equal(means(out),means(current))
    assert np.array_equal(out[8:10,10:12],current[8:10,10:12])
    assert out[10,12,0]<out[2,2,0]-40


def test_transfer_rejects_isolated_index_matches_and_different_hd():
    idx=np.full((8,8),40,np.uint8);refidx=idx.copy();refidx[3,3]=41
    same=matching_context(idx,(0,0),refidx,(0,0))
    assert not same[2:5,2:5].any() and same[6,6]
    body=np.full((16,16,4),200,np.uint8);body[...,3]=255
    layer=np.full((16,16,4),60,np.uint8);layer[...,3]=255
    pal=np.full((256,3),60,np.uint8)
    out,seam=transfer_layer(idx,layer,(0,0),refidx,body,(0,0),pal,own_clean=layer)
    assert np.array_equal(out,layer) and seam is None


if __name__ == "__main__":
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(f) for n, f in list(globals().items()) if n.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
