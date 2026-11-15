"""Deterministic colour-only denoising with immutable coverage and material borders."""

import cv2
import numpy as np


def denoise(rgba, remap, metal_lift=0, dramatic=False):
    """Suppress palette-scale speckle, retaining low-frequency wear and structural edges."""
    if rgba.dtype != np.uint8 or rgba.shape[:2] != remap.shape or rgba.shape[2] != 4:
        raise ValueError("expected uint8 RGBA and matching remap mask")
    rgb = rgba[:, :, :3].astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    interior = cv2.erode((rgba[:, :, 3] > 250).astype(np.uint8), kernel) > 0
    paint = (remap > 128).astype(np.uint8)
    border = cv2.dilate(paint, kernel) != cv2.erode(paint, kernel)
    guide = cv2.cvtColor(cv2.GaussianBlur(rgb, (0, 0), 1.5), cv2.COLOR_RGB2GRAY)
    gradient = np.maximum(abs(cv2.Sobel(guide, -1, 1, 0) / 8), abs(cv2.Sobel(guide, -1, 0, 1) / 8))
    weight = np.clip((18 - gradient) / 10, 0, 1) * interior * (~border)
    median = cv2.medianBlur(rgba[:, :, :3].copy(), 5 if dramatic else 3).astype(np.float32)
    clean = cv2.GaussianBlur(median, (0, 0), 2.5) if dramatic else cv2.bilateralFilter(median, 7, 48, 1.5)
    # Restore broad weathering, rather than mistaking it for removable dirt.
    clean += cv2.GaussianBlur(rgb, (0, 0), 8) - cv2.GaussianBlur(clean, (0, 0), 8)
    if dramatic:
        # Expand the broad metal range; encoded paint remains an index and keeps its original range.
        clean += ((clean - 128) * 0.1) * (1 - paint[:, :, None])
    # Encoded house-paint grey is a palette index, not metal albedo: never lift it.
    clean += (255 - clean) * metal_lift * (1 - paint[:, :, None])
    result = rgba.copy()
    result[:, :, :3] = np.clip(np.rint(rgb + (clean - rgb) * weight[:, :, None] * (1 if dramatic else 0.92)), 0, 255).astype(np.uint8)
    return result


def metrics(before, after, remap):
    """Use source-defined flat interiors; never reselect an easier mask after filtering."""
    rgb = before[:, :, :3].astype(np.float32)
    valid = cv2.erode((before[:, :, 3] > 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    paint = (remap > 128).astype(np.uint8)
    border = cv2.dilate(paint, np.ones((5, 5), np.uint8)) != cv2.erode(paint, np.ones((5, 5), np.uint8))
    guide = cv2.cvtColor(cv2.GaussianBlur(rgb, (0, 0), 1.5), cv2.COLOR_RGB2GRAY)
    gradient = np.maximum(abs(cv2.Sobel(guide, -1, 1, 0) / 8), abs(cv2.Sobel(guide, -1, 0, 1) / 8))
    flat = valid & (~border) & (gradient < 8)
    # A source-defined 3px guard excludes preserved seams from the strict flat-surface measurement.
    safe_flat = cv2.erode(flat.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = []
    for image in [before, after]:
        rgb = image[:, :, :3].astype(np.float32)
        high = rgb - cv2.GaussianBlur(rgb, (0, 0), 1)
        # Luminance variance, not inter-channel chroma variance.
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rows.append({
            "safe_flat_hp_rms": float(np.sqrt(np.mean(high[safe_flat] ** 2))) if safe_flat.any() else None,
            "hp_rms": float(np.sqrt(np.mean(high[flat] ** 2))) if flat.any() else None,
            "variance": [float(np.var(cv2.GaussianBlur(luma, (0, 0), scale)[valid])) if valid.any() else None for scale in [8, 16, 32]],
        })
    return {"flat_pixels": int(flat.sum()), "safe_flat_pixels": int(safe_flat.sum()), "before": rows[0], "after": rows[1],
            "alpha_exact": bool(np.array_equal(before[:, :, 3], after[:, :, 3])),
            "boundary_rgb_exact": bool(np.array_equal(before[~valid | border], after[~valid | border]))}
