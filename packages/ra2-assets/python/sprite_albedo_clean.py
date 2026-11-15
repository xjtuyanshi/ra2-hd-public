#!/usr/bin/env python3
"""Deterministic own-frame sprite cleanup and context-checked texel transfer.

No game policy lives here. Callers supply decoded original RGB, HD straight RGBA,
exact paint indices/masks, offsets and optional emissive masks. Alpha is immutable.
The CLI accepts an explicit JSON list of image paths; it never extracts or downloads art.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from redetail_sprites import srgb_to_lab, lab_to_srgb


def resize(a, size, method=Image.Resampling.LANCZOS):
    return np.asarray(Image.fromarray(np.asarray(a, np.uint8)).resize(size, method)).astype(np.float32)


def clean(rgba, original, strength=0.8, protected=None):
    """Reduce only intra-original-pixel variation, preserving every 2x2 RGB sum.

    Original-scale edges and small contrast features protect their entire neighbourhood.
    Callers additionally protect emissive and glass texels using native-resolution masks.
    No pixel is reconstructed from another frame or from the original palette.
    """
    rgb = rgba[..., :3].astype(np.float32)
    h, w = original.shape[:2]
    if rgb.shape[:2] != (2*h, 2*w):
        raise ValueError('cleanup requires an exact 2x frame')
    lab = srgb_to_lab(original.astype(np.float32))
    span = np.max(cv2.dilate(lab, np.ones((3, 3), np.uint8)) -
                  cv2.erode(lab, np.ones((3, 3), np.uint8)), axis=-1)
    guard = cv2.dilate((span > 22).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    guard |= original.max(-1) - original.min(-1) > 60
    if protected is not None:
        guard |= protected
    guard |= rgba[..., 3].reshape(h, 2, w, 2).min((1, 3)) < 255
    mean = rgb.reshape(h, 2, w, 2, 3).mean((1, 3))
    guide = cv2.resize(mean, (2*w, 2*h), interpolation=cv2.INTER_LINEAR)
    guide -= np.repeat(np.repeat(guide.reshape(h, 2, w, 2, 3).mean((1, 3)), 2, 0), 2, 1)
    center = np.repeat(np.repeat(mean, 2, 0), 2, 1)
    target = center + guide
    amount = np.repeat(np.repeat((~guard)*strength, 2, 0), 2, 1)[..., None]
    out = rgb + amount*(target-rgb)
    # Convex motion towards the guide must stay in gamut without moving the native mean.
    delta = out-rgb
    bound = np.where(delta > 0, (255-rgb)/np.maximum(delta, 1e-9),
                     rgb/np.maximum(-delta, 1e-9))
    scale = np.minimum(1, bound.reshape(h, 2, w, 2, 3).min((1, 3)))
    out = rgb + delta*np.repeat(np.repeat(scale, 2, 0), 2, 1)
    blocks = out.reshape(h, 2, w, 2, 3).transpose(0, 2, 4, 1, 3).reshape(h, w, 3, 4)
    quant = np.floor(blocks).astype(np.int32)
    sums = rgb.reshape(h, 2, w, 2, 3).sum((1, 3)).astype(np.int32)
    need = sums-quant.sum(-1)
    order = np.argsort(-(blocks-quant), axis=-1)
    for rank in range(4):
        add = (need > rank).astype(np.int32)[..., None]
        np.put_along_axis(quant, order[..., rank:rank+1],
                          np.take_along_axis(quant, order[..., rank:rank+1], -1)+add, -1)
    out = quant.reshape(h, w, 3, 2, 2).transpose(0, 3, 1, 4, 2).reshape(2*h, 2*w, 3)
    return np.dstack([out.astype(np.uint8), rgba[..., 3]])


def matching_context(indices, offset, body_indices, body_offset):
    """Exact registered 3x3 index agreement, including the context outside a crop."""
    h, w = indices.shape
    x, y = np.subtract(offset, body_offset)
    padded = window(body_indices, y-1, x-1, h+2, w+2)
    local = np.pad(indices, 1)
    equal = (padded == local).astype(np.uint8)
    same = cv2.erode(equal, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0)[1:-1, 1:-1] > 0
    return same & (indices > 0)


def fractional_mask(indices, low, high, scale=2):
    """Exact indexed coverage, bilinear reconstruction with no Lanczos lobes; R8 supports fractions."""
    binary = ((indices >= low) & (indices <= high)).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(binary).resize((indices.shape[1] * scale, indices.shape[0] * scale),
                                                   Image.Resampling.BILINEAR))


def window(a, y, x, h, w):
    out = np.zeros((h, w) + a.shape[2:], a.dtype)
    y0, x0 = max(0, y), max(0, x); y1, x1 = min(a.shape[0], y+h), min(a.shape[1], x+w)
    if y1 > y0 and x1 > x0:
        out[y0-y:y1-y, x0-x:x1-x] = a[y0:y1, x0:x1]
    return out


def transfer_layer(indices, rgba, offset, body_indices, body_hd, body_offset,
                   original_palette, unlit=None, own_clean=None, protected=None):
    """Optionally copy context-identical texels; every other texel keeps its own cleanup.

    A second colour-compatibility check prevents equal palette indices from copying a
    visibly different historical HD interpretation. Protected texels always stay local.
    """
    h, w = indices.shape; x, y = np.subtract(offset, body_offset)
    ref = window(body_hd, 2*y, 2*x, 2*h, 2*w)
    out = clean(rgba, original_palette[indices], protected=protected) if own_clean is None else own_clean.copy()
    same = matching_context(indices, offset, body_indices, body_offset)
    if unlit is not None: same &= ~unlit
    if protected is not None: same &= ~protected
    error = np.abs(out[..., :3].astype(float)-ref[..., :3]).reshape(h, 2, w, 2, 3).max((1, 3, 4))
    same &= error <= 3
    same2 = np.repeat(np.repeat(same, 2, 0), 2, 1)
    out[same2, :3] = ref[same2, :3]
    seam = float(np.abs(out[..., :3].astype(float)-ref[..., :3])[same2].mean()) if same2.any() else None
    return out, seam


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path, help='JSON frames: original, hd, output paths; optional strength/detail_cap')
    a = p.parse_args(); config = json.loads(a.config.read_text())
    for frame in config['frames']:
        rgba = np.array(Image.open(frame['hd']).convert('RGBA'))
        original = np.array(Image.open(frame['original']).convert('RGB'))
        protected = np.array(Image.open(frame['protected']).convert('L')) > 0 if frame.get('protected') else None
        result = clean(rgba, original, protected=protected, **config.get('settings', {}))
        Image.fromarray(result).save(frame['output'])

if __name__ == '__main__':
    main()
