"""Shape-locked projection pages for voxel vehicles, rendered offline from G-buffer bakes (no image model).

A bake (hd-renderer sandbox ``?vbake=1``, driven per look) holds, per part and facing, the smooth remaster mesh on
black and on white (coverage), the same with the second house's materials (house paint mask), its geometric world
normals, and the engine look's ``debug=voxel-xy`` pass whose red channel is the voxel palette index. This builder
relights every pixel with one fixed sun, palette albedo per voxel, colour-derived material response, curvature bevels
and panel seams at palette-ramp boundaries, then resamples to page scale. Silhouette, parts and proportions are the
mesh's own, so no frame can change shape; all facings come from one deterministic shader, so turning does not
flicker. Output uses the voxel projection manifest of vehicle_projection.py and installs like its builds.
Inputs and outputs are derived from player files and stay outside Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vehicle_projection import BAKE_JACOBIAN, Frame, Registration, mask_iou, pack_frames, paint_channels, rgb

# Toward the sun, world axes (y up): the engine voxel light (hd-renderer ra2/vpl.ts ENGINE_VOXEL_LIGHT) reversed.
SUN = np.array([0.5, 0.70710678, -0.5])
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def view_direction(jacobian: np.ndarray = BAKE_JACOBIAN) -> np.ndarray:
    """Unit vector toward the iso camera: orthogonal to both screen rows, pointing up."""
    v = np.cross(jacobian[0], jacobian[1]).astype(np.float64)
    v /= np.linalg.norm(v)
    return v if v[1] > 0 else -v


def coverage(black: np.ndarray, white: np.ndarray) -> np.ndarray:
    # The backgrounds come back tone-mapped (white is about 245): normalise by the corners, as unmatte does.
    kb = black[:8, :8].mean(axis=(0, 1))
    wb = white[:8, :8].mean(axis=(0, 1))
    alpha = np.clip(1 - ((white - black) / np.maximum(wb - kb, 1)).mean(-1), 0, 1)
    alpha[alpha < 0.03] = 0
    return alpha.astype(np.float32)


def nearest_labels(valid: np.ndarray) -> np.ndarray:
    """Flat index of the nearest valid pixel, for every pixel."""
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    table = np.zeros(int(labels.max()) + 1, np.int64)
    table[labels[valid]] = np.flatnonzero(valid)
    return table[labels]


def nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return values
    flat = values.reshape(valid.size, -1)
    return flat[nearest_labels(valid)].reshape(values.shape)


def stable_index(index: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Drop multisample blends of neighbouring indices: keep local-majority values, refill the rest by nearest."""
    stable = np.zeros_like(valid)
    k = 2 * radius + 1
    for value in np.unique(index[valid]):
        m = (index == value) & valid
        stable |= m & (cv2.blur(m.astype(np.float32), (k, k)) > 0.3)
    return nearest_fill(index, stable if stable.any() else valid)


def materials(palette: np.ndarray) -> dict[str, np.ndarray]:
    """Per palette entry: specular strength and shininess from its colour (no per-type tables)."""
    c = palette.astype(np.float32) / 255
    lum = c @ LUMA
    sat = (c.max(-1) - c.min(-1)) / np.maximum(c.max(-1), 1e-3)
    steel = np.clip(1 - sat * 3, 0, 1) * np.clip((lum - 0.08) / 0.25, 0, 1)
    spec = 0.10 + 0.32 * steel
    shine = 18 + 50 * steel
    spec[16:32], shine[16:32] = 0.22, 26
    spec[lum < 0.06] = 0.03
    return {'spec': spec.astype(np.float32), 'shine': shine.astype(np.float32)}


DEFAULTS = {'key': 0.95, 'ambient': 0.42, 'exposure': 1.0, 'saturation': 1.15, 'bevel_px': 6.0, 'bevel': 0.35,
            'cavity': 0.45, 'seam_px': 3.0, 'seam': 0.42, 'fine': 0.18, 'rim_px': 4.0, 'rim': 0.35, 'mode_radius': 3.0,
            'sharpen': 0.45}


def shade_frame(colour_k, colour_w, paint_k, normal_rgba, debug_k, palette, p):
    """Bake-resolution straight colour (sRGB 0..255), coverage and house mask of one frame."""
    alpha = coverage(colour_k, colour_w)
    solid = alpha > 0.5
    na = normal_rgba[..., 3].astype(np.float32) / 255
    n = (normal_rgba[..., :3].astype(np.float32) / 255 * 2 - 1)
    n = nearest_fill(n, na > 0.9)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-4)
    engine = debug_k.max(-1) > 3
    index = stable_index(np.rint(debug_k[..., 0]).astype(np.int32), engine, int(p['mode_radius']))
    index = np.clip(nearest_fill(index, engine), 0, 255)
    house = (np.abs(paint_k - colour_k).max(-1) > 8) & solid

    pal = palette.astype(np.float32) / 255
    albedo = srgb_to_linear(pal[index])
    mat = materials(palette)
    spec_s, shine = mat['spec'][index], mat['shine'][index]

    sun = SUN / np.linalg.norm(SUN)
    view = view_direction()
    nl = n @ sun
    ndl = np.clip(nl, 0, 1)
    wrap = np.clip((nl + 0.35) / 1.35, 0, 1)
    sky = 0.5 + 0.5 * n[..., 1]
    diffuse = p['key'] * (0.6 * ndl + 0.4 * wrap) + p['ambient'] * (0.55 + 0.45 * sky)
    half = (sun + view) / np.linalg.norm(sun + view)
    specular = spec_s * np.clip(n @ half, 0, 1) ** shine * (shine + 8) / 60 * (nl > 0)

    # Curvature (screen-space divergence of the normal field): convex edges catch light, concave seams darken.
    s = p['bevel_px']
    nb = cv2.GaussianBlur(n, (0, 0), s * 0.5)
    sx = BAKE_JACOBIAN[0] / np.linalg.norm(BAKE_JACOBIAN[0])
    sy = BAKE_JACOBIAN[1] / np.linalg.norm(BAKE_JACOBIAN[1])
    nsx = (nb @ sx).astype(np.float32)
    nsy = (nb @ sy).astype(np.float32)
    curv = cv2.Sobel(nsx, cv2.CV_32F, 1, 0, ksize=3) / 8 + cv2.Sobel(nsy, cv2.CV_32F, 0, 1, ksize=3) / 8
    curv = np.clip(curv * s, -1, 1) * solid
    convex, concave = np.clip(curv, 0, 1), np.clip(-curv, 0, 1)

    # Panel seams where the voxel colour changes ramp (hull plate against track, paint against steel).
    ramp = index // 16
    seam = ((ramp != np.roll(ramp, -1, 1)) | (ramp != np.roll(ramp, -1, 0))).astype(np.float32)
    w = max(1, int(round(p['seam_px'])))
    seam = cv2.GaussianBlur(cv2.dilate(seam, np.ones((w, w), np.uint8)), (0, 0), w * 0.35) * solid
    # Shade steps inside a ramp (the voxel art's own panel lines): a lighter seam.
    lum_idx = (pal @ LUMA)[index].astype(np.float32)
    step = np.abs(cv2.Sobel(lum_idx, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(lum_idx, cv2.CV_32F, 0, 1, ksize=1))
    fine = cv2.GaussianBlur((step > 0.08).astype(np.float32), (0, 0), w * 0.3) * solid

    # Rim: a thin dark contact line along the silhouette, like the original sprite outlines.
    inside = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 5)
    rim = np.clip(1 - inside / p['rim_px'], 0, 1) * solid

    light = diffuse * (1 - p['seam'] * seam - p['fine'] * fine) * (1 - p['cavity'] * concave) * (1 - p['rim'] * rim)
    glint = specular + p['bevel'] * convex * (0.4 + 0.6 * wrap)
    linear = (albedo * light[..., None] + glint[..., None] * (0.35 + 0.65 * albedo)) * p['exposure']
    linear = linear / (1 + np.maximum(linear - 0.8, 0) * 1.2)
    out = linear_to_srgb(linear) * 255
    grey = out.mean(-1, keepdims=True)
    out = np.clip(grey + (out - grey) * p['saturation'], 0, 255)
    return out.astype(np.float32), alpha, house


def downsample(colour, alpha, factor, sharpen):
    h, w = alpha.shape
    size = (max(1, round(w / factor)), max(1, round(h / factor)))
    pm = cv2.resize(colour * alpha[..., None], size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    c = pm / np.maximum(a[..., None], 1e-4)
    if sharpen > 0:
        c = np.clip(c + sharpen * (c - cv2.GaussianBlur(c, (0, 0), 0.8)), 0, 255)
    return c, a


def build(type_name: str, bake: Path, palette_png: Path, out: Path, page_zoom: float = 5.0,
          params: dict | None = None, review: Path | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    meta = json.loads((bake / 'meta.json').read_text())
    factor = float(meta['zoom']) / page_zoom
    with Image.open(palette_png) as image:
        palette = np.asarray(image.convert('RGB')).reshape(-1, 3)[:256]
    parts: dict[str, list[Frame]] = {}
    shape = []
    for f in meta['looks']['rem']['frames']:
        part, k = f['part'], f['k']
        name = f'{part}_{k:03d}'
        k_img, w_img, p_img = (rgb(bake / 'rem' / f'{name}_{s}.png') for s in 'kwp')
        with Image.open(bake / 'rem' / f'{name}_n.png') as image:
            normals = np.asarray(image.convert('RGBA'))
        debug = rgb(bake / 'exy' / f'{name}_k.png')
        colour, alpha, house = shade_frame(k_img, w_img, p_img, normals, debug, palette, p)
        page_colour, page_alpha = downsample(colour, alpha, factor, p['sharpen'])
        size = page_alpha.shape[::-1]
        page_house = (cv2.resize(house.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.3).astype(np.float32)
        mesh_alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        hull_alpha = cv2.resize((debug.max(-1) > 3).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        paint = paint_channels(page_colour, page_alpha, page_colour, page_house, palette[16:32])
        pivot = np.asarray(f['pivot'], np.float64) / factor
        iou = mask_iou(page_alpha, mesh_alpha)
        reg = Registration(np.array([[1, 0, 0], [0, 1, 0]], np.float64), iou, 1.0, 0.0, 0.0)
        parts.setdefault(part, []).append(Frame(k, int(f['facing']), page_colour, page_alpha, paint, pivot,
                                                page_alpha, reg))
        shape.append({'part': part, 'k': k, 'iou_mesh': iou, 'iou_voxel_hull': mask_iou(page_alpha, hull_alpha)})
    out.mkdir(parents=True, exist_ok=True)
    count = len(next(iter(parts.values())))
    if count not in (16, 32):
        raise ValueError(f'expected 16 or 32 facings, got {count}')
    manifest = {'version': 1, 'type': type_name, 'frames': count, 'facingStep': 256 // count,
                'scale': page_zoom / 4, 'jacobian': (BAKE_JACOBIAN * page_zoom / 4).tolist(),
                'house': {'first': 16, 'count': 16}, 'shading': 'baked', 'parts': {}, 'registration': []}
    for part, frames in parts.items():
        frames.sort(key=lambda fr: fr.facing)
        colour_atlas, paint_atlas, packed = pack_frames(frames)
        cn, pn = f'{type_name}_{part}.png', f'{type_name}_{part}_paint.png'
        Image.fromarray(colour_atlas).save(out / cn)
        Image.fromarray(paint_atlas).save(out / pn)
        manifest['parts'][part] = {'colour': cn, 'paint': pn, 'size': [colour_atlas.shape[1], colour_atlas.shape[0]],
                                   'frames': packed}
        manifest['registration'] += [{'part': part, 'k': fr.k, 'iou': fr.registration.iou, 'scale': 1.0, 'dx': 0.0,
                                      'dy': 0.0} for fr in frames]
    (out / f'{type_name}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if review:
        review.mkdir(parents=True, exist_ok=True)
        (review / f'{type_name}_shape.json').write_text(json.dumps(shape, indent=1))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', required=True)
    parser.add_argument('--bake', type=Path, required=True)
    parser.add_argument('--palette', type=Path, required=True, help='256-entry unit palette PNG (a map palettes/unit.png)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--page-zoom', type=float, default=5.0)
    parser.add_argument('--param', action='append', default=[], help='name=value shading override (see DEFAULTS)')
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    params = {}
    for item in args.param:
        key, value = item.split('=', 1)
        if key not in DEFAULTS:
            raise SystemExit(f'unknown param {key}')
        params[key] = float(value)
    build(args.type, args.bake, args.palette, args.out, args.page_zoom, params, args.review)


if __name__ == '__main__':
    main()
