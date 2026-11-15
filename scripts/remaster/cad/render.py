"""Offline boards from the authoritative exported meshes. No game cache discovery or live runtime.

Dependencies: numpy and Pillow. All input/output/font paths are arguments. The source remap mask
is preserved independently of albedo; only that mask receives the explicitly exported house colour.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


PROJECTION = np.array([[30., 0., -30.], [15., -36.742346, 15.]])
VIEW = np.cross(PROJECTION[0], PROJECTION[1])
VIEW /= np.linalg.norm(VIEW)
if VIEW[1] < 0:
    VIEW *= -1
SUN = np.array([-0.62, 0.78, -0.20])
SUN /= np.linalg.norm(SUN)


def raster(screen, depth, triangles, height, width):
    """Two-sided orthographic z buffer and perspective-free barycentric weights."""
    zbuffer = np.full((height, width), -np.inf)
    ids = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 2))
    for i, triangle in enumerate(triangles):
        q = screen[triangle]
        xmin, ymin = np.maximum(np.floor(q.min(0)).astype(int), 0)
        xmax, ymax = np.minimum(np.ceil(q.max(0)).astype(int), [width - 1, height - 1])
        if xmin > xmax or ymin > ymax:
            continue
        (x0, y0), (x1, y1), (x2, y2) = q
        area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
        if abs(area) < 1e-12:
            continue
        x = np.arange(xmin, xmax+1)[None] + .5
        y = np.arange(ymin, ymax+1)[:, None] + .5
        a = ((x1-x)*(y2-y) - (x2-x)*(y1-y)) / area
        b = ((x2-x)*(y0-y) - (x0-x)*(y2-y)) / area
        c = 1-a-b
        inside = (a >= -1e-8) & (b >= -1e-8) & (c >= -1e-8)
        z = a*depth[triangle[0]] + b*depth[triangle[1]] + c*depth[triangle[2]]
        rect = np.s_[ymin:ymax+1, xmin:xmax+1]
        update = inside & (z > zbuffer[rect])
        zbuffer[rect][update] = z[update]
        ids[rect][update] = i
        bary[rect][update] = np.stack(np.broadcast_arrays(a, b), -1)[update]
    return ids, bary


def interpolate(values, triangles, ids, bary, flat=False):
    covered = ids >= 0
    t = triangles[ids[covered]]
    if flat:
        return values[t[:, 0]]
    w = np.column_stack([bary[covered], 1-bary[covered].sum(1)])
    if values.ndim == 1:
        return (values[t]*w).sum(1)
    return (values[t]*w[..., None]).sum(1)


def paint_maps(part):
    """Nearest original painted exposed face, per direction; all maps are derived, never geometry."""
    rows = np.array(part['paint'], float)
    size = np.array(part['solid']['size'])
    solid = np.array(part['solid']['data'], bool).reshape(size)
    lookup = np.full(size, -1, int)
    lookup[tuple(rows[:, :3].astype(int).T)] = np.arange(len(rows))
    query = np.indices(size + 6).reshape(3, -1).T - 3 + .5
    maps = []
    for axis in range(3):
        for sign in (-1, 1):
            neighbor = np.roll(solid, -sign, axis)
            boundary = [slice(None)]*3
            boundary[axis] = -1 if sign == 1 else 0
            neighbor[tuple(boundary)] = False
            cells = np.argwhere(solid & ~neighbor & (lookup >= 0))
            if not len(cells):
                cells = rows[:, :3].astype(int)
            faces = cells.astype(float) + .5
            faces[:, axis] += .5*sign
            best = np.empty(len(query), np.int32)
            for start in range(0, len(query), 256):
                distance = ((query[start:start+256, None] - faces[None])**2).sum(2)
                best[start:start+256] = np.argmin(distance, 1)
            maps.append(lookup[tuple(cells[best].T)].reshape(size+6))
    return rows, maps


def yaw(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def geometry(part, cad=None):
    affine = np.array(part['affine']).reshape(3, 4)
    if cad is None:
        e = part['engine']
        position = np.array(e['position']).reshape(-1, 3)
        position[:, 0] += part['pivot']
        return dict(position=position, normal=np.array(e['normal']).reshape(-1, 3),
                    lattice=np.array(e['vpos']).reshape(-1, 3),
                    triangles=np.array(e['index']).reshape(-1, 3))
    positions, normals, lattice, indices = [], [], [], []
    base = 0
    for mesh in cad['meshes']:
        p = np.array(mesh['positions']).reshape(-1, 3)
        n = np.array(mesh['normals']).reshape(-1, 3) @ np.linalg.inv(affine[:, :3])
        n /= np.maximum(1e-12, np.linalg.norm(n, axis=1, keepdims=True))
        lattice.append(p)
        positions.append(p @ affine[:, :3].T + affine[:, 3])
        normals.append(n)
        indices.append(np.array(mesh['indices']).reshape(-1, 3) + base)
        base += len(p)
    return dict(position=np.concatenate(positions), normal=np.concatenate(normals),
                lattice=np.concatenate(lattice), triangles=np.concatenate(indices))


def render(parts, paint, theta, house, scale, cell):
    supersample = 3
    height = width = cell*supersample
    offset = np.array([width*.5, height*.65])
    rotation = yaw(theta)
    acc = {k: [] for k in ('position', 'normal', 'lattice', 'part')}
    triangles, base = [], 0
    for i, (_, part) in enumerate(parts.items()):
        n = len(part['position'])
        acc['position'].append(part['position'] @ rotation.T)
        acc['normal'].append(part['normal'] @ rotation.T)
        acc['lattice'].append(part['lattice'])
        acc['part'].append(np.full(n, i))
        triangles.append(part['triangles']+base)
        base += n
    mesh = {k: np.concatenate(v) for k, v in acc.items()}
    triangles = np.concatenate(triangles)
    screen = mesh['position'] @ PROJECTION.T * (scale*supersample) + offset
    if (screen.min(0) < 2).any() or (screen.max(0) > [width-2, height-2]).any():
        raise ValueError('Board tile clips model; increase --cell without changing the model scale')
    ids, bary = raster(screen, mesh['position']@VIEW, triangles, height, width)
    covered = ids >= 0
    lattice = interpolate(mesh['lattice'], triangles, ids, bary)
    normals = interpolate(mesh['normal'], triangles, ids, bary)
    normals /= np.maximum(1e-12, np.linalg.norm(normals, axis=1, keepdims=True))
    ownership = interpolate(mesh['part'], triangles, ids, bary, flat=True).astype(int)
    colours = np.zeros((covered.sum(), 3))
    remap_mask = np.zeros(covered.sum(), bool)
    for i, (key, (rows, maps, affine)) in enumerate(paint.items()):
        selected = ownership == i
        if not selected.any():
            continue
        # World normal -> section plane normal; row-vector inverse-transpose inverse.
        local_normal = (normals[selected] @ rotation) @ affine[:, :3]
        dominant = np.argmax(abs(local_normal), axis=1)
        sign = local_normal[np.arange(len(dominant)), dominant] > 0
        direction = dominant*2 + sign
        cells = np.floor(lattice[selected]).astype(int) + 3
        cells = np.minimum(np.maximum(cells, 0), np.array(maps[0].shape)-1)
        rowids = np.empty(len(cells), int)
        for d in range(6):
            chosen = direction == d
            rowids[chosen] = maps[d][tuple(cells[chosen].T)]
        source = rows[rowids]
        # Mask is exact source palette remap membership, not a paint-class or procedural stripe rule.
        mask = source[:, 7].astype(bool)
        if not np.array_equal(mask, (source[:, 3] >= 16) & (source[:, 3] <= 31)):
            raise ValueError('Source remap mask mismatch')
        albedo = source[:, 4:7]/255
        # Source ramp position supplies only shade; house hue always comes from the caller.
        level = .45 + .55*(31-source[:, 3])/15
        albedo[mask] = np.asarray(house)[None]*level[mask, None]
        colours[selected] = albedo
        remap_mask[selected] = mask
    light = .42 + .78*np.maximum(0, normals@SUN) + .12*np.maximum(0, normals[:, 1])
    shaded = np.clip(colours*light[:, None], 0, 1)
    ground = np.full((height, width, 3), [0.45, 0.52, 0.30])
    # A shared projected contact/cast shadow, solely from the rendered geometry.
    ground_y = min(p['position'][:, 1].min() for p in parts.values())
    shadow_p = mesh['position'].copy()
    above = np.maximum(0, shadow_p[:, 1]-ground_y)
    shadow_p[:, 0] -= above*SUN[0]/SUN[1]
    shadow_p[:, 2] -= above*SUN[2]/SUN[1]
    shadow_p[:, 1] = ground_y
    shadow_screen = shadow_p@PROJECTION.T*(scale*supersample)+offset
    shadow_ids, _ = raster(shadow_screen, np.zeros(len(shadow_p)), triangles, height, width)
    shadow = Image.fromarray(((shadow_ids >= 0)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(supersample*.65))
    ground *= (1-.34*np.asarray(shadow)[..., None]/255)
    ground[covered] = shaded
    picture = Image.fromarray((ground*255).round().clip(0, 255).astype(np.uint8)).resize((cell, cell), Image.Resampling.LANCZOS)
    return picture, dict(remapPixels=int(remap_mask.sum()), coveredPixels=int(covered.sum()),
                         clipped=False, sourceMaskVerified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--cell', type=int, default=190)
    args = parser.parse_args()
    start = time.perf_counter()
    export = json.loads((args.input/'export.json').read_text())
    cad = json.loads((args.input/'cad_mesh.json').read_text())
    silhouettes = json.loads((args.input/'silhouette.json').read_text())
    current, proposed, paint = {}, {}, {}
    for key, part in export['parts'].items():
        current[key] = geometry(part)
        proposed[key] = geometry(part, cad[key])
        rows, maps = paint_maps(part)
        paint[key] = rows, maps, np.array(part['affine']).reshape(3, 4)
    chinese = args.font is not None
    font = ImageFont.truetype(str(args.font), 12) if chinese else ImageFont.load_default(size=12)
    cell = args.cell
    board = Image.new('RGB', (cell*8, cell*2+84), (28, 31, 36))
    draw = ImageDraw.Draw(board)
    title = (f"{export['id']}｜当前引擎几何 / 新 CAD｜同相机、同尺度、同研究光照｜house={export['colours']['parameter']}"
             if chinese else f"{export['id']} | Current engine geometry / CAD | Same camera, scale, light | house={export['colours']['parameter']}")
    draw.text((6, 4), title, font=font, fill='white')
    audit = {}
    for i, theta in enumerate(export['camera']['facings']):
        before, a = render(current, paint, theta, export['colours']['rgb'], export['camera']['scale'], cell)
        after, b = render(proposed, paint, theta, export['colours']['rgb'], export['camera']['scale'], cell)
        board.paste(before, (i*cell, 26))
        board.paste(after, (i*cell, cell+48))
        draw.text((i*cell+4, 28), f"{i*45}° " + ('当前' if chinese else 'Current'), font=font, fill='white')
        value = silhouettes['assembled'][i]['iou']
        draw.text((i*cell+4, cell+50), f"CAD IoU {value:.4f}", font=font, fill='white')
        audit[str(i*45)] = dict(current=a, cad=b)
        print(f'rendered {i*45} degrees, {time.perf_counter()-start:.3f}s', flush=True)
    caption = '离线研究图，非实机截图；上/下：当前/新；1× 为既有研究视距 2.5，2× 为最近邻放大。' if chinese else 'Offline research, not a runtime capture. Current / CAD. 1x uses research zoom 2.5; 2x is nearest-neighbour.'
    draw.text((6, cell*2+53), caption, font=font, fill='white')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, picture in [('board_当前_新', board)]:
        picture.save(args.output/(name+'.png'))
        picture.resize((picture.width*2, picture.height*2), Image.Resampling.NEAREST).save(args.output/(name+'_2x.png'))
    table = Image.new('RGB', (cell*8, cell+24*(len(silhouettes)+1)), (28, 31, 36))
    td = ImageDraw.Draw(table)
    for i in range(8):
        table.paste(board.crop((i*cell, cell+48, (i+1)*cell, cell*2+48)), (i*cell, 0))
        for j, (key, values) in enumerate(silhouettes.items()):
            td.text((i*cell+4, cell+j*22), f'{key}: {values[i]["iou"]:.5f}', font=font, fill='white')
    table.save(args.output/'turntable_8.png')
    table.resize((table.width*2, table.height*2), Image.Resampling.NEAREST).save(args.output/'turntable_8_2x.png')
    (args.output/'render_metadata.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        camera=export['camera'], colour=export['colours'], audit=audit, offline=True, sourceMaskVerified=True,
        note='Paint is source-face lookup; no procedural panels, rings, wheels or decoration.'), indent=1)+'\n')


if __name__ == '__main__':
    main()
