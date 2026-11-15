"""Native-scale residual sample projection. Diagnostic only; never changes acceptance masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import PROJECTION, VIEW, yaw


def project_samples(exported, samples, population, column, theta, cell):
    rows, values = [], []
    rotation = yaw(theta)
    for key, part in exported['parts'].items():
        source = np.asarray(samples[key][population], float)
        affine = np.asarray(part['affine']).reshape(3, 4)
        world = source[:, :3] @ affine[:, :3].T + affine[:, 3]
        rows.append(world @ rotation.T)
        values.append(source[:, column])
    world, error = np.concatenate(rows), np.concatenate(values)
    screen = world @ PROJECTION.T * exported['camera']['scale'] + [cell*.5, cell*.65]
    pixel = np.floor(screen).astype(int)
    if (pixel < 1).any() or (pixel >= cell-1).any():
        raise ValueError('Residual board clips model')
    depth = world @ VIEW
    flat = pixel[:, 1]*cell + pixel[:, 0]
    buffer = np.full(cell*cell, -np.inf)
    np.maximum.at(buffer, flat, depth)
    front = depth >= buffer[flat] - 1e-8
    residual = np.full(cell*cell, -1.)
    np.maximum.at(residual, flat[front], error[front])
    knots = np.array([0, .25, .5, .85, 1., 1.5, 3.])
    colours = np.array([[35, 80, 210], [30, 170, 200], [30, 200, 90], [235, 215, 40],
                        [255, 125, 25], [225, 35, 35], [245, 40, 220]])
    rgb = np.full((cell*cell, 3), [28, 31, 36], np.uint8)
    valid = residual >= 0
    for c in range(3):
        rgb[valid, c] = np.interp(residual[valid], knots, colours[:, c]).round().astype(np.uint8)
    return Image.fromarray(rgb.reshape(cell, cell, 3)), int(valid.sum()), int((residual > .85).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--residual', type=Path, required=True)
    parser.add_argument('--cell', type=int, default=190)
    args = parser.parse_args()
    exported = json.loads((args.input/'export.json').read_text())
    samples = json.loads((args.residual/'residual_samples.json').read_text())
    cell = args.cell
    board = Image.new('RGB', (8*cell, 4*(cell+22)+90), (28, 31, 36))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=12)
    draw.text((6, 4), f"{exported['id']} residual samples | 1x = research zoom 2.5 | 4x = nearest neighbour", font=font, fill='white')
    draw.text((6, 22), 'Blue 0 / cyan .25 / green .5 / yellow .85 GATE / orange 1.0 / red 1.5 / magenta 3+ voxels', font=font, fill='white')
    labels = [('voxel', 6, 'Voxel -> CAD formal NN'), ('cad', 6, 'CAD -> voxel formal NN'),
              ('voxel', 7, 'Voxel -> all CAD triangles LOWER BOUND'), ('cad', 7, 'CAD -> voxel exact triangle distance')]
    audit = []
    for row, (population, column, label) in enumerate(labels):
        y = 46+row*(cell+22)
        draw.text((6, y), label, font=font, fill='white')
        for facing, theta in enumerate(exported['camera']['facings']):
            picture, covered, over = project_samples(exported, samples, population, column, theta, cell)
            board.paste(picture, (facing*cell, y+20))
            draw.text((facing*cell+3, y+20), f'{facing*45} deg', font=font, fill='white')
            audit.append(dict(row=label, facing=facing*45, sampledPixels=covered, overGatePixels=over))
    draw.text((6, board.height-22), 'Closest depth sample per native pixel. Dark = unsampled. Offline diagnostic; source and CAD use the same affine/camera.', font=font, fill='white')
    board.save(args.residual/'residual_heatmap_1x.png')
    board.resize((board.width*4, board.height*4), Image.Resampling.NEAREST).save(args.residual/'residual_heatmap_4x.png')
    (args.residual/'board_metadata.json').write_text(json.dumps(dict(camera=exported['camera'], audit=audit,
        representation='Projected formal/diagnostic surface samples, nearest depth at native research scale; not a raster IoU mask.'), indent=1)+'\n')


if __name__ == '__main__':
    main()
