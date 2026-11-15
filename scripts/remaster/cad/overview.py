"""Assemble existing native board tiles without rescaling, repainting, or changing camera/light."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path)
    args = p.parse_args()
    status = json.loads((args.input/'round_batch_status.json').read_text())
    ids = status['ids']
    first = Image.open(args.input/'units'/ids[0]/'board_当前_新.png')
    cell = first.width//8
    cols = 5
    stride = cell*2+54
    result = Image.new('RGB', (cols*cell, ((len(ids)+cols-1)//cols)*stride), (28, 31, 36))
    d = ImageDraw.Draw(result)
    font = ImageFont.truetype(str(args.font), 12) if args.font else ImageFont.load_default(size=12)
    audit = {}
    for i, model in enumerate(ids):
        source = args.input/'units'/model/'board_当前_新.png'
        board = Image.open(source)
        if board.width != cell*8 or board.height != cell*2+84:
            raise ValueError('Mismatched native board contract')
        x, y = (i % cols)*cell, (i//cols)*stride
        d.text((x+4, y+3), f'{model} | 0° | 1x', fill='white', font=font)
        before = board.crop((0, 26, cell, 26+cell))
        after = board.crop((0, cell+48, cell, cell*2+48))
        result.paste(before, (x, y+24))
        result.paste(after, (x, y+cell+38))
        d.text((x+4, y+cell+25), '当前 / 新 CAD' if args.font else 'Current / CAD', fill='white', font=font)
        audit[model] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            facing=0, scale=1, nativeTile=cell, rescaled=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(offline=True, units=audit), indent=1)+'\n')


if __name__ == '__main__':
    main()
