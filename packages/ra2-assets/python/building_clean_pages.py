"""Install opt-in building colour variants into an existing mkfarm overlay, never the source farm."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image

from cache_farm import install_file
from shape_locked_denoise import denoise, metrics


def sprite_keys(value, sprites):
    if isinstance(value, str) and value in sprites:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from sprite_keys(item, sprites)
    elif isinstance(value, list):
        for item in value:
            yield from sprite_keys(item, sprites)


def run(source, farm, theater, report, preset="tuned"):
    if source.resolve() == farm.resolve() or not farm.is_dir() or farm.is_symlink():
        raise ValueError("use a distinct existing mkfarm overlay")
    base = source / "_library" / theater
    library = json.loads((base / "library.json").read_text())
    refs = library.get("spriteAtlasesHd")
    if not refs:
        sidecar = json.loads((base / "sprites-hd.json").read_text())
        refs = sidecar if isinstance(sidecar, list) else sidecar["spriteAtlasesHd"]
    selected = set()
    for value in library["types"].values():
        if value["kind"] == "building":
            selected.update(sprite_keys(value, library["sprites"]))
    pages = {}
    for key in sorted(selected):
        for frame in library["sprites"][key]["frames"]:
            if not frame or not frame.get("rect") or min(frame["rect"][2:]) <= 0:
                continue
            page = frame.get("page", 0)
            if refs[page]:
                pages.setdefault(page, {}).setdefault(tuple(frame["rect"]), []).append(key)
    files = [None] * len(refs)
    log = {"theater": theater, "algorithm": "median5-gaussian2.5-coarse8-edge-lock-contrast10-v4" if preset == "dramatic" else "median3-bilateral7-sigma48-coarse8-edge-lock-metal035-v3", "pages": [], "bodies": []}
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="building-clean-") as scratch:
        scratch = Path(scratch)
        for index, rects in sorted(pages.items()):
            ref = refs[index]
            path = base / ref["file"]
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            image = Image.open(path).convert("RGBA")
            remap_image = Image.open(base / ref["remap"]).convert("L")
            scale = ref.get("scale", 2)
            for rect, keys in rects.items():
                x, y, width, height = [round(v * scale) for v in rect]
                box = (x, y, x + width, y + height)
                before = np.array(image.crop(box))
                remap = np.array(remap_image.crop(box))
                after = denoise(before, remap, metal_lift=0 if preset == "dramatic" else 0.035, dramatic=preset == "dramatic")
                if any(key.startswith("building:") for key in keys):
                    log["bodies"].append({"sprites": keys, "rect": rect, "page": index, **metrics(before, after, remap)})
                image.paste(Image.fromarray(after), box)
            filename = Path(ref["file"]).stem + "-cleanC.png"
            destination = scratch / filename
            image.save(destination, compress_level=3)
            install_file(farm, f"_library/{theater}/{filename}", destination)
            files[index] = filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash, "source page changed"
            log["pages"].append({"index": index, "file": filename, "rectangles": len(rects), "source_sha256": before_hash})
            print(theater, index, len(rects), flush=True)
            image.close()
            remap_image.close()
            report.write_text(json.dumps(log, ensure_ascii=False, indent=2))
        sidecar = scratch / "buildings-clean.json"
        sidecar.write_text(json.dumps({"version": 1, "files": files}, indent=2))
        install_file(farm, f"_library/{theater}/buildings-clean.json", sidecar)
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["tuned", "dramatic"], default="tuned")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--farm", type=Path, required=True)
    parser.add_argument("--theater", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.farm, args.theater, args.report, args.preset)
