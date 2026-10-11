"""Command line: python -m ghibli_converter INPUT [INPUT ...] [-o OUTPUT_DIR]"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .core import DEFAULT_MODEL, GhibliConverter

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            files += sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif item.is_file():
            files.append(item)
        else:
            print(f"skipping {item}: not found", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ghibli_converter", description="Convert photos to a Ghibli-inspired style (AnimeGANv3 Hayao).")
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("-o", "--output", default="output", help="output folder (default: output)")
    parser.add_argument("--max-side", type=int, default=768, help="longest side the model runs at; smaller = stronger painted look, larger = more detail (default: 768, 0 = no limit)")
    parser.add_argument("--model-size", action="store_true", help="save at the size the model ran at instead of the original size")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the ONNX model")
    args = parser.parse_args(argv)

    files = collect(args.inputs)
    if not files:
        print("no images to convert", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = GhibliConverter(args.model)
    failures = 0
    for path in files:
        start = time.perf_counter()
        try:
            with Image.open(path) as image:
                result = converter.convert(image, max_side=args.max_side or None, keep_size=not args.model_size)
        except (UnidentifiedImageError, OSError) as err:
            print(f"skipping {path}: {err}", file=sys.stderr)
            failures += 1
            continue
        target = out_dir / f"{path.stem}_ghibli.png"
        result.save(target)
        print(f"{path} -> {target} ({result.width}x{result.height}, {time.perf_counter() - start:.2f}s)")
    return 1 if failures == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
