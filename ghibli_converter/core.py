"""Turn photos into a Ghibli-inspired (Hayao Miyazaki) style with AnimeGANv3.

The model works on NHWC float32 tensors in [-1, 1]; both sides must be a
multiple of 8 and at least 256 px, matching the reference inference script
in the AnimeGANv3 repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "AnimeGANv3_Hayao_36.onnx"
MIN_SIDE = 256
MULTIPLE = 8


def model_size(width: int, height: int, max_side: int | None = None) -> tuple[int, int]:
    """Size the model runs at: optionally capped, then snapped to the model's grid."""
    if max_side and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width, height = round(width * scale), round(height * scale)

    def snap(x: int) -> int:
        return MIN_SIDE if x < MIN_SIDE else x - x % MULTIPLE

    return snap(width), snap(height)


def preprocess(image: Image.Image, max_side: int | None = None) -> np.ndarray:
    """PIL image -> (1, H, W, 3) float32 tensor in [-1, 1]."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = model_size(*image.size, max_side=max_side)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 127.5 - 1.0)[np.newaxis]


def postprocess(output: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """(1, H, W, 3) tensor in [-1, 1] -> PIL image, optionally resized to `size` (w, h)."""
    pixels = np.clip((np.squeeze(output, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    result = Image.fromarray(pixels, "RGB")
    if size and result.size != size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


class GhibliConverter:
    """Loads the ONNX model once and converts any number of images."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        import onnxruntime as ort  # imported here so the pure helpers above stay importable without it

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def convert(self, image: Image.Image, max_side: int | None = 768, keep_size: bool = True) -> Image.Image:
        upright = ImageOps.exif_transpose(image)
        tensor = preprocess(upright, max_side=max_side)
        (output,) = self.session.run(None, {self.input_name: tensor})
        return postprocess(output, upright.size if keep_size else None)
