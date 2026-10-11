from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ghibli_converter.core import DEFAULT_MODEL, model_size, postprocess, preprocess


@pytest.mark.parametrize(
    "size, max_side, expected",
    [
        ((1280, 853), None, (1280, 848)),   # snapped down to a multiple of 8
        ((100, 300), None, (256, 296)),     # sides under 256 are raised to 256
        ((4000, 3000), 768, (768, 576)),    # longest side capped first
        ((3000, 4000), 768, (576, 768)),
        ((500, 400), 768, (496, 400)),      # already under the cap: unchanged apart from snapping
    ],
)
def test_model_size(size, max_side, expected):
    assert model_size(*size, max_side=max_side) == expected


def test_model_size_is_always_valid():
    for w, h in [(257, 999), (8191, 13), (640, 480)]:
        mw, mh = model_size(w, h, max_side=1024)
        assert mw % 8 == 0 and mh % 8 == 0
        assert mw >= 256 and mh >= 256


def test_preprocess_shape_and_range():
    image = Image.new("RGB", (640, 480), (255, 0, 128))
    tensor = preprocess(image)
    assert tensor.shape == (1, 480, 640, 3)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, 0, 0], [1.0, -1.0, 128 / 127.5 - 1.0], atol=1e-6)


def test_preprocess_handles_alpha_and_greyscale():
    assert preprocess(Image.new("RGBA", (300, 300))).shape == (1, 296, 296, 3)
    assert preprocess(Image.new("L", (300, 300))).shape == (1, 296, 296, 3)


def test_preprocess_respects_exif_rotation():
    image = Image.new("RGB", (640, 480))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90°, so the photo is really portrait
    image.info["exif"] = exif.tobytes()
    rotated = Image.open(_roundtrip_jpeg(image, exif))
    assert preprocess(rotated).shape == (1, 640, 480, 3)


def test_postprocess_maps_back_to_pixels():
    tensor = np.stack([np.full((256, 256), v, np.float32) for v in (-1.0, 0.0, 1.0)], axis=-1)[None]
    image = postprocess(tensor)
    assert image.size == (256, 256)
    assert image.getpixel((0, 0)) == (0, 127, 255)


def test_postprocess_clips_and_resizes():
    tensor = np.full((1, 256, 256, 3), 5.0, np.float32)
    image = postprocess(tensor, size=(300, 200))
    assert image.size == (300, 200)
    assert image.getpixel((10, 10)) == (255, 255, 255)


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model file not present")
def test_end_to_end_conversion_keeps_size():
    pytest.importorskip("onnxruntime")
    from ghibli_converter import GhibliConverter

    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (333, 517, 3), dtype=np.uint8))
    converter = GhibliConverter()
    result = converter.convert(photo, max_side=512)
    assert result.size == photo.size
    assert converter.convert(photo, max_side=512, keep_size=False).size == model_size(517, 333, 512)


def _roundtrip_jpeg(image, exif):
    import io

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif.tobytes())
    buffer.seek(0)
    return buffer
