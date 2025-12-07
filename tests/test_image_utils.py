"""Tests covering flexible image decoding utilities."""

import io

import numpy as np
import pytest
from PIL import Image, features

from src import image_utils


def test_load_image_from_bytes_supports_webp():
    if not features.check("webp"):
        pytest.skip("Pillow build lacks WebP support")

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="WEBP")

    decoded = image_utils.ImageProcessor.load_image_from_bytes(buffer.getvalue())

    assert decoded is not None
    assert decoded.shape == (4, 4, 3)
    assert np.all(decoded[:, :, 2] == 255)  # BGR ordering


def test_load_image_from_bytes_falls_back_to_pillow(monkeypatch):
    class DummyCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(*_args, **_kwargs):
            return None

    monkeypatch.setattr(image_utils, "HAS_CV2", True, raising=False)
    monkeypatch.setattr(image_utils, "cv2", DummyCV2, raising=False)

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(0, 128, 255)).save(buffer, format="PNG")

    decoded = image_utils.ImageProcessor.load_image_from_bytes(buffer.getvalue())

    assert decoded is not None
    assert decoded.shape == (2, 2, 3)
    # Validate fallback produced BGR array
    assert decoded[0, 0, 0] == 255
    assert decoded[0, 0, 1] == 128
    assert decoded[0, 0, 2] == 0
