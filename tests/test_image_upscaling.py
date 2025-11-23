"""Tests for the new super-resolution preprocessing pipeline."""

import numpy as np

from src import image_utils


def test_prepare_input_image_invokes_upscaler(monkeypatch):
    """The preprocessing path should always call the configured upscaler first."""

    class DummyUpscaler:
        def __init__(self):
            self.calls = 0

        def upscale(self, image: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.clip(image + 10, 0, 255)

    dummy = DummyUpscaler()
    monkeypatch.setattr(image_utils, "get_image_upscaler", lambda: dummy)

    original = np.ones((8, 8, 3), dtype=np.uint8)
    processed = image_utils.ImageProcessor.prepare_input_image(
        original, max_width=8, max_height=8
    )

    assert dummy.calls == 1
    np.testing.assert_array_equal(processed, original + 10)


def test_enhance_image_gracefully_handles_failure(monkeypatch):
    """If the upscaler raises, the original frame should pass through untouched."""

    class FailingUpscaler:
        def upscale(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - raised
            raise RuntimeError("boom")

    monkeypatch.setattr(image_utils, "get_image_upscaler", lambda: FailingUpscaler())

    sample = np.zeros((4, 4, 3), dtype=np.uint8)
    enhanced = image_utils.ImageProcessor.enhance_image(sample)

    np.testing.assert_array_equal(enhanced, sample)
