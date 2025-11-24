"""Tests for the new super-resolution preprocessing pipeline."""

import numpy as np

from src import image_utils, image_upscaler


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


def test_image_upscaler_prefers_ibm_max_backend(monkeypatch):
    class DummyNCNN:
        def __init__(self):
            self.enabled = True

        def upscale(self, image):  # pragma: no cover - should not run
            raise AssertionError("NCNN backend should not run when IBM MAX succeeds")

    class DummyMax:
        def __init__(self):
            self.enabled = True

        def upscale(self, image):
            return np.clip(image + 5, 0, 255)

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyNCNN())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyMax())

    upscaler = image_upscaler.ImageUpscaler()
    upscaler._realesrgan = None
    upscaler._opencv_sr = None

    sample = np.zeros((4, 4, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    assert np.all(result == 5)
    assert upscaler.last_backend == "ibm_max"


def test_image_upscaler_uses_ncnn_when_ibm_disabled(monkeypatch):
    class DummyNCNN:
        def __init__(self):
            self.enabled = True

        def upscale(self, image):
            return np.clip(image + 7, 0, 255)

    class DummyMax:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - disabled
            raise AssertionError("IBM MAX backend disabled")

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyNCNN())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyMax())

    upscaler = image_upscaler.ImageUpscaler()
    upscaler._realesrgan = None
    upscaler._opencv_sr = None

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    assert np.all(result == 7)
    assert upscaler.last_backend == "ncnn"
