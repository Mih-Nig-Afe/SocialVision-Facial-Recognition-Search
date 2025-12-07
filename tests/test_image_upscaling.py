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


def test_image_upscaler_prefers_opencv_backend(monkeypatch):
    """Test that OpenCV is the preferred backend in the new priority order (memory-efficient)."""

    class DummyNCNN:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - should not run
            raise AssertionError("NCNN backend should not run")

    class DummyMax:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - should not run
            raise AssertionError("IBM MAX backend should not run")

    class DummySuperRes:
        def __init__(self):
            self.calls = 0

        def upsample(self, image):
            self.calls += 1
            return np.clip(image + 5, 0, 255)

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyNCNN())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyMax())

    upscaler = image_upscaler.ImageUpscaler()
    upscaler._realesrgan = None  # Disable Real-ESRGAN
    dummy_superres = DummySuperRes()
    upscaler._opencv_sr = dummy_superres

    sample = np.zeros((4, 4, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    # Result should be processed using OpenCV (preferred memory-efficient backend)
    assert upscaler.last_backend == "opencv"
    assert dummy_superres.calls == 1


def test_image_upscaler_uses_lanczos_fallback(monkeypatch):
    """Test that Lanczos fallback works when other backends are unavailable."""

    class DummyNCNN:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - disabled
            raise AssertionError("NCNN backend disabled")

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

    # Should fall back to lanczos when no other backends available
    assert upscaler.last_backend == "lanczos"
    # Result should be upscaled (larger than original)
    assert result.shape[0] >= sample.shape[0]
    assert result.shape[1] >= sample.shape[1]


def test_image_upscaler_falls_back_to_lanczos(monkeypatch):
    """Test that Lanczos fallback works when OpenCV fails."""

    class DummyNCNN:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):
            return None

    class DummyMax:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):
            return None

    class FailingSuperRes:
        def upsample(self, image):
            return None  # Simulate failure

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyNCNN())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyMax())

    upscaler = image_upscaler.ImageUpscaler()
    upscaler._realesrgan = None  # Disable Real-ESRGAN
    upscaler._opencv_sr = FailingSuperRes()

    sample = np.zeros((3, 3, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    # Should fall back to lanczos when opencv fails
    assert upscaler.last_backend == "lanczos"
    # Result should be upscaled
    assert result.shape[0] >= sample.shape[0]


def test_image_upscaler_opencv_is_primary_backend(monkeypatch):
    """Test that OpenCV is used as the primary backend (memory-efficient default)."""

    class DummyBackend:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - disabled
            raise AssertionError("Should not be called")

    class DummySuperRes:
        def __init__(self):
            self.calls = 0

        def upsample(self, image):
            self.calls += 1
            return np.clip(image + 13, 0, 255)

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyBackend())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyBackend())

    upscaler = image_upscaler.ImageUpscaler()
    upscaler._realesrgan = None  # Real-ESRGAN disabled for memory efficiency
    dummy_superres = DummySuperRes()
    upscaler._opencv_sr = dummy_superres

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    assert np.all(result == 13)
    assert upscaler.last_backend == "opencv"
    assert dummy_superres.calls == 1


def test_max_api_disables_after_consecutive_failures(monkeypatch):
    monkeypatch.setattr(image_upscaler.config, "IBM_MAX_ENABLED", True, raising=False)
    monkeypatch.setattr(
        image_upscaler.config, "IBM_MAX_URL", "http://ibm-max:5000", raising=False
    )
    monkeypatch.setattr(image_upscaler.config, "IBM_MAX_TIMEOUT", 1, raising=False)
    monkeypatch.setattr(
        image_upscaler.config, "IBM_MAX_FAILURE_THRESHOLD", 2, raising=False
    )
    monkeypatch.setattr(
        image_upscaler.config, "IBM_MAX_PROBE_ON_START", False, raising=False
    )

    client = image_upscaler.MaxAPIUpscaler()
    assert client.enabled is True

    client._register_failure()
    assert client.enabled is True

    client._register_failure()
    assert client.enabled is False


def test_max_api_probe_disables_when_unreachable(monkeypatch):
    monkeypatch.setattr(image_upscaler.config, "IBM_MAX_ENABLED", True, raising=False)
    monkeypatch.setattr(
        image_upscaler.config, "IBM_MAX_URL", "http://ibm-max:5000", raising=False
    )
    monkeypatch.setattr(image_upscaler.config, "IBM_MAX_TIMEOUT", 2, raising=False)
    monkeypatch.setattr(
        image_upscaler.config, "IBM_MAX_PROBE_ON_START", True, raising=False
    )

    class DummyRequests:
        @staticmethod
        def get(*_args, **_kwargs):
            raise RuntimeError("probe failure")

    monkeypatch.setattr(
        image_upscaler.MaxAPIUpscaler,
        "_get_requests_module",
        lambda self: DummyRequests,
        raising=False,
    )

    client = image_upscaler.MaxAPIUpscaler()
    assert client.enabled is False
