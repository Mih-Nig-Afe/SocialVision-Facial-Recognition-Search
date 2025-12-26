"""Tests for the new super-resolution preprocessing pipeline."""

import numpy as np

from src import image_utils, image_upscaler


def _tune_scaler(upscaler):
    """Keep unit tests fast by shrinking target scales."""
    upscaler.min_edge = 1
    upscaler.max_edge = 64
    upscaler.target_scale = 2.0
    upscaler.max_passes = 2
    upscaler.target_tiles = 0
    upscaler.min_realesrgan_scale = 1.05


def test_cpu_optimization_clamps_settings(monkeypatch):
    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    upscaler._device = "cpu"
    upscaler.target_scale = 7.5
    upscaler.max_passes = 3
    upscaler.tile = 128
    upscaler.tile_pad = 4

    upscaler._optimize_realesrgan_for_cpu()

    assert upscaler.target_scale == 4.0
    assert upscaler.max_passes == 1
    assert upscaler.tile >= 224
    assert upscaler.tile_pad >= 12


def test_cpu_tile_bump_skipped_when_target_tiles_defined(monkeypatch):
    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    upscaler._device = "cpu"
    upscaler.tile = 96
    upscaler.target_tiles = 25

    upscaler._optimize_realesrgan_for_cpu()

    assert upscaler.tile == 96


def test_realesrgan_small_target_uses_interpolation(monkeypatch):
    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.min_realesrgan_scale = 1.2
    upscaler._realesrgan = object()  # should never be called

    calls = {}

    def fake_resize(self, image, scale):
        calls["scale"] = scale
        return image

    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_resize_with_factor", fake_resize, raising=False
    )

    sample = np.ones((2, 2, 3), dtype=np.uint8)
    result = upscaler._upscale_with_realesrgan(sample, outscale=1.1)

    assert np.array_equal(result, sample)
    assert abs(calls["scale"] - 1.1) < 1e-6


def test_realesrgan_min_scale_can_be_lowered(monkeypatch):
    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.min_realesrgan_scale = 1.0
    upscaler._realesrgan_scale = 4
    upscaler.max_passes = 1

    calls = {}

    def fake_run(self, image, requested_scale):
        calls.setdefault("scales", []).append(requested_scale)
        return image

    monkeypatch.setattr(
        image_upscaler.ImageUpscaler,
        "_run_realesrgan_pass",
        fake_run,
        raising=False,
    )

    sample = np.ones((2, 2, 3), dtype=np.uint8)
    result = upscaler._upscale_with_realesrgan(sample, outscale=1.05)

    np.testing.assert_array_equal(result, sample)
    assert calls["scales"][0] >= 1.04


def test_target_tiles_adjust_runtime_tile_count(monkeypatch):
    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.tile = 0
    upscaler.target_tiles = 25

    class DummyRealESRGAN:
        def __init__(self):
            self.tile = 0

    upscaler._realesrgan = DummyRealESRGAN()

    upscaler._maybe_adjust_tile_for_image(width=1000, height=1000)

    chosen_tile = upscaler._realesrgan.tile
    assert chosen_tile >= 32
    tiles = image_upscaler.ImageUpscaler._estimate_tile_count(1000, 1000, chosen_tile)
    assert abs(tiles - 25) <= 5


def test_realesrgan_remainder_uses_resize(monkeypatch):
    class DummyRealESRGAN:
        def enhance(self, image, outscale=1.0):
            factor = max(1, int(round(outscale)))
            boosted = np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
            return boosted, None

    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_initialize_backends", lambda self: None
    )
    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.max_passes = 1
    upscaler._realesrgan = DummyRealESRGAN()
    upscaler._realesrgan_scale = 4

    resize_calls = {}

    def fake_resize(self, image, scale):
        resize_calls["scale"] = scale
        return np.zeros((int(image.shape[0] * scale), int(image.shape[1] * scale), 3))

    monkeypatch.setattr(
        image_upscaler.ImageUpscaler, "_resize_with_factor", fake_resize, raising=False
    )

    sample = np.ones((2, 2, 3), dtype=np.uint8)
    result = upscaler._upscale_with_realesrgan(sample, outscale=6.0)

    assert result.shape[0] >= 6
    assert abs(resize_calls["scale"] - 1.5) < 1e-6


def test_prepare_input_image_invokes_upscaler(monkeypatch):
    """The preprocessing path should call the upscaler when enhancement is requested."""

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
        original,
        max_width=8,
        max_height=8,
        enhance=True,
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
    _tune_scaler(upscaler)
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
    _tune_scaler(upscaler)
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
    _tune_scaler(upscaler)
    upscaler._realesrgan = None  # Disable Real-ESRGAN
    upscaler._opencv_sr = FailingSuperRes()

    sample = np.zeros((3, 3, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    # Should fall back to lanczos when opencv fails
    assert upscaler.last_backend == "lanczos"
    # Result should be upscaled
    assert result.shape[0] >= sample.shape[0]


def test_image_upscaler_prefers_realesrgan_when_available(monkeypatch):
    """Real-ESRGAN should be preferred when the backend is available."""

    class DummyBackend:
        def __init__(self):
            self.enabled = False

        def upscale(self, image):  # pragma: no cover - disabled
            raise AssertionError("Should not be called")

    class DummyRealESRGAN:
        def __init__(self):
            self.calls = 0

        def enhance(self, image, outscale=1.0):
            self.calls += 1
            factor = max(1, int(round(outscale)))
            boosted = np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
            boosted = np.clip(boosted + 21, 0, 255).astype(np.uint8)
            return boosted, None

    monkeypatch.setattr(image_upscaler, "NCNNUpscaler", lambda: DummyBackend())
    monkeypatch.setattr(image_upscaler, "MaxAPIUpscaler", lambda: DummyBackend())

    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    dummy_realesrgan = DummyRealESRGAN()
    upscaler._realesrgan = dummy_realesrgan
    upscaler._opencv_sr = None

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    assert np.all(result == 21)
    assert upscaler.last_backend == "realesrgan"
    assert dummy_realesrgan.calls == 1


def test_image_upscaler_falls_back_to_opencv_when_realesrgan_missing(monkeypatch):
    """If Real-ESRGAN is unavailable, OpenCV should be the next fallback."""

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
    _tune_scaler(upscaler)
    upscaler._realesrgan = None  # Explicitly disable to force OpenCV fallback
    dummy_superres = DummySuperRes()
    upscaler._opencv_sr = dummy_superres

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    result = upscaler.upscale(sample)

    assert np.all(result == 13)
    assert upscaler.last_backend == "opencv"
    assert dummy_superres.calls == 1


def test_realesrgan_model_alias_exposes_preferred_scale():
    model = image_upscaler.REAL_ESRGAN_MODELS["realesrgan_x6plus"]
    assert model["scale"] == 4
    assert model["preferred_outscale"] == 8


def test_realesrgan_memory_fallback_retries_lower_scale():
    class DummyRealESRGAN:
        def __init__(self):
            self.calls = []

        def enhance(self, image, outscale=1.0):
            self.calls.append(outscale)
            if outscale >= 4:
                raise RuntimeError("CUDA out of memory")
            factor = max(1, int(round(outscale)))
            boosted = np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
            boosted = np.clip(boosted + 7, 0, 255).astype(np.uint8)
            return boosted, None

    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.max_passes = 1
    upscaler._realesrgan = DummyRealESRGAN()
    upscaler._realesrgan_scale = 4
    upscaler._realesrgan_preferred_outscale = 6

    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    result = upscaler._upscale_with_realesrgan(sample, outscale=6)

    assert result.shape[0] == 6  # Fallback scale of 3 applied
    assert upscaler._realesrgan.calls == [4, 3]


def test_realesrgan_multipass_hits_high_target():
    class DummyRealESRGAN:
        def __init__(self):
            self.calls = []

        def enhance(self, image, outscale=1.0):
            self.calls.append(outscale)
            factor = max(1, int(round(outscale)))
            boosted = np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
            boosted = np.clip(boosted + 5, 0, 255).astype(np.uint8)
            return boosted, None

    upscaler = image_upscaler.ImageUpscaler()
    _tune_scaler(upscaler)
    upscaler.target_scale = 8.0
    upscaler.max_passes = 2
    upscaler._realesrgan = DummyRealESRGAN()
    upscaler._realesrgan_scale = 4
    upscaler._realesrgan_preferred_outscale = 8

    sample = np.ones((1, 1, 3), dtype=np.uint8)
    result = upscaler._upscale_with_realesrgan(sample, outscale=8.0)

    assert result.shape[0] == 8
    assert upscaler._realesrgan.calls == [4, 2]


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
