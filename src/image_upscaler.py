"""High-detail image upscaling utilities for SocialVision.

This module provides multiple upscaling backends with automatic fallback:
1. Real-ESRGAN (PyTorch) - High-quality AI upscaling with tiling for memory efficiency
2. OpenCV EDSR - Fast DNN-based super-resolution
3. Lanczos/Bicubic - Basic interpolation fallback

Note: IBM MAX and NCNN backends have been deprioritized due to compatibility issues.
"""

from __future__ import annotations

import gc
import io
import math
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from src.config import get_config
from src.logger import setup_logger
from src.compat.torchvision_patch import ensure_functional_tensor_shim

logger = setup_logger(__name__)
config = get_config()

# Memory limit for image processing (in bytes) - prevents OOM
MAX_IMAGE_MEMORY_BYTES = 50 * 1024 * 1024  # 50MB per image max
MAX_PIXEL_COUNT = (
    4 * 1024 * 1024
)  # 4 megapixels max before downscale (more conservative)

REAL_ESRGAN_MODELS = {
    "realesrgan_x4plus": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "blocks": 23,
        "feat": 64,
        "grow": 32,
    },
    "realesrgan_x4plus_anime_6B": {
        "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth",
        "blocks": 6,
        "feat": 64,
        "grow": 32,
    },
    "realesrgan_x6plus": {
        "scale": 4,
        "preferred_outscale": 8,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "blocks": 23,
        "feat": 64,
        "grow": 32,
    },
    "realesrgan_x2plus": {
        "scale": 2,
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x2plus.pth",
        "blocks": 23,
        "feat": 64,
        "grow": 32,
    },
}

OPENCV_EDSR_MODEL = {
    "filename": "EDSR_x4.pb",
    "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
    "scale": 4,
    "name": "edsr",
}

# Backend priority now favors Real-ESRGAN when enough GPU memory (>=12GB) is available
# with OpenCV and Lanczos as progressively lighter fallbacks.
SUPPORTED_BACKENDS = ("ibm_max", "ncnn", "realesrgan", "opencv", "lanczos")
DEFAULT_BACKEND_PRIORITY = ("realesrgan", "opencv", "lanczos")


class MaxAPIUpscaler:
    """Client for the IBM MAX Image Resolution Enhancer microservice."""

    def __init__(self) -> None:
        self.enabled = getattr(config, "IBM_MAX_ENABLED", False)
        self.base_url = (getattr(config, "IBM_MAX_URL", "") or "").rstrip("/")
        self.timeout = float(getattr(config, "IBM_MAX_TIMEOUT", 120.0))
        self.failure_threshold = max(
            1, int(getattr(config, "IBM_MAX_FAILURE_THRESHOLD", 3))
        )
        self.failure_count = 0
        self.probe_on_start = bool(getattr(config, "IBM_MAX_PROBE_ON_START", True))

        if self.enabled and not self.base_url:
            logger.warning("IBM MAX upscaling enabled but IBM_MAX_URL is missing")
            self.enabled = False
        elif self.enabled:
            logger.info("IBM MAX upscaler enabled (%s)", self.base_url)

        if self.enabled and self.probe_on_start:
            self._probe_service()

    def upscale(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled:
            return None

        try:
            requests = self._get_requests_module()
        except Exception as exc:  # pragma: no cover - import guard
            logger.error("requests library unavailable for IBM MAX call: %s", exc)
            return None

        try:
            rgb_image = image[:, :, ::-1]
            payload = io.BytesIO()
            Image.fromarray(rgb_image).save(payload, format="PNG")
            payload.seek(0)

            endpoint = f"{self.base_url}/model/predict"
            response = requests.post(
                endpoint,
                files={"image": ("input.png", payload.getvalue(), "image/png")},
                timeout=self.timeout,
            )
            response.raise_for_status()

            if not response.content:
                logger.warning("IBM MAX response was empty")
                self._register_failure()
                return None

            result = Image.open(io.BytesIO(response.content)).convert("RGB")
            self._reset_failures()
            return np.array(result)[:, :, ::-1]
        except Exception as exc:
            logger.error("IBM MAX upscaling failed: %s", exc, exc_info=True)
            self._register_failure()
            return None

    def _reset_failures(self) -> None:
        self.failure_count = 0

    def _register_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            if self.enabled:
                logger.warning(
                    "Disabling IBM MAX upscaler after %s consecutive failures",
                    self.failure_count,
                )
            self.enabled = False

    def _probe_service(self) -> None:
        try:
            requests = self._get_requests_module()
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("requests unavailable; disabling IBM MAX probe: %s", exc)
            self.enabled = False
            return

        try:
            response = requests.get(
                f"{self.base_url}/model/metadata",
                timeout=min(self.timeout, 5.0),
            )
            response.raise_for_status()
            logger.info("IBM MAX startup probe succeeded")
            self._reset_failures()
        except Exception as exc:
            logger.warning(
                "Disabling IBM MAX after startup probe failure: %s",
                exc,
            )
            self.enabled = False

    def _get_requests_module(self):  # pragma: no cover - small helper
        import requests

        return requests


class NCNNUpscaler:
    """Wrapper around the Real-ESRGAN NCNN Vulkan executable used by gimp_upscale."""

    def __init__(self) -> None:
        self.enabled = getattr(config, "NCNN_UPSCALING_ENABLED", False)
        exec_path = getattr(config, "NCNN_EXEC_PATH", None)
        self.exec_path = Path(exec_path).expanduser() if exec_path else None
        self.model_name = getattr(config, "NCNN_MODEL_NAME", "realesrgan-x4plus")
        self.scale = float(getattr(config, "NCNN_SCALE", 4.0))
        self.tiles = int(getattr(config, "NCNN_TILES", 0))
        self.tile_pad = int(getattr(config, "NCNN_TILE_PAD", 10))
        self.timeout = float(getattr(config, "NCNN_TIMEOUT", 240.0))

        if self.enabled and not self.exec_path:
            logger.warning("NCNN upscaling enabled but NCNN_EXEC_PATH is missing")
            self.enabled = False
        if self.enabled and self.exec_path and not self.exec_path.exists():
            logger.warning(
                "NCNN upscaler executable not found at %s",
                self.exec_path,
            )
            self.enabled = False

    def upscale(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled or self.exec_path is None:
            return None

        src_path: Optional[str] = None
        dst_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as src:
                Image.fromarray(image[:, :, ::-1]).save(src.name)
                src_path = src.name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as dst:
                dst_path = dst.name

            cmd = [str(self.exec_path), "-i", src_path, "-o", dst_path]
            if self.scale > 0:
                cmd.extend(["-s", str(self.scale)])
            if self.model_name:
                cmd.extend(["-n", self.model_name])
            if self.tiles > 0:
                cmd.extend(["-t", str(self.tiles)])
            if self.tile_pad > 0:
                cmd.extend(["-p", str(self.tile_pad)])

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )

            result = Image.open(dst_path).convert("RGB")
            return np.array(result)[:, :, ::-1]
        except subprocess.CalledProcessError as exc:
            logger.error(
                "NCNN upscaler process failed (code %s): %s",
                exc.returncode,
                exc.stderr.decode("utf-8", errors="ignore"),
            )
            return None
        except Exception as exc:
            logger.error("NCNN upscaler error: %s", exc, exc_info=True)
            return None
        finally:
            for path in (src_path, dst_path):
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except OSError:
                        pass


_UPSCALER: Optional["ImageUpscaler"] = None
_UPSCALER_LOCK = threading.Lock()


def get_image_upscaler() -> "ImageUpscaler":
    """Return a singleton ImageUpscaler instance."""

    global _UPSCALER
    if _UPSCALER is None:
        with _UPSCALER_LOCK:
            if _UPSCALER is None:
                _UPSCALER = ImageUpscaler()
    return _UPSCALER


class ImageUpscaler:
    """Facade around high-quality upscaling backends with memory management.

    This class provides intelligent image upscaling with:
    - Automatic backend selection (Real-ESRGAN preferred, OpenCV EDSR fallback)
    - Memory-efficient tiling for large images
    - Pre-processing for optimal quality
    - Post-processing with sharpening and enhancement
    - Graceful fallback to Lanczos interpolation
    """

    def __init__(self) -> None:
        self.enabled = getattr(config, "IMAGE_UPSCALING_ENABLED", True)
        self.backend_name = getattr(
            config, "IMAGE_UPSCALING_BACKEND", "realesrgan_x4plus"
        )
        self.target_scale = float(getattr(config, "IMAGE_UPSCALING_TARGET_SCALE", 2.0))
        self.min_edge = int(getattr(config, "IMAGE_UPSCALING_MIN_EDGE", 256))
        self.max_edge = int(getattr(config, "IMAGE_UPSCALING_MAX_EDGE", 1536))
        # Default tile size to 256 for memory efficiency (0 = no tiling)
        self.tile = int(getattr(config, "IMAGE_UPSCALING_TILE", 256))
        self.tile_pad = int(getattr(config, "IMAGE_UPSCALING_TILE_PAD", 10))
        self.max_passes = max(1, int(getattr(config, "IMAGE_UPSCALING_MAX_PASSES", 1)))
        self.target_tiles = max(
            0, int(getattr(config, "IMAGE_UPSCALING_TARGET_TILES", 0))
        )
        self.min_realesrgan_scale = max(
            1.0,
            float(getattr(config, "IMAGE_UPSCALING_MIN_REALESRGAN_SCALE", 1.05)),
        )
        self.use_half = bool(getattr(config, "IMAGE_UPSCALING_HALF_PRECISION", False))
        priority_raw = getattr(
            config,
            "IMAGE_UPSCALING_BACKEND_PRIORITY",
            ",".join(DEFAULT_BACKEND_PRIORITY),
        )
        self.backend_priority = self._parse_backend_priority(priority_raw)

        self._realesrgan = None
        self._realesrgan_scale = 1
        self._realesrgan_preferred_outscale = None
        self._opencv_sr = None
        self._opencv_scale = 1
        self._device = "cpu"
        self._current_tile = max(0, self.tile)
        self._ncnn_client = NCNNUpscaler()
        self._max_client = MaxAPIUpscaler()
        self._last_backend = "uninitialized"

        if self.enabled:
            self._initialize_backends()
        else:
            logger.info("Image upscaling disabled via configuration")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upscale(
        self, image: np.ndarray, *, minimum_outscale: float = 1.0
    ) -> np.ndarray:
        """Upscale an image while preserving detail with memory management.

        This method:
        1. Validates image size and memory constraints
        2. Applies pre-processing for optimal quality
        3. Upscales using available backends (Real-ESRGAN > OpenCV > Lanczos)
        4. Applies post-processing with subtle sharpening
        5. Cleans up memory after processing
        """

        if not self.enabled or image is None:
            self._last_backend = "disabled"
            return image

        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if image.ndim == 3 else 1
        except Exception:
            return image

        # Memory guard: Check if image is too large
        pixel_count = width * height
        if pixel_count > MAX_PIXEL_COUNT:
            logger.warning(
                "Image too large (%dx%d = %.1fMP), downscaling first",
                width,
                height,
                pixel_count / 1e6,
            )
            image = self._safe_downscale(image, MAX_PIXEL_COUNT)
            height, width = image.shape[:2]

        if max(height, width) >= self.max_edge:
            self._last_backend = "size_guard"
            return image

        base_outscale = self._compute_outscale(width, height)
        max_allowed = self.max_edge / float(max(1, max(height, width)))
        requested_min = max(1.0, float(minimum_outscale))
        outscale = max(base_outscale, requested_min)
        outscale = min(outscale, max_allowed)

        if outscale <= 1.0 + 1e-6:
            self._last_backend = "no_scale"
            return image

        # Pre-process image for better quality
        preprocessed = self._preprocess_image(image)

        # Try backends in priority order
        for backend_name, handler in self._iter_backends():
            try:
                upscaled = handler(preprocessed, outscale)
                finalized = self._finalize_resolution(image, upscaled, outscale)
                if finalized is not None:
                    # Post-process for enhanced quality
                    enhanced = self._postprocess_image(finalized)
                    self._last_backend = backend_name
                    self._log_backend_success(backend_name)
                    # Clean up memory
                    self._cleanup_memory()
                    return enhanced
            except MemoryError:
                logger.warning(
                    "Backend '%s' ran out of memory, trying next", backend_name
                )
                self._cleanup_memory()
                continue
            except Exception as exc:
                logger.warning("Backend '%s' failed: %s", backend_name, exc)
                continue

        # Ultimate fallback: high-quality Lanczos
        self._last_backend = "lanczos"
        result = self._lanczos_upscale(preprocessed, outscale)
        enhanced = self._postprocess_image(result)
        self._cleanup_memory()
        return enhanced

    def _safe_downscale(self, image: np.ndarray, max_pixels: int) -> np.ndarray:
        """Downscale image to fit within pixel budget while maintaining aspect ratio."""
        height, width = image.shape[:2]
        current_pixels = width * height
        if current_pixels <= max_pixels:
            return image

        scale = (max_pixels / current_pixels) ** 0.5
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        try:
            import cv2

            return cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        except Exception:
            pil_image = self._to_pil(image)
            if pil_image:
                resized = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                return np.array(resized)[:, :, ::-1]
            return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply pre-processing to improve upscaling quality."""
        try:
            pil_image = self._to_pil(image)
            if pil_image is None:
                return image

            # Slight denoise/blur to reduce artifacts in source
            # This helps AI upscalers produce cleaner results
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))

            # Convert back to numpy BGR
            result = np.array(pil_image)
            if result.ndim == 3 and result.shape[2] == 3:
                return result[:, :, ::-1]
            return result
        except Exception as exc:
            logger.debug("Pre-processing failed: %s", exc)
            return image

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply post-processing to enhance upscaled image quality."""
        try:
            pil_image = self._to_pil(image)
            if pil_image is None:
                return image

            # Apply subtle unsharp mask for detail enhancement
            # This compensates for any softness from upscaling
            sharpened = pil_image.filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=2)
            )

            # Subtle contrast enhancement
            enhancer = ImageEnhance.Contrast(sharpened)
            enhanced = enhancer.enhance(1.05)

            # Convert back to numpy BGR
            result = np.array(enhanced)
            if result.ndim == 3 and result.shape[2] == 3:
                return result[:, :, ::-1]
            return result
        except Exception as exc:
            logger.debug("Post-processing failed: %s", exc)
            return image

    def _cleanup_memory(self) -> None:
        """Force garbage collection to free memory after upscaling."""
        try:
            gc.collect()
            # Also try to clear PyTorch cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    def _lanczos_upscale(self, image: np.ndarray, outscale: float) -> np.ndarray:
        """High-quality Lanczos upscaling fallback."""
        if outscale <= 1.0:
            return image

        try:
            height, width = image.shape[:2]
            new_width = max(1, int(round(width * outscale)))
            new_height = max(1, int(round(height * outscale)))

            pil_image = self._to_pil(image)
            if pil_image is None:
                # Fallback to OpenCV
                try:
                    import cv2

                    return cv2.resize(
                        image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
                    )
                except Exception:
                    return self._resize_to_dimensions(image, new_width, new_height)

            # Use high-quality Lanczos resampling
            upscaled = pil_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Convert back to numpy BGR
            result = np.array(upscaled)
            if result.ndim == 3 and result.shape[2] == 3:
                return result[:, :, ::-1]
            return result
        except Exception as exc:
            logger.error("Lanczos upscaling failed: %s", exc)
            return self._fallback_resize(image, outscale)

    def _parse_backend_priority(self, raw: Optional[str]) -> List[str]:
        if not raw:
            return list(DEFAULT_BACKEND_PRIORITY)

        parsed: List[str] = []
        for token in raw.split(","):
            name = token.strip().lower()
            if not name:
                continue
            if name not in SUPPORTED_BACKENDS:
                logger.warning("Unknown upscaling backend '%s' in priority list", name)
                continue
            if name not in parsed:
                parsed.append(name)

        return parsed or list(DEFAULT_BACKEND_PRIORITY)

    def _iter_backends(self):
        seen = set()

        for backend_name in self.backend_priority:
            handler = self._backend_handler(backend_name)
            if handler is None:
                continue
            seen.add(backend_name)
            yield backend_name, handler

        for backend_name in DEFAULT_BACKEND_PRIORITY:
            if backend_name in seen:
                continue
            handler = self._backend_handler(backend_name)
            if handler is None:
                continue
            yield backend_name, handler

    def _backend_handler(
        self, backend_name: str
    ) -> Optional[Callable[[np.ndarray, float], Optional[np.ndarray]]]:
        mapping = {
            "ibm_max": self._attempt_ibm_max,
            "ncnn": self._attempt_ncnn,
            "realesrgan": self._attempt_realesrgan_backend,
            "opencv": self._attempt_opencv_backend,
            "lanczos": self._attempt_lanczos_backend,
        }
        return mapping.get(backend_name)

    def _get_backend_handler(
        self, backend_name: str
    ) -> Optional[Callable[[np.ndarray, float], Optional[np.ndarray]]]:
        """Public alias for _backend_handler for multi-extraction module."""
        return self._backend_handler(backend_name)

    def _attempt_ibm_max(
        self, image: np.ndarray, _outscale: float
    ) -> Optional[np.ndarray]:
        if not self._max_client.enabled:
            return None
        return self._max_client.upscale(image)

    def _attempt_ncnn(
        self, image: np.ndarray, _outscale: float
    ) -> Optional[np.ndarray]:
        if not self._ncnn_client.enabled:
            return None
        return self._ncnn_client.upscale(image)

    def _attempt_realesrgan_backend(
        self, image: np.ndarray, outscale: float
    ) -> Optional[np.ndarray]:
        if not self._realesrgan:
            return None
        return self._upscale_with_realesrgan(image, outscale)

    def _attempt_opencv_backend(
        self, image: np.ndarray, outscale: float
    ) -> Optional[np.ndarray]:
        if not self._opencv_sr:
            return None
        return self._upscale_with_opencv(image, outscale)

    def _attempt_lanczos_backend(
        self, image: np.ndarray, outscale: float
    ) -> Optional[np.ndarray]:
        """Always-available Lanczos upscaling backend."""
        try:
            return self._lanczos_upscale(image, outscale)
        except Exception as exc:
            logger.warning("Lanczos backend failed: %s", exc)
            return None

    def _log_backend_success(self, backend_name: str) -> None:
        messages = {
            "ibm_max": "Upscaled frame via IBM MAX microservice",
            "ncnn": "Upscaled frame via Real-ESRGAN NCNN backend",
            "realesrgan": "Upscaled frame via native Real-ESRGAN backend",
            "opencv": "Upscaled frame via OpenCV super-resolution backend",
            "lanczos": "Upscaled frame via high-quality Lanczos interpolation",
        }
        message = messages.get(backend_name)
        if message:
            logger.info(message)

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------
    def _initialize_backends(self) -> None:
        realesrgan_ready = self._init_realesrgan_backend()
        opencv_ready = self._init_opencv_superres_backend()

        if not realesrgan_ready and opencv_ready:
            logger.warning(
                "Real-ESRGAN backend unavailable. Falling back to OpenCV super-resolution"
            )
        if not realesrgan_ready and not opencv_ready:
            logger.warning(
                "No dedicated upscaling backend available; using bicubic fallback"
            )

    def _init_realesrgan_backend(self) -> bool:
        model_config = REAL_ESRGAN_MODELS.get(self.backend_name)
        if model_config is None and self.backend_name == "auto":
            for candidate in ("realesrgan_x4plus", "realesrgan_x2plus"):
                model_config = REAL_ESRGAN_MODELS.get(candidate)
                if model_config and self._build_realesrgan(model_config):
                    return True
            return False

        if not model_config:
            logger.warning("Unknown Real-ESRGAN backend '%s'", self.backend_name)
            return False

        return self._build_realesrgan(model_config)

    def _build_realesrgan(self, model_config: dict) -> bool:
        ensure_functional_tensor_shim()
        try:
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        except Exception as exc:
            logger.warning("Real-ESRGAN libraries unavailable: %s", exc)
            return False

        model_path = self._ensure_weights(
            model_config["filename"], model_config["url"], "Real-ESRGAN"
        )
        if not model_path:
            return False

        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            half_precision = self.use_half and self._device == "cuda"
        except Exception:
            self._device = "cpu"
            half_precision = False

        self._optimize_realesrgan_for_cpu()

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_config["feat"],
            num_block=model_config["blocks"],
            num_grow_ch=model_config["grow"],
            scale=model_config["scale"],
        )

        try:
            self._realesrgan = RealESRGANer(
                scale=model_config["scale"],
                model_path=str(model_path),
                model=net,
                tile=max(0, self.tile),
                tile_pad=max(0, self.tile_pad),
                pre_pad=0,
                half=half_precision,
                device=self._device,
            )
            self._realesrgan_scale = model_config["scale"]
            preferred = model_config.get("preferred_outscale")
            if preferred:
                self._realesrgan_preferred_outscale = float(preferred)
                if self.target_scale < self._realesrgan_preferred_outscale:
                    self.target_scale = self._realesrgan_preferred_outscale
            else:
                self._realesrgan_preferred_outscale = self._realesrgan_scale
            logger.info(
                "Real-ESRGAN backend initialized (%s, device=%s)",
                model_path.name,
                self._device,
            )
            return True
        except Exception as exc:
            logger.error("Failed to initialize Real-ESRGAN backend: %s", exc)
            self._realesrgan = None
            return False

    def _init_opencv_superres_backend(self) -> bool:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            logger.warning("OpenCV not available for super-resolution: %s", exc)
            return False

        try:
            from cv2 import dnn_superres  # type: ignore
        except Exception as exc:
            logger.warning("cv2.dnn_superres unavailable: %s", exc)
            return False

        model_path = self._ensure_weights(
            OPENCV_EDSR_MODEL["filename"],
            OPENCV_EDSR_MODEL["url"],
            "OpenCV-EDSR",
        )
        if not model_path:
            return False

        try:
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model_path))
            sr.setModel(OPENCV_EDSR_MODEL["name"], OPENCV_EDSR_MODEL["scale"])
            self._opencv_sr = sr
            self._opencv_scale = OPENCV_EDSR_MODEL["scale"]
            logger.info("OpenCV EDSR super-resolution backend initialized")
            return True
        except Exception as exc:
            logger.error("Failed to initialize OpenCV super-resolution: %s", exc)
            self._opencv_sr = None
            return False

    # ------------------------------------------------------------------
    # Backend helpers
    # ------------------------------------------------------------------
    def _upscale_with_realesrgan(
        self, image: np.ndarray, outscale: float
    ) -> Optional[np.ndarray]:
        target = max(1.0, outscale)
        min_scale = max(1.0, self.min_realesrgan_scale)
        if target <= min_scale:
            return self._resize_with_factor(image, target)

        if self._realesrgan_preferred_outscale:
            target = min(target, self._realesrgan_preferred_outscale)

        self._maybe_adjust_tile_for_image(image.shape[1], image.shape[0])

        current = image
        accumulated = 1.0
        passes = 0
        max_passes = max(1, self.max_passes)

        # Tracks whether the last Real-ESRGAN pass had to downshift due to memory.
        # When that happens and we've exhausted our pass budget, we should avoid
        # forcing the remaining scale via interpolation.
        self._last_realesrgan_pass_used_fallback = False

        while passes < max_passes:
            remaining = target / accumulated
            if remaining <= min_scale:
                break

            pass_scale = min(self._realesrgan_scale, remaining)
            result = self._run_realesrgan_pass(current, pass_scale)
            if result is None:
                if passes == 0:
                    return None
                break

            width_ratio = result.shape[1] / max(1, current.shape[1])
            height_ratio = result.shape[0] / max(1, current.shape[0])
            actual_scale = max(width_ratio, height_ratio, 1.0)
            accumulated *= actual_scale
            current = result
            passes += 1

            if accumulated >= target - 1e-2:
                break

        if passes == 0:
            return None

        if accumulated < target - 1e-2 and not (
            getattr(self, "_last_realesrgan_pass_used_fallback", False)
            and passes >= max_passes
        ):
            scale_factor = target / accumulated
            current = self._resize_with_factor(current, scale_factor)

        return current

    def _is_memory_error(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        if "memory" in name:
            return True
        message = str(exc).lower()
        return "out of memory" in message or "cuda oom" in message

    def _run_realesrgan_pass(
        self, image: np.ndarray, requested_scale: float
    ) -> Optional[np.ndarray]:
        requested_scale = float(requested_scale)
        requested_scale = max(1.0, min(requested_scale, float(self._realesrgan_scale)))

        disable_tiling = False
        original_tile: Optional[int] = None
        if self._should_disable_tiling_for_scale(requested_scale):
            original_tile = getattr(self._realesrgan, "tile", None)
            if original_tile and original_tile > 0:
                disable_tiling = True
                self._realesrgan.tile = 0
                logger.debug(
                    "Disabling Real-ESRGAN tiling for fractional scale %.3f to avoid tensor rounding errors",
                    requested_scale,
                )

        try:
            result, _ = self._realesrgan.enhance(image, outscale=requested_scale)
            self._last_realesrgan_pass_used_fallback = False
            return result
        except Exception as exc:
            if self._is_memory_error(exc) and requested_scale > 2.0:
                fallback_scale = min(
                    float(self._realesrgan_scale), max(2.0, requested_scale - 1.0)
                )
                logger.warning(
                    "Real-ESRGAN pass at %sx exhausted memory, retrying at %sx",
                    requested_scale,
                    fallback_scale,
                )
                try:
                    result, _ = self._realesrgan.enhance(image, outscale=fallback_scale)
                    self._last_realesrgan_pass_used_fallback = True
                    return result
                except Exception as fallback_exc:
                    logger.error(
                        "Real-ESRGAN fallback pass (%sx) failed: %s",
                        fallback_scale,
                        fallback_exc,
                        exc_info=True,
                    )
                    self._last_realesrgan_pass_used_fallback = False
                    return None

            logger.error(
                "Real-ESRGAN pass failed (%sx): %s",
                requested_scale,
                exc,
                exc_info=True,
            )
            self._last_realesrgan_pass_used_fallback = False
            return None
        finally:
            if disable_tiling and original_tile is not None:
                self._realesrgan.tile = original_tile

    def _optimize_realesrgan_for_cpu(self) -> None:
        if self._device != "cpu":
            return

        if self.max_passes > 1:
            logger.info(
                "CPU detected; limiting Real-ESRGAN to a single pass for responsiveness"
            )
            self.max_passes = 1

        if self.target_scale > 4.0:
            logger.info(
                "CPU detected; clamping target scale from %.1f to 4.0",
                self.target_scale,
            )
            self.target_scale = 4.0

        if self.target_tiles <= 0:
            if self.tile == 0 or self.tile < 224:
                logger.info(
                    "CPU detected; increasing Real-ESRGAN tile size to 224 for fewer passes"
                )
                self.tile = 224

        if self.tile_pad < 12:
            self.tile_pad = 12

    def _maybe_adjust_tile_for_image(self, width: int, height: int) -> None:
        if not self._realesrgan or width <= 0 or height <= 0:
            return

        tile = self._calculate_tile_for_dimensions(width, height)
        if tile is None:
            return

        previous = getattr(self._realesrgan, "tile", None)
        if previous == tile:
            return

        self._realesrgan.tile = tile
        self._current_tile = tile

        if self.target_tiles > 0:
            tiles = self._estimate_tile_count(width, height, tile)
            logger.info(
                "Real-ESRGAN tiling set to %dpx (~%d tiles) for %dx%d frame",
                tile,
                tiles,
                width,
                height,
            )

    def _calculate_tile_for_dimensions(self, width: int, height: int) -> Optional[int]:
        tile = max(0, self.tile)
        desired_tiles = max(0, self.target_tiles)

        if desired_tiles > 0:
            approx_tile = int(round(((width * height) / float(desired_tiles)) ** 0.5))
            tile = max(32, approx_tile)

        if tile <= 0:
            return None

        if self._device == "cpu" and desired_tiles == 0:
            tile = max(tile, 224)

        return max(32, tile)

    def _should_disable_tiling_for_scale(self, requested_scale: float) -> bool:
        if not self._realesrgan:
            return False

        tile = getattr(self._realesrgan, "tile", 0)
        if tile <= 0:
            return False

        # Real-ESRGAN's tiler can mis-compute tensor sizes for non-integer scales.
        return not math.isclose(
            requested_scale, round(requested_scale), rel_tol=1e-3, abs_tol=1e-3
        )

    @staticmethod
    def _estimate_tile_count(width: int, height: int, tile: int) -> int:
        if tile <= 0:
            return 1
        cols = math.ceil(width / tile)
        rows = math.ceil(height / tile)
        return max(1, rows * cols)

    def _upscale_with_opencv(
        self, image: np.ndarray, outscale: float
    ) -> Optional[np.ndarray]:
        try:
            upscaled = self._opencv_sr.upsample(image)
            if not np.isclose(outscale, self._opencv_scale):
                upscaled = self._resize_with_factor(
                    upscaled, outscale / self._opencv_scale
                )
            return upscaled
        except Exception as exc:
            logger.error("OpenCV super-resolution failed: %s", exc, exc_info=True)
            return None

    def _fallback_resize(self, image: np.ndarray, outscale: float) -> np.ndarray:
        return self._resize_with_factor(image, outscale)

    def _finalize_resolution(
        self, original: np.ndarray, upscaled: Optional[np.ndarray], outscale: float
    ) -> Optional[np.ndarray]:
        if upscaled is None:
            return None

        expected_width = max(1, int(round(original.shape[1] * outscale)))
        expected_height = max(1, int(round(original.shape[0] * outscale)))

        if upscaled.shape[1] == expected_width and upscaled.shape[0] == expected_height:
            return upscaled

        return self._resize_to_dimensions(upscaled, expected_width, expected_height)

    def _resize_with_factor(self, image: np.ndarray, scale: float) -> np.ndarray:
        if scale <= 1.0:
            return image

        new_width = max(1, int(round(image.shape[1] * scale)))
        new_height = max(1, int(round(image.shape[0] * scale)))
        return self._resize_to_dimensions(image, new_width, new_height)

    def _resize_to_dimensions(
        self, image: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        if image is None:
            return image

        try:
            import cv2  # type: ignore

            resized = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            return resized
        except Exception:
            pil_image = self._to_pil(image)
            if pil_image is None:
                return image
            resized = pil_image.resize((width, height))
            return np.array(resized)[:, :, ::-1]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _compute_outscale(self, width: int, height: int) -> float:
        longest_edge = max(width, height)
        shortest_edge = min(width, height)

        desired = max(1.0, self.target_scale)
        if shortest_edge < self.min_edge:
            desired = max(desired, self.min_edge / float(shortest_edge))

        max_allowed = self.max_edge / float(longest_edge)
        return min(desired, max_allowed)

    def _ensure_weights(self, filename: str, url: str, label: str) -> Optional[Path]:
        target = Path(config.MODELS_DIR) / filename
        if target.exists():
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".download")

        try:
            import requests  # type: ignore

            logger.info("Downloading %s weights to %s", label, target)
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(target)
            return target
        except Exception as exc:
            logger.error("Failed to download %s weights: %s", label, exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return None

    @staticmethod
    def _to_pil(image: np.ndarray):
        try:
            if image.ndim == 2:
                data = image
            else:
                data = image[:, :, ::-1]
            return Image.fromarray(data)
        except Exception:
            return None

    @property
    def last_backend(self) -> str:
        return self._last_backend
