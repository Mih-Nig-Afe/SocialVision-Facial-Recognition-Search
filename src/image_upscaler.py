"""High-detail image upscaling utilities for SocialVision."""

from __future__ import annotations

import io
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.config import get_config
from src.logger import setup_logger
from src.compat.torchvision_patch import ensure_functional_tensor_shim

logger = setup_logger(__name__)
config = get_config()

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
    """Facade around high-quality upscaling backends (Real-ESRGAN preferred)."""

    def __init__(self) -> None:
        self.enabled = getattr(config, "IMAGE_UPSCALING_ENABLED", True)
        self.backend_name = getattr(
            config, "IMAGE_UPSCALING_BACKEND", "realesrgan_x4plus"
        )
        self.target_scale = float(getattr(config, "IMAGE_UPSCALING_TARGET_SCALE", 2.0))
        self.min_edge = int(getattr(config, "IMAGE_UPSCALING_MIN_EDGE", 512))
        self.max_edge = int(getattr(config, "IMAGE_UPSCALING_MAX_EDGE", 2048))
        self.tile = int(getattr(config, "IMAGE_UPSCALING_TILE", 0))
        self.tile_pad = int(getattr(config, "IMAGE_UPSCALING_TILE_PAD", 10))
        self.use_half = bool(getattr(config, "IMAGE_UPSCALING_HALF_PRECISION", False))

        self._realesrgan = None
        self._realesrgan_scale = 1
        self._opencv_sr = None
        self._opencv_scale = 1
        self._device = "cpu"
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
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale an image while preserving detail."""

        if not self.enabled or image is None:
            self._last_backend = "disabled"
            return image

        try:
            height, width = image.shape[:2]
        except Exception:
            return image

        if max(height, width) >= self.max_edge:
            self._last_backend = "size_guard"
            return image

        outscale = self._compute_outscale(width, height)
        if outscale <= 1.0:
            self._last_backend = "no_scale"
            return image

        if self._max_client.enabled:
            upscaled = self._max_client.upscale(image)
            finalized = self._finalize_resolution(image, upscaled, outscale)
            if finalized is not None:
                self._last_backend = "ibm_max"
                logger.info("Upscaled frame via IBM MAX microservice")
                return finalized

        if self._ncnn_client.enabled:
            upscaled = self._ncnn_client.upscale(image)
            finalized = self._finalize_resolution(image, upscaled, outscale)
            if finalized is not None:
                self._last_backend = "ncnn"
                logger.info("Upscaled frame via Real-ESRGAN NCNN backend")
                return finalized

        if self._realesrgan:
            upscaled = self._upscale_with_realesrgan(image, outscale)
            finalized = self._finalize_resolution(image, upscaled, outscale)
            if finalized is not None:
                self._last_backend = "realesrgan"
                logger.info("Upscaled frame via native Real-ESRGAN backend")
                return finalized

        if self._opencv_sr:
            upscaled = self._upscale_with_opencv(image, outscale)
            finalized = self._finalize_resolution(image, upscaled, outscale)
            if finalized is not None:
                self._last_backend = "opencv"
                return finalized

        self._last_backend = "resize"
        return self._fallback_resize(image, outscale)

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------
    def _initialize_backends(self) -> None:
        initialized = self._init_realesrgan_backend()
        if initialized:
            return

        logger.warning(
            "Real-ESRGAN backend unavailable. Falling back to OpenCV super-resolution"
        )
        if self._init_opencv_superres_backend():
            return

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
        try:
            result, _ = self._realesrgan.enhance(image, outscale=outscale)
            return result
        except Exception as exc:
            logger.error("Real-ESRGAN upscaling failed: %s", exc, exc_info=True)
            return None

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
