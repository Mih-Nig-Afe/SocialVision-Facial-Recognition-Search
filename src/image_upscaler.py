"""High-detail image upscaling utilities for SocialVision."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import get_config
from src.logger import setup_logger

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
            return image

        try:
            height, width = image.shape[:2]
        except Exception:
            return image

        if max(height, width) >= self.max_edge:
            return image

        outscale = self._compute_outscale(width, height)
        if outscale <= 1.0:
            return image

        if self._realesrgan:
            upscaled = self._upscale_with_realesrgan(image, outscale)
            if upscaled is not None:
                return upscaled

        if self._opencv_sr:
            upscaled = self._upscale_with_opencv(image, outscale)
            if upscaled is not None:
                return upscaled

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

    def _resize_with_factor(self, image: np.ndarray, scale: float) -> np.ndarray:
        if scale <= 1.0:
            return image

        try:
            import cv2  # type: ignore

            new_width = max(1, int(round(image.shape[1] * scale)))
            new_height = max(1, int(round(image.shape[0] * scale)))
            resized = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC,
            )
            return resized
        except Exception:
            pil_image = self._to_pil(image)
            if pil_image is None:
                return image
            new_width = max(1, int(round(pil_image.width * scale)))
            new_height = max(1, int(round(pil_image.height * scale)))
            resized = pil_image.resize((new_width, new_height))
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
            from PIL import Image  # type: ignore

            if image.ndim == 2:
                mode = "L"
            else:
                mode = "RGB"
                image = image[:, :, ::-1]
            return Image.fromarray(image, mode=mode)
        except Exception:
            return None
