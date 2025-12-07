"""
Image processing utilities
"""

import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple
from PIL import Image, ImageDraw
import io
from src.logger import setup_logger
from src.config import get_config
from src.image_upscaler import get_image_upscaler

logger = setup_logger(__name__)
config = get_config()

# Try to import cv2, but make it optional
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageProcessor:
    """Image processing utilities"""

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format) or None
        """
        try:
            decoders = []
            if HAS_CV2:
                decoders.append(
                    lambda: ImageProcessor._decode_with_cv2_path(image_path)
                )
            decoders.append(lambda: ImageProcessor._decode_with_pillow(image_path))

            image = ImageProcessor._try_decoders(decoders)
            if image is not None:
                logger.info(f"Loaded image: {image_path}")
            else:
                logger.error(
                    f"Error loading image: unsupported or corrupted file ({image_path})"
                )
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load image from bytes

        Args:
            image_bytes: Image data as bytes

        Returns:
            Image as numpy array (BGR format) or None
        """
        try:
            decoders = []
            if HAS_CV2:
                decoders.append(
                    lambda: ImageProcessor._decode_with_cv2_bytes(image_bytes)
                )

            def pillow_decode():
                return ImageProcessor._decode_with_pillow(io.BytesIO(image_bytes))

            decoders.append(pillow_decode)

            image = ImageProcessor._try_decoders(decoders)
            if image is None:
                logger.error("Failed to decode image from bytes")
            return image
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            return None

    @staticmethod
    def resize_image(
        image: np.ndarray, max_width: int = 1280, max_height: int = 720
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio

        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height

        Returns:
            Resized image
        """
        try:
            height, width = image.shape[:2]

            # Calculate scaling factor
            scale = min(max_width / width, max_height / height, 1.0)

            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)

                if HAS_CV2:
                    image = cv2.resize(image, (new_width, new_height))
                else:
                    # Use PIL for resizing
                    pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
                    pil_image = pil_image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )
                    image = np.array(pil_image)[:, :, ::-1]  # RGB to BGR

                logger.info(f"Resized image to {new_width}x{new_height}")

            return image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image

    @staticmethod
    def prepare_input_image(
        image: np.ndarray,
        max_width: int = 1400,
        max_height: int = 1400,
    ) -> np.ndarray:
        """Resize image before sending to recognition pipeline."""
        enhanced = ImageProcessor.enhance_image(image)
        return ImageProcessor.resize_image(enhanced, max_width, max_height)

    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Run the configured super-resolution backend before downstream processing."""

        try:
            upscaler = get_image_upscaler()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Unable to initialize image upscaler: %s", exc)
            return image

        try:
            enhanced = upscaler.upscale(image)
            backend = getattr(upscaler, "last_backend", "unknown")
            logger.info("Image enhanced via %s backend", backend)
            return enhanced
        except Exception as exc:
            logger.error("Image upscaling failed: %s", exc)
            return image

    @staticmethod
    def frame_image_for_display(
        image: np.ndarray,
        frame_size: Tuple[int, int] = (520, 520),
        padding: int = 18,
        background_color: Tuple[int, int, int] = (12, 12, 12),
        border_color: Tuple[int, int, int] = (80, 80, 80),
    ) -> np.ndarray:
        """Place an image inside a fixed-size frame for consistent UI presentation."""

        try:
            target_w, target_h = frame_size
            target_w = max(target_w, 10)
            target_h = max(target_h, 10)
            pad_w = max(target_w - 2 * padding, 1)
            pad_h = max(target_h - 2 * padding, 1)

            height, width = image.shape[:2]
            scale = min(pad_w / width, pad_h / height, 1.0)
            new_w = max(int(width * scale), 1)
            new_h = max(int(height * scale), 1)

            if HAS_CV2:
                resized = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                resized = np.array(pil_image)[:, :, ::-1]

            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            frame[:] = background_color

            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            frame[top : top + new_h, left : left + new_w] = resized

            if HAS_CV2:
                cv2.rectangle(
                    frame,
                    (left - 4 if left - 4 > 0 else 0, top - 4 if top - 4 > 0 else 0),
                    (
                        (
                            left + new_w + 4
                            if left + new_w + 4 < target_w
                            else target_w - 1
                        ),
                        top + new_h + 4 if top + new_h + 4 < target_h else target_h - 1,
                    ),
                    border_color,
                    1,
                )
            else:
                pil_frame = Image.fromarray(frame[:, :, ::-1])
                draw = ImageDraw.Draw(pil_frame)
                draw.rectangle(
                    [
                        left - 4 if left - 4 > 0 else 0,
                        top - 4 if top - 4 > 0 else 0,
                        (
                            left + new_w + 4
                            if left + new_w + 4 < target_w
                            else target_w - 1
                        ),
                        top + new_h + 4 if top + new_h + 4 < target_h else target_h - 1,
                    ],
                    outline=border_color,
                    width=2,
                )
                frame = np.array(pil_frame)[:, :, ::-1]

            return frame
        except Exception as exc:
            logger.error(f"Error framing image for display: {exc}")
            return image

    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 85) -> bool:
        """
        Save image to file

        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (1-100)

        Returns:
            True if successful
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            extension = Path(output_path).suffix.lstrip(".").lower() or "jpg"

            if HAS_CV2 and extension in {"jpg", "jpeg", "png"}:
                params = (
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                    if extension in {"jpg", "jpeg"}
                    else []
                )
                success = cv2.imwrite(output_path, image, params)
                if not success:
                    raise ValueError("cv2.imwrite returned False")
            else:
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
                pil_kwargs = (
                    {"quality": quality} if extension in {"jpg", "jpeg", "webp"} else {}
                )
                pil_image.save(
                    output_path,
                    format=ImageProcessor._normalize_pil_format(extension),
                    **pil_kwargs,
                )

            logger.info(f"Saved image: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    @staticmethod
    def draw_faces(
        image: np.ndarray, face_locations: list, labels: Optional[list] = None
    ) -> np.ndarray:
        """
        Draw rectangles around detected faces

        Args:
            image: Input image
            face_locations: List of face locations (top, right, bottom, left)
            labels: Optional labels for each face

        Returns:
            Image with drawn faces
        """
        try:
            image_copy = image.copy()

            if HAS_CV2:
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Draw rectangle
                    cv2.rectangle(
                        image_copy, (left, top), (right, bottom), (0, 255, 0), 2
                    )

                    # Draw label if provided
                    if labels and i < len(labels):
                        label = labels[i]
                        cv2.putText(
                            image_copy,
                            label,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
            else:
                # Use PIL for drawing
                pil_image = Image.fromarray(image_copy[:, :, ::-1])  # BGR to RGB
                draw = ImageDraw.Draw(pil_image)

                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Draw rectangle (green color in RGB)
                    draw.rectangle(
                        [left, top, right, bottom], outline=(0, 255, 0), width=2
                    )

                    # Draw label if provided
                    if labels and i < len(labels):
                        label = labels[i]
                        draw.text((left, top - 10), label, fill=(0, 255, 0))

                image_copy = np.array(pil_image)[:, :, ::-1]  # RGB to BGR

            logger.info(f"Drew {len(face_locations)} faces on image")
            return image_copy
        except Exception as e:
            logger.error(f"Error drawing faces: {e}")
            return image

    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = "jpg") -> bytes:
        """
        Convert image to bytes

        Args:
            image: Input image
            format: Output format (jpg, png)

        Returns:
            Image as bytes
        """
        try:
            fmt = (format or "jpg").lower()
            if HAS_CV2 and fmt in {"jpg", "jpeg", "png"}:
                ext = ".jpg" if fmt in {"jpg", "jpeg"} else ".png"
                params = (
                    [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt in {"jpg", "jpeg"} else []
                )
                ok, buffer = cv2.imencode(ext, image, params)
                if not ok:
                    raise ValueError("cv2.imencode failed")
                return buffer.tobytes()

            pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            buffer = io.BytesIO()
            pil_kwargs = {"quality": 85} if fmt in {"jpg", "jpeg", "webp"} else {}
            pil_image.save(
                buffer,
                format=ImageProcessor._normalize_pil_format(fmt),
                **pil_kwargs,
            )
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return b""

    @staticmethod
    def validate_image(file_path: str) -> Tuple[bool, str]:
        """
        Validate image file

        Args:
            file_path: Path to image file

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check file exists
            if not Path(file_path).exists():
                return False, "File does not exist"

            # Check file size
            file_size = Path(file_path).stat().st_size
            if file_size > config.MAX_IMAGE_SIZE:
                return False, f"File size exceeds {config.MAX_IMAGE_SIZE} bytes"

            # Check file format
            file_ext = Path(file_path).suffix.lower().lstrip(".")
            if file_ext not in config.ALLOWED_IMAGE_FORMATS:
                return (
                    False,
                    f"File format not supported. Allowed: {config.ALLOWED_IMAGE_FORMATS}",
                )

            # Try to load image
            if HAS_CV2:
                image = cv2.imread(file_path)
                if image is None:
                    return False, "Failed to load image"
            else:
                # Use PIL for validation
                try:
                    Image.open(file_path)
                except Exception:
                    return False, "Failed to load image"

            return True, "Valid image"

        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, str(e)

    @staticmethod
    def _try_decoders(
        decoders: Iterable[Callable[[], Optional[np.ndarray]]],
    ) -> Optional[np.ndarray]:
        for decode in decoders:
            try:
                image = decode()
            except Exception as exc:
                logger.debug(f"Decoder failed: {exc}")
                continue
            if image is not None:
                return image
        return None

    @staticmethod
    def _decode_with_cv2_path(image_path: str) -> Optional[np.ndarray]:
        if not HAS_CV2:
            return None
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _decode_with_cv2_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        if not HAS_CV2:
            return None
        if not image_bytes:
            return None
        nparr = np.frombuffer(image_bytes, np.uint8)
        if nparr.size == 0:
            return None
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def _decode_with_pillow(source) -> Optional[np.ndarray]:
        try:
            if isinstance(source, io.BytesIO):
                source.seek(0)
            with Image.open(source) as pil_image:
                if getattr(pil_image, "is_animated", False):
                    pil_image.seek(0)
                rgb_image = pil_image.convert("RGB")
                data = np.array(rgb_image)
                return data[:, :, ::-1]
        except Exception as exc:
            logger.debug(f"Pillow decode failed: {exc}")
            return None

    @staticmethod
    def _normalize_pil_format(extension: str) -> str:
        mapping = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "gif": "GIF",
            "bmp": "BMP",
            "webp": "WEBP",
            "tiff": "TIFF",
            "tif": "TIFF",
            "ico": "ICO",
            "ppm": "PPM",
            "pgm": "PPM",
            "pbm": "PPM",
        }
        normalized = mapping.get(extension.lower(), extension.upper())
        return normalized or "JPEG"
