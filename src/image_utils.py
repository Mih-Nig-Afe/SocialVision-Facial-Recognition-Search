"""
Image processing utilities
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import io
from src.logger import setup_logger
from src.config import get_config

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
            # Try using cv2 first if available
            if HAS_CV2:
                image = cv2.imread(image_path)
                if image is not None:
                    logger.info(f"Loaded image: {image_path}")
                    return image

            # Fallback to PIL
            pil_image = Image.open(image_path).convert("RGB")
            # Convert PIL RGB to numpy BGR for compatibility
            image = np.array(pil_image)
            image = image[:, :, ::-1]  # RGB to BGR
            logger.info(f"Loaded image (PIL): {image_path}")
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
            if HAS_CV2:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    logger.error("Failed to decode image from bytes")
                    return None
                return image
            else:
                # Use PIL for decoding
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = np.array(pil_image)
                image = image[:, :, ::-1]  # RGB to BGR
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
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better face detection

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        try:
            if not HAS_CV2:
                logger.warning("cv2 not available, skipping image enhancement")
                return image

            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            logger.info("Image enhanced")
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
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

            if HAS_CV2:
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                # Use PIL for saving
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
                pil_image.save(output_path, quality=quality)

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
            if HAS_CV2:
                if format.lower() == "jpg":
                    _, buffer = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                else:
                    _, buffer = cv2.imencode(".png", image)
                return buffer.tobytes()
            else:
                # Use PIL for encoding
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
                buffer = io.BytesIO()
                pil_image.save(buffer, format=format.upper(), quality=85)
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
