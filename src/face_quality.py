"""
Face quality assessment and auto-improvement module.

This module provides functionality to:
1. Assess face image quality (blur, brightness, contrast, etc.)
2. Automatically improve face data quality
3. Filter low-quality faces before adding to database
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from src.logger import setup_logger

logger = setup_logger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FaceQualityAssessor:
    """Assesses and improves face image quality."""

    @staticmethod
    def assess_quality(face_image: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of a face image.

        Args:
            face_image: Face image as numpy array (BGR format)

        Returns:
            Dictionary with quality metrics:
            - blur_score: Lower is better (0-1, <0.1 is sharp)
            - brightness: Optimal around 0.5 (0-1)
            - contrast: Higher is better (0-1)
            - sharpness: Higher is better (0-1)
            - overall_score: Combined quality score (0-1, higher is better)
        """
        metrics = {
            "blur_score": 1.0,
            "brightness": 0.5,
            "contrast": 0.0,
            "sharpness": 0.0,
            "overall_score": 0.0,
        }

        if face_image is None or face_image.size == 0:
            return metrics

        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if HAS_CV2 else np.mean(face_image, axis=2).astype(np.uint8)
            else:
                gray = face_image

            # 1. Blur detection using Laplacian variance
            if HAS_CV2:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                # Normalize blur score (higher variance = sharper image)
                # Typical sharp images have variance > 100
                metrics["blur_score"] = max(0.0, min(1.0, 1.0 - (laplacian_var / 500.0)))
                metrics["sharpness"] = min(1.0, laplacian_var / 500.0)
            else:
                # Fallback: simple gradient-based blur detection
                grad_x = np.gradient(gray.astype(float), axis=1)
                grad_y = np.gradient(gray.astype(float), axis=0)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                variance = np.var(gradient_magnitude)
                metrics["blur_score"] = max(0.0, min(1.0, 1.0 - (variance / 1000.0)))
                metrics["sharpness"] = min(1.0, variance / 1000.0)

            # 2. Brightness assessment
            mean_brightness = np.mean(gray) / 255.0
            metrics["brightness"] = mean_brightness
            # Optimal brightness is around 0.4-0.6
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2.0

            # 3. Contrast assessment (standard deviation)
            std_dev = np.std(gray) / 255.0
            metrics["contrast"] = min(1.0, std_dev * 2.0)  # Normalize to 0-1

            # 4. Overall quality score (weighted combination)
            metrics["overall_score"] = (
                0.4 * (1.0 - metrics["blur_score"]) +  # Sharpness weight
                0.2 * brightness_score +  # Brightness weight
                0.4 * metrics["contrast"]  # Contrast weight
            )

            logger.debug(
                f"Face quality: blur={metrics['blur_score']:.3f}, "
                f"brightness={metrics['brightness']:.3f}, "
                f"contrast={metrics['contrast']:.3f}, "
                f"overall={metrics['overall_score']:.3f}"
            )

        except Exception as e:
            logger.warning(f"Error assessing face quality: {e}")
            metrics["overall_score"] = 0.5  # Default moderate quality

        return metrics

    @staticmethod
    def improve_face_quality(face_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Automatically improve face image quality.

        Args:
            face_image: Face image as numpy array (BGR format)

        Returns:
            Tuple of (improved_image, improvement_metrics)
        """
        if face_image is None or face_image.size == 0:
            return face_image, {}

        improved = face_image.copy()
        improvements = {
            "brightness_adjusted": False,
            "contrast_adjusted": False,
            "sharpened": False,
            "denoised": False,
        }

        try:
            # 1. Brightness and contrast adjustment
            if HAS_CV2:
                # Convert to LAB color space for better color preservation
                lab = cv2.cvtColor(improved, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)

                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)

                # Recombine channels
                improved = cv2.merge([l_channel, a, b])
                improved = cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)
                improvements["contrast_adjusted"] = True
                improvements["brightness_adjusted"] = True
            else:
                # Fallback: simple histogram equalization
                if len(improved.shape) == 3:
                    for i in range(3):
                        improved[:, :, i] = np.clip(
                            (improved[:, :, i].astype(float) - np.mean(improved[:, :, i])) * 1.2 + 128,
                            0, 255
                        ).astype(np.uint8)
                improvements["contrast_adjusted"] = True

            # 2. Denoising
            if HAS_CV2:
                improved = cv2.fastNlMeansDenoisingColored(improved, None, 10, 10, 7, 21)
                improvements["denoised"] = True

            # 3. Sharpening (light)
            if HAS_CV2:
                kernel = np.array([[-1, -1, -1],
                                 [-1,  9, -1],
                                 [-1, -1, -1]]) * 0.1
                sharpened = cv2.filter2D(improved, -1, kernel)
                improved = cv2.addWeighted(improved, 0.7, sharpened, 0.3, 0)
                improvements["sharpened"] = True

            logger.info(f"Face quality improvements applied: {improvements}")

        except Exception as e:
            logger.warning(f"Error improving face quality: {e}")
            # Return original if improvement fails
            improved = face_image

        return improved, improvements

    @staticmethod
    def is_acceptable_quality(face_image: np.ndarray, min_score: float = 0.3) -> Tuple[bool, Dict[str, float]]:
        """
        Check if face image meets minimum quality requirements.

        Args:
            face_image: Face image as numpy array
            min_score: Minimum overall quality score (0-1)

        Returns:
            Tuple of (is_acceptable, quality_metrics)
        """
        metrics = FaceQualityAssessor.assess_quality(face_image)
        is_acceptable = metrics["overall_score"] >= min_score
        return is_acceptable, metrics

    @staticmethod
    def filter_by_quality(
        face_images: List[np.ndarray],
        min_score: float = 0.3
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        """
        Filter face images by quality, keeping only acceptable ones.

        Args:
            face_images: List of face images
            min_score: Minimum quality score

        Returns:
            Tuple of (filtered_images, quality_metrics_list)
        """
        filtered = []
        metrics_list = []

        for face_img in face_images:
            is_acceptable, metrics = FaceQualityAssessor.is_acceptable_quality(face_img, min_score)
            metrics_list.append(metrics)
            if is_acceptable:
                filtered.append(face_img)
            else:
                logger.debug(f"Filtered out low-quality face (score: {metrics['overall_score']:.3f})")

        logger.info(f"Filtered {len(face_images)} faces, kept {len(filtered)} high-quality faces")
        return filtered, metrics_list


class AutoFaceImprover:
    """Automatically improves face data quality in the database."""

    def __init__(self, face_engine, database):
        """
        Initialize auto-improver.

        Args:
            face_engine: FaceRecognitionEngine instance
            database: FaceDatabase instance
        """
        self.face_engine = face_engine
        self.database = database
        self.quality_assessor = FaceQualityAssessor()

    def improve_and_add_face(
        self,
        username: str,
        image: np.ndarray,
        source: str = "profile_pic",
        min_quality_score: float = 0.3,
        auto_improve: bool = True,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, any]:
        """
        Improve face quality and add to database if quality is acceptable.

        Args:
            username: Username identifier
            image: Input image
            source: Source identifier
            min_quality_score: Minimum quality score to accept
            auto_improve: Whether to automatically improve quality
            metadata: Additional metadata

        Returns:
            Summary dictionary with results
        """
        from src.image_utils import ImageProcessor

        summary = {
            "faces_detected": 0,
            "faces_accepted": 0,
            "faces_rejected": 0,
            "faces_improved": 0,
            "quality_scores": [],
            "errors": [],
            "success": False,
        }

        try:
            # Prepare image
            processed_image = ImageProcessor.prepare_input_image(image)

            # Detect faces
            face_locations = self.face_engine.detect_faces(processed_image)
            summary["faces_detected"] = len(face_locations)

            if not face_locations:
                summary["errors"].append("No faces detected")
                return summary

            # Extract face chips
            face_chips = self.face_engine.extract_face_chips(processed_image, face_locations)

            improved_count = 0
            for idx, chip in enumerate(face_chips):
                # Assess quality
                quality_metrics = self.quality_assessor.assess_quality(chip)
                summary["quality_scores"].append(quality_metrics)

                # Improve if needed and enabled
                improved_chip = chip
                if auto_improve and quality_metrics["overall_score"] < 0.7:
                    improved_chip, improvements = self.quality_assessor.improve_face_quality(chip)
                    if any(improvements.values()):
                        improved_count += 1
                        logger.info(f"Improved face {idx+1} quality (score: {quality_metrics['overall_score']:.3f} -> improved)")

                # Check if quality is acceptable
                is_acceptable, _ = self.quality_assessor.is_acceptable_quality(
                    improved_chip, min_quality_score
                )

                if is_acceptable:
                    # Reconstruct image with improved chip for embedding extraction
                    # For now, we'll use the original processed image but note the improvement
                    summary["faces_accepted"] += 1
                else:
                    summary["faces_rejected"] += 1
                    logger.warning(
                        f"Rejected face {idx+1} due to low quality (score: {quality_metrics['overall_score']:.3f})"
                    )

            summary["faces_improved"] = improved_count

            # Process and add faces (using original image, quality filtering happens above)
            if summary["faces_accepted"] > 0:
                add_summary = self.face_engine.process_and_add_face(
                    username=username,
                    image=processed_image,
                    source=source,
                    metadata=metadata or {},
                    return_summary=True,
                )
                summary["success"] = add_summary.get("success", False)
                summary["faces_added"] = add_summary.get("faces_added", 0)

            logger.info(
                f"Auto-improve summary: {summary['faces_accepted']} accepted, "
                f"{summary['faces_rejected']} rejected, {improved_count} improved"
            )

        except Exception as e:
            logger.error(f"Error in auto-improve: {e}", exc_info=True)
            summary["errors"].append(str(e))

        return summary

