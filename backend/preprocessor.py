"""Intelligent image preprocessing for PruneVision."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from rembg import remove as rembg_remove

    REMBG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    rembg_remove = None
    REMBG_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

MIN_WIDTH = 100
MIN_HEIGHT = 100
BLUR_THRESHOLD = 100.0
BRIGHTNESS_LOW_THRESHOLD = 40.0
BRIGHTNESS_HIGH_THRESHOLD = 220.0
QUALITY_SCORE_MAX = 100.0
QUALITY_SCORE_MIN = 0.0
TARGET_SIZE = (224, 224)
GREEN_HUE_LOW = 25
GREEN_HUE_HIGH = 95
CONTOUR_PADDING = 10
CENTER_CROP_RATIO = 0.8
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAMMA_EPSILON = 1e-6
U2NET_MODEL_PATH = Path.home() / ".u2net" / "u2net.onnx"
ALLOW_REMBG_DOWNLOAD = False


@dataclass(slots=True)
class QualityCheckResult:
    """Represents the result of the quality gate."""

    quality_score: float
    quality_passed: bool
    reason: Optional[str]
    blur_variance: float
    brightness_mean: float
    original_size: Tuple[int, int]


@dataclass(slots=True)
class PreprocessingReport:
    """Summarizes all preprocessing steps and diagnostics."""

    quality_score: float
    quality_passed: bool
    quality_reason: Optional[str]
    enhancements_applied: list[str]
    leaf_detected: bool
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    processing_time_ms: float


@dataclass(slots=True)
class ConfidenceResult:
    """Represents the confidence-based recommendation output."""

    status: str
    label: str
    confidence: float
    recommendation: str
    warning: Optional[str]


class IntelligentPreprocessor:
    """Run quality checking, enhancement, segmentation, and confidence gating."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""

        self.target_size = TARGET_SIZE

    def _decode_image(self, image_input: bytes | bytearray | str | Path | np.ndarray) -> np.ndarray:
        """Decode supported inputs into a BGR OpenCV image."""

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            return image_input.copy()

        if isinstance(image_input, (str, Path)):
            image_bytes = Path(image_input).read_bytes()
        else:
            image_bytes = bytes(image_input)

        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            decoded = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return decoded

    def _check_quality(self, image_bgr: np.ndarray) -> QualityCheckResult:
        """Compute image quality metrics and decide whether the image is usable."""

        height, width = image_bgr.shape[:2]
        original_size = (width, height)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(np.mean(gray))

        score = QUALITY_SCORE_MAX
        reasons: list[str] = []

        if blur_variance < BLUR_THRESHOLD:
            score -= min(40.0, (BLUR_THRESHOLD - blur_variance) * 0.4)
            reasons.append("Image appears blurry")

        if brightness_mean < BRIGHTNESS_LOW_THRESHOLD:
            score -= min(30.0, (BRIGHTNESS_LOW_THRESHOLD - brightness_mean) * 0.75)
            reasons.append("Image is too dark")
        elif brightness_mean > BRIGHTNESS_HIGH_THRESHOLD:
            score -= min(30.0, (brightness_mean - BRIGHTNESS_HIGH_THRESHOLD) * 0.75)
            reasons.append("Image is too bright")

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            score -= 50.0
            reasons.append("Image is too small")

        score = float(np.clip(score, QUALITY_SCORE_MIN, QUALITY_SCORE_MAX))
        quality_passed = not reasons
        reason = "; ".join(reasons) if reasons else None

        LOGGER.info(
            "Quality check: blur_variance=%.2f brightness_mean=%.2f score=%.2f passed=%s",
            blur_variance,
            brightness_mean,
            score,
            quality_passed,
        )

        return QualityCheckResult(
            quality_score=score,
            quality_passed=quality_passed,
            reason=reason,
            blur_variance=blur_variance,
            brightness_mean=brightness_mean,
            original_size=original_size,
        )

    def _auto_enhance(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Apply denoising, contrast enhancement, sharpening, and gamma correction."""

        enhancements_applied: list[str] = []
        enhanced = image_bgr.copy()

        try:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            enhancements_applied.append("denoising")
        except Exception as exc:
            LOGGER.warning("Denoising skipped: %s", exc)

        try:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge((l_channel, a_channel, b_channel))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhancements_applied.append("clahe_contrast")
        except Exception as exc:
            LOGGER.warning("CLAHE skipped: %s", exc)

        try:
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
            enhancements_applied.append("sharpening")
        except Exception as exc:
            LOGGER.warning("Sharpening skipped: %s", exc)

        gray_mean = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
        normalized_mean = max(gray_mean / 255.0, GAMMA_EPSILON)
        gamma = 1.0
        if normalized_mean < 0.5:
            gamma = 0.75 + (0.5 - normalized_mean)
        elif normalized_mean > 0.75:
            gamma = 1.0 + (normalized_mean - 0.75)

        inv_gamma = 1.0 / max(gamma, GAMMA_EPSILON)
        lookup_table = np.array(
            [((index / 255.0) ** inv_gamma) * 255.0 for index in range(256)]
        ).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, lookup_table)
        enhancements_applied.append("gamma_correction")

        LOGGER.info("Enhancement stage completed with steps: %s", ", ".join(enhancements_applied))
        return enhanced, enhancements_applied

    def _segment_leaf(self, image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Try to isolate the leaf using rembg, green masking, and contour cropping."""

        leaf_detected = False
        processed = image_bgr.copy()

        if REMBG_AVAILABLE and (ALLOW_REMBG_DOWNLOAD or U2NET_MODEL_PATH.exists()):
            try:
                rgba = rembg_remove(Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)))
                if isinstance(rgba, Image.Image):
                    rgba = np.array(rgba)
                if rgba.ndim == 3 and rgba.shape[2] == 4:
                    alpha = rgba[:, :, 3]
                    rgb = rgba[:, :, :3]
                    mask_from_alpha = (alpha > 0).astype(np.uint8) * 255
                    processed = cv2.bitwise_and(rgb, rgb, mask=mask_from_alpha)
                    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    leaf_detected = True
                LOGGER.info("Background removal completed using rembg")
            except Exception as exc:
                LOGGER.warning("rembg segmentation skipped: %s", exc)
        elif REMBG_AVAILABLE:
            LOGGER.info(
                "Skipping rembg model download during inference. "
                "Model file not found at %s; using HSV contour segmentation fallback.",
                U2NET_MODEL_PATH,
            )

        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([GREEN_HUE_LOW, 20, 20], dtype=np.uint8),
            np.array([GREEN_HUE_HIGH, 255, 255], dtype=np.uint8),
        )
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 0:
                x, y, width, height = cv2.boundingRect(largest_contour)
                x = max(0, x - CONTOUR_PADDING)
                y = max(0, y - CONTOUR_PADDING)
                width = min(processed.shape[1] - x, width + 2 * CONTOUR_PADDING)
                height = min(processed.shape[0] - y, height + 2 * CONTOUR_PADDING)
                processed = processed[y : y + height, x : x + width]
                leaf_detected = True
                LOGGER.info("Leaf contour detected and cropped at x=%d y=%d w=%d h=%d", x, y, width, height)

        if not leaf_detected:
            LOGGER.info("Leaf not detected; applying center crop fallback")
            height, width = processed.shape[:2]
            crop_size = int(min(width, height) * CENTER_CROP_RATIO)
            crop_size = max(crop_size, 1)
            x_start = max(0, (width - crop_size) // 2)
            y_start = max(0, (height - crop_size) // 2)
            processed = processed[y_start : y_start + crop_size, x_start : x_start + crop_size]

        processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)
        LOGGER.info("Segmentation output resized to %s", self.target_size)
        return processed, leaf_detected

    def process(self, image_input: bytes | bytearray | str | Path | np.ndarray) -> tuple[np.ndarray, PreprocessingReport]:
        """Run the full preprocessing pipeline and return the processed image and report."""

        start_time = time.perf_counter()
        image_bgr = self._decode_image(image_input)
        original_size = (image_bgr.shape[1], image_bgr.shape[0])

        quality_result = self._check_quality(image_bgr)
        enhanced_image, enhancements_applied = self._auto_enhance(image_bgr)
        segmented_image, leaf_detected = self._segment_leaf(enhanced_image)

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        report = PreprocessingReport(
            quality_score=quality_result.quality_score,
            quality_passed=quality_result.quality_passed,
            quality_reason=quality_result.reason,
            enhancements_applied=enhancements_applied,
            leaf_detected=leaf_detected,
            original_size=original_size,
            processed_size=(segmented_image.shape[1], segmented_image.shape[0]),
            processing_time_ms=processing_time_ms,
        )

        LOGGER.info(
            "Preprocessing finished in %.2f ms, leaf_detected=%s, passed=%s",
            processing_time_ms,
            leaf_detected,
            quality_result.quality_passed,
        )
        return segmented_image, report

    def gate_confidence(self, label: str, confidence: float) -> ConfidenceResult:
        """Convert a raw confidence score into a final decision payload."""

        if confidence > 0.75:
            status = "high_confidence"
            recommendation = "Prediction is strong. Proceed with the suggested treatment path."
            warning = None
        elif confidence >= 0.50:
            status = "medium_confidence"
            recommendation = "Prediction is plausible, but a clearer image may improve certainty."
            warning = "Confidence is moderate. Consider retaking the image for confirmation."
        else:
            status = "low_confidence"
            recommendation = "Image quality or model certainty is low. Please retake the image."
            warning = "Confidence is low. The image should be retaken before relying on the result."

        LOGGER.info("Confidence gating result: label=%s confidence=%.4f status=%s", label, confidence, status)
        return ConfidenceResult(
            status=status,
            label=label,
            confidence=float(confidence),
            recommendation=recommendation,
            warning=warning,
        )

    def report_to_dict(self, report: PreprocessingReport) -> dict[str, Any]:
        """Convert a preprocessing report into a JSON-serializable dictionary."""

        return asdict(report)
