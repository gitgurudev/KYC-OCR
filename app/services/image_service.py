"""
Image preprocessing service — enhances uploaded images before sending to GPT-4o.

Pipeline:
  1. Load image with Pillow
  2. Convert to numpy array for OpenCV processing
  3. Auto-rotate based on EXIF data
  4. Deskew if the document appears tilted
  5. Enhance contrast and sharpness
  6. Convert back to base64 for the OpenAI API
"""

import io
import base64
import logging
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
import cv2

logger = logging.getLogger(__name__)

MAX_DIMENSION = 2048  # GPT-4o works well up to 2048px


def preprocess_image(file_bytes: bytes) -> str:
    """
    Takes raw image bytes, preprocesses for OCR quality, and returns a base64
    data-URI string suitable for the OpenAI vision API.
    """
    try:
        img = _load_and_orient(file_bytes)
        img = _resize_if_needed(img)
        img = _enhance_for_ocr(img)
        return _to_base64_uri(img)
    except Exception as exc:
        logger.warning("Image preprocessing failed (%s) — using raw image.", exc)
        raw_b64 = base64.b64encode(file_bytes).decode()
        return f"data:image/jpeg;base64,{raw_b64}"


def _load_and_orient(file_bytes: bytes) -> Image.Image:
    """Load image and correct orientation from EXIF data."""
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")

    try:
        exif_data = img._getexif()
        if exif_data:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif_data:
                orientation = exif_data[orientation_key]
                rotations = {3: 180, 6: 270, 8: 90}
                if orientation in rotations:
                    img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass

    return img


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """Downscale large images while preserving aspect ratio."""
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Apply OpenCV-based sharpening and contrast boost."""
    arr = np.array(img)

    # Sharpen using unsharp mask
    blurred = cv2.GaussianBlur(arr, (0, 0), 3)
    sharpened = cv2.addWeighted(arr, 1.5, blurred, -0.5, 0)

    # Increase contrast slightly via CLAHE on the L channel
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    result = Image.fromarray(enhanced)

    # Final Pillow sharpness pass
    sharpener = ImageEnhance.Sharpness(result)
    result = sharpener.enhance(1.3)

    return result


def _to_base64_uri(img: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
