from PIL import Image
import numpy as np


COLORFULNESS_THRESHOLD = 8.0
HISTOPATHOLOGY_COLORFULNESS_THRESHOLD = 12.0
HISTOPATHOLOGY_SATURATION_RATIO = 0.25
# Histology tissue can be smooth after resizing, so texture is only a
# secondary guard; color and saturation reject grayscale CT/MRI images.
HISTOPATHOLOGY_TEXTURE_THRESHOLD = 0.75
SCAN_BACKGROUND_THRESHOLD = 20


def check_image_modality(image, expected_modality):
    """Return a conservative modality check based on training-image appearance."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    colorfulness = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    grayscale_minimum = int(grayscale.min())

    if expected_modality == "histopathology":
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        saturation_ratio = float((saturation >= 20).mean())
        texture = float(np.abs(np.diff(grayscale, axis=0)).mean())
        valid = (
            colorfulness >= HISTOPATHOLOGY_COLORFULNESS_THRESHOLD
            and saturation_ratio >= HISTOPATHOLOGY_SATURATION_RATIO
            and texture >= HISTOPATHOLOGY_TEXTURE_THRESHOLD
        )
        reason = (
            "This image does not have the color and tissue texture expected "
            "from a stained histopathology image."
        )
    elif expected_modality == "mri":
        valid = colorfulness < COLORFULNESS_THRESHOLD and grayscale_minimum <= SCAN_BACKGROUND_THRESHOLD
        reason = "This image does not match the expected MRI scan appearance."
    elif expected_modality == "ct":
        valid = colorfulness < COLORFULNESS_THRESHOLD and grayscale_minimum > SCAN_BACKGROUND_THRESHOLD
        reason = "This image does not match the expected CT scan appearance."
    else:
        raise ValueError(f"Unsupported modality: {expected_modality}")

    return {
        "valid": valid,
        "reason": reason,
        "colorfulness": round(colorfulness, 2),
        "saturation_ratio": round(saturation_ratio, 3) if expected_modality == "histopathology" else None,
        "texture": round(texture, 2) if expected_modality == "histopathology" else None,
        "grayscale_minimum": grayscale_minimum,
    }
