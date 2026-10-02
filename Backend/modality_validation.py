from PIL import Image
import numpy as np


COLORFULNESS_THRESHOLD = 8.0
SCAN_BACKGROUND_THRESHOLD = 20


def check_image_modality(image, expected_modality):
    """Return a conservative modality check based on training-image appearance."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    grayscale = np.asarray(image.convert("L"), dtype=np.int16)
    colorfulness = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    grayscale_minimum = int(grayscale.min())

    if expected_modality == "histopathology":
        valid = colorfulness >= COLORFULNESS_THRESHOLD
        reason = "This image looks like a grayscale scan rather than a stained tissue image."
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
        "grayscale_minimum": grayscale_minimum,
    }
