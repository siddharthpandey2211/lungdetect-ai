import os
import uuid

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    from .modality_validation import check_image_modality
except ImportError:
    from modality_validation import check_image_modality


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung_resnet18.pth"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "histopathology",
    "uploads",
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "lung_aca",
    "lung_n",
    "lung_scc"
]

# Minimum confidence required to automatically assign
# an uploaded image to one of the trained classes.
CONFIDENCE_THRESHOLD = float(
    os.getenv("HISTOPATHOLOGY_CONFIDENCE_THRESHOLD", "60")
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_image_model():

    print("Loading histopathology ResNet18 model...")
    print("Model:", MODEL_PATH)
    print("Device:", DEVICE)

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Histopathology CNN loaded successfully")

    return model


image_model = load_image_model()


# ============================================================
# SAVE IMAGE
# ============================================================

def save_uploaded_image(image):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(UPLOAD_DIR, filename)
    image.convert("RGB").save(save_path, format="JPEG", quality=95)
    return save_path, filename


# ============================================================
# PREDICT HISTOPATHOLOGY
# ============================================================

def predict_histopathology(
    image,
    original_filename=None,
    saved_upload_path=None,
):

    try:

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        image = image.convert("RGB")
        if saved_upload_path is None:
            _, saved_filename = save_uploaded_image(image)
        else:
            saved_filename = os.path.basename(saved_upload_path)

        modality_check = check_image_modality(image, "histopathology")

        if not modality_check["valid"]:
            return {
                "Model": "ResNet18 Histopathology CNN",
                "Prediction": "Unknown / Needs Manual Verification",
                "Class": "unknown",
                "Model Prediction": "unknown",
                "Confidence (%)": 0.0,
                "Threshold (%)": CONFIDENCE_THRESHOLD,
                "Class Probabilities (%)": {class_name: 0.0 for class_name in CLASS_NAMES},
                "Definition": "This image does not match the expected histopathology appearance and needs manual verification.",
                "Storage Class": "uploads",
                "Saved Filename": saved_filename,
                "Storage Status": (
                    "Image retained in uploads; it does not match the expected "
                    "histopathology modality."
                ),
                "Unknown Reason": modality_check["reason"],
                "Manual Verification Required": True,
                "Device": str(DEVICE),
            }

        image_tensor = image_transform(
            image
        ).unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = image_model(
                image_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()


        probabilities = (
            probabilities[0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Get prediction
        # ----------------------------------------------------

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )


        # ----------------------------------------------------
        # Get all probabilities
        # ----------------------------------------------------

        class_probabilities = {
            CLASS_NAMES[i]:
            round(
                float(probabilities[i] * 100),
                2
            )
            for i in range(len(CLASS_NAMES))
        }


        if confidence >= CONFIDENCE_THRESHOLD:
            final_class = predicted_class
            storage_status = (
                "Image retained in uploads; prediction does not change its "
                "training-data location."
            )
        else:
            final_class = "unknown"
            storage_status = (
                f"Image retained in uploads; confidence is below the "
                f"{CONFIDENCE_THRESHOLD}% threshold and needs manual review."
            )


        # ----------------------------------------------------
        # Display-friendly result
        # ----------------------------------------------------

        if final_class == "lung_aca":
            result = "Cancer"
            definition = (
                "Adenocarcinoma is defined as a malignant cancer "
                "that starts in gland cells."
            )
        elif final_class == "lung_scc":
            result = "Cancer"
            definition = (
                "Squamous cell carcinoma is a malignant tumor "
                "arising from squamous cells."
            )
        elif final_class == "lung_n":
            result = "No Cancer"
            definition = (
                "The image is classified as normal lung tissue, "
                "with no cancer pattern detected."
            )
        else:
            result = "Unknown / Needs Manual Verification"
            definition = (
                "This image does not match the expected histopathology "
                "appearance and needs manual verification."
            )


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {

            "Model":
                "ResNet18 Histopathology CNN",

            "Prediction":
                result,

            "Definition":
                definition,

            "Cancer Status":
                result,

            "Class":
                final_class,

            "Model Prediction":
                predicted_class,

            "Confidence (%)":
                round(confidence, 2),

            "Threshold (%)":
                CONFIDENCE_THRESHOLD,

            "Class Probabilities (%)":
                class_probabilities,

            "Storage Class":
                "uploads",

            "Saved Filename":
                saved_filename,

            "Storage Status":
                storage_status,

            "Device":
                str(DEVICE),

            "Manual Verification Required":
                final_class == "unknown",

            "Disclaimer":
                "Educational/prototype result only. "
                "This image classification does not "
                "constitute a medical diagnosis."
        }


    except Exception as e:

        raise RuntimeError(
            f"Image prediction failed: {str(e)}"
        )