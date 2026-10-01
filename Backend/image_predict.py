import os
import uuid

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung_resnet18.pth"
)

SAVE_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "histopathology"
)

UNKNOWN_DIR = os.path.join(
    SAVE_DIR,
    "unknown"
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
    os.getenv("HISTOPATHOLOGY_CONFIDENCE_THRESHOLD", "70")
)


# Create unknown folder if it doesn't exist
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# Make sure class folders exist
for class_name in CLASS_NAMES:
    os.makedirs(
        os.path.join(SAVE_DIR, class_name),
        exist_ok=True
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

    print("✓ Histopathology CNN loaded successfully")

    return model


image_model = load_image_model()


# ============================================================
# SAVE IMAGE
# ============================================================

def save_uploaded_image(image, original_filename, folder):

    # Generate unique filename
    unique_id = uuid.uuid4().hex

    # Always save as JPG
    filename = f"{unique_id}.jpg"

    save_path = os.path.join(
        folder,
        filename
    )

    # Convert to RGB
    image = image.convert("RGB")

    # Save image
    image.save(
        save_path,
        format="JPEG",
        quality=95
    )

    return save_path, filename


# ============================================================
# PREDICT HISTOPATHOLOGY
# ============================================================

def predict_histopathology(
    image,
    original_filename=None
):

    try:

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        image = image.convert("RGB")

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


        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            # Model is confident enough
            final_class = predicted_class

            save_folder = os.path.join(
                SAVE_DIR,
                final_class
            )

            storage_status = (
                f"Image classified as {final_class}"
            )

        else:

            # Model is not confident enough
            final_class = "unknown"

            save_folder = UNKNOWN_DIR

            storage_status = (
                "Image sent to unknown because "
                "model confidence is below the "
                f"{CONFIDENCE_THRESHOLD}% threshold."
            )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        saved_path, saved_filename = (
            save_uploaded_image(
                image,
                original_filename,
                save_folder
            )
        )


        # ----------------------------------------------------
        # Display-friendly result
        # ----------------------------------------------------

        if final_class == "lung_n":

            result = "Normal Lung Tissue"

        elif final_class == "lung_aca":

            result = "Lung Adenocarcinoma Pattern"

        elif final_class == "lung_scc":

            result = "Lung Squamous Cell Carcinoma Pattern"

        else:

            result = "Unknown / Needs Manual Verification"


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {

            "Model":
                "ResNet18 Histopathology CNN",

            "Prediction":
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
                final_class,

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