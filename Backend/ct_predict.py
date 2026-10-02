from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

try:
    from .modality_validation import check_image_modality
except ImportError:
    from modality_validation import check_image_modality


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "ct_resnet18.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

CLASS_NAMES = checkpoint["class_names"]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_ct(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"CT image not found: {image_path}"
        )

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")
    modality_check = check_image_modality(image, "ct")

    if not modality_check["valid"]:
        return {
            "Model": "ResNet18",
            "Modality": "Lung CT Scan",
            "Prediction": "Unknown / Needs Manual Verification",
            "Class": "unknown",
            "Confidence (%)": 0.0,
            "Class Probabilities (%)": {class_name: 0.0 for class_name in CLASS_NAMES},
            "Device": str(DEVICE),
            "Manual Verification Required": True,
            "Unknown Reason": modality_check["reason"],
        }

    # Preprocess
    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # Inference
    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()

    confidence = confidence.item() * 100

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # All class probabilities
    class_probabilities = {}

    for index, class_name in enumerate(CLASS_NAMES):

        class_probabilities[class_name] = round(
            probabilities[0][index].item() * 100,
            2
        )

    # User-friendly prediction
    if predicted_class == "malignant":

        prediction = "Malignant Pattern Detected"

    elif predicted_class == "benign":

        prediction = "Benign Pattern Detected"

    else:

        prediction = "Normal Pattern Detected"


    return {
        "Model": "ResNet18",
        "Modality": "Lung CT Scan",

        "Prediction": prediction,

        "Class": predicted_class,

        "Confidence (%)": round(
            confidence,
            2
        ),

        "Class Probabilities (%)": class_probabilities,

        "Device": str(DEVICE)
    }