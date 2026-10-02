from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    from .modality_validation import check_image_modality
except ImportError:
    from modality_validation import check_image_modality


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lung_mri_convmixer.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CONVMIXER
# Same architecture used during training
# ============================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        groups=dim,
                        padding=kernel_size // 2
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )
            ),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.block(x)


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim=256,
        depth=8,
        kernel_size=5,
        patch_size=4,
        n_classes=2
    ):
        super().__init__()

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                3,
                dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(dim=dim, kernel_size=kernel_size)
                for _ in range(depth)
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# ============================================================
# IMAGE TRANSFORM
# Same validation preprocessing used during training
# ============================================================

transform = transforms.Compose([
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

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# Your training script saved the model state in a checkpoint.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state = checkpoint["model_state_dict"]

    classes = checkpoint.get(
        "classes",
        ["cancer", "no_cancer"]
    )

    model = ConvMixer(
        dim=checkpoint.get("dim", 256),
        depth=checkpoint.get("depth", 8),
        kernel_size=checkpoint.get("kernel_size", 5),
        patch_size=checkpoint.get("patch_size", 4),
        n_classes=len(classes)
    )

else:
    # Fallback if only state_dict was saved
    model_state = checkpoint
    classes = ["cancer", "no_cancer"]

    model = ConvMixer(
        dim=256,
        depth=8,
        kernel_size=5,
        patch_size=4,
        n_classes=2
    )


model.load_state_dict(model_state)
model.to(DEVICE)
model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_mri(image_path):
    """
    Predict lung MRI image.

    Returns:
        dict containing prediction, confidence,
        and class probabilities.
    """

    image = Image.open(image_path).convert("RGB")
    modality_check = check_image_modality(image, "mri")

    if not modality_check["valid"]:
        return {
            "Model": "ConvMixer",
            "Modality": "Lung MRI",
            "Prediction": "Unknown / Needs Manual Verification",
            "Class": "unknown",
            "Confidence (%)": 0.0,
            "Class Probabilities (%)": {class_name: 0.0 for class_name in classes},
            "Device": str(DEVICE),
            "Manual Verification Required": True,
            "Unknown Reason": modality_check["reason"],
        }

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()
    confidence = confidence.item() * 100

    predicted_class = classes[predicted_index]

    class_probabilities = {
        classes[i]: round(
            probabilities[0][i].item() * 100,
            2
        )
        for i in range(len(classes))
    }

    # User-friendly result
    if predicted_class == "cancer":
        result = "Cancer Detected"
    else:
        result = "No Cancer Detected"

    return {
        "Model": "ConvMixer",
        "Modality": "Lung MRI",
        "Prediction": result,
        "Class": predicted_class,
        "Confidence (%)": round(confidence, 2),
        "Class Probabilities (%)": class_probabilities,
        "Device": str(DEVICE)
    }