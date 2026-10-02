from pathlib import Path
import zipfile
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from tqdm import tqdm


# ============================================================
# LungDetect AI - Lung MRI ConvMixer
# ============================================================

print("=" * 70)
print("LungDetect AI - Lung MRI ConvMixer")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is inside:
# Early Cancer Detection/Backend/train_mri_convmixer.py
#
# Therefore parent.parent = Early Cancer Detection

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "Data" / "mri"
DOWNLOAD_ROOT = DATA_ROOT / "downloads"
EXTRACT_ROOT = DATA_ROOT / "extracted"

MODEL_ROOT = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_ROOT / "lung_mri_convmixer.pth"

ZIP_PATH = DOWNLOAD_ROOT / "lung-cancer-mri-images.zip"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# Important for Windows.
# We previously encountered multiprocessing/DataLoader problems.
NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "CUDA version:",
        torch.version.cuda
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DATA_ROOT.mkdir(parents=True, exist_ok=True)
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND DATASET ROOT
# ============================================================

def find_dataset_root(search_root):
    """
    Searches recursively for a directory containing:

        train/
        validate/

    This allows the ZIP file to contain an extra parent folder.
    """

    search_root = Path(search_root)

    if not search_root.exists():
        return None

    # First check the root itself
    if (
        (search_root / "train").is_dir()
        and
        (search_root / "validate").is_dir()
    ):
        return search_root

    # Then search recursively
    for directory in search_root.rglob("*"):

        if not directory.is_dir():
            continue

        train_dir = directory / "train"
        validate_dir = directory / "validate"

        if train_dir.is_dir() and validate_dir.is_dir():
            return directory

    return None


# ============================================================
# EXTRACT DATASET
# ============================================================

def extract_dataset():

    print()
    print("=" * 70)
    print("Checking MRI dataset")
    print("=" * 70)

    # Check whether it is already extracted
    existing_root = find_dataset_root(DATA_ROOT)

    if existing_root is not None:

        print("Dataset already extracted.")
        print("Dataset root:", existing_root)

        return existing_root

    # ZIP must exist
    if not ZIP_PATH.exists():

        print()
        print("ERROR: MRI dataset ZIP was not found.")
        print()
        print("Expected ZIP location:")
        print(ZIP_PATH)
        print()
        print("Place the downloaded file here:")
        print()
        print(
            "Data\\mri\\downloads\\lung-cancer-mri-images.zip"
        )
        print()

        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH}"
        )

    print()
    print("ZIP found:")
    print(ZIP_PATH)

    print()
    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:

        zip_ref.extractall(EXTRACT_ROOT)

    print("Extraction completed.")

    dataset_root = find_dataset_root(EXTRACT_ROOT)

    if dataset_root is None:

        raise RuntimeError(
            "Could not find train/ and validate/ folders "
            "after extracting the ZIP."
        )

    print()
    print("Dataset root:")
    print(dataset_root)

    return dataset_root


# ============================================================
# DATASET
# ============================================================

DATASET_ROOT = extract_dataset()

TRAIN_DIR = DATASET_ROOT / "train"
VALIDATE_DIR = DATASET_ROOT / "validate"

print()
print("Training directory:")
print(TRAIN_DIR)

print()
print("Validation directory:")
print(VALIDATE_DIR)


# ============================================================
# TRANSFORMS
# ============================================================

# Same basic preprocessing used by the source approach:
#
# - Resize to 224 x 224
# - Random horizontal flip for training
# - ImageNet normalization

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validate_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 70)
print("Loading MRI images")
print("=" * 70)

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

validate_dataset = datasets.ImageFolder(
    VALIDATE_DIR,
    transform=validate_transform
)


# ============================================================
# CHECK CLASSES
# ============================================================

print()
print("Training classes:")
print(train_dataset.classes)

print()
print("Training class mapping:")
print(train_dataset.class_to_idx)

print()
print("Validation classes:")
print(validate_dataset.classes)

print()
print("Validation class mapping:")
print(validate_dataset.class_to_idx)


if train_dataset.class_to_idx != validate_dataset.class_to_idx:

    raise RuntimeError(
        "Training and validation class mappings are different."
    )


NUM_CLASSES = len(train_dataset.classes)

print()
print("Number of classes:", NUM_CLASSES)

print("Training images:", len(train_dataset))
print("Validation images:", len(validate_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

validate_loader = DataLoader(
    validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# CONVMIXER COMPONENTS
# ============================================================

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):

        return self.fn(x) + x


class ConvMixerBlock(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size
    ):

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

            nn.Conv2d(
                dim,
                dim,
                kernel_size=1
            ),

            nn.GELU(),

            nn.BatchNorm2d(dim)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# CONVMIXER MODEL
# ============================================================

class ConvMixer(nn.Module):

    def __init__(
        self,
        dim=256,
        depth=8,
        kernel_size=5,
        patch_size=4,
        num_classes=2
    ):

        super().__init__()

        # Patch embedding
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

        # ConvMixer blocks
        self.blocks = nn.Sequential(

            *[
                ConvMixerBlock(
                    dim=dim,
                    kernel_size=kernel_size
                )

                for _ in range(depth)
            ]

        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # Classification head
        self.classifier = nn.Linear(
            dim,
            num_classes
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = self.blocks(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("Creating ConvMixer")
print("=" * 70)

model = ConvMixer(
    dim=256,
    depth=8,
    kernel_size=5,
    patch_size=4,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


print()
print("Model configuration:")
print("Image size :", IMAGE_SIZE)
print("Dimension  : 256")
print("Depth      : 8")
print("Kernel     : 5")
print("Patch size : 4")
print("Classes    :", NUM_CLASSES)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        progress.set_postfix(
            loss=loss.item()
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        progress = tqdm(
            validate_loader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    )

    validation_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("Starting ConvMixer Training")
print("=" * 70)

best_f1 = -1.0
best_accuracy = 0.0


for epoch in range(EPOCHS):

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _
    ) = validate()

    print()
    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Acc  : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation F1   : "
        f"{val_f1:.4f}"
    )

    # Save best model based on macro F1
    if val_f1 > best_f1:

        best_f1 = val_f1
        best_accuracy = val_accuracy

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "classes":
                train_dataset.classes,

            "class_to_idx":
                train_dataset.class_to_idx,

            "image_size":
                IMAGE_SIZE,

            "dim":
                256,

            "depth":
                8,

            "kernel_size":
                5,

            "patch_size":
                4,

            "best_validation_accuracy":
                val_accuracy,

            "best_validation_f1":
                val_f1,

            "epoch":
                epoch + 1
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print()
        print(
            "Best model saved:"
        )

        print(
            MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("Loading Best Model")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("Final MRI Model Evaluation")
print("=" * 70)

(
    final_loss,
    final_accuracy,
    final_f1,
    true_labels,
    predicted_labels
) = validate()


# ============================================================
# METRICS
# ============================================================

precision_f1_report = classification_report(
    true_labels,
    predicted_labels,
    target_names=train_dataset.classes,
    digits=4,
    zero_division=0
)

conf_matrix = confusion_matrix(
    true_labels,
    predicted_labels
)

print()
print("Final Loss:")
print(f"{final_loss:.4f}")

print()
print("Final Accuracy:")
print(f"{final_accuracy:.4f}")

print(
    f"Final Accuracy (%): "
    f"{final_accuracy * 100:.2f}%"
)

print()
print("Macro F1:")
print(f"{final_f1:.4f}")

print()
print("=" * 70)
print("Classification Report")
print("=" * 70)

print(
    precision_f1_report
)

print()
print("=" * 70)
print("Confusion Matrix")
print("=" * 70)

print(
    conf_matrix
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Model:")
print("ConvMixer")

print()
print("Best model:")
print(MODEL_PATH)

print()
print("Classes:")
print(train_dataset.classes)

print()
print(
    f"Validation Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{final_f1:.4f}"
)

print()
print("Device used:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU used:"
    )

    print(
        torch.cuda.get_device_name(0)
    )

print()
print("=" * 70)