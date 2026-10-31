# ==========================================================
# LungDetect AI
# File: train_image_model.py
# Purpose: Train ResNet18 on LC25000 lung histopathology images
# ==========================================================

import os
from pathlib import Path
import random
import re

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# Configuration
# ==========================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

IMAGE_SIZE = 224

NUM_WORKERS = 0

# ==========================================================
# Reproducibility
# ==========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "Data" / "histopathology"

MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ==========================================================
# Device
# ==========================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("LungDetect AI - Histopathology CNN")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================================
# Check Dataset
# ==========================================================

print("\nChecking dataset...")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )

required_classes = [
    "lung_n",
    "lung_aca",
    "lung_scc"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEGACY_UPLOAD_FILENAME = re.compile(r"^[0-9a-f]{32}\.jpg$", re.IGNORECASE)


class TrainingImageDataset(Dataset):
    def __init__(self, root, class_names, transform):
        self.classes = sorted(class_names)
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
        self.transform = transform
        self.samples = []

        for class_name in self.classes:
            class_path = root / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Missing class folder:\n{class_path}")
            self.samples.extend(
                (path, self.class_to_idx[class_name])
                for path in sorted(class_path.iterdir())
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and not LEGACY_UPLOAD_FILENAME.fullmatch(path.name)
                )
            )

        if not self.samples:
            raise RuntimeError(f"No training images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


for class_name in required_classes:

    class_path = DATASET_DIR / class_name

    if not class_path.exists():
        raise FileNotFoundError(
            f"Missing class folder:\n{class_path}"
        )

    image_count = len([
        x for x in class_path.iterdir()
        if (
            x.is_file()
            and x.suffix.lower() in IMAGE_EXTENSIONS
            and not LEGACY_UPLOAD_FILENAME.fullmatch(x.name)
        )
    ])

    print(f"{class_name}: {image_count} images")


# ==========================================================
# Image Transformations
# ==========================================================

weights = ResNet18_Weights.DEFAULT

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test transformation
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading images...")

full_dataset = TrainingImageDataset(
    DATASET_DIR,
    required_classes,
    train_transform,
)

evaluation_dataset = TrainingImageDataset(
    DATASET_DIR,
    required_classes,
    test_transform,
)

print("\nClasses:")
print(full_dataset.class_to_idx)

print("\nTotal images:", len(full_dataset))


# ==========================================================
# Dataset Split
# ==========================================================

# 70% train
# 15% validation
# 15% test

total_size = len(full_dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

# Validation and test should NOT use augmentation.
# Reuse the same class-scoped sample order with an evaluation transform.
val_dataset = torch.utils.data.Subset(
    evaluation_dataset,
    val_dataset.indices
)

test_dataset = torch.utils.data.Subset(
    evaluation_dataset,
    test_dataset.indices
)

print("\nDataset split:")
print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Testing    :", len(test_dataset))


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ==========================================================
# Load ResNet18
# ==========================================================

print("\nLoading ResNet18...")

model = models.resnet18(weights=weights)

# Freeze the pretrained layers first.
for param in model.parameters():
    param.requires_grad = False


# Replace final classification layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(full_dataset.classes)
)

model = model.to(DEVICE)


# ==========================================================
# Loss and Optimizer
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# Training Function
# ==========================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================================
# Validation Function
# ==========================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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

    loss = running_loss / total

    accuracy = correct / total

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return loss, accuracy, f1


# ==========================================================
# Training
# ==========================================================

print("\n")
print("=" * 60)
print("Starting CNN Training")
print("=" * 60)

best_f1 = -1

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": []
}

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy, val_f1 = validate()

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    history["val_f1"].append(val_f1)

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: {val_f1:.4f}"
    )

    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "class_names": full_dataset.classes,

                "class_to_idx": full_dataset.class_to_idx,

                "image_size": IMAGE_SIZE,

                "model_name": "ResNet18"
            },

            MODELS_DIR / "lung_resnet18.pth"
        )

        print("✓ Best model saved.")


# ==========================================================
# Load Best Model
# ==========================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODELS_DIR / "lung_resnet18.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ==========================================================
# Test Evaluation
# ==========================================================

print("\n")
print("=" * 60)
print("Testing CNN")
print("=" * 60)

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

print("\nTest Results")

print(
    "Accuracy :", round(accuracy, 4)
)

print(
    "Precision:", round(precision, 4)
)

print(
    "Recall   :", round(recall, 4)
)

print(
    "Macro F1 :", round(f1, 4)
)


# ==========================================================
# Classification Report
# ==========================================================

print("\n")
print("=" * 60)
print("Classification Report")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=full_dataset.classes,
        zero_division=0
    )
)


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=full_dataset.classes,
    yticklabels=full_dataset.classes
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "ResNet18 Histopathology Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "histopathology_confusion_matrix.png"
)

plt.close()


# ==========================================================
# Training Curves
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "ResNet18 Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "histopathology_accuracy.png"
)

plt.close()


# ==========================================================
# Save Training History
# ==========================================================

history_df = __import__("pandas").DataFrame(history)

history_df.to_csv(
    MODELS_DIR / "histopathology_training_history.csv",
    index=False
)


# ==========================================================
# Final Information
# ==========================================================

print("\n")
print("=" * 60)
print("CNN Training Completed Successfully")
print("=" * 60)

print(
    "Model:",
    MODELS_DIR / "lung_resnet18.pth"
)

print(
    "Classes:",
    full_dataset.classes
)

print(
    "Best Validation Macro F1:",
    round(best_f1, 4)
)

print(
    "Test Accuracy:",
    round(accuracy, 4)
)

print(
    "Test Macro F1:",
    round(f1, 4)
)

print("\n")
print("Next step:")
print("Create Backend/image_predict.py")
