from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "Data"
    / "ct_Scan"
    / "extracted"
    / "The IQ-OTHNCCD lung cancer dataset"
)

MODEL_ROOT = PROJECT_ROOT / "models"
PLOT_ROOT = PROJECT_ROOT / "plots"

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_ROOT / "ct_resnet18.pth"
RESULTS_CSV = MODEL_ROOT / "ct_model_results.csv"
REPORT_PATH = MODEL_ROOT / "ct_classification_report.txt"
CONFUSION_MATRIX_PATH = PLOT_ROOT / "ct_confusion_matrix.png"


# Original dataset folder names
CLASS_FOLDERS = [
    "Bengin cases",
    "Malignant cases",
    "Normal cases",
]

# Clean names used by the model
CLASS_NAMES = [
    "benign",
    "malignant",
    "normal",
]

CLASS_MAPPING = {
    "Bengin cases": 0,
    "Malignant cases": 1,
    "Normal cases": 2,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("CT SCAN LUNG CANCER CLASSIFICATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Dataset: {DATA_ROOT}")
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_ROOT}"
    )


for folder in CLASS_FOLDERS:
    folder_path = DATA_ROOT / folder

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Missing class folder:\n{folder_path}"
        )


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

image_paths = []
labels = []

print("Scanning dataset...")

for folder_name in CLASS_FOLDERS:

    folder_path = DATA_ROOT / folder_name

    files = [
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    label = CLASS_MAPPING[folder_name]

    for file_path in files:
        image_paths.append(str(file_path))
        labels.append(label)

    print(
        f"{CLASS_NAMES[label]:10s}: "
        f"{len(files):4d} images"
    )


image_paths = np.array(image_paths)
labels = np.array(labels)

print()
print(f"Total images: {len(image_paths)}")


# ============================================================
# DATASET STATISTICS
# ============================================================

print()
print("Class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):
    count = np.sum(labels == class_id)
    print(f"{class_name:10s}: {count}")


# ============================================================
# STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

indices = np.arange(len(image_paths))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels,
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_indices],
)


train_paths = image_paths[train_indices]
train_labels = labels[train_indices]

val_paths = image_paths[val_indices]
val_labels = labels[val_indices]

test_paths = image_paths[test_indices]
test_labels = labels[test_indices]


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Train:      {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Test:       {len(test_paths)}")
print(f"Total:      {len(image_paths)}")


def print_split_distribution(name, split_labels):

    print(f"\n{name} distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = np.sum(split_labels == class_id)

        print(
            f"  {class_name:10s}: {count}"
        )


print_split_distribution("Train", train_labels)
print_split_distribution("Validation", val_labels)
print_split_distribution("Test", test_labels)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# CUSTOM DATASET
# ============================================================

class CTDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = int(self.labels[index])

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not read image: {image_path}\n{e}"
            )

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = CTDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = CTDataset(
    val_paths,
    val_labels,
    eval_transform
)

test_dataset = CTDataset(
    test_paths,
    test_labels,
    eval_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_labels,
    minlength=len(CLASS_NAMES)
)

total_train = len(train_labels)
num_classes = len(CLASS_NAMES)

class_weights = (
    total_train
    / (num_classes * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

for name, count, weight in zip(
    CLASS_NAMES,
    class_counts,
    class_weights.cpu().numpy()
):
    print(
        f"{name:10s}: "
        f"count={count:4d}, "
        f"weight={weight:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING RESNET18")
print("=" * 70)

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Replace final classifier
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels_batch in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels_batch = labels_batch.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            batch_size = labels_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

    avg_loss = (
        total_loss / total_samples
    )

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

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_val_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0

history = []


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels_batch.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    train_loss = running_loss / total

    train_accuracy = correct / total

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_precision_macro": val_precision,
        "val_recall_macro": val_recall,
        "val_f1_macro": val_f1,
        "learning_rate": current_lr
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f} "
        f"| Val F1: {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "class_mapping": CLASS_MAPPING,
            "image_size": IMAGE_SIZE,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "model_name": "ResNet18",
            "dataset": "IQ-OTHNCCD",
            "split": {
                "train": len(train_paths),
                "validation": len(val_paths),
                "test": len(test_paths)
            },
            "seed": SEED
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Macro F1 = {best_val_f1:.4f})"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            f"Early stopping triggered at epoch {epoch}."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_csv = MODEL_ROOT / "ct_training_history.csv"

history_df.to_csv(
    history_csv,
    index=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

print(
    f"Best epoch: {checkpoint['best_epoch']}"
)

print(
    f"Best validation macro F1: "
    f"{checkpoint['best_val_f1']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    y_true,
    y_pred
) = evaluate(
    model,
    test_loader
)


print()
print(f"Test Loss:              {test_loss:.4f}")
print(f"Test Accuracy:          {test_accuracy:.4f}")
print(f"Test Precision Macro:   {test_precision:.4f}")
print(f"Test Recall Macro:      {test_recall:.4f}")
print(f"Test Macro F1:          {test_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CT Scan ResNet18 Classification Report\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Validation Macro F1: "
        f"{best_val_f1:.4f}\n\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Precision Macro: "
        f"{test_precision:.4f}\n"
    )

    f.write(
        f"Test Recall Macro: "
        f"{test_recall:.4f}\n"
    )

    f.write(
        f"Test Macro F1: {test_f1:.4f}\n\n"
    )

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "CT Scan ResNet18 - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=30
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):

    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results = {
    "Model": "ResNet18",
    "Dataset": "IQ-OTHNCCD",
    "Device": str(DEVICE),
    "Best Epoch": best_epoch,
    "Best Validation Macro F1": best_val_f1,
    "Test Loss": test_loss,
    "Test Accuracy": test_accuracy,
    "Test Precision Macro": test_precision,
    "Test Recall Macro": test_recall,
    "Test Macro F1": test_f1,
    "Train Samples": len(train_paths),
    "Validation Samples": len(val_paths),
    "Test Samples": len(test_paths),
    "Total Samples": len(image_paths),
    "Seed": SEED
}

results_df = pd.DataFrame(
    [results]
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best epoch:              {best_epoch}")
print(f"Validation Macro F1:    {best_val_f1:.4f}")
print(f"Test Accuracy:           {test_accuracy:.4f}")
print(f"Test Precision Macro:    {test_precision:.4f}")
print(f"Test Recall Macro:       {test_recall:.4f}")
print(f"Test Macro F1:           {test_f1:.4f}")

print()
print("Saved files:")

print(f"Model:")
print(MODEL_PATH)

print()
print(f"Training history:")
print(history_csv)

print()
print(f"Classification report:")
print(REPORT_PATH)

print()
print(f"Confusion matrix:")
print(CONFUSION_MATRIX_PATH)

print()
print("Classes:")
print(CLASS_NAMES)

print("=" * 70)