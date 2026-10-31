
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "Data" / "mri" / "lung_cancer_MRI_dataset"

TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "validate"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_convmixer.pth"

CLASS_NAMES = ["cancer", "no_cancer"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TrainingImageDataset(Dataset):
    def __init__(self, root, transform):
        self.classes = CLASS_NAMES
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
        self.transform = transform
        self.samples = []

        for class_name in self.classes:
            class_path = root / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Missing MRI training class folder: {class_path}"
                )
            self.samples.extend(
                (path, self.class_to_idx[class_name])
                for path in sorted(class_path.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

        if not self.samples:
            raise RuntimeError(f"No MRI images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), label


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. IMAGE TRANSFORMS
# --------------------------------------------------

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    "validate": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

train_data = TrainingImageDataset(
    root=TRAIN_PATH,
    transform=data_transforms["train"],
)

val_data = TrainingImageDataset(
    root=VAL_PATH,
    transform=data_transforms["validate"],
)

class_names = train_data.classes

print("\nClasses:", class_names)
print("Class mapping:", train_data.class_to_idx)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

assert train_data.class_to_idx == val_data.class_to_idx, (
    "Training and validation class mappings do not match."
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# --------------------------------------------------
# 4. CONVMIXER BLOCK
# --------------------------------------------------

class ConvMixerBlock(nn.Module):

    def __init__(self, dim, kernel_size):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            groups=dim,
            padding="same"
        )

        self.bn1 = nn.BatchNorm2d(dim)

        self.pwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=1
        )

        self.bn2 = nn.BatchNorm2d(dim)

        self.activation = nn.GELU()

    def forward(self, x):

        residual = x

        out = self.dwconv(x)
        out = self.activation(out)
        out = self.bn1(out)

        out = out + residual

        out = self.pwconv(out)
        out = self.bn2(out)
        out = self.activation(out)

        return out


# --------------------------------------------------
# 5. CONVMIXER MODEL
# --------------------------------------------------

class ConvMixer(nn.Module):

    def __init__(
        self,
        input_shape=(3, 224, 224),
        num_classes=2,
        dim=256,
        kernel_size=5,
        depth=8,
        patch_size=4
    ):

        super(ConvMixer, self).__init__()

        in_channels = input_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(dim, kernel_size)
                for _ in range(depth)
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):

        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# --------------------------------------------------
# 6. INITIALIZE MODEL
# --------------------------------------------------

model = ConvMixer().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

print("\nModel initialized successfully.")


# --------------------------------------------------
# 7. TRAINING FUNCTION
# --------------------------------------------------

def train_loop(model, loader):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    for X, y in progress:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(X)

        loss = loss_fn(predictions, y)

        loss.backward()

        optimizer.step()

        batch_size = y.size(0)

        total_loss += loss.item() * batch_size

        predicted_classes = predictions.argmax(dim=1)

        correct += (predicted_classes == y).sum().item()
        total += batch_size

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# --------------------------------------------------
# 8. VALIDATION FUNCTION
# --------------------------------------------------

def validate_loop(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for X, y in tqdm(loader, desc="Validation"):

            X = X.to(device)
            y = y.to(device)

            predictions = model(X)

            loss = loss_fn(predictions, y)

            batch_size = y.size(0)

            total_loss += loss.item() * batch_size

            predicted_classes = predictions.argmax(dim=1)

            correct += (predicted_classes == y).sum().item()
            total += batch_size

            all_predictions.extend(
                predicted_classes.cpu().numpy()
            )

            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return (
        avg_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# --------------------------------------------------
# 9. TRAINING FOR 10 EPOCHS
# --------------------------------------------------

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_accuracy = -1.0

for epoch in range(EPOCHS):

    print(f"\n{'=' * 50}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'=' * 50}")

    train_loss, train_accuracy = train_loop(
        model,
        train_loader
    )

    val_loss, val_accuracy, _, _ = validate_loop(
        model,
        val_loader
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_data.class_to_idx,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            },
            MODEL_PATH
        )

        print("Best model saved:", MODEL_PATH)


# --------------------------------------------------
# 10. LOAD BEST MODEL
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nBest epoch:", checkpoint["epoch"])
print("Best validation accuracy:", checkpoint["val_accuracy"])


# --------------------------------------------------
# 11. FINAL VALIDATION METRICS
# --------------------------------------------------

val_loss, val_accuracy, labels, predictions = validate_loop(
    model,
    val_loader
)

print("\nFinal Validation Results")
print("------------------------")

print("Loss:", round(val_loss, 4))
print("Accuracy:", round(val_accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    labels,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")

plt.title("ConvMixer Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.show()


# --------------------------------------------------
# 12. TRAINING CURVES
# --------------------------------------------------

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_curves.png"
)

plt.show()

print("\nTraining completed.")
print("Model saved at:", MODEL_PATH)
