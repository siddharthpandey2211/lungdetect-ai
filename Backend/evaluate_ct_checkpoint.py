"""Evaluate the existing CT checkpoint on a reconstructed stratified test split."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "ct_Scan"
    / "extracted"
    / "The IQ-OTHNCCD lung cancer dataset"
)
MODEL_PATH = PROJECT_ROOT / "models" / "ct_resnet18.pth"
RESULTS_DIR = PROJECT_ROOT / "models" / "ct_checkpoint_evaluation"

CLASS_FOLDERS = ["Bengin cases", "Malignant cases", "Normal cases"]
CLASS_NAMES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PathImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def resolve_data_dir():
    """Support the dataset's optional extra extraction-folder level."""
    candidate_dirs = [
        DATA_DIR,
        DATA_DIR / "The IQ-OTHNCCD lung cancer dataset",
    ]
    for candidate in candidate_dirs:
        if all((candidate / folder).is_dir() for folder in CLASS_FOLDERS):
            return candidate

    expected = ", ".join(CLASS_FOLDERS)
    raise FileNotFoundError(
        f"Could not find all CT class folders ({expected}) under {DATA_DIR} "
        "or its nested extraction directory."
    )


def collect_samples(data_dir):
    samples = []
    for label, folder in enumerate(CLASS_FOLDERS):
        class_dir = data_dir / folder
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"No CT images found in class folder: {class_dir}")
        samples.extend((path, label) for path in files)

    return samples


def make_test_split(samples):
    labels = np.asarray([label for _, label in samples])
    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=SEED,
        stratify=labels,
    )

    train_val_labels = np.asarray([label for _, label in train_val_samples])
    train_test_split(
        train_val_samples,
        test_size=0.1764705882,
        random_state=SEED,
        stratify=train_val_labels,
    )
    return test_samples


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"CT checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported CT checkpoint format: {MODEL_PATH}"
        )
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError(
            "Checkpoint classes do not match the evaluation class order: "
            f"{checkpoint.get('class_names')}"
        )
    if checkpoint.get("seed") != SEED:
        raise ValueError(
            "Checkpoint seed does not match the reconstructed split seed: "
            f"{checkpoint.get('seed')}"
        )

    data_dir = resolve_data_dir()
    samples = collect_samples(data_dir)
    test_samples = make_test_split(samples)
    image_size = int(checkpoint.get("image_size", 224))

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    test_loader = DataLoader(
        PathImageDataset(test_samples, test_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(DEVICE))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    labels = list(range(len(CLASS_NAMES)))
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    metrics = {
        "evaluation_type": "reconstructed_split_not_verified_historical_test_set",
        "checkpoint": str(MODEL_PATH),
        "dataset_directory": str(data_dir),
        "split_procedure": {
            "description": "Stratified 70/15/15 split reconstructed from the training script",
            "seed": SEED,
            "test_size": 0.15,
            "validation_size_of_train_val": 0.1764705882,
            "test_samples": len(test_samples),
        },
        "checkpoint_metadata": {
            "model_name": checkpoint.get("model_name"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_macro_f1": checkpoint.get("best_val_f1"),
        },
        "class_names": CLASS_NAMES,
        "dataset_samples": len(samples),
        "test_class_counts": {
            class_name: sum(label == index for _, label in test_samples)
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    report_header = (
        "CT checkpoint evaluation — reconstructed split\n"
        "This is not verified to be the historical test split used for training.\n"
        f"Checkpoint: {MODEL_PATH}\n"
        f"Dataset: {data_dir}\n"
        f"Device: {DEVICE}\n\n"
        "Classification report:\n"
    )
    confusion_text = np.array2string(
        np.asarray(metrics["confusion_matrix"]),
        separator=" ",
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        report_header + report_text + "\nConfusion matrix:\n" + confusion_text + "\n",
        encoding="utf-8",
    )

    print(f"Device: {DEVICE}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\n" + report_text)
    print("Confusion matrix:")
    print(confusion_text)
    print(f"\nMetrics saved to: {RESULTS_DIR / 'test_metrics.json'}")
    print(f"Report saved to: {RESULTS_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
