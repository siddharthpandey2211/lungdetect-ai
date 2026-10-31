"""
Improved CT training script for LungDetect AI.
Compatible with ct_predict.py:
  - models/ct_resnet18.pth
  - checkpoint["model_state_dict"]
  - checkpoint["class_names"] in benign, malignant, normal order
Dataset expected at:
  Data/ct_Scan/extracted/The IQ-OTHNCCD lung cancer dataset/
    Bengin cases/
    Malignant cases/
    Normal cases/
"""
from pathlib import Path
import random, json, copy, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "ct_Scan" / "extracted" / "The IQ-OTHNCCD lung cancer dataset"
NESTED_DATA_DIR = DATA_DIR / "The IQ-OTHNCCD lung cancer dataset"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "ct_resnet18.pth"
RESULTS_DIR = ROOT / "training_results" / "ct"
IMAGE_SIZE, BATCH_SIZE, EPOCHS, PATIENCE = 224, 32, 30, 7
SEED, NUM_WORKERS = 42, 0
CLASS_FOLDERS = ["Bengin cases", "Malignant cases", "Normal cases"]
CLASS_NAMES = ["benign", "malignant", "normal"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PathImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im: image = im.convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

def collect_samples():
    data_dir = next(
        (
            candidate
            for candidate in (DATA_DIR, NESTED_DATA_DIR)
            if all((candidate / folder).is_dir() for folder in CLASS_FOLDERS)
        ),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(
            "CT dataset class folders were not found in either "
            f"{DATA_DIR} or {NESTED_DATA_DIR}"
        )

    samples = []
    for label, folder in enumerate(CLASS_FOLDERS):
        class_dir = data_dir / folder
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )
        samples.extend((p, label) for p in files)
    if not samples: raise RuntimeError(f"No CT images found under {data_dir}")
    counts = {name: sum(y == i for _, y in samples) for i, name in enumerate(CLASS_NAMES)}
    print("Dataset counts:", counts)
    return samples

def make_loader(samples, transform, shuffle):
    ds = PathImageDataset(samples, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")

def evaluate(model, loader, criterion=None):
    model.eval(); ys, preds, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if criterion:
                total_loss += criterion(logits, y).item() * y.size(0)
            ys.extend(y.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist()); n += y.size(0)
    return {"loss": total_loss / max(n, 1) if criterion else None,
            "accuracy": accuracy_score(ys, preds),
            "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
            "y_true": ys, "y_pred": preds}

def main():
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    labels = np.array([y for _, y in samples])
    train_val, test = train_test_split(samples, test_size=0.15, random_state=SEED, stratify=labels)
    train_labels = np.array([y for _, y in train_val])
    train, val = train_test_split(train_val, test_size=0.1764705882, random_state=SEED, stratify=train_labels)
    # Result is approximately 70/15/15, stratified.
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomRotation(7), transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256,256)), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    train_loader = make_loader(train, train_tf, True)
    val_loader = make_loader(val, eval_tf, False)
    test_loader = make_loader(test, eval_tf, False)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)),
                                         y=np.array([y for _, y in train]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    # Fine-tune the full pretrained network at a conservative learning rate.
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if not n.startswith("fc.")], "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); running, seen = 0.0, 0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits,y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()*y.size(0); seen += y.size(0)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["macro_f1"])
        row = {"epoch":epoch,"train_loss":running/max(seen,1),"val_loss":val_metrics["loss"],
               "val_accuracy":val_metrics["accuracy"],"val_macro_f1":val_metrics["macro_f1"],
               "lr_backbone":optimizer.param_groups[0]["lr"],"lr_head":optimizer.param_groups[1]["lr"]}
        history.append(row); print(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            torch.save({"model_state_dict":model.state_dict(),"class_names":CLASS_NAMES,
                        "class_mapping":{f:i for i,f in enumerate(CLASS_FOLDERS)},
                        "image_size":IMAGE_SIZE,"best_val_f1":best_f1,"best_epoch":best_epoch,
                        "model_name":"ResNet18","dataset":"IQ-OTHNCCD","split":"70/15/15 stratified",
                        "seed":SEED}, MODEL_PATH)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}."); break
    (RESULTS_DIR/"history.json").write_text(json.dumps(history, indent=2))
    # Final held-out evaluation uses the best validation-selected checkpoint.
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    result = evaluate(model, test_loader)
    report = classification_report(result["y_true"], result["y_pred"], labels=list(range(3)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=list(range(3))).tolist()
    summary = {"test_accuracy":result["accuracy"],"test_macro_f1":result["macro_f1"],
               "best_epoch":best_epoch,"best_val_macro_f1":best_f1,"classification_report":report,
               "confusion_matrix":cm,"class_names":CLASS_NAMES}
    (RESULTS_DIR/"test_metrics.json").write_text(json.dumps(summary, indent=2))
    print("Test accuracy:", result["accuracy"], "Test macro F1:", result["macro_f1"])
    print(classification_report(result["y_true"], result["y_pred"], target_names=CLASS_NAMES, zero_division=0))
    print("Saved compatible checkpoint:", MODEL_PATH)

if __name__ == "__main__": main()
