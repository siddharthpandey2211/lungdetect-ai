"""
Improved histopathology training script for LungDetect AI.
Keeps the inference-compatible checkpoint models/lung_resnet18.pth and
class order: lung_aca, lung_n, lung_scc.
Expected dataset:
  Data/histopathology/lung_aca/
  Data/histopathology/lung_n/
  Data/histopathology/lung_scc/
Uploaded prediction images should NOT be stored in these class folders.
"""
from pathlib import Path
import re
import random, json
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
DATA_DIR = ROOT / "Data" / "histopathology"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lung_resnet18.pth"
RESULTS_DIR = ROOT / "training_results" / "histopathology"
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
IMAGE_SIZE, BATCH_SIZE, EPOCHS, PATIENCE = 224, 32, 25, 6
SEED, NUM_WORKERS = 42, 0
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEGACY_UPLOAD_FILENAME = re.compile(r"^[0-9a-f]{32}\.jpg$", re.IGNORECASE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class PathDataset(Dataset):
    def __init__(self, samples, transform): self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,y = self.samples[i]
        with Image.open(p) as im: im = im.convert("RGB")
        return self.transform(im), y

def collect_samples():
    samples=[]
    for label, name in enumerate(CLASS_NAMES):
        folder = DATA_DIR / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}")
        files = sorted(
            p
            for p in folder.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in EXTS
                and not LEGACY_UPLOAD_FILENAME.fullmatch(p.name)
            )
        )
        samples.extend((p,label) for p in files)
        print(f"{name}: {len(files)} images")
    if not samples: raise RuntimeError(f"No training images found in {DATA_DIR}")
    return samples

def loader(samples, tf, shuffle):
    return DataLoader(PathDataset(samples,tf), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=DEVICE.type=="cuda")

def evaluate(model, dl, criterion=None):
    model.eval(); ys=[]; ps=[]; loss_sum=0.; n=0
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x)
            if criterion: loss_sum += criterion(logits,y).item()*y.size(0)
            ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist(); n += y.size(0)
    return {"loss":loss_sum/max(n,1) if criterion else None,"accuracy":accuracy_score(ys,ps),
            "macro_f1":f1_score(ys,ps,average="macro",zero_division=0),"y_true":ys,"y_pred":ps}

def main():
    seed_everything(); MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    samples=collect_samples(); y=np.array([v for _,v in samples])
    train_val,test=train_test_split(samples,test_size=.15,random_state=SEED,stratify=y)
    y_tv=np.array([v for _,v in train_val])
    train,val=train_test_split(train_val,test_size=.1764705882,random_state=SEED,stratify=y_tv)
    train_tf=transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE,scale=(.85,1.0)),
        transforms.RandomHorizontalFlip(.5), transforms.RandomVerticalFlip(.15),
        transforms.RandomRotation(15), transforms.ColorJitter(brightness=.08,contrast=.08,saturation=.05),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    eval_tf=transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    train_dl=loader(train,train_tf,True); val_dl=loader(val,eval_tf,False); test_dl=loader(test,eval_tf,False)
    weights=compute_class_weight(class_weight="balanced",classes=np.arange(3),y=np.array([v for _,v in train]))
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=DEVICE))
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Fine-tune the last residual stage and classification head; keep earlier generic features frozen.
    for p in model.parameters(): p.requires_grad=False
    for p in model.layer4.parameters(): p.requires_grad=True
    model.fc=nn.Linear(model.fc.in_features,3)
    for p in model.fc.parameters(): p.requires_grad=True
    model=model.to(DEVICE)
    optimizer=torch.optim.AdamW([
        {"params":model.layer4.parameters(),"lr":1e-4},
        {"params":model.fc.parameters(),"lr":5e-4}
    ],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2)
    best_f1=-1.; best_epoch=0; stale=0; history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0.; seen=0
        for x,yb in train_dl:
            x,yb=x.to(DEVICE),yb.to(DEVICE); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            total+=loss.item()*yb.size(0); seen+=yb.size(0)
        vm=evaluate(model,val_dl,criterion); scheduler.step(vm["macro_f1"])
        row={"epoch":epoch,"train_loss":total/max(seen,1),"val_loss":vm["loss"],
             "val_accuracy":vm["accuracy"],"val_macro_f1":vm["macro_f1"]}
        history.append(row); print(row)
        if vm["macro_f1"]>best_f1:
            best_f1,best_epoch,stale=vm["macro_f1"],epoch,0
            torch.save({"model_state_dict":model.state_dict(),"class_names":CLASS_NAMES,
                        "class_to_idx":{name:i for i,name in enumerate(CLASS_NAMES)},
                        "image_size":IMAGE_SIZE,"model_name":"ResNet18",
                        "best_val_f1":best_f1,"best_epoch":best_epoch,"dataset":"LC25000",
                        "split":"70/15/15 stratified","seed":SEED},MODEL_PATH)
        else:
            stale+=1
            if stale>=PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}."); break
    (RESULTS_DIR/"history.json").write_text(json.dumps(history,indent=2))
    ckpt=torch.load(MODEL_PATH,map_location=DEVICE,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    tm=evaluate(model,test_dl)
    report=classification_report(tm["y_true"],tm["y_pred"],labels=[0,1,2],
                                 target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    summary={"test_accuracy":tm["accuracy"],"test_macro_f1":tm["macro_f1"],
             "best_epoch":best_epoch,"best_val_macro_f1":best_f1,
             "classification_report":report,
             "confusion_matrix":confusion_matrix(tm["y_true"],tm["y_pred"],labels=[0,1,2]).tolist(),
             "class_names":CLASS_NAMES}
    (RESULTS_DIR/"test_metrics.json").write_text(json.dumps(summary,indent=2))
    print("Test accuracy:",tm["accuracy"],"Test macro F1:",tm["macro_f1"])
    print(classification_report(tm["y_true"],tm["y_pred"],target_names=CLASS_NAMES,zero_division=0))
    print("Saved compatible checkpoint:",MODEL_PATH)

if __name__=="__main__": main()
