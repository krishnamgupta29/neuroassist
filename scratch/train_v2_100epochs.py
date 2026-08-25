"""
NeuroAssist — Production 3D ResNet-10 Training v2 (100+ Epochs)
================================================================
Key Improvements over v1:
 1. 100 epochs with Cosine Annealing warm restart
 2. Heavy 3D Data Augmentation (Random Flip, Rotation90, Intensity Jitter, Gaussian Noise, Random Crop+Pad)
 3. Label Smoothing (0.1) to prevent overconfident CN bias
 4. Dropout (0.5) before final FC layer
 5. Gradient clipping to stabilize training
 6. Early stopping with patience=20 on val balanced accuracy
 7. Per-epoch logging of train/val loss + balanced accuracy
 8. Best checkpoint saved by val balanced accuracy (not val loss)
"""

import os
import sys
import csv
import time
import logging
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix

# --- Paths ---
RAW_DATA_DIR = r"C:\Users\krish\Downloads\MRI\MRI"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLINICAL_CSV = os.path.join(BASE_DIR, "backend", "ml", "clinical.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "02_Deep_Learning_Models", "processed_volumes")
WEIGHTS_OUT = os.path.join(BASE_DIR, "backend", "ml", "neuroassist_resnet10.pth")
REPORTS_DIR = os.path.join(BASE_DIR, "02_Deep_Learning_Models", "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Hyperparameters ---
TARGET_SHAPE = (128, 128, 128)
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4          # effective batch = 8
NUM_EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
PATIENCE = 20                 # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainV2")


# ============================================================
#  DATA LOADING
# ============================================================
def load_labels():
    labels = {}
    with open(CLINICAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"].strip().upper()
            lbl = row["label"].strip().upper()
            if lbl in LABEL_MAP:
                labels[sid] = LABEL_MAP[lbl]
                labels[sid.replace("_", "-")] = LABEL_MAP[lbl]
                labels[sid.replace("_", "")] = LABEL_MAP[lbl]
    logger.info(f"Loaded {len(labels)} subject labels from clinical registry.")
    return labels


# ============================================================
#  3D DATA AUGMENTATION  (operates on numpy arrays)
# ============================================================
class Augment3D:
    """Heavy online 3D augmentation for small medical datasets."""

    def __init__(self, p_flip=0.5, p_rot90=0.4, p_noise=0.3, p_intensity=0.5, p_crop=0.3):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_noise = p_noise
        self.p_intensity = p_intensity
        self.p_crop = p_crop

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        # 1. Random flips along each axis
        for axis in range(3):
            if np.random.rand() < self.p_flip:
                vol = np.flip(vol, axis=axis).copy()

        # 2. Random 90-degree rotations in random plane
        if np.random.rand() < self.p_rot90:
            k = int(np.random.choice([1, 2, 3]))
            planes = [(0, 1), (0, 2), (1, 2)]
            plane = planes[int(np.random.randint(len(planes)))]
            vol = np.rot90(vol, k=k, axes=plane).copy()

        # 3. Additive Gaussian noise
        if np.random.rand() < self.p_noise:
            sigma = np.random.uniform(0.01, 0.04)
            vol = vol + np.random.normal(0, sigma, vol.shape).astype(np.float32)

        # 4. Intensity scaling + shift
        if np.random.rand() < self.p_intensity:
            scale = np.random.uniform(0.85, 1.15)
            shift = np.random.uniform(-0.08, 0.08)
            vol = vol * scale + shift

        # 5. Random 3D crop + zero-pad back to original shape
        if np.random.rand() < self.p_crop:
            D, H, W = vol.shape
            crop_frac = np.random.uniform(0.85, 0.95)
            nd, nh, nw = int(D * crop_frac), int(H * crop_frac), int(W * crop_frac)
            sd = np.random.randint(0, D - nd + 1)
            sh = np.random.randint(0, H - nh + 1)
            sw = np.random.randint(0, W - nw + 1)
            cropped = vol[sd:sd+nd, sh:sh+nh, sw:sw+nw]
            padded = np.zeros_like(vol)
            pd = (D - nd) // 2
            ph = (H - nh) // 2
            pw = (W - nw) // 2
            padded[pd:pd+nd, ph:ph+nh, pw:pw+nw] = cropped
            vol = padded

        vol = np.clip(vol, 0.0, 1.0)
        return vol


# ============================================================
#  DATASET
# ============================================================
class MRIDataset(Dataset):
    def __init__(self, file_paths, labels, augment=None):
        self.paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = sitk.ReadImage(self.paths[idx], sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        if self.augment is not None:
            arr = self.augment(arr)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
#  3D RESNET-10  (with Dropout before FC)
# ============================================================
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet10(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
#  MAIN TRAINING LOOP
# ============================================================
def main():
    start_total = time.time()
    logger.info("=" * 70)
    logger.info(f"  NEUROASSIST v2 TRAINING — {NUM_EPOCHS} EPOCHS ON {DEVICE.type.upper()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"  Augmentation: Flip + Rot90 + Noise + Intensity + Crop")
    logger.info(f"  Label Smoothing: {LABEL_SMOOTHING}  Dropout: {DROPOUT}")
    logger.info(f"  Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info("=" * 70)

    # 1. Load labels & collect preprocessed volumes
    labels = load_labels()
    cached_paths, cached_labels = [], []

    # Check processed dir directly first
    if os.path.exists(PROCESSED_DIR):
        for f in sorted(os.listdir(PROCESSED_DIR)):
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                sid = f.replace(".nii.gz", "").replace(".nii", "").strip().upper()
                lbl = labels.get(sid) or labels.get(sid.replace("_", "-")) or labels.get(sid.replace("_", ""))
                if lbl is not None:
                    cached_paths.append(os.path.join(PROCESSED_DIR, f))
                    cached_labels.append(lbl)

    # Fallback to RAW_DATA_DIR if processed_dir had nothing
    if len(cached_paths) == 0 and os.path.exists(RAW_DATA_DIR):
        subjects = sorted(os.listdir(RAW_DATA_DIR))
        for sid in subjects:
            s_dir = os.path.join(RAW_DATA_DIR, sid)
            if not os.path.isdir(s_dir):
                continue
            lbl = labels.get(sid.upper())
            if lbl is None:
                continue
            out_nii = os.path.join(PROCESSED_DIR, f"{sid}.nii.gz")
            if os.path.exists(out_nii):
                cached_paths.append(out_nii)
                cached_labels.append(lbl)

    logger.info(f"Total Preprocessed Scans: {len(cached_paths)}")
    c_cn = cached_labels.count(0)
    c_mci = cached_labels.count(1)
    c_ad = cached_labels.count(2)
    logger.info(f"Class Distribution: CN={c_cn}, MCI={c_mci}, AD={c_ad}")

    if len(cached_paths) < 10:
        logger.error("Not enough preprocessed volumes! Run preprocessing first.")
        return

    # 2. Stratified Split (70/15/15)
    train_p, temp_p, train_y, temp_y = train_test_split(
        cached_paths, cached_labels, test_size=0.30, stratify=cached_labels, random_state=42
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=0.50, stratify=temp_y, random_state=42
    )
    logger.info(f"Splits: Train={len(train_p)} Val={len(val_p)} Test={len(test_p)}")

    # 3. Inverse class weights
    class_counts = [train_y.count(0), train_y.count(1), train_y.count(2)]
    weights = torch.tensor(
        [len(train_y) / (3.0 * max(c, 1)) for c in class_counts],
        dtype=torch.float32
    ).to(DEVICE)
    logger.info(f"Loss Weights: CN={weights[0]:.3f} MCI={weights[1]:.3f} AD={weights[2]:.3f}")

    # 4. Datasets & Loaders
    augmenter = Augment3D()
    train_ds = MRIDataset(train_p, train_y, augment=augmenter)
    val_ds = MRIDataset(val_p, val_y, augment=None)
    test_ds = MRIDataset(test_p, test_y, augment=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # 5. Model, Loss, Optimizer, Scheduler
    model = ResNet10(num_classes=3, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_bal_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_bal_acc": [], "lr": []}

    logger.info("=" * 70)
    logger.info(f"  TRAINING STARTED — {NUM_EPOCHS} EPOCHS")
    logger.info("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()

                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS
                loss.backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * GRAD_ACCUM_STEPS * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)

            cur_step_loss = running_loss / total
            cur_step_acc = correct / total * 100.0
            step_pct = (step + 1) / len(train_loader) * 100.0
            sys.stdout.write(
                f"\r>> Epoch [{epoch:2d}/{NUM_EPOCHS}] "
                f"Batch [{step+1:2d}/{len(train_loader)}] ({step_pct:3.0f}%) | "
                f"Loss: {cur_step_loss:.4f} | Acc: {cur_step_acc:5.1f}%"
            )
            sys.stdout.flush()

        train_loss = running_loss / total
        train_acc = correct / total * 100.0

        # --- Validation ---
        model.eval()
        v_loss, v_preds_all, v_labels_all = 0.0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                v_preds_all.extend(preds.cpu().numpy())
                v_labels_all.extend(targets.cpu().numpy())

        val_loss = v_loss / len(v_labels_all)
        val_acc = sum(1 for p, l in zip(v_preds_all, v_labels_all) if p == l) / len(v_labels_all) * 100
        val_bal_acc = balanced_accuracy_score(v_labels_all, v_preds_all) * 100

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["lr"].append(current_lr)

        improved = ""
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_bal_acc': val_bal_acc,
                'val_loss': val_loss,
            }, WEIGHTS_OUT)
            improved = f" ★ BEST ({val_bal_acc:.1f}%) → Saved!"
        else:
            epochs_no_improve += 1

        sys.stdout.write("\n")
        sys.stdout.flush()

        logger.info(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}] ({elapsed:.1f}s) "
            f"TrLoss={train_loss:.4f} TrAcc={train_acc:.1f}% | "
            f"VLoss={val_loss:.4f} VAcc={val_acc:.1f}% VBalAcc={val_bal_acc:.1f}% "
            f"LR={current_lr:.2e}{improved}"
        )

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping triggered — no improvement for {PATIENCE} epochs.")
            break

    # 6. Final Test Evaluation on best checkpoint
    logger.info("=" * 70)
    logger.info("  EVALUATING BEST MODEL ON BLIND TEST SET")
    logger.info("=" * 70)

    ckpt = torch.load(WEIGHTS_OUT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class sensitivity
    sens = []
    class_names = ['CN', 'MCI', 'AD']
    for i in range(3):
        row_sum = cm[i].sum()
        s = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
        sens.append(s)
        logger.info(f"  {class_names[i]} Sensitivity: {s:.1f}% ({cm[i, i]}/{row_sum})")

    total_time = (time.time() - start_total) / 60.0
    logger.info(f"★ TEST BALANCED ACCURACY: {bal_acc:.2f}%")
    logger.info(f"★ TEST MACRO F1 SCORE:    {macro_f1:.2f}%")
    logger.info(f"★ TEST MACRO AUC (OvR):   {auc:.2f}%")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Total Training Time: {total_time:.1f} minutes")

    # 7. Save report
    report_file = os.path.join(REPORTS_DIR, "training_metrics.txt")
    with open(report_file, "w") as f:
        f.write(f"NeuroAssist ResNet-10 v2 Production Training Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Device: {DEVICE.type.upper()}\n")
        f.write(f"Epochs: {epoch}/{NUM_EPOCHS}\n")
        f.write(f"Augmentation: Flip + Rot90 + Noise + Intensity + Crop\n")
        f.write(f"Label Smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Best Val Balanced Accuracy: {best_val_bal_acc:.2f}%\n")
        f.write(f"Test Balanced Accuracy: {bal_acc:.2f}%\n")
        f.write(f"Test Macro F1: {macro_f1:.2f}%\n")
        f.write(f"Test AUC: {auc:.2f}%\n")
        f.write(f"{'='*50}\n")
        for i, cn in enumerate(class_names):
            f.write(f"{cn} Sensitivity: {sens[i]:.1f}%\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Weights Path: {WEIGHTS_OUT}\n")
        f.write(f"Training Time: {total_time:.1f} minutes\n")
    logger.info(f"Report saved to {report_file}")

    # 8. Save training curves as CSV
    curves_file = os.path.join(REPORTS_DIR, "training_curves.csv")
    with open(curves_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_bal_acc", "lr"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.2f}",
                f"{history['val_acc'][i]:.2f}",
                f"{history['val_bal_acc'][i]:.2f}",
                f"{history['lr'][i]:.8f}",
            ])
    logger.info(f"Training curves saved to {curves_file}")

    # 9. Plot and save visualization curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_arr = list(range(1, len(history["train_loss"]) + 1))

        # Loss curves
        ax1.plot(epochs_arr, history["train_loss"], label="Train Loss", color="#3b82f6", lw=2)
        ax1.plot(epochs_arr, history["val_loss"], label="Val Loss", color="#ef4444", lw=2, linestyle="--")
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CrossEntropy Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Accuracy curves
        ax2.plot(epochs_arr, history["train_acc"], label="Train Accuracy", color="#10b981", lw=2)
        ax2.plot(epochs_arr, history["val_bal_acc"], label="Val Balanced Accuracy", color="#8b5cf6", lw=2, linestyle="--")
        ax2.set_title("Training & Validation Balanced Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "training_curves_multi.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        logger.info(f"Training curves plot saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
