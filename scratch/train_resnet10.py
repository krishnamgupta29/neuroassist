"""
NeuroAssist — Full Training Pipeline
=====================================
1. Reads raw ADNI DICOM subjects from C:\\Users\\krish\\Downloads\\MRI\\MRI
2. Maps each subject to its clinical label from clinical.csv
3. Preprocesses each volume (SimpleITK → 128³ NIfTI)
4. Splits into train/val/test
5. Trains a 3D ResNet-10 for 20 epochs
6. Saves best checkpoint as neuroassist_resnet10.pth
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

# ── Paths ───────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"C:\Users\krish\Downloads\MRI\MRI"
CLINICAL_CSV = os.path.join(os.path.dirname(__file__), "..", "02_Deep_Learning_Models", "clinical.csv")
if not os.path.exists(CLINICAL_CSV):
    CLINICAL_CSV = os.path.join(os.path.dirname(__file__), "..", "backend", "ml", "clinical.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed_volumes")
WEIGHTS_OUT = os.path.join(OUTPUT_DIR, "neuroassist_resnet10.pth")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────────
TARGET_SHAPE = (128, 128, 128)
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
BATCH_SIZE = 2
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("train")

# ═════════════════════════════════════════════════════════════════════════════════
# 1. CLINICAL LABELS
# ═════════════════════════════════════════════════════════════════════════════════
def load_labels():
    labels = {}
    with open(CLINICAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"].strip()
            lbl = row["label"].strip().upper()
            if lbl in LABEL_MAP:
                labels[sid] = LABEL_MAP[lbl]
    log.info("Loaded %d clinical labels from %s", len(labels), CLINICAL_CSV)
    return labels

# ═════════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING (DICOM → 128³ NIfTI)
# ═════════════════════════════════════════════════════════════════════════════════
def preprocess_subject(subject_dir: str, out_path: str) -> bool:
    """Load DICOM series, N4 correct, skull strip, resample to 128³, save NIfTI."""
    try:
        # Find DICOM series recursively
        reader = sitk.ImageSeriesReader()
        series_found = False
        for root, dirs, files in os.walk(subject_dir):
            dicom_names = reader.GetGDCMSeriesFileNames(root)
            if dicom_names:
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                series_found = True
                break
        
        if not series_found:
            # Try loading as single file
            for root, dirs, files in os.walk(subject_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        image = sitk.ReadImage(fp, sitk.sitkFloat32)
                        if image.GetDimension() == 3 and min(image.GetSize()) > 10:
                            series_found = True
                            break
                    except:
                        continue
                if series_found:
                    break

        if not series_found:
            log.warning("No DICOM series found in %s", subject_dir)
            return False

        image = sitk.Cast(image, sitk.sitkFloat32)

        # Ensure valid spacing
        spacing = list(image.GetSpacing())
        spacing = [max(0.5, s) for s in spacing]
        image.SetSpacing(tuple(spacing))

        # N4 Bias Correction (fast)
        try:
            mask = sitk.OtsuThreshold(image, 0, 1, 200)
            sz = image.GetSize()
            sf = [4 if s >= 16 else (2 if s >= 4 else 1) for s in sz]
            down = sitk.Shrink(image, sf)
            dm = sitk.Shrink(mask, sf)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([10, 5])
            corrector.Execute(down, dm)
            log_bias = corrector.GetLogBiasFieldAsImage(image)
            image = sitk.Exp(log_bias) * image
        except Exception:
            pass

        # Curvature flow denoising
        try:
            image = sitk.CurvatureFlow(image, timeStep=0.125, numberOfIterations=2)
        except Exception:
            pass

        # Skull stripping (morphological)
        try:
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(0)
            otsu.SetOutsideValue(1)
            brain_mask = otsu.Execute(image)
            brain_mask = sitk.BinaryFillhole(brain_mask)
            brain_mask = sitk.BinaryErode(brain_mask, [2] * 3)
            labeled = sitk.ConnectedComponent(brain_mask)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(labeled)
            if stats.GetNumberOfLabels() > 0:
                largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
                final_mask = sitk.Equal(labeled, largest)
            else:
                final_mask = brain_mask
            final_mask = sitk.BinaryDilate(final_mask, [2] * 3)
            image = sitk.Mask(image, final_mask)
        except Exception:
            pass

        # Resample to 128³
        in_size = image.GetSize()
        in_spacing = image.GetSpacing()
        target_spacing = [
            max(0.2, (in_size[i] * max(0.1, in_spacing[i])) / TARGET_SHAPE[i])
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(TARGET_SHAPE)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        image = resampler.Execute(image)

        # Intensity normalization
        stats_f = sitk.StatisticsImageFilter()
        stats_f.Execute(image)
        mn, mx = stats_f.GetMinimum(), stats_f.GetMaximum()
        if mx > mn + 1e-6:
            image = (image - mn) / (mx - mn)

        sitk.WriteImage(image, out_path)
        return True

    except Exception as e:
        log.error("Failed to preprocess %s: %s", subject_dir, e)
        return False


# ═════════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═════════════════════════════════════════════════════════════════════════════════
class BrainDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = sitk.ReadImage(self.paths[idx], sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # Normalize
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        tensor = torch.tensor(arr).unsqueeze(0)  # (1, 128, 128, 128)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


# ═════════════════════════════════════════════════════════════════════════════════
# 4. MODEL: 3D ResNet-10 (same arch as backend/ml/medicalnet.py)
# ═════════════════════════════════════════════════════════════════════════════════
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
    def __init__(self, num_classes=3):
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
        return self.fc(x.view(x.size(0), -1))


# ═════════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════════
def train():
    start = time.time()

    # ── Labels ──
    labels = load_labels()

    # ── Preprocessing ──
    log.info("="*60)
    log.info("PHASE 1: Preprocessing DICOM → 128³ NIfTI")
    log.info("="*60)

    paths, ys = [], []
    subjects = sorted(os.listdir(RAW_DATA_DIR))
    for sid in subjects:
        sub_dir = os.path.join(RAW_DATA_DIR, sid)
        if not os.path.isdir(sub_dir):
            continue
        if sid not in labels:
            log.warning("Subject %s not in clinical.csv — skipping", sid)
            continue

        out_nii = os.path.join(PROCESSED_DIR, f"{sid}.nii.gz")
        if os.path.exists(out_nii):
            paths.append(out_nii)
            ys.append(labels[sid])
            continue

        log.info("Preprocessing %s ...", sid)
        ok = preprocess_subject(sub_dir, out_nii)
        if ok:
            paths.append(out_nii)
            ys.append(labels[sid])
        else:
            log.warning("Skipping %s (preprocessing failed)", sid)

    log.info("Total preprocessed volumes: %d", len(paths))
    counts = {v: 0 for v in LABEL_MAP.values()}
    for y in ys:
        counts[y] += 1
    log.info("Class distribution: CN=%d  MCI=%d  AD=%d", counts[0], counts[1], counts[2])

    if len(paths) < 10:
        log.error("Too few samples (%d). Aborting.", len(paths))
        return

    # ── Split ──
    train_p, temp_p, train_y, temp_y = train_test_split(paths, ys, test_size=0.3, stratify=ys, random_state=42)
    val_p, test_p, val_y, test_y = train_test_split(temp_p, temp_y, test_size=0.5, stratify=temp_y, random_state=42)
    log.info("Split: train=%d  val=%d  test=%d", len(train_p), len(val_p), len(test_p))

    # ── Class weights ──
    total = len(train_y)
    n_classes = 3
    class_counts = [train_y.count(i) for i in range(n_classes)]
    weights = torch.tensor([total / (n_classes * max(c, 1)) for c in class_counts], dtype=torch.float32).to(DEVICE)
    log.info("Class weights: %s", weights.cpu().numpy())

    # ── Loaders ──
    train_ds = BrainDataset(train_p, train_y)
    val_ds = BrainDataset(val_p, val_y)
    test_ds = BrainDataset(test_p, test_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ──
    model = ResNet10(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    log.info("="*60)
    log.info("PHASE 2: Training 3D ResNet-10 for %d epochs on %s", NUM_EPOCHS, DEVICE)
    log.info("="*60)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = total_samples = 0

        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == lbls).sum().item()
            total_samples += imgs.size(0)

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss_sum += loss.item() * imgs.size(0)
                _, preds = torch.max(out, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        log.info(
            "Epoch %2d/%d │ Train Loss: %.4f  Acc: %.1f%% │ Val Loss: %.4f  Acc: %.1f%%",
            epoch, NUM_EPOCHS, train_loss, train_acc * 100, val_loss, val_acc * 100
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, WEIGHTS_OUT)
            log.info("  ✓ Saved best checkpoint (val_loss=%.4f, val_acc=%.1f%%)", val_loss, val_acc * 100)

    # ── Test ──
    log.info("="*60)
    log.info("PHASE 3: Evaluating on held-out test set")
    log.info("="*60)

    checkpoint = torch.load(WEIGHTS_OUT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)

    log.info("Balanced Accuracy: %.2f%%", bal_acc * 100)
    log.info("Macro F1 Score:    %.2f%%", macro_f1 * 100)
    log.info("AUC (Macro OvR):   %.2f%%", auc * 100)
    log.info("Confusion Matrix:\n%s", cm)

    elapsed = time.time() - start
    log.info("Total training time: %.1f minutes", elapsed / 60)
    log.info("Weights saved to: %s", WEIGHTS_OUT)

    # Write summary report
    report_path = os.path.join(OUTPUT_DIR, "reports", "training_report.txt")
    with open(report_path, "w") as f:
        f.write(f"NeuroAssist ResNet-10 Training Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Train/Val/Test: {len(train_p)}/{len(val_p)}/{len(test_p)}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Val Acc: {best_val_acc*100:.1f}%\n")
        f.write(f"Test Balanced Accuracy: {bal_acc*100:.2f}%\n")
        f.write(f"Test Macro F1: {macro_f1*100:.2f}%\n")
        f.write(f"Test AUC: {auc*100:.2f}%\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Training Time: {elapsed/60:.1f} min\n")
        f.write(f"Weights: {WEIGHTS_OUT}\n")

    log.info("Report saved to: %s", report_path)


if __name__ == "__main__":
    train()
