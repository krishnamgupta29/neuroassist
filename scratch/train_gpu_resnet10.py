"""
NeuroAssist — High-Speed GPU Training Pipeline (RTX 2050 Optimized)
===================================================================
- Automatic Mixed Precision (AMP / FP16) for 4GB VRAM compatibility
- SimpleITK Volume Caching (128x128x128)
- 3D ResNet-10 with Inverse Class Frequency Loss
- Saves production weights to backend/ml/neuroassist_resnet10.pth
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
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 20
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainGPU")

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

def preprocess_subject(subject_dir: str, out_path: str) -> bool:
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(subject_dir)
        if dicom_names:
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            # Search subdirectories
            found = False
            for root, _, files in os.walk(subject_dir):
                d_names = reader.GetGDCMSeriesFileNames(root)
                if d_names:
                    reader.SetFileNames(d_names)
                    image = reader.Execute()
                    found = True
                    break
            if not found:
                # Try single file
                for root, _, files in os.walk(subject_dir):
                    for f in files:
                        try:
                            fp = os.path.join(root, f)
                            image = sitk.ReadImage(fp, sitk.sitkFloat32)
                            if image.GetDimension() in [2, 3]:
                                found = True
                                break
                        except:
                            continue
                    if found:
                        break
            if not found:
                return False

        if image.GetDimension() == 2:
            image = sitk.JoinSeries([image] * 128)

        # Sanitize spacing
        curr_spacing = list(image.GetSpacing())
        valid_spacing = tuple([max(0.5, float(s)) for s in curr_spacing])
        image.SetSpacing(valid_spacing)

        # Fast N4 Bias Correction
        try:
            mask_img = sitk.OtsuThreshold(image, 0, 1, 200)
            sz = image.GetSize()
            sf = [4 if s >= 16 else (2 if s >= 4 else 1) for s in sz]
            down_img = sitk.Shrink(image, sf)
            down_mask = sitk.Shrink(mask_img, sf)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([10, 5])
            corrector.Execute(down_img, down_mask)
            log_bias = corrector.GetLogBiasFieldAsImage(image)
            image = sitk.Exp(log_bias) * image
        except:
            pass

        # Skull Stripping
        try:
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(0)
            otsu.SetOutsideValue(1)
            bmask = otsu.Execute(image)
            bmask = sitk.BinaryFillhole(bmask)
            bmask = sitk.BinaryErode(bmask, [2, 2, 2])
            labeled = sitk.ConnectedComponent(bmask)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(labeled)
            if stats.GetNumberOfLabels() > 0:
                largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
                final_mask = sitk.Equal(labeled, largest)
            else:
                final_mask = bmask
            final_mask = sitk.BinaryDilate(final_mask, [2, 2, 2])
            image = sitk.Mask(image, final_mask)
        except:
            pass

        # Resample to 128x128x128
        in_size = image.GetSize()
        in_spacing = image.GetSpacing()
        target_spacing = [
            max(0.2, float((in_size[i] * max(0.1, in_spacing[i])) / TARGET_SHAPE[i]))
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

        # Intensity Normalization (0.0 to 1.0)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        min_v, max_v = stats.GetMinimum(), stats.GetMaximum()
        if max_v > min_v + 1e-6:
            image = (image - min_v) / (max_v - min_v)

        sitk.WriteImage(image, out_path)
        return True
    except Exception as e:
        logger.warning(f"Preprocessing error on {subject_dir}: {e}")
        return False

class MRIDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = sitk.ReadImage(self.paths[idx], sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) # (1, 128, 128, 128)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)

# --- 3D ResNet-10 Architecture ---
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

def main():
    start_total = time.time()
    logger.info("=" * 60)
    logger.info(f"STARTING 3D RESNET-10 ACCELERATED TRAINING ON {DEVICE.type.upper()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    logger.info("=" * 60)

    # 1. Preprocess & Cache Data
    labels = load_labels()
    cached_paths, cached_labels = [], []
    subjects = sorted(os.listdir(RAW_DATA_DIR))
    
    logger.info(f"Scanning {len(subjects)} subjects from ADNI repository...")
    for idx, sid in enumerate(subjects):
        s_dir = os.path.join(RAW_DATA_DIR, sid)
        if not os.path.isdir(s_dir):
            continue
        
        lbl = labels.get(sid.upper())
        if lbl is None:
            continue
            
        out_nii = os.path.join(PROCESSED_DIR, f"{sid}.nii.gz")
        if not os.path.exists(out_nii):
            logger.info(f"[{idx+1}/{len(subjects)}] Preprocessing {sid}...")
            ok = preprocess_subject(s_dir, out_nii)
            if not ok:
                continue
        
        cached_paths.append(out_nii)
        cached_labels.append(lbl)

    logger.info(f"Total Preprocessed Scans Ready: {len(cached_paths)}")
    c_cn = cached_labels.count(0)
    c_mci = cached_labels.count(1)
    c_ad = cached_labels.count(2)
    logger.info(f"Class Breakdown: CN={c_cn}, MCI={c_mci}, AD={c_ad}")

    # 2. Stratified Dataset Split (70% Train, 15% Val, 15% Test)
    train_p, temp_p, train_y, temp_y = train_test_split(cached_paths, cached_labels, test_size=0.30, stratify=cached_labels, random_state=42)
    val_p, test_p, val_y, test_y = train_test_split(temp_p, temp_y, test_size=0.50, stratify=temp_y, random_state=42)

    logger.info(f"Data Splits: Train={len(train_p)} | Validation={len(val_p)} | Test={len(test_p)}")

    # 3. Class Weights for Loss Balance
    class_counts = [train_y.count(0), train_y.count(1), train_y.count(2)]
    weights = torch.tensor([len(train_y) / (3.0 * max(c, 1)) for c in class_counts], dtype=torch.float32).to(DEVICE)
    logger.info(f"Inverse Class Weights: CN={weights[0]:.2f}, MCI={weights[1]:.2f}, AD={weights[2]:.2f}")

    train_ds = MRIDataset(train_p, train_y)
    val_ds = MRIDataset(val_p, val_y)
    test_ds = MRIDataset(test_p, test_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model & Optimizer Setup
    model = ResNet10(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    logger.info("=" * 60)
    logger.info("TRAINING IN PROGRESS (20 EPOCHS)...")
    logger.info("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                
                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS
                loss.backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * GRAD_ACCUM_STEPS * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == targets).sum().item()
                v_total += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc = (v_correct / v_total) * 100.0
        scheduler.step()
        epoch_dur = time.time() - t_epoch_start

        logger.info(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] ({epoch_dur:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, WEIGHTS_OUT)
            logger.info(f"  >>> Best Model Checkpoint Saved to {WEIGHTS_OUT} (Val Acc: {val_acc:.1f}%)")

    # 5. Final Test Set Evaluation
    logger.info("=" * 60)
    logger.info("EVALUATING BEST MODEL ON BLIND TEST SET...")
    logger.info("=" * 60)

    checkpoint = torch.load(WEIGHTS_OUT, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(DEVICE)
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

    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100.0
    except:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"🏆 TEST BALANCED ACCURACY: {bal_acc:.2f}%")
    logger.info(f"🏆 TEST MACRO F1 SCORE:    {macro_f1:.2f}%")
    logger.info(f"🏆 TEST MACRO AUC (OvR):   {auc:.2f}%")
    logger.info(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")
    logger.info(f"Total Time Elapsed: {(time.time() - start_total) / 60:.1f} minutes")

    report_file = os.path.join(REPORTS_DIR, "training_metrics.txt")
    with open(report_file, "w") as f:
        f.write(f"NeuroAssist ResNet-10 Production Training Report\n")
        f.write(f"Device: {DEVICE.type.upper()}\n")
        f.write(f"Test Balanced Accuracy: {bal_acc:.2f}%\n")
        f.write(f"Test Macro F1: {macro_f1:.2f}%\n")
        f.write(f"Test AUC: {auc:.2f}%\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Weights Path: {WEIGHTS_OUT}\n")
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
