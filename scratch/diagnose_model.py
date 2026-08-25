"""
Deep diagnosis of the inference pipeline:
  1. Raw CNN logits (before softmax)
  2. CNN softmax probabilities
  3. Morphometric features from actual voxels
  4. Morphometric priors
  5. Final blended output
Run on 12 different scans to see if the CNN is actually discriminating.
"""
import sys, os, glob
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from ml.medicalnet import get_multiclass_model, weights_loaded
from ml.inference import run_preprocessing
import SimpleITK as sitk

# Load model once
model = get_multiclass_model()
model.eval()
print(f"Weights loaded: {weights_loaded()}")
print(f"Model fc layer: {model.fc}")
print(f"fc.weight shape: {model.fc.weight.shape}")
print(f"fc.bias: {model.fc.bias.data.numpy()}")
print()

# Get scans
scan_dir = r"D:\neuroassist\02_Deep_Learning_Models\processed_volumes"
scans = sorted(glob.glob(os.path.join(scan_dir, "*.nii.gz")))[:12]

# Load clinical labels for ground truth
csv_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'ml', 'clinical.csv')
gt_labels = {}
if os.path.exists(csv_path):
    import csv
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sid = row.get("subject_id", "").strip().upper()
            lbl = row.get("label", "").strip().upper()
            if sid and lbl:
                gt_labels[sid] = lbl

print(f"{'Scan':<18} {'GT':>4} | {'Logit_CN':>9} {'Logit_MCI':>9} {'Logit_AD':>9} | {'CNN_CN':>7} {'CNN_MCI':>7} {'CNN_AD':>7} | {'Morph_CN':>8} {'Morph_MCI':>9} {'Morph_AD':>8} | {'Final_CN':>8} {'Final_MCI':>9} {'Final_AD':>8} | {'Pred':>5}")
print("=" * 175)

for scan_path in scans:
    basename = os.path.basename(scan_path).replace(".nii.gz", "")
    gt = gt_labels.get(basename.upper(), "??")
    
    try:
        # Preprocess
        preprocessed_img = run_preprocessing(scan_path)
        vol_array = sitk.GetArrayFromImage(preprocessed_img).astype(np.float32)
        
        if vol_array.shape != (128, 128, 128):
            print(f"{basename:<18} SHAPE ERROR: {vol_array.shape}")
            continue
        
        min_a, max_a = vol_array.min(), vol_array.max()
        if max_a > min_a:
            vol_array = (vol_array - min_a) / (max_a - min_a)
        
        tensor_img = torch.tensor(vol_array).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logits = model(tensor_img)
            raw_logits = logits.numpy()[0]
            cnn_probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # Morphometric features
        cx, cy, cz = 64, 64, 64
        vent_region = vol_array[cx-14:cx+14, cy-12:cy+12, cz-16:cz+16]
        ventricle_csf_ratio = float(np.mean(vent_region < 0.30))
        
        brain_parenchyma = vol_array[vol_array > 0.15]
        parenchyma_density = float(np.mean(brain_parenchyma)) if len(brain_parenchyma) > 0 else 0.5
        
        left_hippo = vol_array[max(0,cx-22):min(128,cx-6), max(0,cy-16):min(128,cy+6), max(0,cz-12):min(128,cz+12)]
        right_hippo = vol_array[max(0,cx+6):min(128,cx+22), max(0,cy-16):min(128,cy+6), max(0,cz-12):min(128,cz+12)]
        hippo_tissue_mean = float((np.mean(left_hippo) + np.mean(right_hippo)) / 2.0)
        
        hippo_atrophy = float(np.clip(1.0 - (hippo_tissue_mean / 0.70), 0.05, 0.95))
        ventricle_enlargement = float(np.clip(ventricle_csf_ratio * 2.2, 0.05, 0.95))
        cortical_thinning = float(np.clip(1.0 - (parenchyma_density / 0.65), 0.05, 0.95))
        
        morpho_severity = float(ventricle_enlargement * 0.45 + hippo_atrophy * 0.40 + cortical_thinning * 0.15)
        
        morph_cn = float(np.clip(1.2 - (morpho_severity * 2.0), 0.05, 0.95))
        morph_ad = float(np.clip((morpho_severity * 2.0) - 0.4, 0.05, 0.95))
        morph_mci = float(np.clip(1.0 - abs(0.50 - morpho_severity) * 2.2, 0.08, 0.92))
        
        morph_total = morph_cn + morph_mci + morph_ad
        morph_probs = np.array([morph_cn/morph_total, morph_mci/morph_total, morph_ad/morph_total])
        
        # Blended
        combined = (cnn_probs * 0.65) + (morph_probs * 0.35)
        combined = combined / np.sum(combined)
        
        pred = ["CN", "MCI", "AD"][int(np.argmax(combined))]
        match = "Y" if pred == gt else "N"
        
        print(f"{basename:<18} {gt:>4} | {raw_logits[0]:>9.4f} {raw_logits[1]:>9.4f} {raw_logits[2]:>9.4f} | {cnn_probs[0]:>7.3f} {cnn_probs[1]:>7.3f} {cnn_probs[2]:>7.3f} | {morph_probs[0]:>8.3f} {morph_probs[1]:>9.3f} {morph_probs[2]:>8.3f} | {combined[0]:>8.3f} {combined[1]:>9.3f} {combined[2]:>8.3f} | {pred:>4} {match}")
        
    except Exception as e:
        print(f"{basename:<18} ERROR: {e}")

print()
print("=== Analysis Summary ===")
print("If CNN logits are nearly identical across all scans -> model collapsed (not learning discriminative features)")
print("If morph probs are similar -> morphometric thresholds may need calibration against actual anatomy")
