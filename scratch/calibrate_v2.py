"""
Extract 512-dim features from CNN penultimate layer + richer morphometric features.
Train a properly calibrated classifier using these combined features.
"""
import sys, os, glob, csv, pickle, time
import numpy as np
import torch
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from ml.medicalnet import get_multiclass_model, weights_loaded

scan_dir = r"D:\neuroassist\02_Deep_Learning_Models\processed_volumes"
csv_path = r"D:\neuroassist\backend\ml\clinical.csv"

# Load ground truth
gt_labels = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        sid = row.get("subject_id", "").strip().upper()
        lbl = row.get("label", "").strip().upper()
        if sid and lbl in ["CN", "MCI", "AD"]:
            gt_labels[sid] = lbl

scans = sorted(glob.glob(os.path.join(scan_dir, "*.nii.gz")))
print(f"Total scans: {len(scans)}, Labels: {len(gt_labels)}")

# Load model and create feature extractor (remove final fc layer)
model = get_multiclass_model()
model.eval()
print(f"Weights loaded: {weights_loaded()}")

# Hook to capture penultimate features
features_cache = {}
def hook_fn(module, input, output):
    features_cache['feat'] = output.detach().cpu().numpy()

# Register hook on avgpool (output is [batch, 512, 1, 1, 1])
hook = model.avgpool.register_forward_hook(hook_fn)

def extract_rich_features(vol):
    """Extract 30+ handcrafted volumetric features that don't depend on exact spatial alignment."""
    feats = []
    
    brain_mask = vol > 0.05
    brain_voxels = vol[brain_mask]
    
    if len(brain_voxels) < 100:
        return np.zeros(32)
    
    # 1-5: Intensity histogram statistics
    feats.append(float(np.mean(brain_voxels)))       # mean intensity
    feats.append(float(np.std(brain_voxels)))         # std
    feats.append(float(np.median(brain_voxels)))      # median
    mu = np.mean(brain_voxels)
    sigma = max(np.std(brain_voxels), 1e-6)
    feats.append(float(np.mean(((brain_voxels - mu)/sigma)**3)))  # skewness
    feats.append(float(np.mean(((brain_voxels - mu)/sigma)**4)))  # kurtosis
    
    # 6-10: Intensity percentiles
    for p in [5, 25, 75, 95, 99]:
        feats.append(float(np.percentile(brain_voxels, p)))
    
    # 11-14: Volume fractions at different thresholds
    total_vox = vol.size
    for thresh in [0.1, 0.3, 0.5, 0.7]:
        feats.append(float(np.sum(vol > thresh)) / total_vox)
    
    # 15: Brain volume fraction
    feats.append(float(np.sum(brain_mask)) / total_vox)
    
    # 16-17: Low vs high intensity ratio (proxy for CSF vs tissue)
    low = float(np.sum((vol > 0.05) & (vol < 0.25))) / max(float(np.sum(brain_mask)), 1)
    high = float(np.sum(vol > 0.60)) / max(float(np.sum(brain_mask)), 1)
    feats.append(low)
    feats.append(high)
    
    # 18-20: Spatial distribution (center of mass shift)
    coords = np.array(np.where(brain_mask))
    com = coords.mean(axis=1) / 128.0  # normalized center of mass
    feats.extend([float(com[0]), float(com[1]), float(com[2])])
    
    # 21-23: Spatial spread (standard deviation of coordinates)
    spread = coords.std(axis=1) / 128.0
    feats.extend([float(spread[0]), float(spread[1]), float(spread[2])])
    
    # 24-26: Intensity in anatomical thirds (axial)
    third = 128 // 3
    for s in range(3):
        slab = vol[s*third:(s+1)*third]
        slab_brain = slab[slab > 0.05]
        feats.append(float(np.mean(slab_brain)) if len(slab_brain) > 10 else 0.0)
    
    # 27-29: Gradient magnitude features (edge density = structural complexity)
    gx = np.diff(vol, axis=0)
    gy = np.diff(vol, axis=1)
    gz = np.diff(vol, axis=2)
    grad_mag = np.sqrt(gx[:127,:127,:127]**2 + gy[:127,:127,:127]**2 + gz[:127,:127,:127]**2)
    feats.append(float(np.mean(grad_mag)))      # avg gradient
    feats.append(float(np.std(grad_mag)))        # gradient variability
    feats.append(float(np.percentile(grad_mag, 95)))  # strong edges
    
    # 30-32: Entropy-based features
    hist, _ = np.histogram(brain_voxels, bins=50, range=(0, 1))
    hist = hist / max(hist.sum(), 1)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    feats.append(float(entropy))
    
    # Texture: local variance
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(vol, size=5)
    local_sq_mean = uniform_filter(vol**2, size=5)
    local_var = local_sq_mean - local_mean**2
    feats.append(float(np.mean(local_var[brain_mask])))
    
    feats.append(float(np.max(brain_voxels) - np.min(brain_voxels)))  # dynamic range
    
    return np.array(feats[:32])


label_map = {"CN": 0, "MCI": 1, "AD": 2}
X_cnn = []
X_morph = []
y_all = []
scan_ids = []

t0 = time.time()
for i, scan_path in enumerate(scans):
    basename = os.path.basename(scan_path).replace(".nii.gz", "")
    gt = gt_labels.get(basename.upper(), None)
    if gt is None:
        continue
    
    img = sitk.ReadImage(scan_path, sitk.sitkFloat32)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    if vol.shape != (128, 128, 128):
        continue
    
    mn, mx = vol.min(), vol.max()
    if mx > mn:
        vol = (vol - mn) / (mx - mn)
    
    # CNN features
    tensor_img = torch.tensor(vol).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _ = model(tensor_img)
    cnn_feat = features_cache['feat'].flatten()  # 512-dim
    X_cnn.append(cnn_feat)
    
    # Rich morphometric features
    morph_feat = extract_rich_features(vol)
    X_morph.append(morph_feat)
    
    y_all.append(label_map[gt])
    scan_ids.append(basename)
    
    if (i + 1) % 30 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(scans)}] {elapsed:.0f}s elapsed...")

hook.remove()

X_cnn = np.array(X_cnn)
X_morph = np.array(X_morph)
y_all = np.array(y_all)

print(f"\nFeature shapes: CNN={X_cnn.shape}, Morph={X_morph.shape}")
print(f"Labels: CN={np.sum(y_all==0)}, MCI={np.sum(y_all==1)}, AD={np.sum(y_all==2)}")

# Combine features
X_combined = np.hstack([X_cnn, X_morph])
print(f"Combined feature dim: {X_combined.shape[1]}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Also scale CNN-only and morph-only for comparison
scaler_cnn = StandardScaler()
X_cnn_scaled = scaler_cnn.fit_transform(X_cnn)

scaler_morph = StandardScaler()
X_morph_scaled = scaler_morph.fit_transform(X_morph)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test 1: CNN features only
y_pred_cnn = cross_val_predict(
    LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs'),
    X_cnn_scaled, y_all, cv=skf
)
print(f"\n=== CNN Features Only (512-dim) ===")
print(f"5-Fold CV Balanced Accuracy: {balanced_accuracy_score(y_all, y_pred_cnn)*100:.1f}%")

# Test 2: Morph features only
y_pred_morph = cross_val_predict(
    LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs'),
    X_morph_scaled, y_all, cv=skf
)
print(f"\n=== Morphometric Features Only (32-dim) ===")
print(f"5-Fold CV Balanced Accuracy: {balanced_accuracy_score(y_all, y_pred_morph)*100:.1f}%")

# Test 3: Combined
y_pred_combo = cross_val_predict(
    LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs'),
    X_scaled, y_all, cv=skf
)
print(f"\n=== Combined CNN + Morphometric ({X_combined.shape[1]}-dim) ===")
print(f"5-Fold CV Balanced Accuracy: {balanced_accuracy_score(y_all, y_pred_combo)*100:.1f}%")
print("\nDetailed Report:")
print(classification_report(y_all, y_pred_combo, target_names=["CN", "MCI", "AD"]))
print("Confusion Matrix:")
print(confusion_matrix(y_all, y_pred_combo))

# Use whichever is best
accuracies = {
    'cnn': balanced_accuracy_score(y_all, y_pred_cnn),
    'morph': balanced_accuracy_score(y_all, y_pred_morph),
    'combo': balanced_accuracy_score(y_all, y_pred_combo),
}
best = max(accuracies, key=accuracies.get)
print(f"\nBest approach: {best} ({accuracies[best]*100:.1f}%)")

# Train final best classifier
if best == 'cnn':
    final_scaler = scaler_cnn
    X_final = X_cnn_scaled
    feat_type = 'cnn'
elif best == 'morph':
    final_scaler = scaler_morph
    X_final = X_morph_scaled
    feat_type = 'morph'
else:
    final_scaler = scaler
    X_final = X_scaled
    feat_type = 'combo'

clf_final = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42, solver='lbfgs')
clf_final.fit(X_final, y_all)

out_dir = r"D:\neuroassist\backend\ml"
pickle.dump(clf_final, open(os.path.join(out_dir, "calibrated_classifier.pkl"), "wb"))
pickle.dump(final_scaler, open(os.path.join(out_dir, "calibrated_scaler.pkl"), "wb"))
pickle.dump({'type': feat_type, 'cnn_dim': X_cnn.shape[1], 'morph_dim': X_morph.shape[1]}, 
            open(os.path.join(out_dir, "calibrated_meta.pkl"), "wb"))

print(f"\nSaved {feat_type} classifier to {out_dir}/calibrated_classifier.pkl")
print(f"Saved scaler to {out_dir}/calibrated_scaler.pkl")

# Quick sanity check on first 5 scans
print(f"\n=== Sanity Check: First 5 predictions ===")
probs_final = clf_final.predict_proba(X_final[:5])
for i in range(5):
    gt_name = ["CN", "MCI", "AD"][y_all[i]]
    pred_name = ["CN", "MCI", "AD"][np.argmax(probs_final[i])]
    print(f"  {scan_ids[i]}: GT={gt_name} Pred={pred_name} P=[CN:{probs_final[i][0]:.3f} MCI:{probs_final[i][1]:.3f} AD:{probs_final[i][2]:.3f}]")
