"""
Calibrate morphometric features against ground truth labels.
Compute actual distributions of hippocampal density, ventricle CSF ratio,
and parenchyma density for CN vs MCI vs AD volumes.
Then train a simple logistic regression classifier on these features.
"""
import sys, os, glob, csv
import numpy as np
import SimpleITK as sitk

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
print(f"Total scans: {len(scans)}, Labels available: {len(gt_labels)}")

# Collect features per class
features_by_class = {"CN": [], "MCI": [], "AD": []}

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
    
    cx, cy, cz = 64, 64, 64
    
    # Feature 1: Ventricle CSF ratio (low-intensity central region)
    vent = vol[cx-14:cx+14, cy-12:cy+12, cz-16:cz+16]
    f_vent = float(np.mean(vent < 0.30))
    
    # Feature 2: Parenchyma density
    brain = vol[vol > 0.15]
    f_paren = float(np.mean(brain)) if len(brain) > 0 else 0.5
    
    # Feature 3: Hippocampal tissue density
    lh = vol[max(0,cx-22):min(128,cx-6), max(0,cy-16):min(128,cy+6), max(0,cz-12):min(128,cz+12)]
    rh = vol[max(0,cx+6):min(128,cx+22), max(0,cy-16):min(128,cy+6), max(0,cz-12):min(128,cz+12)]
    f_hippo = float((np.mean(lh) + np.mean(rh)) / 2.0)
    
    # Feature 4: Brain volume fraction (non-zero voxels)
    f_brain_vol = float(np.mean(vol > 0.10))
    
    # Feature 5: Cortical intensity variance (outer shell variability)
    outer = vol[10:118, 10:118, 10:118]
    f_cortex_var = float(np.std(outer))
    
    # Feature 6: Central-to-peripheral ratio
    center = vol[44:84, 44:84, 44:84]
    periph = vol[10:30, 10:30, 10:30]
    f_cp_ratio = float(np.mean(center)) / max(float(np.mean(periph)), 0.01)
    
    # Feature 7: Intensity histogram skewness
    brain_vals = vol[vol > 0.05].flatten()
    if len(brain_vals) > 100:
        mu = np.mean(brain_vals)
        sigma = max(np.std(brain_vals), 1e-6)
        f_skew = float(np.mean(((brain_vals - mu) / sigma) ** 3))
    else:
        f_skew = 0.0
    
    # Feature 8: High-intensity fraction (potential white matter)
    f_hi = float(np.mean(vol > 0.70))
    
    features_by_class[gt].append([f_vent, f_paren, f_hippo, f_brain_vol, f_cortex_var, f_cp_ratio, f_skew, f_hi])
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(scans)}...")

print(f"\nSamples: CN={len(features_by_class['CN'])}, MCI={len(features_by_class['MCI'])}, AD={len(features_by_class['AD'])}")

feat_names = ["vent_csf", "paren_dens", "hippo_dens", "brain_vol", "cortex_var", "cp_ratio", "skewness", "hi_frac"]

print(f"\n{'Feature':<14} | {'CN mean':>9} {'CN std':>8} | {'MCI mean':>9} {'MCI std':>8} | {'AD mean':>9} {'AD std':>8} | {'Discriminative?'}")
print("=" * 120)

for fi, fname in enumerate(feat_names):
    cn_vals = [f[fi] for f in features_by_class["CN"]]
    mci_vals = [f[fi] for f in features_by_class["MCI"]]
    ad_vals = [f[fi] for f in features_by_class["AD"]]
    
    cn_m, cn_s = np.mean(cn_vals), np.std(cn_vals)
    mci_m, mci_s = np.mean(mci_vals), np.std(mci_vals)
    ad_m, ad_s = np.mean(ad_vals), np.std(ad_vals)
    
    # Check if means differ significantly
    spread = max(abs(cn_m - mci_m), abs(cn_m - ad_m), abs(mci_m - ad_m))
    avg_std = (cn_s + mci_s + ad_s) / 3
    disc = "YES" if spread > avg_std * 0.5 else "no"
    
    print(f"{fname:<14} | {cn_m:>9.4f} {cn_s:>8.4f} | {mci_m:>9.4f} {mci_s:>8.4f} | {ad_m:>9.4f} {ad_s:>8.4f} | {disc}")

# Now train a simple logistic regression on these features
print("\n=== Training Logistic Regression on Morphometric Features ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report
import pickle

X_all = []
y_all = []
label_map = {"CN": 0, "MCI": 1, "AD": 2}
for cls in ["CN", "MCI", "AD"]:
    for feat in features_by_class[cls]:
        X_all.append(feat)
        y_all.append(label_map[cls])

X_all = np.array(X_all)
y_all = np.array(y_all)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Cross-validated evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(
    LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    X_scaled, y_all, cv=skf
)

bal_acc = balanced_accuracy_score(y_all, y_pred_cv)
print(f"\n5-Fold CV Balanced Accuracy: {bal_acc*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_all, y_pred_cv, target_names=["CN", "MCI", "AD"]))

# Train final model on all data
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_scaled, y_all)

# Save the classifier and scaler
out_dir = r"D:\neuroassist\backend\ml"
pickle.dump(clf, open(os.path.join(out_dir, "morpho_classifier.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(out_dir, "morpho_scaler.pkl"), "wb"))
pickle.dump(feat_names, open(os.path.join(out_dir, "morpho_features.pkl"), "wb"))

print(f"\nSaved classifier to {out_dir}/morpho_classifier.pkl")
print(f"Saved scaler to {out_dir}/morpho_scaler.pkl")
print("\nLogistic Regression coefficients:")
for i, cls in enumerate(["CN", "MCI", "AD"]):
    print(f"  {cls}: {dict(zip(feat_names, clf.coef_[i].round(3)))}")
print(f"  Intercepts: {clf.intercept_.round(3)}")
