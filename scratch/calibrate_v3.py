"""
Recalibrate morphometric features using the EXACT same preprocessing
pipeline as inference.py (run_preprocessing → feature extraction).
This ensures calibration ↔ inference consistency.
"""
import sys, os, glob, csv, pickle, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from ml.inference import run_preprocessing, _extract_morphometric_features
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
print(f"Total scans: {len(scans)}, Labels: {len(gt_labels)}")

label_map = {"CN": 0, "MCI": 1, "AD": 2}
X_all = []
y_all = []
scan_ids = []

t0 = time.time()
for i, scan_path in enumerate(scans):
    basename = os.path.basename(scan_path).replace(".nii.gz", "")
    gt = gt_labels.get(basename.upper(), None)
    if gt is None:
        continue
    
    # Use EXACT same preprocessing as inference.py
    preprocessed_img = run_preprocessing(scan_path)
    vol = sitk.GetArrayFromImage(preprocessed_img).astype(np.float32)
    
    if vol.shape != (128, 128, 128):
        print(f"  SKIP {basename}: shape={vol.shape}")
        continue
    
    mn, mx = vol.min(), vol.max()
    if mx > mn:
        vol = (vol - mn) / (mx - mn)
    
    # Extract EXACT same features as inference.py
    feats = _extract_morphometric_features(vol)
    X_all.append(feats)
    y_all.append(label_map[gt])
    scan_ids.append(basename)
    
    if (i + 1) % 30 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(scans)}] {elapsed:.0f}s elapsed...")

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"\nFeatures: {X_all.shape}, Labels: CN={np.sum(y_all==0)}, MCI={np.sum(y_all==1)}, AD={np.sum(y_all==2)}")

# Check feature discrimination after preprocessing
feat_names = ["mean","std","median","skew","kurt","p5","p25","p75","p95","p99",
              "v01","v03","v05","v07","brain_frac","low_frac","hi_frac",
              "com_x","com_y","com_z","sp_x","sp_y","sp_z",
              "ax1","ax2","ax3","grad_m","grad_s","grad_95","entropy","texture","dynrng"]

print(f"\n{'Feature':<12} | {'CN mean':>8} | {'MCI mean':>8} | {'AD mean':>8} | {'Spread':>7}")
print("=" * 65)
for fi in range(min(len(feat_names), X_all.shape[1])):
    cn_m = np.mean(X_all[y_all==0, fi])
    mci_m = np.mean(X_all[y_all==1, fi])
    ad_m = np.mean(X_all[y_all==2, fi])
    spread = max(abs(cn_m-mci_m), abs(cn_m-ad_m), abs(mci_m-ad_m))
    print(f"{feat_names[fi]:<12} | {cn_m:>8.4f} | {mci_m:>8.4f} | {ad_m:>8.4f} | {spread:>7.4f}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test multiple classifiers
classifiers = {
    "LogReg": LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42),
    "RF": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    "SVM": SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42),
}

best_name = None
best_acc = 0
best_clf = None

for name, clf in classifiers.items():
    y_pred = cross_val_predict(clf, X_scaled, y_all, cv=skf)
    acc = balanced_accuracy_score(y_all, y_pred)
    print(f"\n{name}: 5-Fold CV Balanced Accuracy = {acc*100:.1f}%")
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_clf = clf

print(f"\n{'='*60}")
print(f"BEST: {best_name} with {best_acc*100:.1f}% balanced accuracy")
print(f"{'='*60}")

# Detailed report for best
y_pred_best = cross_val_predict(best_clf, X_scaled, y_all, cv=skf)
print(classification_report(y_all, y_pred_best, target_names=["CN", "MCI", "AD"]))

# Train final model on ALL data
best_clf.fit(X_scaled, y_all)

# Save
out_dir = r"D:\neuroassist\backend\ml"
pickle.dump(best_clf, open(os.path.join(out_dir, "calibrated_classifier.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(out_dir, "calibrated_scaler.pkl"), "wb"))
print(f"Saved {best_name} to {out_dir}")

# Sanity check: predict on first 8 scans
print(f"\n=== Sanity Check ===")
probs = best_clf.predict_proba(X_scaled[:8])
for i in range(8):
    gt_name = ["CN", "MCI", "AD"][y_all[i]]
    pred_name = ["CN", "MCI", "AD"][np.argmax(probs[i])]
    match = "Y" if pred_name == gt_name else "N"
    print(f"  {scan_ids[i]}: GT={gt_name} Pred={pred_name} CN={probs[i][0]:.1%} MCI={probs[i][1]:.1%} AD={probs[i][2]:.1%} {match}")
