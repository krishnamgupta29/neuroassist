"""
Train LogisticRegression on direct-read volumes (no preprocessing).
LogReg gives diverse predictions unlike SVM which collapses to majority class.
"""
import sys, os, glob, csv, pickle
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from ml.inference import _extract_morphometric_features

scan_dir = r"D:\neuroassist\02_Deep_Learning_Models\processed_volumes"
csv_path = r"D:\neuroassist\backend\ml\clinical.csv"

gt = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        sid = row.get("subject_id","").strip().upper()
        lbl = row.get("label","").strip().upper()
        if sid and lbl in ["CN","MCI","AD"]:
            gt[sid] = lbl

scans = sorted(glob.glob(os.path.join(scan_dir, "*.nii.gz")))
lm = {"CN":0, "MCI":1, "AD":2}
X, y, ids = [], [], []

for sp in scans:
    bn = os.path.basename(sp).replace(".nii.gz","")
    if bn.upper() not in gt: continue
    img = sitk.ReadImage(sp, sitk.sitkFloat32)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    if vol.shape != (128,128,128): continue
    mn, mx = vol.min(), vol.max()
    if mx > mn: vol = (vol-mn)/(mx-mn)
    X.append(_extract_morphometric_features(vol))
    y.append(lm[gt[bn.upper()]])
    ids.append(bn)

X, y = np.array(X), np.array(y)
print(f"Samples: {len(X)}, CN={sum(y==0)}, MCI={sum(y==1)}, AD={sum(y==2)}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test multiple C values for LogReg to find most diverse
for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced', random_state=42)
    yp = cross_val_predict(clf, Xs, y, cv=skf)
    acc = balanced_accuracy_score(y, yp)
    # Check diversity on training set
    clf.fit(Xs, y)
    train_pred = clf.predict(Xs[:20])
    unique = len(set(train_pred))
    print(f"  C={C}: CV={acc*100:.1f}%, Train diversity on first 20={unique} classes, preds={list(train_pred[:10])}")

# Use C=0.01 (most regularized = most diverse, less overfit to majority)
best_C = 0.01
clf = LogisticRegression(C=best_C, max_iter=3000, class_weight='balanced', random_state=42)
yp = cross_val_predict(clf, Xs, y, cv=skf)
print(f"\nFinal LogReg C={best_C}: CV Balanced Acc = {balanced_accuracy_score(y, yp)*100:.1f}%")
print(classification_report(y, yp, target_names=["CN","MCI","AD"]))

clf.fit(Xs, y)
out = r"D:\neuroassist\backend\ml"
pickle.dump(clf, open(os.path.join(out, "calibrated_classifier.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(out, "calibrated_scaler.pkl"), "wb"))

# Full diversity check
probs = clf.predict_proba(Xs)
preds = clf.predict(Xs)
print(f"Full dataset prediction distribution: CN={sum(preds==0)}, MCI={sum(preds==1)}, AD={sum(preds==2)}")

print(f"\n=== 15-Scan Diversity Check ===")
for i in range(min(15, len(X))):
    gn = ["CN","MCI","AD"][y[i]]
    pn = ["CN","MCI","AD"][preds[i]]
    m = "Y" if pn==gn else "N"
    print(f"  {ids[i]}: GT={gn} Pred={pn} CN={probs[i][0]:.1%} MCI={probs[i][1]:.1%} AD={probs[i][2]:.1%} {m}")
