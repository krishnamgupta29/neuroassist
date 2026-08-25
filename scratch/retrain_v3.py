"""
Retrain a much stronger classifier using GradientBoosting + enhanced anatomical features.
Target: Replace the weak LogisticRegression (50% acc, near-random entropy) with a 
GradientBoosting model that produces confident, diverse predictions.
"""
import sys, os, glob, pickle, csv
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.abspath('backend'))

VOLUMES_DIR = '02_Deep_Learning_Models/processed_volumes'
CLINICAL_CSV = 'backend/ml/clinical.csv'
OUTPUT_DIR = 'backend/ml'


def extract_enhanced_features(vol):
    feats = []
    brain_mask = vol > 0.05
    brain_voxels = vol[brain_mask]
    if len(brain_voxels) < 100:
        return np.zeros(48)
    
    # 1-5: Global intensity
    feats.append(float(np.mean(brain_voxels)))
    feats.append(float(np.std(brain_voxels)))
    feats.append(float(np.median(brain_voxels)))
    mu = np.mean(brain_voxels)
    sigma = max(np.std(brain_voxels), 1e-6)
    feats.append(float(np.mean(((brain_voxels - mu)/sigma)**3)))
    feats.append(float(np.mean(((brain_voxels - mu)/sigma)**4)))
    
    # 6-10: Percentiles
    for p in [5, 25, 75, 95, 99]:
        feats.append(float(np.percentile(brain_voxels, p)))
    
    # 11-14: Volume fractions
    total_vox = vol.size
    for thresh in [0.1, 0.3, 0.5, 0.7]:
        feats.append(float(np.sum(vol > thresh)) / total_vox)
    
    # 15: Brain fraction
    feats.append(float(np.sum(brain_mask)) / total_vox)
    
    # 16-17: Low/high tissue
    low = float(np.sum((vol > 0.05) & (vol < 0.25))) / max(float(np.sum(brain_mask)), 1)
    high = float(np.sum(vol > 0.60)) / max(float(np.sum(brain_mask)), 1)
    feats.append(low)
    feats.append(high)
    
    # 18-20: Center of mass
    coords = np.array(np.where(brain_mask))
    com = coords.mean(axis=1) / 128.0
    feats.extend([float(com[0]), float(com[1]), float(com[2])])
    
    # 21-23: Spatial spread
    spread = coords.std(axis=1) / 128.0
    feats.extend([float(spread[0]), float(spread[1]), float(spread[2])])
    
    # 24-26: Axial thirds
    third = 128 // 3
    for s in range(3):
        slab = vol[s*third:(s+1)*third]
        sb = slab[slab > 0.05]
        feats.append(float(np.mean(sb)) if len(sb) > 10 else 0.0)
    
    # 27-29: Gradient
    gx = np.diff(vol, axis=0)
    gy = np.diff(vol, axis=1)
    gz = np.diff(vol, axis=2)
    grad_mag = np.sqrt(gx[:127,:127,:127]**2 + gy[:127,:127,:127]**2 + gz[:127,:127,:127]**2)
    feats.append(float(np.mean(grad_mag)))
    feats.append(float(np.std(grad_mag)))
    feats.append(float(np.percentile(grad_mag, 95)))
    
    # 30: Entropy
    hist, _ = np.histogram(brain_voxels, bins=50, range=(0, 1))
    hist = hist / max(hist.sum(), 1)
    hist = hist[hist > 0]
    feats.append(float(-np.sum(hist * np.log2(hist))))
    
    # 31: Local texture variance
    local_mean = uniform_filter(vol, size=5)
    local_sq_mean = uniform_filter(vol**2, size=5)
    local_var = local_sq_mean - local_mean**2
    feats.append(float(np.mean(local_var[brain_mask])))
    
    # 32: Dynamic range
    feats.append(float(np.max(brain_voxels) - np.min(brain_voxels)))
    
    # === NEW 16 anatomical features (33-48) ===
    cx, cy, cz = 64, 64, 64
    
    # 33-34: Hippocampal ROI
    lh = vol[max(0,cx-20):cx-4, cy-14:cy+8, cz-10:cz+10]
    rh = vol[cx+4:min(128,cx+20), cy-14:cy+8, cz-10:cz+10]
    lhb = lh[lh > 0.05]
    rhb = rh[rh > 0.05]
    feats.append(float(np.mean(lhb)) if len(lhb) > 10 else 0.0)
    feats.append(float(np.mean(rhb)) if len(rhb) > 10 else 0.0)
    
    # 35: Hippocampal asymmetry
    lm = float(np.mean(lhb)) if len(lhb) > 10 else 0.0
    rm = float(np.mean(rhb)) if len(rhb) > 10 else 0.0
    feats.append(abs(lm - rm) / max(lm + rm, 1e-6))
    
    # 36-37: Ventricle core
    vc = vol[cx-12:cx+12, cy-10:cy+10, cz-14:cz+14]
    vv = vc[vc > 0.01]
    feats.append(float(np.mean(vv < 0.25)) if len(vv) > 10 else 0.0)
    feats.append(float(np.std(vv)) if len(vv) > 10 else 0.0)
    
    # 38: Ventricle-to-brain ratio
    csf_count = float(np.sum((vol > 0.01) & (vol < 0.25)))
    brain_count = max(float(np.sum(brain_mask)), 1)
    feats.append(csf_count / brain_count)
    
    # 39-40: Cortical ribbon
    X, Y, Z = np.mgrid[0:128, 0:128, 0:128]
    outer = ((X-cx)**2/48**2 + (Y-cy)**2/42**2 + (Z-cz)**2/52**2) <= 1.0
    inner = ((X-cx)**2/40**2 + (Y-cy)**2/35**2 + (Z-cz)**2/44**2) <= 1.0
    shell = outer & ~inner
    cv = vol[shell & brain_mask]
    feats.append(float(np.mean(cv)) if len(cv) > 10 else 0.0)
    feats.append(float(np.std(cv)) if len(cv) > 10 else 0.0)
    
    # 41: Gray/White matter ratio
    gm = float(np.sum((vol > 0.30) & (vol < 0.55) & brain_mask))
    wm = max(float(np.sum((vol > 0.55) & brain_mask)), 1)
    feats.append(gm / wm)
    
    # 42-43: Superior vs inferior
    sup = vol[:64][vol[:64] > 0.05]
    inf = vol[64:][vol[64:] > 0.05]
    feats.append(float(np.mean(sup)) if len(sup) > 10 else 0.0)
    feats.append(float(np.mean(inf)) if len(inf) > 10 else 0.0)
    
    # 44: Left-right asymmetry
    lb = vol[:, :64, :][vol[:, :64, :] > 0.05]
    rb = vol[:, 64:, :][vol[:, 64:, :] > 0.05]
    lm2 = float(np.mean(lb)) if len(lb) > 10 else 0.0
    rm2 = float(np.mean(rb)) if len(rb) > 10 else 0.0
    feats.append(abs(lm2 - rm2) / max(lm2 + rm2, 1e-6))
    
    # 45: Temporal lobe
    tl = vol[cx-16:cx+16, :32, cz-16:cz+16]
    tb = tl[tl > 0.05]
    feats.append(float(np.mean(tb)) if len(tb) > 10 else 0.0)
    
    # 46: Frontal lobe
    fl = vol[:40, cy-20:cy+20, 64:]
    fb = fl[fl > 0.05]
    feats.append(float(np.mean(fb)) if len(fb) > 10 else 0.0)
    
    # 47: High-gradient boundary density
    hg = grad_mag > np.percentile(grad_mag, 90)
    feats.append(float(np.sum(hg)) / total_vox)
    
    # 48: Tissue compactness
    if np.sum(brain_mask) > 0:
        bc = np.argwhere(brain_mask)
        bbox = np.prod(bc.max(axis=0) - bc.min(axis=0) + 1)
        feats.append(float(np.sum(brain_mask)) / max(float(bbox), 1))
    else:
        feats.append(0.0)
    
    return np.array(feats[:48])


def main():
    gt_labels = {}
    with open(CLINICAL_CSV, 'r') as f:
        for row in csv.DictReader(f):
            sid = row.get('subject_id', '').strip().upper()
            lbl = row.get('label', '').strip().upper()
            if sid and lbl in ['CN', 'MCI', 'AD']:
                gt_labels[sid] = lbl
    
    print(f"Ground truth: {len(gt_labels)} entries")
    files = sorted(glob.glob(os.path.join(VOLUMES_DIR, '*.nii.gz')))
    print(f"Volume files: {len(files)}")
    
    X_all, y_all, names = [], [], []
    for i, fpath in enumerate(files):
        bn = os.path.basename(fpath).replace('.nii.gz', '')
        label = gt_labels.get(bn.upper())
        if label is None:
            continue
        img = sitk.ReadImage(fpath, sitk.sitkFloat32)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        mn, mx = vol.min(), vol.max()
        if mx > mn:
            vol = (vol - mn) / (mx - mn)
        feats = extract_enhanced_features(vol)
        X_all.append(feats)
        y_all.append(['CN', 'MCI', 'AD'].index(label))
        names.append(bn)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(files)}...")
    
    X = np.array(X_all)
    y = np.array(y_all)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: CN={np.sum(y==0)}, MCI={np.sum(y==1)}, AD={np.sum(y==2)}")
    
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, balanced_accuracy_score
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifiers = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            min_samples_leaf=5, subsample=0.85, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_clf = None, 0, None
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')
        ms = scores.mean()
        print(f"\n{name}: CV bal_acc = {ms:.4f} (+/- {scores.std():.4f})")
        if ms > best_score:
            best_score, best_name, best_clf = ms, name, clf
    
    print(f"\nBest: {best_name} (CV = {best_score:.4f})")
    
    best_clf.fit(X_scaled, y)
    y_pred = best_clf.predict(X_scaled)
    train_acc = balanced_accuracy_score(y, y_pred)
    print(f"Train bal_acc: {train_acc:.4f}")
    print(classification_report(y, y_pred, target_names=['CN', 'MCI', 'AD']))
    
    probs_all = best_clf.predict_proba(X_scaled)
    ent = -np.sum(probs_all * np.log(probs_all + 1e-10), axis=1)
    sp = np.max(probs_all, axis=1) - np.min(probs_all, axis=1)
    print(f"Avg entropy: {ent.mean():.4f} (random={np.log(3):.4f})")
    print(f"Avg spread: {sp.mean()*100:.1f}%")
    print(f"max_prob>60%: {np.sum(np.max(probs_all,axis=1)>0.60)}/{len(probs_all)}")
    print(f"max_prob>50%: {np.sum(np.max(probs_all,axis=1)>0.50)}/{len(probs_all)}")
    
    print(f"\nSample predictions:")
    for i in range(min(15, len(names))):
        p = probs_all[i]
        gt = ['CN','MCI','AD'][y[i]]
        pr = ['CN','MCI','AD'][y_pred[i]]
        print(f"  {names[i]:>14} | {gt:>3} -> {pr:>3} | {p[0]*100:5.1f} {p[1]*100:5.1f} {p[2]*100:5.1f}")
    
    pickle.dump(best_clf, open(os.path.join(OUTPUT_DIR, 'calibrated_classifier.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(OUTPUT_DIR, 'calibrated_scaler.pkl'), 'wb'))
    pickle.dump({'n_features': 48, 'classifier': best_name, 'cv_bal_acc': best_score,
                 'train_bal_acc': train_acc, 'n_samples': len(y)},
                open(os.path.join(OUTPUT_DIR, 'calibrated_meta.pkl'), 'wb'))
    print(f"\nModels saved! DONE!")


if __name__ == '__main__':
    main()
