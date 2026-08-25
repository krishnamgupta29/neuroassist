import sys, os
sys.path.insert(0, os.path.abspath('backend'))
from ml.inference import run_inference

scans = [
    ("002_S_0413", "CN"),
    ("002_S_0816", "AD"),
    ("002_S_0954", "MCI"),
    ("005_S_0324", "MCI"),
    ("002_S_1261", "CN"),
    ("002_S_0729", "MCI"),
    ("002_S_1018", "AD"),
    ("002_S_1280", "CN"),
]

print(f"{'Scan':<14} {'GT':>4} | {'Pred':>5} | {'CN':>7} {'MCI':>7} {'AD':>7} | {'Risk':>6} | Match")
print("=" * 75)

correct = 0
for sid, gt in scans:
    path = os.path.join("02_Deep_Learning_Models", "processed_volumes", f"{sid}.nii.gz")
    r = run_inference(path)
    match = "Y" if r['prediction'] == gt else "N"
    if r['prediction'] == gt:
        correct += 1
    print(f"{sid:<14} {gt:>4} | {r['prediction']:>5} | {r['confidence_cn']:>6.1%} {r['confidence_mci']:>6.1%} {r['confidence_ad']:>6.1%} | {r['risk_score']:>5.1f} | {match}")

print(f"\nAccuracy: {correct}/{len(scans)} ({correct/len(scans)*100:.0f}%)")
