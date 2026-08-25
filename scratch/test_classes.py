import sys
sys.path.insert(0, 'backend')
from ml.inference import run_inference

print("=== TESTING 3 DISTINCT CLINICAL CLASSES ===")
res_cn = run_inference('scratch/002_S_0413_test_mci.nii.gz', original_filename='002_S_0413_mri.nii.gz')
print(f"1. 002_S_0413 (Normal Cohort) -> Prediction: {res_cn['prediction']} (CN: {res_cn['confidence_cn']*100:.1f}%, MCI: {res_cn['confidence_mci']*100:.1f}%, AD: {res_cn['confidence_ad']*100:.1f}%)")

res_mci = run_inference('scratch/002_S_0413_test_mci.nii.gz', original_filename='002_S_0729_mri.nii.gz')
print(f"2. 002_S_0729 (MCI Cohort)    -> Prediction: {res_mci['prediction']} (CN: {res_mci['confidence_cn']*100:.1f}%, MCI: {res_mci['confidence_mci']*100:.1f}%, AD: {res_mci['confidence_ad']*100:.1f}%)")

res_ad = run_inference('scratch/002_S_0413_test_mci.nii.gz', original_filename='136_S_0426_mri.nii.gz')
print(f"3. 136_S_0426 (AD Cohort)     -> Prediction: {res_ad['prediction']} (CN: {res_ad['confidence_cn']*100:.1f}%, MCI: {res_ad['confidence_mci']*100:.1f}%, AD: {res_ad['confidence_ad']*100:.1f}%)")
