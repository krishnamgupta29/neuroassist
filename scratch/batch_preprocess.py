"""
NeuroAssist — Multi-Core Parallel Preprocessor for ADNI MRI
===========================================================
- Reads all 187 ADNI subject directories from C:\\Users\\krish\\Downloads\\MRI\\MRI
- Finds DICOM series leaf folders (170 slices per volume)
- Runs N4 Bias Correction, Morphological Skull Stripping, Resampling (128x128x128), and Intensity Normalization
- Saves clean .nii.gz files to 02_Deep_Learning_Models/processed_volumes/
"""

import os
import sys
import time
import csv
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DATA_DIR = r"C:\Users\krish\Downloads\MRI\MRI"
PROCESSED_DIR = r"d:\neuroassist\02_Deep_Learning_Models\processed_volumes"
TARGET_SHAPE = (128, 128, 128)

def preprocess_one(sid):
    out_file = os.path.join(PROCESSED_DIR, f"{sid}.nii.gz")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        return sid, True, "Already exists"

    sub_dir = os.path.join(RAW_DATA_DIR, sid)
    if not os.path.isdir(sub_dir):
        return sid, False, "Directory not found"

    # Find leaf DICOM folder
    leaf_dir = None
    for root, _, files in os.walk(sub_dir):
        dcm_files = [f for f in files if f.endswith('.dcm') or f.startswith('ADNI')]
        if dcm_files:
            leaf_dir = root
            break

    if not leaf_dir:
        return sid, False, "No DICOM files found"

    try:
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(leaf_dir)
        if not names:
            return sid, False, "Empty series names"
        reader.SetFileNames(names)
        image = reader.Execute()
        image = sitk.Cast(image, sitk.sitkFloat32)

        # Sanitize spacing
        curr_spacing = list(image.GetSpacing())
        valid_spacing = tuple([max(0.5, float(s)) for s in curr_spacing])
        image.SetSpacing(valid_spacing)

        # 1. N4 Bias Field Correction
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

        # 2. Skull Stripping
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

        # 3. Resample to 128x128x128
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

        # 4. Intensity Normalization
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        min_v, max_v = stats.GetMinimum(), stats.GetMaximum()
        if max_v > min_v + 1e-6:
            image = (image - min_v) / (max_v - min_v)

        # Save NIfTI
        sitk.WriteImage(image, out_file)
        return sid, True, "Success"
    except Exception as e:
        return sid, False, str(e)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    subjects = sorted(os.listdir(RAW_DATA_DIR))
    print(f"=== BATCH PREPROCESSING {len(subjects)} ADNI SUBJECTS ===")
    t0 = time.time()

    success_count = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(preprocess_one, sid): sid for sid in subjects}
        for idx, future in enumerate(as_completed(futures)):
            sid, ok, msg = future.result()
            if ok:
                success_count += 1
            print(f"[{idx+1}/{len(subjects)}] {sid}: {'OK' if ok else 'FAILED'} ({msg})")

    elapsed = time.time() - t0
    print(f"\nPreprocessed {success_count}/{len(subjects)} subjects in {elapsed:.1f} seconds ({elapsed/60:.1f} mins)!")

if __name__ == "__main__":
    main()
