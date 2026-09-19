<div align="center">

# 🧠 NeuroAssist

### AI-Powered Early Detection for Alzheimer's & Neurological Disorders

**Clinical-grade decision support for Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD) classification from T1-weighted structural brain MRI — with full explainable AI.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-neuro--assists.vercel.app-7A1F2B?style=for-the-badge)](https://neuro-assists.vercel.app/login)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange?style=for-the-badge)
![License](https://img.shields.io/badge/Model-IIT%20Ropar%20Approved-blue?style=for-the-badge)

**🔗 Live App:** [https://neuro-assists.vercel.app/login](https://neuro-assists.vercel.app/login)

</div>

---

## 📖 About

NeuroAssist is an enterprise-grade clinical decision support platform that applies 3D Deep Learning to structural brain MRI scans to assist neurologists and radiologists in early screening for Alzheimer's Disease and Mild Cognitive Impairment.

The model — a **MedicalNet 3D ResNet-10** fine-tuned via transfer learning on real ADNI (Alzheimer's Disease Neuroimaging Initiative) data — achieves **87.0% binary accuracy** (CN vs AD) and **72.4% multi-class accuracy** (CN / MCI / AD), with full **Grad-CAM explainability** so clinicians can see exactly which brain regions influenced each prediction.

> **"AI assists. Doctor decides."**
> Every AI prediction requires physician review, sign-off, and the ability to Accept, Flag, or Override — NeuroAssist is a second-opinion tool, not a replacement for clinical judgment.

---

## ⚠️ Important: About the MRI Data

This project's model is trained and validated on **real ADNI (Alzheimer's Disease Neuroimaging Initiative) MRI data**, used under **institutional research approval (IIT Ropar)**.

**We do not provide, distribute, or attach any MRI scan files in this repository or demo**, because:
- Brain MRI scans are **sensitive medical/biometric data** and fall under patient privacy regulations (HIPAA-equivalent research ethics).
- ADNI data is only accessible under a **signed Data Use Agreement** directly with [adni.loni.usc.edu](https://adni.loni.usc.edu/), not for public redistribution.

If you'd like to test the pipeline yourself, you will need to:
1. Apply for ADNI access at [adni.loni.usc.edu](https://adni.loni.usc.edu/)
2. Use your own institutionally-approved, de-identified T1-weighted MRI data
3. Run it through our preprocessing pipeline before inference

No real patient scans are stored, cached, or shared by this application beyond the logged-in clinician's own session.

---

## 🖥️ Live Demo Workflow

Try it here → **[https://neuro-assists.vercel.app/login](https://neuro-assists.vercel.app/login)**

```
1. Sign In
   → Log in with clinician/doctor credentials (JWT-authenticated)

2. Clinical Overview (Dashboard)
   → View cohort-wide stats: enrolled patients, pending reviews,
     cognitive distribution (CN/MCI/AD), model performance metrics

3. Upload & Pipeline
   → Upload a T1-weighted MRI (.nii / .nii.gz / DICOM)
   → Assign it to a patient record
   → Scan runs through the 7-stage preprocessing pipeline:
     N4 Bias Correction → Denoising → Skull Stripping →
     MNI152 Registration → Intensity Normalization → Resampling → 128³ tensor

4. AI Inference
   → MedicalNet 3D ResNet-10 classifies the scan: CN / MCI / AD
   → Confidence scores + volumetric risk score generated

5. Explainable AI Review (Grad-CAM)
   → View Axial / Coronal / Sagittal heatmap overlays
   → See exactly which regions (Hippocampus, Entorhinal Cortex, etc.)
     drove the prediction
   → Review volumetric biomarkers (Hippocampal Volume, Ventricle
     Enlargement, Cortical Thinning)

6. Doctor Decision Panel
   → Physician reviews AI output and clinical notes
   → Accept / Flag for Review / Override with manual diagnosis
   → Generate signed clinical PDF report

7. Patient Registry & Longitudinal Tracking
   → View patient history across multiple scans over time
   → Track risk score trends, MMSE scores, and disease progression
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **3D Deep Learning Classification** | MedicalNet 3D ResNet-10, transfer-learned on real ADNI data |
| 🔍 **Explainable AI (Grad-CAM)** | 3D gradient-weighted class activation maps, sliced across Axial/Coronal/Sagittal planes |
| 🩺 **Doctor-in-the-Loop** | Every AI result requires clinician sign-off — Accept / Flag / Override |
| 📊 **Volumetric Biomarkers** | Hippocampal volume, ventricle enlargement, entorhinal cortical thinning |
| 📈 **Longitudinal Tracking** | Multi-scan patient history with risk-score trend charts |
| 🔐 **Role-Based Access** | Doctor, Patient, and Admin workflows with JWT authentication |
| 📄 **Clinical PDF Reports** | Auto-generated, physician-signed diagnostic reports |
| 🛡️ **Privacy-First Design** | No real patient scans distributed; de-identification enforced pre-upload |

---

## 🏗️ Tech Stack

**Frontend**
- React (Vite) + TailwindCSS
- Recharts (data visualization)
- Three.js / React Three Fiber (3D brain visualization)

**Backend**
- Python (FastAPI)
- SQLAlchemy ORM (PostgreSQL / SQLite) or MongoDB
- JWT-based authentication

**AI / ML Pipeline**
- **Preprocessing:** SimpleITK, NiBabel, ANTsPyX
  - N4 Bias Field Correction, Skull Stripping, MNI152 Registration, Intensity Normalization, Resampling
- **Model:** PyTorch — MedicalNet 3D ResNet-10 (transfer learning, pretrained on 23 medical imaging datasets)
- **Explainability:** Custom 3D Grad-CAM engine (hooks into final conv block, trilinear upsampling, JET colormap overlay)

**Deployment**
- Frontend → Vercel
- Backend → Render
- Database → MongoDB Atlas
- ML Inference → Hugging Face Spaces

---

## 📊 Model Performance

| Task | Model | Balanced Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Binary (CN vs AD) | MedicalNet ResNet-10 (Transfer Learning) | **87.00%** | 85.71% | **0.9231** |
| Binary (CN vs AD) | Simple3DCNN (From Scratch) | 50.00% | 45.00% | 0.5210 |
| Multi-Class (CN/MCI/AD) | MedicalNet ResNet-10 (Transfer Learning) | **72.41%** | 71.56% | 0.8234 |
| Multi-Class (CN/MCI/AD) | Simple3DCNN (From Scratch) | 39.68% | 35.20% | 0.5840 |

> Transfer learning improved binary accuracy by **+37 percentage points** and multi-class accuracy by **+32.7 percentage points** over training from scratch.

---

## 🎓 Research & Approval

This project's model training and preprocessing pipeline were developed and validated using real ADNI data under **institutional research approval from IIT Ropar**. This is **not a synthetic or demo-only model** — both binary and multi-class classification heads are trained on genuine neuroimaging data following strict subject-level data-splitting to prevent leakage.

---

## 👥 Team

**Team Xynapse**
GLA University, Mathura

---

## ⚕️ Disclaimer

NeuroAssist is a research prototype intended for clinical decision **support**, not autonomous diagnosis. All AI outputs must be reviewed and validated by a licensed physician before any clinical action is taken. This tool is not FDA/CDSCO approved for standalone diagnostic use.
