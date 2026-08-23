from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Request
from fastapi.responses import FileResponse
from bson import ObjectId
from datetime import datetime
import uuid, os, json, logging

import models
from database import scans_col, patients_col, review_queue_col
from auth import get_current_user, require_role
from ml.inference import run_inference
from ml.gradcam_engine import generate_brain_heatmap_slices, get_slice_image_path
from utils.audit import log_audit
from utils.pdf_report import generate_pdf_report

router = APIRouter(prefix="/api/scan", tags=["scans"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
SCAN_DIR = os.path.join(UPLOAD_DIR, "mri_scans")
GRADCAM_DIR = os.path.join(UPLOAD_DIR, "gradcam")
REPORTS_DIR = os.path.join(UPLOAD_DIR, "reports")
os.makedirs(SCAN_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


async def assert_scan_access(scan: dict, current_user: dict) -> None:
    """Raise 403 unless this user may touch this scan.

    Admin sees everything, a doctor only scans belonging to them, a patient
    only scans attached to their own linked profile. Every scan-scoped route
    must call this — looking a scan up by its id is not authorisation.
    """
    role = current_user.get("role")
    if role == "admin":
        return
    if role == "doctor":
        if scan.get("doctor_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Access denied: scan belongs to another clinician")
        return
    if role == "patient":
        profile = await patients_col.find_one({"user_id": current_user["id"]})
        if not profile or str(profile["_id"]) != scan.get("patient_id"):
            raise HTTPException(status_code=403, detail="Access denied: scan belongs to another patient")
        return
    raise HTTPException(status_code=403, detail="Unauthorized role")


@router.post("/upload")
async def upload_scan(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    current_user: dict = Depends(require_role(["doctor", "admin", "patient"]))
):
    # Verify the patient belongs to this doctor, or is the patient themselves, or is admin
    try:
        query = {"_id": ObjectId(patient_id)}
        if current_user["role"] == "doctor":
            query["doctor_id"] = current_user["id"]
        elif current_user["role"] == "patient":
            patient_profile = await patients_col.find_one({"user_id": current_user["id"]})
            if not patient_profile or str(patient_profile["_id"]) != patient_id:
                raise HTTPException(status_code=403, detail="Unauthorized to upload scans for other patient records")
        patient = await patients_col.find_one(query)
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid patient ID")

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or unauthorized")

    # Validate file extension
    filename = file.filename or "unknown.nii.gz"
    allowed_exts = {".nii", ".nii.gz", ".dcm", ".mha", ".mhd", ".nrrd", ".mnc"}
    ext = ""
    if filename.endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = os.path.splitext(filename)[1].lower()

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed_exts}")

    scan_id_str = f"SCN-{uuid.uuid4().hex[:6].upper()}"
    scan_dir = os.path.join(SCAN_DIR, scan_id_str)
    os.makedirs(scan_dir, exist_ok=True)

    safe_filename = f"{scan_id_str}{ext}"
    file_path = os.path.join(scan_dir, safe_filename)

    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    # Create initial scan document
    scan_doc = {
        "scan_id_string": scan_id_str,
        "patient_id": patient_id,
        "doctor_id": patient.get("doctor_id") or current_user["id"],
        "filename": safe_filename,
        "original_filename": filename,
        "file_path": file_path,
        "upload_date": datetime.utcnow(),
        "status": "uploaded",
        "model_used": None,
        "prediction": None,
        "conf_cn": None, "conf_mci": None, "conf_ad": None,
        "risk_score": None, "urgency": None,
        "processing_time": None, "file_hash": None,
        "biomarker_hippocampal": None, "biomarker_amyloid": None, "biomarker_ventricle": None,
        "gradcam_axial": None, "gradcam_coronal": None, "gradcam_sagittal": None,
        "brain_regions": {},
        "doctor_diagnosis": None, "doctor_notes": None, "reviewed_at": None,
    }

    result = await scans_col.insert_one(scan_doc)

    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="SCAN_UPLOAD",
        details=f"Uploaded scan {scan_id_str} for patient {patient.get('patient_code')} ({patient_id})"
    )

    return {"scan_id": scan_id_str, "status": "uploaded", "message": "File uploaded successfully"}


@router.post("/analyze")
async def analyze_scan(
    scan_id: str = Form(...),
    model_type: str = Form("multiclass"),
    current_user: dict = Depends(require_role(["doctor", "admin", "patient"]))
):
    query = {"scan_id_string": scan_id}
    if current_user["role"] == "doctor":
        query["doctor_id"] = current_user["id"]
    elif current_user["role"] == "patient":
        patient_profile = await patients_col.find_one({"user_id": current_user["id"]})
        if not patient_profile:
            raise HTTPException(status_code=403, detail="Patient profile not found")
        query["patient_id"] = str(patient_profile["_id"])
    scan = await scans_col.find_one(query)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found or unauthorized")

    file_path = scan.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Scan file missing from storage")

    # --- Run real ML inference ---
    inference_result = run_inference(file_path, model_type)

    pred_idx = {"CN": 0, "MCI": 1, "AD": 2}.get(inference_result["prediction"], 1)
    preprocessed_volume = inference_result.pop("preprocessed_volume", None)
    # --- Generate real Grad-CAM ---
    gradcam_result = None
    try:
        gradcam_result = generate_brain_heatmap_slices(
            scan_id=scan_id,
            prediction_class=pred_idx,
            brain_regions=inference_result["brain_regions"],
            preprocessed_volume=preprocessed_volume,
            output_dir=GRADCAM_DIR,
        )
    except Exception as e:
        logger.warning(f"[WARN] Grad-CAM generation failed: {e}")

    gradcam_axial = gradcam_coronal = gradcam_sagittal = None
    if gradcam_result and "slice_paths" in gradcam_result:
        gradcam_axial = gradcam_result["slice_paths"].get("axial")
        gradcam_coronal = gradcam_result["slice_paths"].get("coronal")
        gradcam_sagittal = gradcam_result["slice_paths"].get("sagittal")

    biomarkers = inference_result["biomarkers"]
    update_doc = {
        "model_used": model_type,
        "prediction": inference_result["prediction"],
        "conf_cn": inference_result["confidence_cn"],
        "conf_mci": inference_result["confidence_mci"],
        "conf_ad": inference_result["confidence_ad"],
        "risk_score": inference_result["risk_score"],
        "urgency": inference_result["urgency"],
        "processing_time": inference_result["processing_time"],
        "file_hash": inference_result.get("file_hash", ""),
        "biomarker_hippocampal": biomarkers["hippocampal_atrophy"],
        "biomarker_amyloid": biomarkers["amyloid_plaque_load"],
        "biomarker_ventricle": biomarkers["ventricle_enlargement"],
        "brain_regions": inference_result["brain_regions"],
        "gradcam_axial": gradcam_axial,
        "gradcam_coronal": gradcam_coronal,
        "gradcam_sagittal": gradcam_sagittal,
        "model_trained": inference_result.get("model_trained", False),
        "status": "analyzed",
    }

    await scans_col.update_one({"scan_id_string": scan_id}, {"$set": update_doc})

    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="SCAN_ANALYZE",
        details=f"Analyzed scan {scan_id}. Prediction: {inference_result['prediction']} (risk: {inference_result['risk_score']}%)"
    )

    return {"scan_id": scan_id, "status": "analyzed", "prediction": inference_result["prediction"]}


@router.get("/history")
async def get_scan_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    role = current_user.get("role")
    user_id = current_user.get("id")

    if role == "doctor":
        query = {"doctor_id": user_id}
    elif role == "patient":
        # Fetch patient profile linked to this user
        patient = await patients_col.find_one({"user_id": user_id})
        if not patient:
            return {"items": [], "total": 0}
        query = {"patient_id": str(patient["_id"])}
    else:
        query = {}  # Admin sees all

    cursor = scans_col.find(query).sort("upload_date", -1).limit(limit)
    scans = await cursor.to_list(length=limit)

    result = []
    for s in scans:
        patient_doc = await patients_col.find_one({"_id": ObjectId(s.get("patient_id", "000000000000000000000000"))})
        patient_name = patient_doc.get("full_name", "Unknown") if patient_doc else "Unknown"
        patient_code = patient_doc.get("patient_code", "") if patient_doc else ""
        max_conf = max(s.get("conf_cn") or 0, s.get("conf_mci") or 0, s.get("conf_ad") or 0)

        result.append({
            "id": s.get("scan_id_string"),
            "date": s.get("upload_date").isoformat() if s.get("upload_date") else None,
            "patient": patient_name,
            "patient_code": patient_code,
            "patient_id": s.get("patient_id"),
            "diagnosis": s.get("prediction"),
            "confidence": round(max_conf * 100, 1),
            "status": s.get("status"),
            "urgency": s.get("urgency"),
            "model": s.get("model_used"),
        })

    return {"items": result, "total": len(result)}


@router.get("/result/{scan_id}")
async def get_scan_result(scan_id: str, current_user: dict = Depends(get_current_user)):
    return await _get_scan_detail_by_id(scan_id, current_user)


@router.get("/{scan_id}")
async def get_scan_detail(scan_id: str, current_user: dict = Depends(get_current_user)):
    return await _get_scan_detail_by_id(scan_id, current_user)


@router.delete("/{scan_id}", status_code=204)
async def delete_scan(
    scan_id: str,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if scan:
        # No unscoped fallback here: it used to delete scans the caller did not own.
        await assert_scan_access(scan, current_user)
        await scans_col.delete_one({"_id": scan["_id"]})
        await review_queue_col.delete_one({"scan_id_string": scan_id})
        await log_audit(
            user_id=current_user["id"],
            email=current_user["email"],
            action="SCAN_DELETE",
            details=f"Deleted scan {scan_id}"
        )
    return

@router.put("/{scan_id}/review")
async def review_scan(
    scan_id: str,
    review_data: models.ReviewRequest,
    current_user: dict = Depends(require_role(["doctor", "admin"]))
):
    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    await assert_scan_access(scan, current_user)

    if scan.get("status") in ["accepted", "flagged", "overridden"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This scan has already been signed off and is locked."
        )

    action = review_data.action
    update = {"reviewed_at": datetime.utcnow(), "doctor_notes": review_data.doctor_notes or ""}

    if action == "ACCEPT FINDING":
        update["status"] = "accepted"
        update["doctor_diagnosis"] = scan.get("prediction")
    elif action == "FLAG FOR REVIEW":
        update["status"] = "flagged"
        # Add to review queue
        await review_queue_col.update_one(
            {"scan_id_string": scan_id},
            {"$set": {
                "scan_id_string": scan_id,
                "patient_id": scan.get("patient_id"),
                "doctor_id": current_user["id"],
                "ai_prediction": scan.get("prediction"),
                "corrected_diagnosis": review_data.doctor_diagnosis or scan.get("prediction"),
                "doctor_notes": review_data.doctor_notes or "",
                "review_status": "pending_admin",
                "flagged_at": datetime.utcnow(),
                "approved_for_training": False,
            }},
            upsert=True
        )
    elif action == "OVERRIDE DIAGNOSIS":
        update["status"] = "overridden"
        update["doctor_diagnosis"] = review_data.doctor_diagnosis

    await scans_col.update_one({"scan_id_string": scan_id}, {"$set": update})

    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action=f"SCAN_REVIEW_{action.replace(' ', '_')}",
        details=f"Reviewed scan {scan_id}. Action: {action}"
    )

    return {
        "message": "Review recorded successfully",
        "status": update.get("status"),
        "doctor": current_user["full_name"],
    }


@router.get("/{scan_id}/report")
async def download_report(scan_id: str, current_user: dict = Depends(get_current_user)):
    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    await assert_scan_access(scan, current_user)

    patient_doc = None
    if scan.get("patient_id"):
        try:
            patient_doc = await patients_col.find_one({"_id": ObjectId(scan["patient_id"])})
        except Exception:
            pass

    pdf_path = os.path.join(REPORTS_DIR, f"{scan_id}_report.pdf")
    generate_pdf_report(scan, models.serialize_doc(patient_doc) if patient_doc else {}, pdf_path)

    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="REPORT_DOWNLOAD",
        details=f"Downloaded PDF report for scan {scan_id}"
    )

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"NeuroAssist_Report_{scan_id}.pdf")


async def _get_scan_detail_by_id(scan_id: str, current_user: dict) -> dict:
    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    await assert_scan_access(scan, current_user)

    patient_doc = None
    if scan.get("patient_id"):
        try:
            patient_doc = await patients_col.find_one({"_id": ObjectId(scan["patient_id"])})
        except Exception:
            pass

    gradcam_slices = {}
    for key, field in [("axial", "gradcam_axial"), ("coronal", "gradcam_coronal"), ("sagittal", "gradcam_sagittal")]:
        v = scan.get(field)
        if v:
            gradcam_slices[key] = "/" + v.replace("\\", "/")

    return {
        "scan_id": scan.get("scan_id_string"),
        "patient_id": scan.get("patient_id"),
        "patient_name": patient_doc.get("full_name", "Unknown") if patient_doc else "Unknown",
        "patient_code": patient_doc.get("patient_code", "") if patient_doc else "",
        "prediction": scan.get("prediction"),
        "confidence_cn": scan.get("conf_cn"),
        "confidence_mci": scan.get("conf_mci"),
        "confidence_ad": scan.get("conf_ad"),
        "risk_score": scan.get("risk_score"),
        "urgency": scan.get("urgency"),
        "processing_time": scan.get("processing_time"),
        "model_used": scan.get("model_used"),
        # False means the network had no trained weights: the probabilities are
        # demo output, not a finding. The UI must say so.
        "model_trained": bool(scan.get("model_trained", False)),
        "file_hash": scan.get("file_hash"),
        "original_filename": scan.get("original_filename"),
        "scan_date": scan.get("upload_date").isoformat() if scan.get("upload_date") else None,
        "status": scan.get("status"),
        "doctor_diagnosis": scan.get("doctor_diagnosis"),
        "doctor_notes": scan.get("doctor_notes"),
        "reviewed_at": scan.get("reviewed_at").isoformat() if scan.get("reviewed_at") else None,
        "biomarkers": {
            "hippocampal_atrophy": scan.get("biomarker_hippocampal"),
            "amyloid_plaque_load": scan.get("biomarker_amyloid"),
            "ventricle_enlargement": scan.get("biomarker_ventricle"),
        },
        "gradcam_slices": gradcam_slices,
        "brain_regions": scan.get("brain_regions", {}),
    }


@router.get("/{scan_id}/slice/{view_name}/{slice_index}")
async def get_scan_slice(
    scan_id: str,
    view_name: str,
    slice_index: int,
    request: Request
):
    # Try query param first, then Authorization header
    token = request.query_params.get("token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")
        
    try:
        from auth import JWT_SECRET, ALGORITHM
        from jose import jwt
        from database import users_col
        import models
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email = payload.get("sub")
        token_type = payload.get("type")
        if not email or token_type != "access":
            raise HTTPException(status_code=401, detail="Invalid token")
            
        user_doc = await users_col.find_one({"email": email})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    if view_name not in ["axial", "coronal", "sagittal"]:
        raise HTTPException(status_code=400, detail="Invalid view name")
    if slice_index < 0 or slice_index > 100:
        raise HTTPException(status_code=400, detail="Slice index must be between 0 and 100")

    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # This route hand-rolls its own token check, so it also has to authorise:
    # a valid token for any account used to render any patient's MRI slices.
    await assert_scan_access(scan, models.serialize_doc(user_doc))

    file_path = scan.get("file_path", "")
        
    prediction = scan.get("prediction") or "MCI"
    brain_regions = scan.get("brain_regions") or {}
    try:
        output_path = get_slice_image_path(
            scan_id=scan_id,
            view_name=view_name,
            slice_percent=slice_index,
            prediction=prediction,
            brain_regions=brain_regions,
            file_path=file_path
        )
        return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        logger.error(f"Error rendering dynamic slice: {e}")
        raise HTTPException(status_code=500, detail=f"Error rendering slice: {str(e)}")

