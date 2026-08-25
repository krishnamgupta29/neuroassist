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
    if role in ["admin", "doctor"]:
        return
    if role == "patient":
        profile = await patients_col.find_one({"user_id": current_user["id"]})
        if profile and str(profile["_id"]) == scan.get("patient_id"):
            return
        # Allow if scan's patient_id matches user id directly
        if scan.get("patient_id") == current_user["id"]:
            return
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
    patient = None
    try:
        if ObjectId.is_valid(patient_id):
            patient = await patients_col.find_one({"_id": ObjectId(patient_id)})
    except Exception:
        pass

    if not patient:
        patient = await patients_col.find_one({
            "$or": [
                {"patient_code": patient_id},
                {"patient_code": patient_id.upper()},
                {"full_name": patient_id},
                {"mrn": patient_id}
            ]
        })

    if not patient:
        from database import generate_unique_patient_code
        code = await generate_unique_patient_code()
        p_name = patient_id if not ObjectId.is_valid(patient_id) else f"Patient {code}"
        p_doc = {
            "patient_code": code,
            "doctor_id": current_user["id"],
            "user_id": current_user["id"] if current_user["role"] == "patient" else None,
            "full_name": p_name,
            "gender": "Unknown",
            "medical_history": "Auto-registered upon MRI scan submission.",
            "created_at": datetime.utcnow()
        }
        res = await patients_col.insert_one(p_doc)
        patient = await patients_col.find_one({"_id": res.inserted_id})
        patient_id = str(patient["_id"])

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
    scan = await scans_col.find_one({"scan_id_string": scan_id})
    if not scan and ObjectId.is_valid(scan_id):
        scan = await scans_col.find_one({"_id": ObjectId(scan_id)})
    if not scan:
        scan = await scans_col.find_one({"$or": [{"scanId": scan_id}, {"id": scan_id}]})
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    await assert_scan_access(scan, current_user)

    file_path = scan.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Scan file missing from storage")

    # --- Run real ML inference ---
    inference_result = run_inference(file_path, model_type, original_filename=scan.get("original_filename", ""))

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
        try:
            patient_doc = await patients_col.find_one({"_id": ObjectId(s.get("patient_id", "000000000000000000000000"))})
        except Exception:
            patient_doc = None
        patient_name = patient_doc.get("full_name", "Unknown") if patient_doc else "Unknown"
        patient_code = patient_doc.get("patient_code", "PT-UNKNOWN") if patient_doc else "PT-UNKNOWN"
        max_conf = max(s.get("confidence_cn") or 0, s.get("confidence_mci") or 0, s.get("confidence_ad") or 0)

        result.append({
            "id": s.get("scan_id_string"),
            "scanId": s.get("scan_id_string"),
            "scan_id_string": s.get("scan_id_string"),
            "date": s.get("upload_date").isoformat() if s.get("upload_date") else None,
            "uploadDate": s.get("upload_date").isoformat() if s.get("upload_date") else None,
            "patient": patient_name,
            "patientName": patient_name,
            "patient_code": patient_code,
            "mrn": patient_code,
            "patient_id": s.get("patient_id"),
            "patientId": s.get("patient_id"),
            "prediction": s.get("prediction"),
            "diagnosis": s.get("prediction"),
            "risk_score": s.get("risk_score"),
            "riskScore": s.get("risk_score"),
            "confidence": round(max_conf * 100, 1) if max_conf else 0.0,
            "status": s.get("status"),
            "doctorStatus": s.get("status"),
            "urgency": s.get("urgency"),
            "model": s.get("model_used"),
            "model_used": s.get("model_used"),
            "reviewed_at": s.get("reviewed_at").isoformat() if s.get("reviewed_at") else None,
            "reviewed_by": s.get("reviewed_by"),
            "doctor_diagnosis": s.get("doctor_diagnosis"),
            "doctor_notes": s.get("doctor_notes"),
            "is_signed_off": s.get("status") in ["accepted", "flagged", "overridden", "signed_off"],
            "isSignedOff": s.get("status") in ["accepted", "flagged", "overridden", "signed_off"],
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
    scan = await scans_col.find_one({"$or": [{"scan_id_string": scan_id}, {"id": scan_id}]})
    if not scan and ObjectId.is_valid(scan_id):
        scan = await scans_col.find_one({"_id": ObjectId(scan_id)})

    if not scan:
        now = datetime.utcnow()
        scan = {
            "scan_id_string": scan_id,
            "patient_id": review_data.patient_id or "PT-DEFAULT",
            "doctor_id": current_user["id"],
            "prediction": review_data.doctor_diagnosis or "CN",
            "risk_score": 18,
            "status": "uploaded",
            "created_at": now,
        }
        await scans_col.insert_one(scan)

    await assert_scan_access(scan, current_user)

    action = (review_data.action or "ACCEPT FINDING").upper()
    update = {
        "reviewed_at": datetime.utcnow(),
        "doctor_notes": review_data.doctor_notes or "",
        "doctor_id": current_user["id"],
        "reviewed_by": current_user.get("full_name") or "Dr. Sarah Smith",
        "is_signed_off": True,
    }

    if "ACCEPT" in action:
        update["status"] = "accepted"
        update["doctor_diagnosis"] = review_data.doctor_diagnosis or scan.get("prediction") or "CN"
    elif "FLAG" in action:
        update["status"] = "flagged"
        # Add to review queue
        await review_queue_col.update_one(
            {"scan_id_string": scan_id},
            {"$set": {
                "scan_id_string": scan_id,
                "patient_id": scan.get("patient_id") or review_data.patient_id,
                "doctor_id": current_user["id"],
                "ai_prediction": scan.get("prediction") or "CN",
                "corrected_diagnosis": review_data.doctor_diagnosis or scan.get("prediction") or "CN",
                "doctor_notes": review_data.doctor_notes or "",
                "review_status": "pending_admin",
                "flagged_at": datetime.utcnow(),
                "approved_for_training": False,
            }},
            upsert=True
        )
    elif "OVERRIDE" in action:
        update["status"] = "overridden"
        update["doctor_diagnosis"] = review_data.doctor_diagnosis or "MCI"

    await scans_col.update_one(
        {"$or": [{"scan_id_string": scan_id}, {"id": scan_id}]},
        {"$set": update}
    )

    # Update patient record in database if patient_id is present
    target_pid = str(scan.get("patient_id") or review_data.patient_id or "")
    if target_pid:
        pat_query = {"$or": [{"patient_code": target_pid}, {"id": target_pid}]}
        if ObjectId.is_valid(target_pid):
            pat_query["$or"].append({"_id": ObjectId(target_pid)})
        
        await patients_col.update_one(
            pat_query,
            {"$set": {
                "doctor_status": update.get("status"),
                "status": update.get("status"),
                "is_signed_off": True,
                "reviewed_at": update.get("reviewed_at").isoformat() if isinstance(update.get("reviewed_at"), datetime) else str(update.get("reviewed_at")),
                "reviewed_by": current_user.get("full_name") or "Clinician",
                "doctor_diagnosis": update.get("doctor_diagnosis"),
                "doctor_notes": update.get("doctor_notes")
            }}
        )

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

