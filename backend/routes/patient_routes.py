from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from datetime import datetime
from typing import List

import models
from database import patients_col, scans_col, users_col, generate_unique_patient_code
from auth import get_current_user, require_role
from utils.audit import log_audit

router = APIRouter(prefix="/api/patients", tags=["patients"])

@router.post("/create", status_code=201)
async def create_patient(
    patient: models.PatientCreate,
    current_user: dict = Depends(require_role(["doctor"]))
):
    # Generate unique patient code: NA-YYYY-NNNN
    code = await generate_unique_patient_code()
    
    # Check if a patient user account exists with this email/contact or if they are created by doctor
    # For now, patient is created directly by doctor.
    patient_dict = {
        "patient_code": code,
        "doctor_id": current_user["id"],
        "user_id": None, # Will link if patient signs up later
        "full_name": patient.full_name,
        "date_of_birth": patient.date_of_birth,
        "gender": patient.gender,
        "contact": patient.contact,
        "medical_history": patient.medical_history,
        "doctor_notes": "",
        "created_at": datetime.utcnow()
    }
    
    result = await patients_col.insert_one(patient_dict)
    new_patient_id = str(result.inserted_id)
    
    # Audit log
    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="PATIENT_CREATE",
        details=f"Created patient profile for: {patient.full_name} ({code})"
    )
    
    return {
        "id": new_patient_id,
        "patient_code": code,
        "full_name": patient.full_name,
        "date_of_birth": patient.date_of_birth,
        "gender": patient.gender,
        "contact": patient.contact,
        "medical_history": patient.medical_history,
        "doctor_id": current_user["id"]
    }

@router.get("/")
async def get_patients(
    current_user: dict = Depends(get_current_user)
):
    role = current_user.get("role")
    user_id = current_user.get("id")
    
    # Query logic depending on user role
    if role == "doctor":
        # Only this clinician's own patients. The previous $or matched every
        # patient with any non-null doctor_id, leaking the whole directory.
        cursor = patients_col.find({"doctor_id": user_id})
    elif role == "patient":
        # Search patient profiles linked to this user ID
        cursor = patients_col.find({"user_id": user_id})
    elif role == "admin":
        cursor = patients_col.find({})
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized role"
        )
        
    patients = await cursor.to_list(length=100)
    
    # Augment with scan counts
    result = []
    for p in patients:
        p_id_str = str(p["_id"])
        scan_count = await scans_col.count_documents({"patient_id": p_id_str})
        p_data = models.serialize_doc(p)
        p_data["scan_count"] = scan_count
        result.append(p_data)
        
    return result

@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        obj_id = ObjectId(patient_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid patient ID format"
        )
        
    patient = await patients_col.find_one({"_id": obj_id})
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
        
    role = current_user.get("role")
    user_id = current_user.get("id")
    
    # Security boundaries
    if role == "patient" and patient.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Unauthorized patient"
        )
    # An unassigned patient (doctor_id None) must not fall through to every doctor.
    if role == "doctor" and patient.get("doctor_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Patient belongs to another clinician"
        )
        
    # Get scans history
    scan_cursor = scans_col.find({"patient_id": patient_id}).sort("upload_date", -1)
    scans = await scan_cursor.to_list(length=50)
    
    scan_list = []
    for s in scans:
        import json
        brain_regions = s.get("brain_regions", {})
        if isinstance(brain_regions, str):
            try:
                brain_regions = json.loads(brain_regions)
            except Exception:
                brain_regions = {}
                
        # Confidence resolution
        max_conf = max(s.get("conf_cn", 0), s.get("conf_mci", 0), s.get("conf_ad", 0))
        
        scan_list.append({
            "id": s.get("scan_id_string"),
            "date": s.get("upload_date").isoformat() if s.get("upload_date") else None,
            "diagnosis": s.get("prediction"),
            "confidence": round(max_conf * 100, 1),
            "conf_cn": s.get("conf_cn"),
            "conf_mci": s.get("conf_mci"),
            "conf_ad": s.get("conf_ad"),
            "status": s.get("status"),
            "urgency": s.get("urgency"),
            "model": s.get("model_used"),
            "brain_regions": brain_regions
        })
        
    # Audit log
    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="PATIENT_VIEW",
        details=f"Viewed patient file for: {patient.get('full_name')} ({patient.get('patient_code')})"
    )
    
    p_data = models.serialize_doc(patient)
    p_data["scans"] = scan_list
    p_data["total_scans"] = len(scan_list)
    
    return p_data

@router.get("/{patient_id}/timeline")
async def get_patient_timeline(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        obj_id = ObjectId(patient_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid patient ID format"
        )
        
    patient = await patients_col.find_one({"_id": obj_id})
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
        
    # Access checks
    role = current_user.get("role")
    user_id = current_user.get("id")
    if role == "patient" and patient.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized"
        )
    if role == "doctor" and patient.get("doctor_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized"
        )
        
    scan_cursor = scans_col.find({"patient_id": patient_id}).sort("upload_date", 1)
    scans = await scan_cursor.to_list(length=100)
    
    timeline = []
    for s in scans:
        pred = s.get("prediction")
        if pred:
            score = {"CN": 0, "MCI": 1, "AD": 2}.get(pred, 0)
            conf = max(s.get("conf_cn", 0), s.get("conf_mci", 0), s.get("conf_ad", 0))
            timeline.append({
                "date": s.get("upload_date").strftime("%b %Y") if s.get("upload_date") else "",
                "date_iso": s.get("upload_date").isoformat() if s.get("upload_date") else "",
                "score": score,
                "result": pred,
                "conf": round(conf * 100, 1),
                "scan_id": s.get("scan_id_string")
            })
            
    return timeline

@router.delete("/{patient_id}", status_code=204)
async def delete_patient(
    patient_id: str,
    current_user: dict = Depends(require_role(["doctor"]))
):
    try:
        obj_id = ObjectId(patient_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid patient ID format"
        )
        
    patient = await patients_col.find_one({"_id": obj_id})
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
        
    # Check if the doctor owns this patient profile
    if patient.get("doctor_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Patient profile owned by another doctor"
        )
        
    # Delete patient profile
    await patients_col.delete_one({"_id": obj_id})
    
    # Delete associated scans
    await scans_col.delete_many({"patient_id": patient_id})
    
    # Log audit
    await log_audit(
        user_id=current_user["id"],
        email=current_user["email"],
        action="PATIENT_DELETE",
        details=f"Deleted patient profile for: {patient.get('full_name')} ({patient.get('patient_code')})"
    )
    
    return None
