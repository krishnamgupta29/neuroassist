from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class RoleEnum(str, Enum):
    doctor = "doctor"
    patient = "patient"
    admin = "admin"

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: RoleEnum

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: RoleEnum
    created_at: datetime

    class Config:
        from_attributes = True

class PatientCreate(BaseModel):
    full_name: str
    date_of_birth: str
    gender: str
    contact: str = ""
    medical_history: str = ""

class PatientResponse(BaseModel):
    id: str
    patient_code: str
    doctor_id: str
    user_id: Optional[str] = None
    full_name: str
    date_of_birth: str
    gender: str
    contact: str = ""
    medical_history: str = ""
    doctor_notes: str = ""
    scan_count: int = 0

class Biomarkers(BaseModel):
    hippocampal_atrophy: Optional[float] = None
    amyloid_plaque_load: Optional[float] = None
    ventricle_enlargement: Optional[float] = None

class ScanResponse(BaseModel):
    scan_id: str
    patient_id: str
    patient_name: str
    patient_code: str
    prediction: str
    confidence_cn: float
    confidence_mci: float
    confidence_ad: float
    risk_score: float
    urgency: str
    processing_time: Optional[float] = None
    model_used: str
    file_hash: Optional[str] = None
    original_filename: Optional[str] = None
    scan_date: Optional[str] = None
    status: str
    doctor_diagnosis: Optional[str] = None
    doctor_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    biomarkers: Biomarkers
    gradcam_slices: Dict[str, str]
    brain_regions: Dict[str, float]

class ReviewRequest(BaseModel):
    action: str  # ACCEPT FINDING, FLAG FOR REVIEW, OVERRIDE DIAGNOSIS
    doctor_diagnosis: Optional[str] = None
    doctor_notes: Optional[str] = None
    patient_id: Optional[str] = None

class AuditLogCreate(BaseModel):
    user_id: str
    user_email: str
    action: str
    details: str

class SettingsUpdate(BaseModel):
    active_model: str
    auto_archive: bool
    notifications_enabled: bool

# Document conversion utilities for MongoDB
def serialize_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert ObjectId '_id' to string 'id' for JSON response."""
    if not doc:
        return None
    serialized = dict(doc)
    if "_id" in serialized:
        serialized["id"] = str(serialized["_id"])
        del serialized["_id"]
    return serialized

def serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert list of MongoDB documents."""
    return [serialize_doc(d) for d in docs if d]
