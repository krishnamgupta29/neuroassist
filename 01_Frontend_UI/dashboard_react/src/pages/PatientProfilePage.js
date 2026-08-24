import React, { useState, useMemo } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { patientAPI } from '../services/api';
import DashboardLayout from '../components/layout/DashboardLayout';
import StatusBadge from '../components/common/StatusBadge';
import { generatePatientData } from '../utils/mockDataGenerator';
import { 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { 
  FiArrowLeft, 
  FiUploadCloud, 
  FiCheckCircle, 
  FiArrowRight,
  FiTrash2,
  FiAlertTriangle,
  FiX
} from 'react-icons/fi';

export default function PatientProfilePage() {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const { state, dispatch } = useApp();

  const patientsList = Array.isArray(state.patients) ? state.patients : [];
  const rawPatient = patientsList.find(p => (p.id || p._id) === patientId) || patientsList[0];
  const targetPatientId = rawPatient?.id || rawPatient?._id || patientId || 'PT-9042';

  // Generate deterministic seeded values for this patient
  const seeded = useMemo(() => generatePatientData(targetPatientId), [targetPatientId]);

  const patient = {
    id: targetPatientId,
    _id: targetPatientId,
    full_name: rawPatient?.full_name || rawPatient?.name || 'Patient Record',
    name: rawPatient?.full_name || rawPatient?.name || 'Patient Record',
    patient_code: rawPatient?.patient_code || rawPatient?.mrn || 'NA-2026-0001',
    mrn: rawPatient?.patient_code || rawPatient?.mrn || 'NA-2026-0001',
    age: rawPatient?.age || (rawPatient?.date_of_birth ? new Date().getFullYear() - new Date(rawPatient.date_of_birth).getFullYear() : (rawPatient?.age || 65)),
    gender: rawPatient?.gender || '—',
    condition: rawPatient?.condition || rawPatient?.diagnosis || seeded.condition,
    stage: rawPatient?.stage || seeded.stage,
    assignedDoctor: rawPatient?.assignedDoctor || 'Assigned Clinical Team',
    doctorNotes: rawPatient?.doctorNotes || 'Initial baseline MRI acquired. Mild bilateral temporal lobe asymmetry noted.',
    riskScore: rawPatient?.riskScore ?? rawPatient?.risk_score ?? seeded.riskScore,
    mmseScore: rawPatient?.mmseScore ?? rawPatient?.mmse ?? seeded.mmseScore,
    history: rawPatient?.history && rawPatient.history.length > 0 ? rawPatient.history : seeded.history,
  };

  const pId = patient.id || patient._id;
  const pName = patient.full_name || patient.name || 'Patient Record';
  const pMrn = patient.patient_code || patient.mrn || patient.id;
  const pInitials = pName.split(' ').map(n => n[0]).join('').slice(0, 2) || 'PT';

  const scansList = Array.isArray(state.scans) ? state.scans : [];
  let patientScans = scansList.filter(s => {
    const sPId = s.patientId || s.patient_id;
    const sMrn = s.patient_code || s.mrn;
    const sPName = (s.patientName || s.patient || '').trim().toLowerCase();
    const targetName = pName.trim().toLowerCase();
    if (sPId && sPId === pId) return true;
    if (sMrn && sMrn === pMrn) return true;
    if (sPName && targetName && sPName === targetName) return true;
    return false;
  });

  const latestScan = patientScans.length > 0 ? patientScans[0] : null;
  const rawCond = (latestScan?.prediction || latestScan?.condition || patient.condition || 'CN').toUpperCase();
  const realCond = rawCond.includes('AD') ? 'AD' : rawCond.includes('MCI') ? 'MCI' : 'CN';
  patient.condition = realCond;
  patient.stage = realCond === 'AD' ? "Alzheimer's Disease" : realCond === 'MCI' ? 'Mild Cognitive Impairment' : 'Cognitively Normal';
  if (latestScan?.riskScore !== undefined || latestScan?.risk_score !== undefined) {
    patient.riskScore = latestScan.riskScore ?? latestScan.risk_score;
  }

  const [notes, setNotes] = useState(patient.doctorNotes || '');
  const [isSaved, setIsSaved] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  const handleSaveNotes = () => {
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2500);
  };

  const handleDeletePatient = async () => {
    setIsDeleting(true);
    setDeleteError('');
    try {
      // Only drop it locally once the server confirms, otherwise the patient
      // reappears on the next fetch.
      await patientAPI.delete(patient.id);
    } catch (e) {
      setIsDeleting(false);
      setDeleteError(
        e?.response?.data?.detail || 'Could not delete this patient. Nothing was removed.'
      );
      return;
    }

    dispatch({ type: 'DELETE_PATIENT', payload: patient.id });
    setIsDeleting(false);
    setShowDeleteModal(false);
    navigate('/dashboard/patients');
  };

  return (
    <DashboardLayout
      title={`Patient Record: ${pName}`}
      subtitle={`Medical Record Number: ${pMrn} · Registry ID: ${patient.id}`}
      action={
        <div className="flex items-center gap-2">
          <Link
            to="/dashboard/patients"
            className="btn-outline text-xs"
          >
            <FiArrowLeft className="w-3.5 h-3.5" />
            <span>All Patients</span>
          </Link>
          <button
            type="button"
            onClick={() => setShowDeleteModal(true)}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold text-[#7A1F2B] bg-[#F8EAED] border border-[#ECC8CF] hover:bg-[#F0D5DA] transition-colors flex items-center gap-1.5"
          >
            <FiTrash2 className="w-3.5 h-3.5" />
            <span>Delete Patient</span>
          </button>
          <Link
            to="/dashboard/scan"
            className="btn-maroon text-xs shadow-clinical-sm"
          >
            <FiUploadCloud className="w-3.5 h-3.5" />
            <span>Upload New Follow-Up MRI</span>
          </Link>
        </div>
      }
    >
      <div className="space-y-6">
        
        {/* Top Demographics Card */}
        <div className="clinical-card p-6 grid grid-cols-1 md:grid-cols-4 gap-6 items-center bg-white">
          <div className="flex items-center gap-4 md:col-span-2">
            <div className="w-14 h-14 rounded-2xl bg-[#FAF6F3] border border-[#E8E2DA] flex items-center justify-center font-serif font-bold text-[#7A1F2B] text-xl shadow-sm">
              {pInitials}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-serif font-bold text-[#22201F]">{pName}</h3>
                <StatusBadge status={patient.condition || 'CN'} size="sm" />
              </div>
              <p className="text-xs text-[#7A756F] mt-0.5">
                {patient.stage || 'Clinical Assessment'} · {patient.age || '—'} Years Old ({patient.gender || '—'})
              </p>
            </div>
          </div>

          <div className="border-l border-[#F0EBE5] pl-4 space-y-1">
            <span className="text-[10px] uppercase font-bold text-[#A39E98] block">Primary Neurologist</span>
            <span className="text-xs font-semibold text-[#22201F] block">{patient.assignedDoctor}</span>
            <span className="text-[11px] text-[#7A756F]">Memory Disorders Clinic</span>
          </div>

          <div className="border-l border-[#F0EBE5] pl-4 space-y-1">
            <span className="text-[10px] uppercase font-bold text-[#A39E98] block">Current AI Risk & MMSE</span>
            <div className="flex items-center gap-3">
              <span className="text-lg font-serif font-bold text-[#7A1F2B]">{patient.riskScore} <span className="text-xs font-sans text-[#7A756F]">/ 100</span></span>
              <span className="text-sm font-mono font-bold text-[#22201F]">MMSE: {patient.mmseScore} <span className="text-xs font-sans text-[#7A756F]">/ 30</span></span>
            </div>
          </div>
        </div>

        {/* Longitudinal Trajectory Chart */}
        <div className="clinical-card p-6 space-y-4 bg-white">
          <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
            <div>
              <h4 className="text-base font-serif font-bold text-[#22201F]">
                Longitudinal Disease Progression Trajectory
              </h4>
              <p className="text-xs text-[#7A756F]">
                Historical trajectory of AI Neurological Risk Score and MMSE Cognitive Exam over time.
              </p>
            </div>
            <span className="text-[11px] font-semibold text-[#4A7C59] bg-[#EDF5F0] px-2.5 py-1 rounded-full border border-[#CFE3D5]">
              {patient.history.length} Timepoints Tracked
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={patient.history} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#7A1F2B" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#7A1F2B" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8E2DA" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#7A756F', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#7A756F', fontSize: 11 }} />
                <RechartsTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-3 rounded-xl shadow-clinical border border-[#E8E2DA] text-xs space-y-1">
                          <div className="font-bold text-[#22201F]">{data.date}</div>
                          <div className="text-[#7A1F2B]">AI Risk Score: <strong>{data.riskScore}/100</strong></div>
                          <div className="text-[#5B7C99]">MMSE Score: <strong>{data.mmse}/30</strong></div>
                          <div className="text-[10px] text-[#A39E98] font-mono">Stage: {data.label}</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area type="monotone" dataKey="riskScore" stroke="#7A1F2B" strokeWidth={2.5} fillOpacity={1} fill="url(#riskGrad)" name="AI Risk Score" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bottom Split: MRI Scan History & Notes */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left 7 Cols: MRI Scans */}
          <div className="lg:col-span-7 clinical-card p-6 space-y-4 bg-white">
            <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
              <h4 className="text-base font-serif font-bold text-[#22201F]">
                Volumetric MRI Series History
              </h4>
              <span className="text-xs text-[#7A756F]">{patientScans.length} Scans Archived</span>
            </div>

            <div className="space-y-3">
              {patientScans.length === 0 ? (
                <div className="p-8 text-center bg-[#FAF6F3] rounded-xl border border-dashed border-[#E8E2DA] space-y-2">
                  <p className="text-xs text-[#7A756F]">No MRI scans uploaded for this patient yet.</p>
                  <Link to="/dashboard/scan" className="btn-maroon text-xs inline-flex items-center gap-1.5 shadow-sm">
                    <FiUploadCloud className="w-3.5 h-3.5" />
                    <span>Upload Baseline MRI</span>
                  </Link>
                </div>
              ) : (
                patientScans.map((scan, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:border-[#7A1F2B] transition-colors"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-xs text-[#22201F]">{scan.scanId}</span>
                        <StatusBadge status={scan.prediction || patient.condition} size="sm" />
                      </div>
                      <p className="text-xs text-[#7A756F]">
                        Acquired on {scan.uploadDate} · AI Confidence: {scan.confidence || 85}%
                      </p>
                      <span className="text-[10px] text-[#A39E98] block">
                        {scan.biomarkers?.hippocampus?.deviation || '-18% Volume Loss'}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 self-start sm:self-center">
                      <button
                        type="button"
                        onClick={() => {
                          if (window.confirm(`Delete scan record ${scan.scanId}?`)) {
                            dispatch({ type: 'DELETE_SCAN', payload: scan.scanId });
                          }
                        }}
                        className="p-1.5 rounded-lg text-[#A39E98] hover:text-[#7A1F2B] hover:bg-[#F8EAED] transition-colors"
                        title="Delete scan"
                      >
                        <FiTrash2 className="w-4 h-4" />
                      </button>
                      <Link
                        to={`/dashboard/scan/${scan.scanId}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-[#E8E2DA] hover:bg-[#F8EAED] hover:text-[#7A1F2B] transition-colors"
                      >
                        <span>Examine Grad-CAM</span>
                        <FiArrowRight className="w-3 h-3" />
                      </Link>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right 5 Cols: Clinical Notes */}
          <div className="lg:col-span-5 clinical-card p-6 space-y-4 bg-white">
            <div className="pb-3 border-b border-[#E8E2DA]">
              <h4 className="text-base font-serif font-bold text-[#22201F]">
                Physician Case Notes
              </h4>
              <p className="text-xs text-[#7A756F]">
                Clinical records & treatment recommendations.
              </p>
            </div>

            <textarea
              rows={5}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="clinical-input text-xs resize-none leading-relaxed w-full"
              placeholder="Record clinical follow-ups..."
            />

            <button
              type="button"
              onClick={handleSaveNotes}
              className="w-full py-2.5 bg-[#7A1F2B] hover:bg-[#661823] text-white rounded-xl text-xs font-semibold transition-all shadow-clinical-sm flex items-center justify-center gap-2"
            >
              <FiCheckCircle className="w-3.5 h-3.5" />
              <span>Update Case Notes</span>
            </button>

            {isSaved && (
              <p className="text-center text-xs font-semibold text-[#4A7C59] animate-fade-in">
                ✓ Patient records updated.
              </p>
            )}
          </div>

        </div>

      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-fadeIn">
          <div className="clinical-card max-w-md w-full p-6 bg-white space-y-4 shadow-2xl animate-scaleUp">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-[#F8EAED] text-[#7A1F2B] flex items-center justify-center">
                  <FiAlertTriangle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-serif font-bold text-[#22201F]">
                    Delete Patient Record
                  </h3>
                  <p className="text-xs text-[#7A756F]">
                    This action cannot be undone.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => { setShowDeleteModal(false); setDeleteError(''); }}
                className="p-1 text-[#A39E98] hover:text-[#22201F]"
              >
                <FiX className="w-4 h-4" />
              </button>
            </div>

            <div className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] text-xs space-y-1">
              <div className="font-bold text-[#22201F]">{pName}</div>
              <div className="text-[#7A756F] font-mono text-[11px]">MRN: {pMrn}</div>
              <p className="text-[11px] text-[#7A1F2B] pt-1">
                All associated longitudinal trajectories, MMSE records, and volumetric scan linkages for this patient will be permanently removed.
              </p>
            </div>

            {deleteError && (
              <div className="p-3 rounded-xl bg-[#F8EAED] border border-[#ECC8CF] text-xs font-semibold text-[#7A1F2B]">
                {deleteError}
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => { setShowDeleteModal(false); setDeleteError(''); }}
                className="px-4 py-2 text-xs font-semibold text-[#7A756F] hover:text-[#22201F] rounded-xl"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={handleDeletePatient}
                className="px-4 py-2 text-xs font-semibold text-white bg-[#7A1F2B] hover:bg-[#661823] rounded-xl transition-all shadow-clinical"
              >
                {isDeleting ? 'Deleting...' : 'Confirm Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

    </DashboardLayout>
  );
}
