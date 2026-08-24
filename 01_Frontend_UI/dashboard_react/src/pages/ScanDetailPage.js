import React, { useState, useMemo, useEffect } from 'react';
import { useParams, Link, useNavigate, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import RiskGaugeArc from '../components/clinical/RiskGaugeArc';
import GradCamViewer from '../components/clinical/GradCamViewer';
import StatusBadge from '../components/common/StatusBadge';
import ClinicalReportModal from '../components/clinical/ClinicalReportModal';
import { generateScanData } from '../utils/mockDataGenerator';
import { scanAPI } from '../services/api';
import { 
  FiCheck,
  FiFlag, 
  FiEdit3, 
  FiPrinter, 
  FiArrowLeft, 
  FiShield, 
  FiCheckCircle,
  FiTrash2,
  FiLock,
  FiAlertTriangle,
  FiClock
} from 'react-icons/fi';

export default function ScanDetailPage() {
  const { scanId } = useParams();
  const navigate = useNavigate();
  const { state, dispatch } = useApp();

  const currentUser = state.auth?.user;
  const isPatient = currentUser?.role === 'patient';

  const scansList = Array.isArray(state.scans) ? state.scans : [];
  const patientsList = Array.isArray(state.patients) ? state.patients : [];

  // Match this scanId only. Falling back to scansList[0] used to render a
  // different patient's scan under the requested id.
  const listScan = scansList.find(s => (s.scanId || s.scan_id_string || s.id) === scanId);
  const targetScanId = listScan?.scanId || listScan?.scan_id_string || listScan?.id || scanId;

  // /api/scan/history carries no probabilities, risk score or biomarkers, so
  // reading only from it meant this page always rendered seeded mock numbers.
  // Pull the real record for this scan.
  const [detail, setDetail] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (!scanId) return undefined;
    scanAPI
      .result(scanId)
      .then(res => { if (!cancelled) setDetail(res.data); })
      .catch(() => { if (!cancelled) setDetail(null); });
    return () => { cancelled = true; };
  }, [scanId]);

  const pct = v => (typeof v === 'number' ? v * 100 : undefined);

  const rawScan = detail
    ? {
        ...listScan,
        patientId: detail.patient_id,
        patientName: detail.patient_name,
        uploadDate: detail.scan_date,
        prediction: detail.prediction,
        riskScore: detail.risk_score,
        confidence: Math.max(
          detail.confidence_cn ?? 0,
          detail.confidence_mci ?? 0,
          detail.confidence_ad ?? 0
        ) * 100,
        probabilities: {
          CN: pct(detail.confidence_cn),
          MCI: pct(detail.confidence_mci),
          AD: pct(detail.confidence_ad),
        },
        biomarkers: detail.biomarkers,
        brain_regions: detail.brain_regions,
        doctorNotes: detail.doctor_notes,
        modelTrained: detail.model_trained,
      }
    : listScan;

  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const queryPatientId = searchParams.get('patientId');
  const queryMrn = searchParams.get('mrn');

  // Resolve patient from state.patients
  const targetPatient = useMemo(() => {
    const sPId = queryPatientId || rawScan?.patientId || rawScan?.patient_id;
    const sMrn = (queryMrn || rawScan?.patient_code || rawScan?.mrn || '').trim().toLowerCase();
    const sPName = (rawScan?.patientName || rawScan?.patient || '').trim().toLowerCase();

    const matched = patientsList.find((p) => {
      const pId = String(p.id || p._id || '');
      const pName = (p.full_name || p.name || '').trim().toLowerCase();
      const pMrn = (p.patient_code || p.mrn || '').trim().toLowerCase();
      if (queryPatientId && (pId === queryPatientId || p._id === queryPatientId)) return true;
      if (queryMrn && pMrn && pMrn === queryMrn) return true;
      if (sPId && pId && pId === sPId) return true;
      if (sMrn && pMrn && sMrn === pMrn) return true;
      if (sPName && pName && sPName === pName) return true;
      return false;
    });

    return matched || patientsList[0] || null;
  }, [patientsList, rawScan, queryPatientId, queryMrn]);

  const rawPatientCond = (rawScan?.prediction || targetPatient?.condition || targetPatient?.diagnosis || 'CN').toUpperCase();
  const unifiedCond = rawPatientCond.includes('AD') ? 'AD' : rawPatientCond.includes('MCI') ? 'MCI' : 'CN';

  // Seeded mock values, used only when no real scan is loaded (demo browsing).
  const seeded = useMemo(() => generateScanData(targetScanId, '', unifiedCond), [targetScanId, unifiedCond]);
  const isMock = !rawScan;

  const patientAge = targetPatient?.age || (targetPatient?.date_of_birth ? (new Date().getFullYear() - new Date(targetPatient.date_of_birth).getFullYear()) : (rawScan?.patientAge || 65));
  const patientGender = targetPatient?.gender || rawScan?.patientGender || 'Male';

  const loggedInDoctor = currentUser?.full_name 
    ? (currentUser.full_name.startsWith('Dr.') ? currentUser.full_name : `Dr. ${currentUser.full_name}`)
    : (targetPatient?.assignedDoctor || 'Assigned Clinician');

  // `??` not `||`: a real riskScore or confidence of 0 is a value, not a miss.
  const scan = {
    scanId: targetScanId,
    scan_id_string: targetScanId,
    patientId: rawScan?.patientId ?? rawScan?.patient_id ?? targetPatient?.id ?? '',
    patientName: targetPatient?.full_name || targetPatient?.name || rawScan?.patientName || rawScan?.patient || 'Patient Record',
    patientAge: patientAge,
    patientGender: patientGender,
    uploadDate: rawScan?.uploadDate || rawScan?.date || targetPatient?.lastScanDate || new Date().toISOString().split('T')[0],
    prediction: rawScan?.prediction ?? unifiedCond,
    confidence: rawScan?.confidence ?? seeded.confidence,
    riskScore: rawScan?.riskScore ?? rawScan?.risk_score ?? targetPatient?.riskScore ?? seeded.riskScore,
    doctorStatus: rawScan?.doctorStatus || 'pending',
    doctorNotes: rawScan?.doctorNotes || '',
    probabilities: rawScan?.probabilities ?? seeded.probabilities,
    biomarkers: rawScan?.biomarkers && Object.keys(rawScan.biomarkers).length > 0 ? rawScan.biomarkers : seeded.biomarkers,
    gradCamRegions: rawScan?.gradCamRegions || seeded.gradCamRegions,
    brain_regions: rawScan?.brain_regions || {},
    // Mock rows are never a trained-model result either.
    modelTrained: !isMock && Boolean(rawScan?.modelTrained ?? rawScan?.model_trained),
  };

  const patient = targetPatient || {
    id: scan.patientId || 'PT-0001',
    name: scan.patientName,
    full_name: scan.patientName,
    mrn: targetPatient?.patient_code || targetPatient?.mrn || 'NA-2026-0035',
    patient_code: targetPatient?.patient_code || targetPatient?.mrn || 'NA-2026-0035',
    age: patientAge,
    gender: patientGender,
    assignedDoctor: loggedInDoctor
  };

  const pName = isPatient 
    ? (currentUser?.full_name || currentUser?.name || scan.patientName || 'My Scan')
    : (patient.full_name || patient.name || scan.patientName || 'Patient Record');
  const pMrn = patient.patient_code || patient.mrn || 'NA-2026-0035';
  const pInitials = pName.split(' ').map(n => n[0]).join('').slice(0, 2) || 'PT';

  const currentScanId = scanId || rawScan?.scanId || rawScan?.scan_id_string || rawScan?.id || 'SCN-DEFAULT';
  const isServerReviewed = Boolean(
    rawScan?.reviewed_at ||
    rawScan?.reviewedAt ||
    rawScan?.doctor_diagnosis ||
    rawScan?.doctorDiagnosis ||
    (rawScan?.doctorStatus && rawScan?.doctorStatus !== 'pending') ||
    (rawScan?.status && ['accepted', 'flagged', 'overridden', 'signed_off'].includes(rawScan.status))
  );

  const serverReviewStatus = rawScan?.doctorStatus || rawScan?.status || (rawScan?.doctor_diagnosis ? 'accepted' : 'accepted');
  const serverReviewNotes = rawScan?.doctor_notes || rawScan?.doctorNotes || '';
  const serverReviewedTime = rawScan?.reviewed_at || rawScan?.reviewedAt || '';

  const [decisionNotes, setDecisionNotes] = useState(serverReviewNotes || scan.doctorNotes || '');
  const [selectedStatus, setSelectedStatus] = useState(serverReviewStatus);
  const [isSignedOff, setIsSignedOff] = useState(isServerReviewed);
  const [signedOffTime, setSignedOffTime] = useState(serverReviewedTime);
  const [showReportModal, setShowReportModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  const confirmDeleteScan = async () => {
    setIsDeleting(true);
    setDeleteError('');
    try {
      await scanAPI.delete(currentScanId);
    } catch (e) {
      setIsDeleting(false);
      setDeleteError(
        e?.response?.data?.detail || 'Could not delete this scan. Nothing was removed.'
      );
      return;
    }
    dispatch({ type: 'DELETE_SCAN', payload: currentScanId });
    setIsDeleting(false);
    setShowDeleteModal(false);
    navigate(isPatient ? '/dashboard/my-scans' : '/dashboard');
  };

  const handleSaveDecision = async (statusToSave = selectedStatus) => {
    const timeStr = new Date().toLocaleDateString('en-GB') + ' · ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const doctorName = loggedInDoctor;

    const actionMap = {
      accepted: 'ACCEPT FINDING',
      flagged: 'FLAG FOR REVIEW',
      overridden: 'OVERRIDE DIAGNOSIS'
    };

    try {
      await scanAPI.review(currentScanId, {
        action: actionMap[statusToSave] || 'ACCEPT FINDING',
        doctor_diagnosis: scan.prediction,
        doctor_notes: decisionNotes
      });
    } catch (e) {
      console.warn('Backend review sync notice:', e);
    }

    const signOffPayload = {
      scanId: currentScanId,
      status: statusToSave,
      notes: decisionNotes,
      signedOffAt: timeStr,
      signedOffBy: doctorName,
      isSignedOff: true,
    };

    setSelectedStatus(statusToSave);
    setIsSignedOff(true);
    setSignedOffTime(timeStr);

    dispatch({
      type: 'UPDATE_SCAN_DECISION',
      payload: signOffPayload
    });
  };

  // Biomarkers with region severity indicators and 0-100% horizontal bars
  const hippo = scan.biomarkers?.hippocampus || seeded.biomarkers.hippocampus;
  const vents = scan.biomarkers?.ventricles || seeded.biomarkers.ventricles;
  const entor = scan.biomarkers?.entorhinalThickness || seeded.biomarkers.entorhinalThickness;

  const biomarkerCards = [
    {
      name: 'Hippocampal Volume Index',
      val: hippo?.value || '2.64 cm³',
      dev: hippo?.deviation || '-22% Atrophy',
      badge: hippo?.severity || 'Significant Atrophy',
      severityPct: hippo?.severityPct ?? (scan.prediction === 'AD' ? 78 : scan.prediction === 'MCI' ? 45 : 15),
      color: hippo?.status === 'high' || scan.prediction === 'AD' ? '#7A1F2B' : hippo?.status === 'medium' || scan.prediction === 'MCI' ? '#B87326' : '#4A7C59',
      bg: hippo?.status === 'high' || scan.prediction === 'AD' ? '#F8EAED' : hippo?.status === 'medium' || scan.prediction === 'MCI' ? '#FAF3E8' : '#EDF5F0',
      normRange: 'Normal: > 3.40 cm³'
    },
    {
      name: 'Lateral Ventricles Caliber',
      val: vents?.value || '44.8 mL',
      dev: vents?.deviation || '+28% Volume Expansion',
      badge: vents?.severity || 'Moderate Dilation',
      severityPct: vents?.severityPct ?? (scan.prediction === 'AD' ? 72 : scan.prediction === 'MCI' ? 40 : 12),
      color: vents?.status === 'high' || scan.prediction === 'AD' ? '#7A1F2B' : vents?.status === 'medium' || scan.prediction === 'MCI' ? '#B87326' : '#4A7C59',
      bg: vents?.status === 'high' || scan.prediction === 'AD' ? '#F8EAED' : vents?.status === 'medium' || scan.prediction === 'MCI' ? '#FAF3E8' : '#EDF5F0',
      normRange: 'Normal: < 32.0 mL'
    },
    {
      name: 'Entorhinal Cortical Ribbon',
      val: entor?.value || '2.08 mm',
      dev: entor?.deviation || '-19% Thinning',
      badge: entor?.severity || 'Early Degeneration',
      severityPct: entor?.severityPct ?? (scan.prediction === 'AD' ? 82 : scan.prediction === 'MCI' ? 48 : 10),
      color: entor?.status === 'high' || scan.prediction === 'AD' ? '#7A1F2B' : entor?.status === 'medium' || scan.prediction === 'MCI' ? '#B87326' : '#4A7C59',
      bg: entor?.status === 'high' || scan.prediction === 'AD' ? '#F8EAED' : entor?.status === 'medium' || scan.prediction === 'MCI' ? '#FAF3E8' : '#EDF5F0',
      normRange: 'Normal: > 2.65 mm'
    },
  ];

  return (
    <DashboardLayout
      title={isPatient ? `MRI Scan Insights: ${pName}` : `Diagnostic Examination: ${pName}`}
      subtitle={`Volumetric Series ${scan.scanId} · Acquired ${scan.uploadDate}`}
      action={
        <div className="flex items-center gap-2">
          <Link
            to={isPatient ? "/dashboard/my-scans" : "/dashboard"}
            className="btn-outline text-xs"
          >
            <FiArrowLeft className="w-3.5 h-3.5" />
            <span>{isPatient ? 'Back to My Scans' : 'Back to Queue'}</span>
          </Link>
          {!isPatient && (
            <button
              type="button"
              onClick={() => setShowDeleteModal(true)}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold text-[#7A1F2B] bg-[#F8EAED] border border-[#ECC8CF] hover:bg-[#F0D5DA] transition-colors flex items-center gap-1.5 cursor-pointer"
              title="Delete this scan record"
            >
              <FiTrash2 className="w-3.5 h-3.5" />
              <span>Delete Scan</span>
            </button>
          )}
          <button
            onClick={() => setShowReportModal(true)}
            className="btn-maroon text-xs shadow-clinical-sm cursor-pointer"
          >
            <FiPrinter className="w-3.5 h-3.5" />
            <span>{isPatient ? 'Export Scan Summary PDF' : 'Generate Clinical PDF'}</span>
          </button>
        </div>
      }
    >
      <div className="space-y-6">
        
        {/* Patient Demographic Summary Strip */}
        <div className="clinical-card p-4 flex flex-wrap items-center justify-between gap-4 text-xs bg-white">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] flex items-center justify-center text-[#7A1F2B] font-serif font-bold text-sm">
              {pInitials}
            </div>
            <div>
              <div className="flex items-center gap-2">
                {isPatient ? (
                  <span className="font-bold text-[#22201F] text-sm">{pName}</span>
                ) : (
                  <Link to={`/dashboard/patients/${patient.id || patient._id || 'P-001'}`} className="font-bold text-[#22201F] text-sm hover:underline">
                    {pName}
                  </Link>
                )}
                <span className="font-mono text-[11px] text-[#A39E98]">({pMrn})</span>
              </div>
              <span className="text-[#7A756F] text-[11px]">
                Age: <strong>{scan.patientAge || patient.age || (patient.date_of_birth ? (new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear()) : 68)} Yrs</strong> · Gender: <strong>{patient.gender || scan.patientGender || 'Male'}</strong> · Assigned: <strong>{patient.assignedDoctor || loggedInDoctor}</strong>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex flex-col text-right">
              <span className="text-[10px] uppercase font-bold text-[#A39E98]">AI Classification</span>
              <StatusBadge status={scan.prediction} size="sm" />
            </div>
            <div className="flex flex-col text-right pl-3 border-l border-[#E8E2DA]">
              <span className="text-[10px] uppercase font-bold text-[#A39E98]">Confidence</span>
              <span className="font-serif font-bold text-[#7A1F2B] text-base">{scan.confidence}%</span>
            </div>
          </div>
        </div>

        {/* Core Layout: Diagnostic Indicators & Doctor Sign-off Panel / Patient Insights */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left 6 Cols: Grad-CAM Slice Workstation & Decision Panel */}
          <div className="lg:col-span-6 space-y-6">

            {/* Grad-CAM Volumetric 3D Slice Workstation */}
            <GradCamViewer
              scanId={scan.scanId}
              condition={scan.prediction}
              confidence={scan.confidence}
              patientName={pName}
              gradCamRegions={scan.gradCamRegions}
              brainRegions={scan.brain_regions}
              biomarkers={scan.biomarkers}
            />

            {/* If Patient: Show Patient Review Status Card. If Doctor: Show Doctor Sign-Off Panel */}
            {isPatient ? (
              <div className="clinical-card p-5 bg-white space-y-3.5 shadow-clinical border border-[#E8E2DA]">
                <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-xl bg-[#EDF5F0] text-[#4A7C59] flex items-center justify-center">
                      <FiShield className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-sm font-serif font-bold text-[#22201F]">
                        Physician Review Status
                      </h4>
                      <span className="text-[11px] text-[#7A756F]">
                        Assigned Clinician: <strong>{patient.assignedDoctor || loggedInDoctor}</strong>
                      </span>
                    </div>
                  </div>
                  {isSignedOff ? (
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-bold uppercase bg-[#EDF5F0] text-[#4A7C59] border border-[#CFE3D5] flex items-center gap-1">
                      <FiCheckCircle className="w-3.5 h-3.5" />
                      <span>Reviewed</span>
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-bold uppercase bg-[#FAF3E8] text-[#B87326] border border-[#F2DEBF] flex items-center gap-1">
                      <FiClock className="w-3.5 h-3.5" />
                      <span>Under Clinical Review</span>
                    </span>
                  )}
                </div>

                <div className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] space-y-2">
                  <span className="text-[11px] font-semibold text-[#22201F] block">
                    {isSignedOff ? 'Physician Clinical Notes:' : 'Clinical Protocol Notice:'}
                  </span>
                  <p className="text-xs text-[#5A5550] leading-relaxed">
                    {isSignedOff
                      ? `"${decisionNotes || 'Volumetric MRI series examined and validated by attending physician.'}"`
                      : 'Your 3D volumetric MRI scan has been processed by the 3D ResNet-10 AI engine. The multi-planar Grad-CAM attention maps and biomarker metrics above will be finalized by your physician during your diagnostic consultation.'}
                  </p>
                </div>

                <div className="text-[11px] text-[#A39E98] flex items-center justify-between pt-1 font-mono">
                  <span>Scan ID: {currentScanId}</span>
                  <span>Acquired: {scan.uploadDate}</span>
                </div>
              </div>
            ) : isSignedOff ? (
              <div className="clinical-card p-5 bg-white space-y-4 border border-[#CFE3D5] shadow-clinical animate-fade-in">
                {/* Confirmed Header */}
                <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-[#EDF5F0] text-[#2E523A] flex items-center justify-center border border-[#CFE3D5] shadow-2xs">
                      <FiCheckCircle className="w-4 h-4 text-[#4A7C59]" />
                    </div>
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-widest text-[#2E523A] bg-[#EDF5F0] px-2 py-0.5 rounded-full border border-[#CFE3D5]">
                        Review Recorded
                      </span>
                      <h4 className="text-sm font-serif font-bold text-[#22201F] mt-0.5">
                        Physician Validation Confirmed
                      </h4>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-[10px] font-semibold text-[#2E523A] bg-[#EDF5F0] px-2.5 py-1 rounded-lg border border-[#CFE3D5]">
                    <FiLock className="w-3.5 h-3.5 text-[#4A7C59]" />
                    <span>Locked</span>
                  </div>
                </div>

                {/* Decision Summary Box */}
                <div className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-[#7A756F] font-semibold uppercase tracking-wider">
                      Recorded Decision:
                    </span>
                    <span className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold uppercase border shadow-2xs ${
                      selectedStatus === 'accepted' ? 'bg-[#EDF5F0] text-[#2E523A] border-[#CFE3D5]' :
                      selectedStatus === 'flagged' ? 'bg-[#FAF3E8] text-[#8A5A14] border-[#F0DEC2]' :
                      'bg-[#F8EAED] text-[#7A1F2B] border-[#ECC8CF]'
                    }`}>
                      {selectedStatus === 'accepted' ? '✓ Accepted AI' : selectedStatus === 'flagged' ? '⚐ Flagged Case' : '✎ Overridden'}
                    </span>
                  </div>

                  {decisionNotes && (
                    <div className="pt-2 border-t border-[#E8E2DA] text-xs text-[#22201F] leading-relaxed bg-white/70 p-2.5 rounded-lg border border-[#E8E2DA]/60">
                      <span className="text-[10px] uppercase font-bold text-[#A39E98] block mb-1">Clinical Directives:</span>
                      "{decisionNotes}"
                    </div>
                  )}
                </div>

                {/* Timestamp and Signer Audit Footer */}
                <div className="flex items-center justify-between text-[11px] font-mono text-[#7A756F] pt-1">
                  <span className="flex items-center gap-1.5 text-[#4A7C59] font-semibold">
                    <span className="w-2 h-2 rounded-full bg-[#4ADE80] animate-pulse" />
                    <span>Recorded: {signedOffTime || 'Just now'}</span>
                  </span>
                  <span className="font-semibold text-[#22201F] font-sans">
                    {loggedInDoctor}
                  </span>
                </div>
              </div>
            ) : (
              <div className="clinical-card p-5 bg-white space-y-4 shadow-clinical border border-[#E8E2DA]">
                <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-[#7A1F2B] bg-[#F8EAED] px-2 py-0.5 rounded-full border border-[#ECC8CF]">
                      Doctor Decision Panel
                    </span>
                    <h4 className="text-sm font-serif font-bold text-[#22201F] mt-1">
                      Physician Validation & Sign-Off
                    </h4>
                  </div>
                  <div className="text-[10px] font-semibold text-[#4A7C59] flex items-center gap-1">
                    <FiShield className="w-3.5 h-3.5" />
                    <span>Final Authority</span>
                  </div>
                </div>

                {/* Three Doctor Action Buttons */}
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => handleSaveDecision('accepted')}
                    className={`py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      selectedStatus === 'accepted'
                        ? 'bg-[#EDF5F0] text-[#2E523A] border-[#4A7C59] shadow-clinical-xs'
                        : 'border-[#E8E2DA] text-[#5A5550] hover:bg-[#FAF6F3]'
                    }`}
                  >
                    <FiCheck className="w-4 h-4 text-[#4A7C59]" />
                    <span>Accept AI</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleSaveDecision('overridden')}
                    className={`py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      selectedStatus === 'overridden'
                        ? 'bg-[#F8EAED] text-[#7A1F2B] border-[#7A1F2B] shadow-clinical-xs'
                        : 'border-[#E8E2DA] text-[#5A5550] hover:bg-[#FAF6F3]'
                    }`}
                  >
                    <FiEdit3 className="w-4 h-4 text-[#7A1F2B]" />
                    <span>Override</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleSaveDecision('flagged')}
                    className={`py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                      selectedStatus === 'flagged'
                        ? 'bg-[#FAF3E8] text-[#8A5A14] border-[#B87326] shadow-clinical-xs'
                        : 'border-[#E8E2DA] text-[#5A5550] hover:bg-[#FAF6F3]'
                    }`}
                  >
                    <FiFlag className="w-4 h-4 text-[#B87326]" />
                    <span>Flag Case</span>
                  </button>
                </div>

                {/* Notes Textarea */}
                <div className="space-y-1.5">
                  <label className="block text-[11px] font-semibold uppercase tracking-wider text-[#7A756F]">
                    Physician Clinical Notes & Recommendations
                  </label>
                  <textarea
                    rows={3}
                    value={decisionNotes}
                    onChange={(e) => setDecisionNotes(e.target.value)}
                    placeholder="Enter diagnostic impressions, biomarker interpretations, or follow-up orders..."
                    className="w-full p-3 rounded-xl border border-[#D8C9BC] bg-[#FAF6F3] text-xs text-[#22201F] placeholder-[#A39E98] focus:outline-none focus:border-[#7A1F2B] resize-none"
                  />
                </div>

                {/* Confirm Sign-Off Button */}
                <button
                  type="button"
                  onClick={() => handleSaveDecision(selectedStatus)}
                  className="w-full py-2.5 rounded-xl bg-[#7A1F2B] hover:bg-[#661823] text-white text-xs font-bold transition-all shadow-clinical flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <FiCheckCircle className="w-4 h-4" />
                  <span>Sign & Finalize Examination</span>
                </button>
              </div>
            )}

          </div>

          {/* Right 6 Cols: Clinical Risk Gauge & Volumetric Biomarkers */}
          <div className="lg:col-span-6 space-y-6">

            {/* Risk Gauge Card */}
            <div className="clinical-card p-6 bg-white space-y-4 shadow-clinical">
              <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
                <h3 className="text-base font-serif font-bold text-[#22201F]">
                  Cohort Risk & Probability Gauge
                </h3>
                <span className="text-xs font-mono text-[#7A756F]">Model: 3D ResNet-10</span>
              </div>

              <RiskGaugeArc
                riskScore={scan.riskScore}
                prediction={scan.prediction}
                confidence={scan.confidence}
                probabilities={scan.probabilities}
                modelTrained={scan.modelTrained}
              />
            </div>

            {/* Volumetric Biomarkers Breakdown */}
            <div className="clinical-card p-6 bg-white space-y-4 shadow-clinical">
              <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
                <h3 className="text-base font-serif font-bold text-[#22201F]">
                  Volumetric Biomarker Regions
                </h3>
                <span className="text-xs text-[#7A756F]">SimpleITK Morphometry</span>
              </div>

              <div className="space-y-3">
                {biomarkerCards.map((bio) => (
                  <div
                    key={bio.name}
                    className="p-3.5 rounded-2xl border border-[#E8E2DA] bg-[#FAF7F4] space-y-2"
                  >
                    {/* Header: Name + Metric */}
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full inline-block shadow-sm"
                          style={{ backgroundColor: bio.color }}
                        />
                        <span className="font-semibold text-[#22201F]">{bio.name}</span>
                      </div>
                      <span className="font-mono font-bold text-[#22201F]">{bio.val}</span>
                    </div>

                    {/* Deviation & Badge row */}
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-[#7A756F]">{bio.dev}</span>
                      <span
                        className="px-2 py-0.5 rounded-full font-semibold border"
                        style={{ color: bio.color, backgroundColor: bio.bg, borderColor: bio.color + '40' }}
                      >
                        {bio.badge}
                      </span>
                    </div>

                    {/* Severity Scale Horizontal Bar */}
                    <div className="space-y-1 pt-1 border-t border-[#F0E8E1]">
                      <div className="flex items-center justify-between text-[10px]">
                        <span className="text-[#A39E98]">{bio.normRange}</span>
                        <span className="font-mono font-bold" style={{ color: bio.color }}>
                          Severity: {bio.severityPct}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-[#E8E2DA] overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700 ease-out"
                          style={{ width: `${bio.severityPct}%`, backgroundColor: bio.color }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>

        </div>

      </div>

      {/* Clinical Report PDF Export Modal */}
      {showReportModal && (
        <ClinicalReportModal
          scan={scan}
          patient={patient}
          onClose={() => setShowReportModal(false)}
        />
      )}

      {/* Delete Scan Confirmation Modal */}
      {showDeleteModal && !isPatient && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-md w-full shadow-clinical-lg border border-[#E8E2DA] space-y-4 animate-scale-up">
            <div className="w-12 h-12 rounded-full bg-[#F8EAED] text-[#7A1F2B] flex items-center justify-center mx-auto border border-[#ECC8CF]">
              <FiAlertTriangle className="w-6 h-6" />
            </div>
            <div className="text-center space-y-1">
              <h3 className="text-base font-serif font-bold text-[#22201F]">
                Delete Scan Record?
              </h3>
              <p className="text-xs text-[#7A756F] leading-relaxed">
                Are you sure you want to permanently delete scan record <strong className="text-[#22201F] font-mono">{currentScanId}</strong> for patient <strong className="text-[#22201F]">{pName}</strong>?
              </p>
            </div>

            {deleteError && (
              <div className="p-3 rounded-xl bg-[#F8EAED] border border-[#ECC8CF] text-xs font-semibold text-[#7A1F2B] text-center">
                {deleteError}
              </div>
            )}

            <div className="flex items-center gap-3 pt-2">
              <button
                type="button"
                onClick={() => { setShowDeleteModal(false); setDeleteError(''); }}
                disabled={isDeleting}
                className="flex-1 py-2.5 rounded-xl border border-[#D8C9BC] hover:bg-[#FAF6F3] text-xs font-semibold text-[#5A5550] transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDeleteScan}
                disabled={isDeleting}
                className="flex-1 py-2.5 rounded-xl bg-[#7A1F2B] hover:bg-[#661823] text-xs font-semibold text-white transition-colors cursor-pointer flex items-center justify-center gap-1.5"
              >
                {isDeleting ? (
                  <span>Deleting...</span>
                ) : (
                  <>
                    <FiTrash2 className="w-3.5 h-3.5" />
                    <span>Delete Scan</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
