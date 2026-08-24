import React, { useState, useMemo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { patientAPI } from '../services/api';
import DashboardLayout from '../components/layout/DashboardLayout';
import StatusBadge from '../components/common/StatusBadge';
import AddPatientModal from '../components/clinical/AddPatientModal';
import { 
  FiSearch, 
  FiArrowRight, 
  FiCalendar, 
  FiUserPlus,
  FiUploadCloud,
  FiTrash2,
  FiAlertTriangle,
  FiX
} from 'react-icons/fi';

export default function PatientsDirectoryPage() {
  const { state, dispatch } = useApp();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const initialQuery = queryParams.get('q') || '';

  const [search, setSearch] = useState(initialQuery);
  const [filterCondition, setFilterCondition] = useState('ALL'); // 'ALL' | 'CN' | 'MCI' | 'AD' | 'FLAGGED'
  const [isAddPatientOpen, setIsAddPatientOpen] = useState(false);
  const [deletePatientTarget, setDeletePatientTarget] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  const filteredPatients = useMemo(() => {
    const rawPatients = state.patients;
    const patientsList = (Array.isArray(rawPatients) ? rawPatients : (rawPatients?.patients || rawPatients?.items || [])).filter(p => {
      const name = (p.full_name || p.name || '').trim().toLowerCase();
      return !name.includes('demo') && !name.includes('arthur pendelton') && !name.includes('helen mirren');
    });
    const allScans = Array.isArray(state.scans) ? state.scans : [];

    return patientsList.map((p) => {
      const pId = p.id || p._id;
      const name = p.name || p.full_name || 'Patient Record';
      const mrn = p.mrn || p.patient_code || pId;

      // Match patient scans by unique patient ID, MRN, or Patient Name
      const patientScans = allScans.filter((s) => {
        const sPId = s.patientId || s.patient_id;
        const sMrn = s.patient_code || s.mrn;
        const sPName = (s.patientName || s.patient || '').trim().toLowerCase();
        const targetName = name.trim().toLowerCase();
        if (sPId && sPId === pId) return true;
        if (sMrn && (sMrn === mrn || sMrn === p.patient_code)) return true;
        if (sPName && targetName && sPName === targetName) return true;
        return false;
      });
      const latestScan = patientScans.length > 0 ? patientScans[0] : null;
      const rawCondition = (latestScan?.prediction || latestScan?.condition || p.condition || p.diagnosis || '').toUpperCase();
      const realCondition = rawCondition.includes('AD') ? 'AD' : rawCondition.includes('MCI') ? 'MCI' : 'CN';
      const defaultRisk = realCondition === 'AD' ? 82 : realCondition === 'MCI' ? 52 : 18;
      const realRisk = latestScan?.riskScore ?? latestScan?.risk_score ?? p.riskScore ?? p.risk_score ?? defaultRisk;
      const isUrgent = p.urgentFlag || realCondition === 'AD' || realRisk >= 75;

      const pAge = p.age || (p.date_of_birth ? new Date().getFullYear() - new Date(p.date_of_birth).getFullYear() : (p.age || 65));
      const pGender = p.gender || '—';
      const pMmse = p.mmseScore ?? p.mmse ?? (realCondition === 'AD' ? 17 : realCondition === 'MCI' ? 24 : 29);

      return {
        ...p,
        _id: pId,
        id: pId,
        full_name: name,
        patient_code: mrn,
        condition: realCondition,
        riskScore: realRisk,
        mmseScore: pMmse,
        age: pAge,
        gender: pGender,
        isUrgent,
        patientScans,
        latestScan,
      };
    }).filter((p) => {
      const matchSearch =
        p.full_name.toLowerCase().includes(search.toLowerCase()) ||
        p.patient_code.toLowerCase().includes(search.toLowerCase()) ||
        String(p.id).toLowerCase().includes(search.toLowerCase());

      if (!matchSearch) return false;

      if (filterCondition === 'ALL') return true;
      if (filterCondition === 'FLAGGED') return p.isUrgent;
      return p.condition === filterCondition;
    });
  }, [state.patients, state.scans, search, filterCondition]);

  const handleDeletePatient = async () => {
    if (!deletePatientTarget) return;
    const targetId = deletePatientTarget.id || deletePatientTarget._id;
    setIsDeleting(true);
    setDeleteError('');
    try {
      await patientAPI.delete(targetId);
    } catch (e) {
      setIsDeleting(false);
      setDeleteError(
        e?.response?.data?.detail || 'Could not delete this patient. Nothing was removed.'
      );
      return;
    }

    dispatch({ type: 'DELETE_PATIENT', payload: targetId });
    setIsDeleting(false);
    setDeletePatientTarget(null);
  };

  return (
    <DashboardLayout
      title="Patient Directory & Longitudinal Registry"
      subtitle="Longitudinal tracking of cognitive cohorts, MMSE trajectory, and volumetric MRI history."
      action={
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsAddPatientOpen(true)}
            className="btn-maroon text-xs shadow-clinical-sm flex items-center gap-1.5"
          >
            <FiUserPlus className="w-4 h-4" />
            <span>Add New Patient</span>
          </button>
          <Link
            to="/dashboard/scan"
            className="btn-outline text-xs hidden sm:inline-flex items-center gap-1.5"
          >
            <FiUploadCloud className="w-4 h-4" />
            <span>Upload Scan</span>
          </Link>
        </div>
      }
    >
      <div className="space-y-6">
        
        {/* Controls Bar: Search & Filter Tabs */}
        <div className="clinical-card p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          
          {/* Search Input */}
          <div className="relative w-full sm:w-80">
            <FiSearch className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A39E98]" />
            <input
              type="text"
              placeholder="Search by patient name, MRN, or ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-[#FAF6F3] border border-[#E8E2DA] rounded-xl text-xs sm:text-sm text-[#22201F] focus:bg-white focus:outline-none focus:border-[#7A1F2B]"
            />
          </div>

          {/* Condition Filter Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 bg-[#FAF6F3] rounded-xl border border-[#E8E2DA] w-full sm:w-auto">
            {[
              { id: 'ALL', label: 'All Patients' },
              { id: 'CN', label: 'CN (Normal)' },
              { id: 'MCI', label: 'MCI (Mild)' },
              { id: 'AD', label: 'AD (Alzheimer’s)' },
              { id: 'FLAGGED', label: 'Urgent Flags' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setFilterCondition(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  filterCondition === tab.id
                    ? 'bg-white text-[#7A1F2B] shadow-clinical-sm border border-[#E8E2DA]'
                    : 'text-[#7A756F] hover:text-[#22201F]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

        </div>

        {/* Patients Table Card */}
        <div className="clinical-card p-4 sm:p-6">
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#E8E2DA] text-[#7A756F] uppercase tracking-wider font-semibold text-[10px]">
                  <th className="py-2.5 px-2.5">Patient / MRN</th>
                  <th className="py-2.5 px-2.5">Demographics</th>
                  <th className="py-2.5 px-2.5">Status</th>
                  <th className="py-2.5 px-2.5">Risk Score</th>
                  <th className="py-2.5 px-2.5">MMSE</th>
                  <th className="py-2.5 px-2.5">Last Scan</th>
                  <th className="py-2.5 px-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F7F1EC]">
                {filteredPatients.map((patient) => {
                  const pId = patient.id || patient._id;
                  const pName = patient.full_name || patient.name || 'Patient Record';
                  const pCode = patient.patient_code || patient.mrn || pId;
                  const pInitials = pName.split(' ').map((n) => n[0]).join('').slice(0, 2) || 'PT';

                  const patientScans = patient.patientScans || [];
                  const latestScan = patient.latestScan;
                  const pCond = patient.condition;
                  const pRisk = patient.riskScore;
                  const pMmse = patient.mmseScore;
                  const pAge = patient.age;
                  const pGender = patient.gender;
                  const pScansCount = Math.max(patientScans.length, patient.scansCount ?? patient.scan_count ?? 0);
                  const rawDateVal = latestScan?.uploadDate || latestScan?.date || patient.lastScanDate || patient.created_at || 'Recent';
                  const formattedScanDate = (() => {
                    if (!rawDateVal || rawDateVal === 'Recent') return 'Recent';
                    try {
                      if (typeof rawDateVal === 'string' && rawDateVal.includes('T')) {
                        return rawDateVal.split('T')[0];
                      }
                      const d = new Date(rawDateVal);
                      return !isNaN(d.getTime()) ? d.toISOString().split('T')[0] : String(rawDateVal);
                    } catch {
                      return String(rawDateVal);
                    }
                  })();

                  return (
                    <tr key={pId} className="hover:bg-[#FAF6F3] transition-colors group">
                      <td className="py-3 px-2.5">
                        <div className="flex items-center gap-2.5">
                          <div className="w-8 h-8 rounded-lg bg-[#F0E8E1] border border-[#D8C9BC] flex items-center justify-center font-serif font-bold text-[#7A1F2B] text-xs flex-shrink-0">
                            {pInitials}
                          </div>
                          <div className="flex flex-col">
                            <Link
                              to={`/dashboard/patients/${pId}`}
                              className="font-bold text-[#22201F] text-xs hover:text-[#7A1F2B] transition-colors whitespace-nowrap"
                            >
                              {pName}
                            </Link>
                            <span className="text-[10px] text-[#A39E98] font-mono whitespace-nowrap">
                              {pCode}
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-2.5 text-[#7A756F] whitespace-nowrap">
                        <span>{pAge} Yrs · {pGender}</span>
                      </td>
                      <td className="py-3 px-2.5 whitespace-nowrap">
                        <StatusBadge status={pCond} size="xs" short={true} />
                      </td>
                      <td className="py-3 px-2.5 font-serif font-bold text-xs whitespace-nowrap">
                        <span className={pRisk >= 70 ? 'text-[#7A1F2B]' : pRisk >= 40 ? 'text-[#B87326]' : 'text-[#4A7C59]'}>
                          {pRisk}
                        </span>
                        <span className="text-[10px] text-[#A39E98] font-sans font-normal"> / 100</span>
                      </td>
                      <td className="py-3 px-2.5 whitespace-nowrap">
                        <span className="font-mono font-bold text-[#22201F]">{pMmse}</span>
                        {pMmse !== '—' && <span className="text-[10px] text-[#A39E98]"> / 30</span>}
                      </td>
                      <td className="py-3 px-2.5 text-[#7A756F] whitespace-nowrap">
                        <div className="flex items-center gap-1 text-[11px]">
                          <FiCalendar className="w-3 h-3 text-[#A39E98] flex-shrink-0" />
                          <span>{formattedScanDate} ({pScansCount})</span>
                        </div>
                      </td>
                      <td className="py-3 px-2.5 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end gap-1">
                          <Link
                            to={`/dashboard/patients/${pId}`}
                            className="btn-outline text-[11px] py-1 px-2.5 inline-flex items-center gap-1 group-hover:border-[#7A1F2B]"
                          >
                            <span>Profile</span>
                            <FiArrowRight className="w-3 h-3" />
                          </Link>
                          <button
                            type="button"
                            onClick={() => setDeletePatientTarget(patient)}
                            title="Delete Patient Record"
                            className="p-1 rounded-lg text-[#A39E98] hover:text-[#7A1F2B] hover:bg-[#F8EAED] transition-colors"
                          >
                            <FiTrash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
                {filteredPatients.length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-xs text-[#A39E98]">
                      No patient records match the selected filter query.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {/* Add Patient Modal */}
      <AddPatientModal
        isOpen={isAddPatientOpen}
        onClose={() => setIsAddPatientOpen(false)}
      />

      {/* Delete Patient Confirmation Modal */}
      {deletePatientTarget && (
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
                onClick={() => { setDeletePatientTarget(null); setDeleteError(''); }}
                className="p-1 text-[#A39E98] hover:text-[#22201F]"
              >
                <FiX className="w-4 h-4" />
              </button>
            </div>

            <div className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] text-xs space-y-1">
              <div className="font-bold text-[#22201F]">
                {deletePatientTarget.full_name || deletePatientTarget.name}
              </div>
              <div className="text-[#7A756F] font-mono text-[11px]">
                MRN: {deletePatientTarget.patient_code || deletePatientTarget.mrn || deletePatientTarget.id}
              </div>
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
                onClick={() => { setDeletePatientTarget(null); setDeleteError(''); }}
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


