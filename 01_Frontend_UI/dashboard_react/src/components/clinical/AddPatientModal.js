import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { patientAPI } from '../../services/api';
import { FiX, FiUser, FiCheckCircle, FiLock, FiShield } from 'react-icons/fi';

/**
 * AddPatientModal — Clinical modal form for registering a new patient into the longitudinal registry.
 * - Auto-generates patient code (MRN) as read-only / locked.
 * - Auto-assigns the logged-in Neurologist as read-only.
 */
export default function AddPatientModal({ isOpen, onClose, onSuccess, redirectToProfile = true }) {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();

  const currentUser = state.auth?.user;
  const defaultDoctor = currentUser?.full_name 
    ? (currentUser.full_name.startsWith('Dr.') ? currentUser.full_name : `Dr. ${currentUser.full_name}`)
    : 'Dr. Sarah Lin, MD';

  // Generate random 4-digit ID
  const randomSuffix = Math.floor(1000 + Math.random() * 9000);
  const autoPatientCode = `NA-2026-${randomSuffix}`;

  const [formData, setFormData] = useState({
    fullName: '',
    age: '',
    gender: 'Female',
    condition: 'CN',
    clinicalNotes: '',
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedName = formData.fullName.trim();
    if (!trimmedName) {
      setError('Patient full name is required.');
      return;
    }

    // Block duplicate patient names in registry (case-insensitive check)
    const existingPatients = Array.isArray(state.patients) ? state.patients : [];
    const isDuplicateName = existingPatients.some(
      (p) => (p.full_name || p.name || '').trim().toLowerCase() === trimmedName.toLowerCase()
    );
    if (isDuplicateName) {
      setError(`A patient with the name "${trimmedName}" is already registered. Please provide a distinct full name.`);
      return;
    }

    if (!formData.age || parseInt(formData.age, 10) < 1 || parseInt(formData.age, 10) > 120) {
      setError('Please enter a valid age (1–120).');
      return;
    }
    if (!formData.gender) {
      setError('Please select patient gender.');
      return;
    }

    setLoading(true);
    setError('');

    const newPatientId = `PT-${randomSuffix}`;
    const ageNum = parseInt(formData.age, 10);
    const dobYear = new Date().getFullYear() - ageNum;
    const dob = `${dobYear}-06-15`;

    const chosenCond = formData.condition || 'CN';

    const conditionStages = {
      CN: 'Cognitively Normal',
      MCI: 'Mild Cognitive Impairment',
      AD: "Alzheimer's Disease",
    };

    const initialRisk = chosenCond === 'AD' ? 82 : chosenCond === 'MCI' ? 52 : 18;
    const initialMmse = chosenCond === 'AD' ? 17 : chosenCond === 'MCI' ? 24 : 29;

    const newPatientObj = {
      id: newPatientId,
      _id: newPatientId,
      full_name: trimmedName,
      name: trimmedName,
      patient_code: autoPatientCode,
      mrn: autoPatientCode,
      date_of_birth: dob,
      age: ageNum,
      gender: formData.gender,
      condition: chosenCond,
      stage: conditionStages[chosenCond],
      diagnosis: chosenCond,
      assignedDoctor: defaultDoctor,
      doctorNotes: formData.clinicalNotes.trim(),
      riskScore: initialRisk,
      mmseScore: initialMmse,
      scan_count: 0,
      scansCount: 0,
      created_at: new Date().toISOString(),
      history: [
        {
          date: new Date().toISOString().slice(0, 7),
          riskScore: initialRisk,
          mmse: initialMmse,
          label: chosenCond,
        },
      ],
    };

    try {
      // Attempt backend persistence
      const res = await patientAPI.create({
        full_name: newPatientObj.full_name,
        patient_code: newPatientObj.patient_code,
        date_of_birth: dob,
        gender: newPatientObj.gender,
        diagnosis: newPatientObj.condition,
        medical_history: newPatientObj.doctorNotes,
      });
      if (res?.data?.id) {
        newPatientObj.id = res.data.id;
        newPatientObj._id = res.data.id;
        if (res.data.patient_code) {
          newPatientObj.patient_code = res.data.patient_code;
          newPatientObj.mrn = res.data.patient_code;
        }
      }
    } catch (err) {
      console.warn('Backend call failed, using client state:', err);
    } finally {
      // Add to client state
      dispatch({ type: 'ADD_PATIENT', payload: newPatientObj });
      setLoading(false);
      onClose();
      if (onSuccess) {
        onSuccess(newPatientObj);
      }
      if (redirectToProfile) {
        navigate(`/dashboard/patients/${newPatientObj.id || newPatientId}`);
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-3xl border border-[#E8E2DA] shadow-2xl w-full max-w-lg overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="p-6 border-b border-[#E8E2DA] flex items-center justify-between bg-[#FAF6F3]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-white border border-[#E8E2DA] text-[#7A1F2B] flex items-center justify-center shadow-clinical-sm">
              <FiUser className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-serif font-bold text-[#22201F]">Register New Patient</h3>
              <p className="text-xs text-[#7A756F]">Add a patient record to your longitudinal clinical registry</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-xl bg-white border border-[#E8E2DA] flex items-center justify-center text-[#7A756F] hover:text-[#22201F] hover:bg-[#F0E8E1] transition-colors"
          >
            <FiX className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4 text-xs overflow-y-auto flex-1">
          {error && (
            <div className="p-3.5 rounded-xl bg-[#F8EAED] border border-[#ECC8CF] text-[#7A1F2B] text-xs flex items-center gap-2">
              <span className="font-bold">Error:</span> {error}
            </div>
          )}

          {/* Full Name */}
          <div className="space-y-1.5">
            <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
              Patient Full Name *
            </label>
            <input
              type="text"
              name="fullName"
              required
              placeholder="e.g. Eleanor Vance"
              value={formData.fullName}
              onChange={handleChange}
              className="clinical-input w-full"
            />
          </div>

          {/* Age & Gender Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
                Age (Years) *
              </label>
              <input
                type="number"
                name="age"
                required
                min="1"
                max="120"
                placeholder="e.g. 72"
                value={formData.age}
                onChange={handleChange}
                className="clinical-input w-full"
              />
            </div>

            <div className="space-y-1.5">
              <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
                Gender
              </label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="clinical-input w-full cursor-pointer"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other / Non-binary</option>
              </select>
            </div>
          </div>

          {/* Baseline Cognitive Status / Diagnosis */}
          <div className="space-y-1.5">
            <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
              Baseline Cognitive Classification *
            </label>
            <select
              name="condition"
              value={formData.condition}
              onChange={handleChange}
              className="clinical-input w-full cursor-pointer font-medium"
            >
              <option value="CN">Cognitively Normal (CN) — Low Risk</option>
              <option value="MCI">Mild Cognitive Impairment (MCI) — Moderate Risk</option>
              <option value="AD">Alzheimer's Disease (AD) — High Risk</option>
            </select>
          </div>



          {/* Auto-generated MRN & Assigned Neurologist (Read-only / Locked) */}
          <div className="grid grid-cols-2 gap-4">
            {/* Auto Generated MRN */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
                  Medical Record ID (MRN)
                </label>
                <span className="text-[10px] text-[#4A7C59] font-semibold flex items-center gap-0.5">
                  <FiLock className="w-3 h-3" /> Auto
                </span>
              </div>
              <div className="px-3 py-2 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] font-mono text-xs font-bold text-[#22201F] flex items-center justify-between select-none">
                <span>{autoPatientCode}</span>
                <span className="text-[10px] uppercase font-semibold text-[#A39E98] tracking-wider">Generated</span>
              </div>
            </div>

            {/* Assigned Neurologist (Read-only) */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
                  Assigned Neurologist
                </label>
                <span className="text-[10px] text-[#5B7C99] font-semibold flex items-center gap-0.5">
                  <FiShield className="w-3 h-3" /> Active
                </span>
              </div>
              <div className="px-3 py-2 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] text-xs font-semibold text-[#22201F] flex items-center justify-between select-none">
                <span>{defaultDoctor}</span>
                <span className="text-[10px] uppercase font-semibold text-[#A39E98] tracking-wider">Logged In</span>
              </div>
            </div>
          </div>

          {/* Clinical Notes / History */}
          <div className="space-y-1.5">
            <label className="block font-semibold uppercase tracking-wider text-[#7A756F]">
              Baseline Medical Notes & History
            </label>
            <textarea
              name="clinicalNotes"
              rows={3}
              placeholder="Enter patient medical background, family history, MMSE score observations..."
              value={formData.clinicalNotes}
              onChange={handleChange}
              className="clinical-input w-full resize-none"
            />
          </div>

          {/* Submit Actions */}
          <div className="pt-3 flex items-center justify-end gap-3 border-t border-[#E8E2DA]">
            <button
              type="button"
              onClick={onClose}
              className="btn-outline text-xs"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn-maroon text-xs shadow-clinical-sm flex items-center gap-1.5"
            >
              {loading ? (
                <span>Registering...</span>
              ) : (
                <>
                  <FiCheckCircle className="w-3.5 h-3.5" />
                  <span>Create Patient Record</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
