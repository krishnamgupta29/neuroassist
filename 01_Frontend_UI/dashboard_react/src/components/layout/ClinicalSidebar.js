import React, { useMemo } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useApp, getStoredDecisions } from '../../context/AppContext';
import { generateScanData } from '../../utils/mockDataGenerator';
import { 
  FiGrid, 
  FiUploadCloud, 
  FiUsers, 
  FiSliders, 
  FiShield, 
  FiLogOut,
  FiFileText
} from 'react-icons/fi';

export default function ClinicalSidebar() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const user = state.auth?.user;
  const isPatient = user?.role === 'patient';
  const userDecisions = useMemo(() => getStoredDecisions(user), [user]);

  const rawPatients = state.patients;
  const patients = Array.isArray(rawPatients) ? rawPatients : (rawPatients?.patients || rawPatients?.items || []);
  const patientById = {};
  const patientByName = {};
  patients.forEach(p => {
    if (p.id || p._id) patientById[p.id || p._id] = p;
    const nm = (p.full_name || p.name || '').toLowerCase();
    if (nm) patientByName[nm] = p;
  });

  const rawScans = Array.isArray(state.scans) ? state.scans : (state.scans?.items || []);
  const uniqueScans = [];
  const seenPatients = new Set();
  const seenScanIds = new Set();

  for (const s of rawScans) {
    const scanId = s.scanId || s.scan_id_string || s.id;
    const pId = s.patientId || s.patient_id;
    const pName = (s.patientName || s.patient || s.patient_name || '').trim().toLowerCase();
    const resolvedPatient = (pId && patientById[pId]) || (pName && patientByName[pName]);

    if (!resolvedPatient && !pName && !scanId) continue;

    const patientKey = (resolvedPatient?.id || resolvedPatient?._id || pId || pName || scanId || 'unknown').toLowerCase();

    // Ensure only 1 entry per unique patient in the sidebar quick access
    if (scanId && !seenScanIds.has(scanId) && !seenPatients.has(patientKey)) {
      seenScanIds.add(scanId);
      seenPatients.add(patientKey);

      const storedDec = userDecisions[scanId] || (resolvedPatient?.id ? userDecisions[resolvedPatient.id] : null) || (resolvedPatient?._id ? userDecisions[resolvedPatient._id] : null);
      const seeded = generateScanData(scanId);

      const rawPred = (storedDec?.prediction || s.prediction || (s.condition && s.condition !== 'CN' ? s.condition : null) || (resolvedPatient?.condition && resolvedPatient.condition !== 'CN' ? resolvedPatient.condition : null) || seeded.prediction || s.condition || 'CN').toUpperCase();
      const pred = rawPred.includes('AD') ? 'AD' : rawPred.includes('MCI') ? 'MCI' : 'CN';
      uniqueScans.push({
        ...s,
        scanId: scanId,
        prediction: pred,
        _resolvedName: resolvedPatient?.full_name || resolvedPatient?.name || s.patientName || s.patient || s.patient_name || 'Patient'
      });
    }
  }
  const scans = uniqueScans;
  const pendingCount = (Array.isArray(state.scans) ? state.scans : []).filter((s) => {
    const isDone = Boolean(
      s.isSignedOff ||
      s.reviewed_at ||
      s.reviewedAt ||
      s.doctor_diagnosis ||
      s.doctorDiagnosis ||
      ['signed_off', 'accepted', 'approved', 'flagged', 'overridden'].includes(s.doctorStatus) ||
      ['accepted', 'flagged', 'overridden'].includes(s.status)
    );
    return !isDone;
  }).length;

  // Role-based Nav Links
  const navLinks = isPatient
    ? [
        { to: '/dashboard/scan', label: 'Upload MRI Scan', icon: FiUploadCloud },
        { to: '/dashboard/my-scans', label: 'My Submitted Scans', icon: FiFileText },
      ]
    : [
        { to: '/dashboard', label: 'Clinical Overview', icon: FiGrid },
        { to: '/dashboard/scan', label: 'Upload & Pipeline', icon: FiUploadCloud },
        { to: '/dashboard/patients', label: 'Patient Roster', icon: FiUsers },
        { to: '/dashboard/settings', label: 'System & Thresholds', icon: FiSliders },
      ];

  const handleLogout = () => {
    dispatch({ type: 'LOGOUT' });
    navigate('/login');
  };

  return (
    <aside className="w-64 bg-[#FAF6F3] border-r border-[#E8E2DA] flex flex-col justify-between py-6 px-4 shrink-0 hidden lg:flex select-none">
      <div className="space-y-6">

        {/* Section: Main Navigation */}
        <div>
          <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-[#A39E98]">
            {isPatient ? 'Patient Portal' : 'Clinical Workspace'}
          </div>
          <nav className="space-y-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  end={link.to === '/dashboard' || link.to === '/dashboard/scan'}
                  className={({ isActive }) =>
                    `flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 ${
                      isActive
                        ? 'bg-[#F8EAED] text-[#7A1F2B] font-semibold border border-[#ECC8CF]'
                        : 'text-[#5A5550] hover:bg-[#F0E8E1] hover:text-[#22201F]'
                    }`
                  }
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{link.label}</span>
                  </div>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Recent Scans Quick Access (Doctor Only) */}
        {!isPatient && scans.length > 0 && (
          <div className="pt-4 border-t border-[#E8E2DA]">
            <div className="px-3 mb-2 flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#A39E98]">
                Recent Scans
              </span>
              {pendingCount > 0 && (
                <span className="text-[10px] font-bold text-[#7A1F2B] bg-[#F8EAED] px-1.5 py-0.5 rounded border border-[#ECC8CF]">
                  {pendingCount} Pending
                </span>
              )}
            </div>
            <div className="space-y-1">
              {scans.slice(0, 3).map((scan) => {
                const scanId = scan.scanId || scan.scan_id_string || scan.id;
                const name = scan._resolvedName || scan.patientName || scan.patient_name || 'Patient';
                const pred = scan.prediction || '—';
                return (
                  <NavLink
                    key={scanId}
                    to={`/dashboard/scan/${scanId}`}
                    className={({ isActive }) =>
                      `flex flex-col px-3 py-2 rounded-xl text-xs transition-all ${
                        isActive
                          ? 'bg-white shadow-clinical border border-[#E8E2DA]'
                          : 'hover:bg-[#F0E8E1] text-[#5A5550]'
                      }`
                    }
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-[#22201F]">{name}</span>
                      <span className={`text-[10px] font-bold uppercase ${pred === 'AD' ? 'text-[#7A1F2B]' : pred === 'MCI' ? 'text-[#B87326]' : 'text-[#4A7C59]'}`}>
                        {pred}
                      </span>
                    </div>
                    <span className="text-[10px] text-[#7A756F] font-mono mt-0.5">{scanId}</span>
                  </NavLink>
                );
              })}
            </div>
          </div>
        )}

        {/* Doctor-in-the-Loop Box (Doctor Only) */}
        {!isPatient && (
          <div className="p-3.5 rounded-xl bg-white border border-[#E8E2DA] text-xs">
            <div className="flex items-center gap-2 text-[#7A1F2B] font-semibold mb-1">
              <FiShield className="w-3.5 h-3.5" />
              <span>Doctor-in-the-Loop</span>
            </div>
            <p className="text-[11px] text-[#7A756F] leading-relaxed">
              AI provides volumetric biomarker assistance. Clinical diagnosis remains under physician sign-off.
            </p>
          </div>
        )}
      </div>

      {/* Bottom: Sign Out */}
      <div className="pt-4 border-t border-[#E8E2DA] space-y-2">
        <button
          type="button"
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-3.5 py-2 rounded-xl text-xs font-medium text-[#7A756F] hover:bg-[#F8EAED] hover:text-[#7A1F2B] transition-colors cursor-pointer"
        >
          <FiLogOut className="w-4 h-4" />
          <span>Exit Session</span>
        </button>
      </div>
    </aside>
  );
}
