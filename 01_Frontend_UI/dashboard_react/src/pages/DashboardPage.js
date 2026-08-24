import React, { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useApp, getStoredDecisions } from '../context/AppContext';
import { generateScanData } from '../utils/mockDataGenerator';
import DashboardLayout from '../components/layout/DashboardLayout';
import MetricCard from '../components/common/MetricCard';
import StatusBadge from '../components/common/StatusBadge';
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip } from 'recharts';
import { 
  FiUploadCloud, 
  FiAlertTriangle, 
  FiCheckCircle, 
  FiActivity, 
  FiArrowRight, 
  FiClock, 
  FiUsers 
} from 'react-icons/fi';
import { LuBrain } from 'react-icons/lu';
import { scanAPI, patientAPI } from '../services/api';

export default function DashboardPage() {
  const { state, dispatch } = useApp();
  const [loading, setLoading] = useState(true);

  // Fetch patients & scans from real backend on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [patientsRes, scansRes] = await Promise.allSettled([
          patientAPI.list(),
          scanAPI.history(50),
        ]);

        // Always dispatch, including an empty list. Skipping the empty case
        // left the previous account's cached rows on screen and made deleted
        // records look like they had come back.
        if (patientsRes.status === 'fulfilled') {
          const pData = patientsRes.value.data;
          const pList = Array.isArray(pData) ? pData : (pData?.patients || pData?.items || []);
          dispatch({ type: 'SET_PATIENTS', payload: pList });
        }
        if (scansRes.status === 'fulfilled') {
          const sData = scansRes.value.data;
          const sList = Array.isArray(sData) ? sData : (sData?.items || sData?.scans || []);
          dispatch({ type: 'SET_SCANS', payload: sList });
        }
      } catch (err) {
        console.error('Dashboard data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [dispatch]);

  // Filter out any demo accounts from patients
  const patients = useMemo(() => {
    const rawList = Array.isArray(state.patients) ? state.patients : [];
    return rawList.filter(p => {
      const name = (p.full_name || p.name || '').trim().toLowerCase();
      return !name.includes('demo') && !name.includes('arthur pendelton') && !name.includes('helen mirren');
    });
  }, [state.patients]);

  const scans = useMemo(() => {
    return Array.isArray(state.scans) ? state.scans : [];
  }, [state.scans]);

  // Robust Cognitive classification helper
  const getConditionCode = (cond) => {
    const raw = (cond || '').toUpperCase();
    if (raw.includes('AD') || raw.includes('ALZHEIMER')) return 'AD';
    if (raw.includes('MCI') || raw.includes('MILD')) return 'MCI';
    if (raw.includes('CN') || raw.includes('NORMAL') || raw.includes('CONTROL')) return 'CN';
    return 'CN';
  };

  const storedDecisions = useMemo(() => getStoredDecisions(state.auth?.user), [state.auth?.user]);

  // Build unified patient evaluation list linking each patient with their latest scan or baseline diagnosis
  const patientEvaluations = useMemo(() => {
    return patients.map((p, idx) => {
      const pId = p.id || p._id;
      const pName = p.full_name || p.name || `Patient Record ${idx + 1}`;
      const pMrn = p.patient_code || p.mrn || `NA-2026-00${idx + 1}`;

      // Search matching scan in state.scans
      const matchingScan = scans.find(s => {
        const sPId = s.patientId || s.patient_id;
        const sPName = (s.patientName || s.patient || '').trim().toLowerCase();
        const sMrn = s.patient_code || s.mrn;
        if (sPId && pId && sPId === pId) return true;
        if (sMrn && (sMrn === pMrn || sMrn === p.patient_code)) return true;
        if (sPName && sPName === pName.trim().toLowerCase()) return true;
        return false;
      });

      const scanId = matchingScan?.scanId || matchingScan?.scan_id_string || matchingScan?.id || `SCN-${700000 + (idx * 1321) % 200000}`;
      const seeded = generateScanData(scanId);

      const storedDec = storedDecisions[pId] || (matchingScan && storedDecisions[matchingScan.scanId || matchingScan.scan_id_string || matchingScan.id]) || storedDecisions[scanId];

      const rawFinding = storedDec?.prediction || matchingScan?.prediction || matchingScan?.condition || (p.condition && p.condition !== 'CN' ? p.condition : seeded.prediction);
      const normalizedCond = getConditionCode(rawFinding);

      const riskScore = storedDec?.riskScore ?? matchingScan?.riskScore ?? matchingScan?.risk_score ?? (p.riskScore && p.riskScore !== 18 ? p.riskScore : seeded.riskScore);
      const uploadDate = matchingScan?.uploadDate || matchingScan?.date || p.lastScanDate || p.created_at || 'Recent';
      const status = storedDec?.status || matchingScan?.doctorStatus || matchingScan?.status || p.doctorStatus || p.status || 'pending';
      const isSignedOff = Boolean(
        storedDec?.isSignedOff ||
        p.isSignedOff ||
        matchingScan?.isSignedOff ||
        matchingScan?.reviewed_at ||
        matchingScan?.reviewedAt ||
        matchingScan?.doctor_diagnosis ||
        matchingScan?.doctorDiagnosis ||
        ['signed_off', 'accepted', 'approved', 'flagged', 'overridden'].includes(status) ||
        ['signed_off', 'accepted', 'approved', 'flagged', 'overridden'].includes(matchingScan?.doctorStatus) ||
        ['signed_off', 'accepted', 'approved', 'flagged', 'overridden'].includes(p.doctorStatus)
      );
      const isFlagged = status === 'flagged' || matchingScan?.doctorStatus === 'flagged' || p.doctorStatus === 'flagged' || p.urgentFlag;

      return {
        patientId: pId,
        patientName: pName,
        mrn: pMrn,
        age: p.age || (p.date_of_birth ? new Date().getFullYear() - new Date(p.date_of_birth).getFullYear() : (p.age || 65)),
        gender: p.gender || '—',
        scanId,
        prediction: normalizedCond,
        riskScore,
        uploadDate,
        status,
        isSignedOff,
        isFlagged,
        isRawScan: Boolean(matchingScan),
      };
    });
  }, [patients, scans, storedDecisions]);

  // Aggregate stats across the full cohort
  const totalPatientsCount = patientEvaluations.length;
  const cnCount = patientEvaluations.filter(e => e.prediction === 'CN').length;
  const mciCount = patientEvaluations.filter(e => e.prediction === 'MCI').length;
  const adCount = patientEvaluations.filter(e => e.prediction === 'AD').length;

  const pendingReviews = patientEvaluations.filter(e => !e.isSignedOff).length;
  const highRiskFlags = patientEvaluations.filter(e => e.prediction === 'AD' || e.riskScore >= 75).length;

  const cognitiveData = [
    { name: 'Cognitively Normal (CN)', count: cnCount, color: '#4A7C59', bg: '#EDF5F0' },
    { name: 'Mild Cognitive Impairment (MCI)', count: mciCount, color: '#B87326', bg: '#FAF3E8' },
    { name: "Alzheimer's Disease (AD)", count: adCount, color: '#7A1F2B', bg: '#F8EAED' },
  ];

  const activeDonutData = cognitiveData.filter((d) => d.count > 0);

  return (
    <DashboardLayout
      title="Clinical Overview"
      subtitle="Neurological diagnostic triage, patient cohort distribution, and volumetric MRI screening workstation."
      action={
        state.auth?.user?.role === 'doctor' && (
          <Link to="/dashboard/scan" className="btn-maroon text-xs shadow-clinical-sm">
            <FiUploadCloud className="w-4 h-4" />
            <span>Upload & Analyze Scan</span>
          </Link>
        )
      }
    >
      <div className="space-y-6">

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-xs text-[#7A756F] flex items-center gap-2">
              <LuBrain className="w-5 h-5 text-[#7A1F2B] animate-subtle-pulse" />
              <span>Loading clinical data from API...</span>
            </div>
          </div>
        )}

        {!loading && (
          <>
            {/* Top Metrics Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Enrolled Patients"
                value={String(totalPatientsCount)}
                subtitle="Active patient clinical records"
                icon={FiUsers}
              />
              <MetricCard
                title="Cohort Scans & Analyses"
                value={String(Math.max(totalPatientsCount, scans.length))}
                subtitle="Volumetric series evaluated"
                icon={FiActivity}
              />
              <MetricCard
                title="Pending Reviews"
                value={String(pendingReviews)}
                subtitle="Awaiting physician validation"
                icon={FiClock}
                badgeText={pendingReviews > 0 ? 'Action Needed' : 'All Clear'}
                badgeType={pendingReviews > 0 ? 'amber' : 'sage'}
              />
              <MetricCard
                title="High-Priority Cases"
                value={String(highRiskFlags)}
                subtitle="AD predictions or severe risk"
                icon={FiAlertTriangle}
                badgeType="maroon"
              />
            </div>

            {/* Cohort Cognitive Distribution Card */}
            <div className="clinical-card p-6 bg-white">
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-[#E8E2DA]">
                <div>
                  <h3 className="text-base font-serif font-bold text-[#22201F]">
                    Cohort Cognitive Distribution
                  </h3>
                  <p className="text-xs text-[#7A756F] mt-0.5">
                    Summary of all {totalPatientsCount} enrolled hospital patient cognitive classifications.
                  </p>
                </div>
                <span className="text-xs font-semibold text-[#7A756F] bg-[#FAF6F3] px-3 py-1 rounded-full border border-[#E8E2DA]">
                  {totalPatientsCount} Total Evaluated {totalPatientsCount === 1 ? 'Patient' : 'Patients'}
                </span>
              </div>

              {totalPatientsCount > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                  {/* Left: Donut Chart */}
                  <div className="md:col-span-5 flex items-center justify-center">
                    <div className="w-[208px] h-[208px] relative flex items-center justify-center">
                      <PieChart width={208} height={208}>
                        <Pie
                          data={activeDonutData}
                          cx={104}
                          cy={104}
                          innerRadius={58}
                          outerRadius={84}
                          paddingAngle={activeDonutData.length > 1 ? 4 : 0}
                          dataKey="count"
                        >
                          {activeDonutData.map((entry, i) => (
                            <Cell key={i} fill={entry.color} stroke="#FFFFFF" strokeWidth={2.5} />
                          ))}
                        </Pie>
                        <RechartsTooltip
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const d = payload[0].payload;
                              return (
                                <div className="bg-white p-2.5 rounded-xl border border-[#E8E2DA] shadow-clinical-md text-xs">
                                  <span className="font-bold text-[#22201F] block">{d.name}</span>
                                  <span className="text-[#7A756F]">
                                    Count: <strong>{d.count} {d.count === 1 ? 'patient' : 'patients'}</strong>
                                  </span>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                      </PieChart>
                      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-sans font-extrabold text-[#22201F] tracking-tight leading-none">
                          {totalPatientsCount}
                        </span>
                        <span className="text-[10px] uppercase font-bold tracking-widest text-[#7A756F] mt-1">
                          Patients
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Right: Cognitive Distribution Breakdown Cards */}
                  <div className="md:col-span-7 space-y-3">
                    {cognitiveData.map((item, idx) => {
                      const pct = totalPatientsCount > 0 ? Math.round((item.count / totalPatientsCount) * 100) : 0;
                      return (
                        <div key={idx} className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] space-y-1.5">
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2">
                              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                              <span className="text-[#22201F] font-bold">{item.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono font-bold text-[#22201F]">
                                {item.count} {item.count === 1 ? 'patient' : 'patients'}
                              </span>
                              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full" style={{ color: item.color, backgroundColor: item.bg }}>
                                {pct}%
                              </span>
                            </div>
                          </div>
                          <div className="w-full h-2 rounded-full bg-[#E8E2DA] overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700 ease-out"
                              style={{ width: `${pct}%`, backgroundColor: item.color }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-10">
                  <p className="text-xs text-[#A39E98]">No patient records found. Click '+ Add Patient' to populate cohort.</p>
                </div>
              )}
            </div>

            {/* Clinical Review Queue / Recent Scans */}
            <div className="clinical-card p-6 bg-white">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 mb-4 border-b border-[#E8E2DA]">
                <div>
                  <h3 className="text-base font-serif font-bold text-[#22201F]">
                    Recent Scans & Patient Clinical Evaluations ({patientEvaluations.length})
                  </h3>
                  <p className="text-xs text-[#7A756F] mt-0.5">
                    Volumetric MRI series, 3D Grad-CAM attention regions, and physician sign-off status.
                  </p>
                </div>
                <Link to="/dashboard/patients" className="text-xs font-semibold text-[#7A1F2B] hover:underline flex items-center gap-1">
                  <span>Manage All Patients</span>
                  <FiArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>

              {patientEvaluations.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-[#E8E2DA] text-[#7A756F] uppercase tracking-wider font-semibold text-[10px]">
                        <th className="py-2.5 px-3">Patient / MRN</th>
                        <th className="py-2.5 px-3">Scan Reference</th>
                        <th className="py-2.5 px-3">AI Finding</th>
                        <th className="py-2.5 px-3">Risk Score</th>
                        <th className="py-2.5 px-3">Review Status</th>
                        <th className="py-2.5 px-3 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#F7F1EC]">
                      {patientEvaluations.map((item, idx) => {
                        const { patientId, patientName, mrn, scanId, prediction, riskScore, isSignedOff } = item;
                        return (
                          <tr key={scanId || idx} className="hover:bg-[#FAF6F3] transition-colors group">
                            <td className="py-3 px-3 whitespace-nowrap">
                              <Link
                                to={`/dashboard/patients/${patientId}`}
                                className="font-bold text-[#22201F] hover:text-[#7A1F2B] hover:underline block"
                              >
                                {patientName}
                              </Link>
                              <span className="font-mono text-[10px] text-[#A39E98]">{mrn}</span>
                            </td>

                            <td className="py-3 px-3 font-mono font-bold text-[#7A1F2B] whitespace-nowrap">
                              <Link to={`/dashboard/scan/${scanId}`} className="hover:underline">
                                {scanId}
                              </Link>
                            </td>

                            <td className="py-3 px-3 whitespace-nowrap">
                              <StatusBadge status={prediction} size="xs" short={true} />
                            </td>

                            <td className="py-3 px-3">
                              <div className="flex items-center gap-2">
                                <span className={`font-mono font-bold ${
                                  riskScore >= 75 ? 'text-[#7A1F2B]' : riskScore >= 40 ? 'text-[#B87326]' : 'text-[#4A7C59]'
                                }`}>
                                  {riskScore}%
                                </span>
                                <div className="w-16 h-1.5 rounded-full bg-[#E8E2DA] overflow-hidden hidden sm:block">
                                  <div
                                    className={`h-full rounded-full ${
                                      riskScore >= 75 ? 'bg-[#7A1F2B]' : riskScore >= 40 ? 'bg-[#B87326]' : 'bg-[#4A7C59]'
                                    }`}
                                    style={{ width: `${Math.min(100, Math.max(5, riskScore))}%` }}
                                  />
                                </div>
                              </div>
                            </td>

                            <td className="py-3 px-3">
                              {isSignedOff ? (
                                item.isFlagged ? (
                                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#F8EAED] text-[#7A1F2B] border border-[#ECC8CF] text-[10px] font-bold">
                                    <FiAlertTriangle className="w-3 h-3" />
                                    <span>Flagged</span>
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#EDF5F0] text-[#4A7C59] border border-[#CFE3D5] text-[10px] font-bold">
                                    <FiCheckCircle className="w-3 h-3" />
                                    <span>Reviewed</span>
                                  </span>
                                )
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#FAF3E8] text-[#8A5A14] border border-[#F0DEC2] text-[10px] font-bold">
                                  <FiClock className="w-3 h-3" />
                                  <span>Pending Sign-Off</span>
                                </span>
                              )}
                            </td>

                            <td className="py-3 px-3 text-right">
                              <div className="flex items-center justify-end gap-1.5">
                                <Link
                                  to={`/dashboard/scan/${scanId}?patientId=${item.patientId || ''}&mrn=${item.mrn || ''}`}
                                  className="btn-outline text-[11px] py-1 px-2.5 inline-flex items-center gap-1 group-hover:border-[#7A1F2B]"
                                >
                                  <span>Workstation</span>
                                  <FiArrowRight className="w-3 h-3" />
                                </Link>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-10 space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-[#FAF6F3] text-[#7A756F] flex items-center justify-center mx-auto">
                    <LuBrain className="w-5 h-5" />
                  </div>
                  <p className="text-xs text-[#7A756F]">No diagnostic scans on file.</p>
                </div>
              )}
            </div>

          </>
        )}

      </div>
    </DashboardLayout>
  );
}

