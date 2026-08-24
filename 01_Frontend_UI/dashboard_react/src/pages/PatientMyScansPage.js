import React from 'react';
import { Link } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import StatusBadge from '../components/common/StatusBadge';
import { 
  FiUploadCloud, 
  FiClock, 
  FiCheckCircle, 
  FiFileText, 
  FiShield,
  FiArrowRight,
  FiActivity,
  FiLayers
} from 'react-icons/fi';
import { LuBrain } from 'react-icons/lu';

export default function PatientMyScansPage() {
  const { state } = useApp();
  const user = state.auth?.user;

  // Filter scans that belong to this patient
  const allScans = Array.isArray(state.scans) ? state.scans : [];
  const userName = (user?.full_name || user?.name || '').toLowerCase();
  const userEmail = (user?.email || '').toLowerCase();

  const myScans = allScans.filter((s) => {
    const sUserId = s.patientUserId || s.userId || s.patient_user_id || s.patientId || s.patient_id;
    const sEmail = (s.patientEmail || s.email || '').trim().toLowerCase();
    const sName = (s.patientName || s.patient || '').trim().toLowerCase();
    
    if (user?.id && sUserId && (user.id === sUserId || String(sUserId) === String(user.id))) return true;
    if (userEmail && sEmail && userEmail === sEmail) return true;
    if (userName && sName && (userName === sName || sName.includes(userName) || userName.includes(sName))) return true;
    return false;
  });

  const latestScan = myScans[0];

  return (
    <DashboardLayout
      title="My Submitted MRI Scans"
      subtitle="Track your uploaded brain MRI scans, 3D Grad-CAM attention maps, and physician diagnostic review status."
      action={
        <Link
          to="/dashboard/scan"
          className="btn-maroon text-xs py-2.5 px-4 shadow-clinical-sm inline-flex items-center gap-2"
        >
          <FiUploadCloud className="w-4 h-4" />
          <span>Upload New Scan</span>
        </Link>
      }
    >
      <div className="space-y-6">

        {/* Quick Patient Stat Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="clinical-card p-4 bg-white flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-[#F8EAED] text-[#7A1F2B] flex items-center justify-center">
              <FiLayers className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-[#7A756F]">Total Submissions</span>
              <h4 className="text-lg font-serif font-bold text-[#22201F]">{myScans.length} Scans</h4>
            </div>
          </div>

          <div className="clinical-card p-4 bg-white flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-[#FAF3E8] text-[#8A5A14] flex items-center justify-center">
              <FiClock className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold text-[#7A756F]">Latest Examination</span>
              <h4 className="text-xs font-bold text-[#22201F] truncate max-w-[220px]">
                {latestScan?.uploadDate || latestScan?.date || 'No Scans Yet'}
              </h4>
            </div>
          </div>
        </div>

        {/* Patient Care Notice */}
        <div className="p-4 sm:p-5 rounded-2xl bg-white border border-[#E8E2DA] shadow-clinical-sm flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#EDF5F0] text-[#4A7C59] flex items-center justify-center shrink-0 mt-0.5">
            <FiShield className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-serif font-bold text-[#22201F]">
              Patient Diagnostic & Explainability Portal
            </h3>
            <p className="text-xs text-[#5A5550] mt-1 leading-relaxed">
              Click on any scan below to open your interactive <strong>3D Grad-CAM visualizer</strong> and view detailed brain region volumetric insights (Hippocampus, Ventricles, Entorhinal Cortex). 
              Your medical doctor will conduct formal evaluation and discuss the findings with you during your next consultation.
            </p>
          </div>
        </div>

        {/* Scans List Table */}
        <div className="clinical-card p-6 bg-white space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-[#E8E2DA]">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#7A1F2B]" />
              <h3 className="text-base font-serif font-bold text-[#22201F]">
                Uploaded MRI Submissions ({myScans.length})
              </h3>
            </div>
            <span className="text-xs text-[#7A756F]">
              Patient: <strong>{user?.full_name || 'Patient Account'}</strong>
            </span>
          </div>

          {myScans.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-[#E8E2DA] text-[#7A756F] uppercase font-bold text-[10px]">
                    <th className="py-3 px-3 whitespace-nowrap">Scan Reference ID</th>
                    <th className="py-3 px-3 whitespace-nowrap">Submission Date</th>
                    <th className="py-3 px-3 whitespace-nowrap">File Format</th>
                    <th className="py-3 px-3 whitespace-nowrap">AI Finding</th>
                    <th className="py-3 px-3 whitespace-nowrap">Confidence</th>
                    <th className="py-3 px-3 whitespace-nowrap">Review Status</th>
                    <th className="py-3 px-3 text-right whitespace-nowrap">Interactive Insights</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#F7F1EC]">
                  {myScans.map((scan, idx) => {
                    const scanId = scan.scanId || scan.scan_id_string || scan.id || `SCN-${idx + 1000}`;
                    const rawDateVal = scan.uploadDate || scan.date || 'Recent';
                    const uploadDate = (() => {
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
                    const fileName = scan.fileFormat || scan.fileName || 'T1_MPRAGE_MRI.nii.gz';
                    const rawPred = (scan.prediction || scan.condition || 'CN').toUpperCase();
                    const prediction = rawPred.includes('AD') ? 'AD' : rawPred.includes('MCI') ? 'MCI' : 'CN';
                    const confidence = scan.confidence || 90;
                    const isReviewed = scan.isSignedOff || scan.doctorStatus === 'signed_off' || scan.doctorStatus === 'accepted' || scan.doctorStatus === 'approved';

                    return (
                      <tr key={scanId} className="hover:bg-[#FAF7F4] transition-colors group">
                        <td className="py-3.5 px-3 whitespace-nowrap">
                          <Link 
                            to={`/dashboard/scan/${scanId}`}
                            className="font-mono font-bold text-[#7A1F2B] hover:underline flex items-center gap-1.5"
                          >
                            <span>{scanId}</span>
                          </Link>
                        </td>

                        <td className="py-3.5 px-3 text-[#5A5550] whitespace-nowrap">
                          <div className="flex items-center gap-1.5">
                            <FiClock className="w-3.5 h-3.5 text-[#A39E98]" />
                            <span>{uploadDate}</span>
                          </div>
                        </td>

                        <td className="py-3.5 px-3 font-mono text-[#22201F] whitespace-nowrap">
                          <div className="flex items-center gap-1.5">
                            <FiFileText className="w-3.5 h-3.5 text-[#7A756F]" />
                            <span className="truncate max-w-[180px]" title={fileName}>{fileName}</span>
                          </div>
                        </td>

                        <td className="py-3.5 px-3 whitespace-nowrap">
                          <StatusBadge status={prediction} size="xs" />
                        </td>

                        <td className="py-3.5 px-3 font-mono font-bold text-[#22201F] whitespace-nowrap">
                          {confidence}%
                        </td>

                        <td className="py-3.5 px-3 whitespace-nowrap">
                          {isReviewed ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#EDF5F0] text-[#4A7C59] border border-[#D5EAD9] font-bold text-[11px]">
                              <FiCheckCircle className="w-3.5 h-3.5" />
                              <span>Reviewed by Physician</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#FAF3E8] text-[#B87326] border border-[#F2DEBF] font-bold text-[11px]">
                              <FiClock className="w-3.5 h-3.5" />
                              <span>Awaiting Clinical Review</span>
                            </span>
                          )}
                        </td>

                        <td className="py-3.5 px-3 text-right whitespace-nowrap">
                          <Link
                            to={`/dashboard/scan/${scanId}`}
                            className="btn-maroon text-xs py-1.5 px-3 inline-flex items-center gap-1.5 shadow-clinical-xs cursor-pointer group-hover:bg-[#661823] whitespace-nowrap"
                          >
                            <FiActivity className="w-3.5 h-3.5" />
                            <span>View Grad-CAM</span>
                            <FiArrowRight className="w-3.5 h-3.5" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-[#FAF6F3] text-[#7A756F] flex items-center justify-center mx-auto">
                <LuBrain className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-serif font-bold text-[#22201F]">
                No MRI Scans Uploaded Yet
              </h4>
              <p className="text-xs text-[#7A756F] max-w-sm mx-auto">
                Upload your first brain MRI volumetric scan (.nii or .dcm) to generate 3D Grad-CAM explainability heatmaps.
              </p>
              <Link
                to="/dashboard/scan"
                className="btn-maroon text-xs py-2 px-4 shadow-clinical-sm inline-flex items-center gap-2 mt-2"
              >
                <FiUploadCloud className="w-4 h-4" />
                <span>Upload Scan Now</span>
              </Link>
            </div>
          )}
        </div>

      </div>
    </DashboardLayout>
  );
}
