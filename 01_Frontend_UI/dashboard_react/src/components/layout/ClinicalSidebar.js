import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
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
