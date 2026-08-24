import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { FiBell, FiUploadCloud, FiSearch, FiCheckCircle, FiAlertTriangle, FiX } from 'react-icons/fi';
import { LuBrain } from 'react-icons/lu';

export default function ClinicalNavbar() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const user = state.auth?.user;
  const isPatient = user?.role === 'patient';

  const initials = user?.full_name
    ? user.full_name.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2)
    : user?.name
    ? user.name.slice(0, 2).toUpperCase()
    : 'U';

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim() && !isPatient) {
      navigate(`/dashboard/patients?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const handleLogout = () => {
    dispatch({ type: 'LOGOUT' });
    navigate('/login');
  };

  return (
    <header className="sticky top-0 z-50 w-full bg-[#FAF6F3]/95 backdrop-blur-md border-b border-[#E8E2DA]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">

        {/* Left: Brand */}
        <div className="flex items-center gap-6">
          <Link to={isPatient ? '/dashboard/scan' : '/dashboard'} className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-[#7A1F2B] text-white flex items-center justify-center shadow-sm group-hover:bg-[#661823] transition-colors">
              <LuBrain className="w-5 h-5 text-white/90" />
            </div>
            <div className="flex flex-col">
              <span className="brand-title text-lg tracking-wider leading-none">
                <span className="brand-bold">NEURO</span>
                <span className="brand-regular">ASSIST</span>
              </span>
              <span className="text-[10px] uppercase font-semibold tracking-widest text-[#7A756F] mt-0.5">
                {isPatient ? 'Patient Diagnostic Portal' : 'Clinical Diagnostic Suite'}
              </span>
            </div>
          </Link>
        </div>

        {/* Center: Search (Doctor Only) */}
        {!isPatient ? (
          <form onSubmit={handleSearch} className="hidden md:flex items-center flex-1 max-w-md relative">
            <FiSearch className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A39E98]" />
            <input
              type="text"
              placeholder="Search patient by name or MRN..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-white border border-[#E8E2DA] rounded-xl text-xs sm:text-sm text-[#22201F] placeholder-[#A39E98] focus:outline-none focus:border-[#7A1F2B] focus:ring-1 focus:ring-[#7A1F2B]"
            />
          </form>
        ) : (
          <div className="hidden md:flex items-center text-xs text-[#7A756F] bg-white px-3.5 py-1.5 rounded-full border border-[#E8E2DA]">
            <span className="w-2 h-2 rounded-full bg-[#4A7C59] mr-2" />
            <span>Secure Patient Portal · Connected to Radiology</span>
          </div>
        )}

        {/* Right: Actions & Profile */}
        <div className="flex items-center gap-3">
          {!isPatient && (
            <Link
              to="/dashboard/scan"
              className="hidden sm:inline-flex items-center gap-2 px-3.5 py-2 bg-[#7A1F2B] hover:bg-[#661823] text-white rounded-xl text-xs font-medium transition-all shadow-sm active:translate-y-0.5"
            >
              <FiUploadCloud className="w-4 h-4" />
              <span>Upload Scan</span>
            </Link>
          )}

          {/* Notifications */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative p-2 rounded-xl bg-white border border-[#E8E2DA] text-[#22201F] hover:bg-[#F7F1EC] transition-colors cursor-pointer"
            >
              <FiBell className="w-4 h-4 text-[#7A756F]" />
              {state.notifications?.length > 0 && (
                <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#7A1F2B]" />
              )}
            </button>

            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 bg-white border border-[#E8E2DA] rounded-2xl shadow-clinical-lg p-4 z-50 animate-slide-up">
                <div className="flex items-center justify-between pb-3 border-b border-[#F0EBE5]">
                  <span className="text-xs font-bold uppercase tracking-wider text-[#22201F]">
                    Notifications ({state.notifications?.length || 0})
                  </span>
                  <button onClick={() => setShowNotifications(false)} className="text-[#A39E98] hover:text-[#22201F] cursor-pointer">
                    <FiX className="w-4 h-4" />
                  </button>
                </div>
                <div className="divide-y divide-[#F7F1EC] max-h-72 overflow-y-auto mt-2">
                  {state.notifications?.map((notif) => (
                    <div key={notif.id} className="py-2.5 flex items-start gap-3">
                      <div className={`mt-0.5 w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${notif.urgent ? 'bg-[#F8EAED] text-[#7A1F2B]' : 'bg-[#EDF5F0] text-[#4A7C59]'}`}>
                        {notif.urgent ? <FiAlertTriangle className="w-3 h-3" /> : <FiCheckCircle className="w-3 h-3" />}
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xs font-semibold text-[#22201F]">{notif.title}</h4>
                        <p className="text-[11px] text-[#7A756F] mt-0.5">{notif.message}</p>
                        <span className="text-[10px] text-[#A39E98] mt-1 block">{notif.time}</span>
                      </div>
                      <button onClick={() => dispatch({ type: 'DISMISS_NOTIFICATION', payload: notif.id })} className="text-[#A39E98] hover:text-[#7A1F2B] text-xs cursor-pointer">
                        <FiX className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                  {(!state.notifications || state.notifications.length === 0) && (
                    <p className="py-6 text-center text-xs text-[#A39E98]">No new notifications.</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* User Profile & Logout */}
          <div className="flex items-center gap-2.5 pl-2 border-l border-[#E8E2DA]">
            <div className="w-8 h-8 rounded-full bg-[#E8DDD4] border border-[#D8C9BC] flex items-center justify-center text-[#7A1F2B] font-serif font-bold text-xs">
              {initials}
            </div>
            <div className="hidden md:flex flex-col text-left">
              <span className="text-xs font-semibold text-[#22201F] leading-tight">
                {user?.full_name || user?.name || 'User'}
              </span>
              <button
                type="button"
                onClick={handleLogout}
                className="text-[10px] text-[#7A1F2B] font-medium leading-tight text-left hover:underline cursor-pointer"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
