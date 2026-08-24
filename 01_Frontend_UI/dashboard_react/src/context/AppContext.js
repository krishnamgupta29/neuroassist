import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { authAPI } from '../services/api';

// Helper to get safe user storage key
export function getUserStorageKey(user) {
  if (!user) return 'guest';
  const id = user.id || user._id || user.email || 'user';
  return String(id).trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
}

// Persistent Decision helper for locking and sign-offs across refreshes
export function getStoredDecisions(user) {
  if (!user) return {};
  const key = `na_decisions_${getUserStorageKey(user)}`;
  try {
    const saved = localStorage.getItem(key);
    if (saved) return JSON.parse(saved) || {};
  } catch (e) {}
  return {};
}

export function saveStoredDecision(user, scanId, decision) {
  if (!user || !scanId) return;
  const key = `na_decisions_${getUserStorageKey(user)}`;
  try {
    const current = getStoredDecisions(user);
    current[scanId] = decision;
    if (decision.patientId) {
      current[decision.patientId] = decision;
    }
    localStorage.setItem(key, JSON.stringify(current));
  } catch (e) {}
}

// Helper to load persisted patients scoped by user account
function getInitialPatients(user) {
  if (!user) return [];
  const key = `na_patients_${getUserStorageKey(user)}`;
  try {
    const saved = localStorage.getItem(key);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch (e) {
    console.warn('Error reading scoped patients from localStorage:', e);
  }
  return [];
}

// Helper to load persisted scans scoped by user account
function getInitialScans(user) {
  if (!user) return [];
  const key = `na_scans_${getUserStorageKey(user)}`;
  try {
    const saved = localStorage.getItem(key);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch (e) {
    console.warn('Error reading scoped scans from localStorage:', e);
  }
  return [];
}

const savedUser = JSON.parse(localStorage.getItem('na_user') || 'null');

const initialState = {
  auth: {
    token: localStorage.getItem('na_token') || null,
    user: savedUser,
    isLoading: true,
  },
  patients: getInitialPatients(savedUser),
  scans: getInitialScans(savedUser),
  notifications: [],
  settings: {
    mciThreshold: 50,
    adAlertThreshold: 75,
    gradcamSensitivity: 0.85,
    activeModel: 'medicalnet-resnet10',
    autoGeneratePdf: true,
    clinicName: '',
    physicianName: '',
    emailAlerts: true,
  },
  activeScanId: null,
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_AUTH': {
      const u = action.payload.user;
      const uPatients = getInitialPatients(u);
      const uScans = getInitialScans(u);

      return {
        ...state,
        auth: {
          token: action.payload.token,
          user: u,
          isLoading: false,
        },
        patients: uPatients,
        scans: uScans,
      };
    }

    case 'AUTH_LOADED':
      return {
        ...state,
        auth: { ...state.auth, isLoading: false },
      };

    case 'LOGOUT': {
      localStorage.removeItem('na_token');
      localStorage.removeItem('na_refresh');
      localStorage.removeItem('na_user');
      return {
        ...initialState,
        auth: { token: null, user: null, isLoading: false },
        patients: [],
        scans: [],
      };
    }

    case 'SET_PATIENTS': {
      const p = action.payload;
      const fetchedList = Array.isArray(p) ? p : (p?.patients || p?.items || []);
      const user = state.auth?.user;
      const uKey = getUserStorageKey(user);
      const decisions = getStoredDecisions(user);

      const mergedPatients = fetchedList.map((pat) => {
        const pId = pat.id || pat._id;
        const dec = decisions[pId] || (pat.lastScanId && decisions[pat.lastScanId]);
        if (dec && dec.isSignedOff) {
          return {
            ...pat,
            isSignedOff: true,
            status: dec.status,
            doctorStatus: dec.status,
            doctorNotes: dec.notes,
            reviewed_at: dec.signedOffAt,
            reviewed_by: dec.signedOffBy,
          };
        }
        return pat;
      });

      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_patients_${uKey}`, JSON.stringify(mergedPatients));
        } catch (e) {}
      }

      return { ...state, patients: mergedPatients };
    }

    case 'ADD_PATIENT': {
      const newPatient = action.payload;
      const current = Array.isArray(state.patients) ? state.patients : [];
      const newId = newPatient.id || newPatient._id;
      const newNameLower = (newPatient.full_name || newPatient.name || '').trim().toLowerCase();

      const updated = [
        newPatient,
        ...current.filter((p) => {
          const id = p.id || p._id;
          const nameLower = (p.full_name || p.name || '').trim().toLowerCase();
          return id !== newId && nameLower !== newNameLower;
        }),
      ];

      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_patients_${uKey}`, JSON.stringify(updated));
        } catch (e) {}
      }

      return { ...state, patients: updated };
    }

    case 'DELETE_PATIENT': {
      const targetId = action.payload;
      const updatedPatients = (state.patients || []).filter(
        (p) => (p.id || p._id) !== targetId
      );
      const updatedScans = (state.scans || []).filter(
        (s) => (s.patientId || s.patient_id) !== targetId
      );

      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_patients_${uKey}`, JSON.stringify(updatedPatients));
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(updatedScans));
        } catch (e) {}
      }

      return {
        ...state,
        patients: updatedPatients,
        scans: updatedScans,
      };
    }

    case 'SET_SCANS': {
      const s = action.payload;
      const fetchedList = Array.isArray(s) ? s : (s?.scans || s?.items || []);
      const user = state.auth?.user;
      const uKey = getUserStorageKey(user);
      const decisions = getStoredDecisions(user);

      const mergedScans = [...fetchedList];
      Object.keys(decisions).forEach((k) => {
        const dec = decisions[k];
        if (dec && dec.isSignedOff && dec.scanId) {
          const matchIdx = mergedScans.findIndex(
            (sc) => (sc.scanId || sc.scan_id_string || sc.id) === dec.scanId || (dec.patientId && (sc.patientId === dec.patientId || sc.patient_id === dec.patientId))
          );
          if (matchIdx >= 0) {
            mergedScans[matchIdx] = {
              ...mergedScans[matchIdx],
              doctorStatus: dec.status,
              status: dec.status,
              doctorNotes: dec.notes,
              isSignedOff: true,
              signedOffAt: dec.signedOffAt,
              signedOffBy: dec.signedOffBy,
            };
          } else {
            mergedScans.unshift({
              scanId: dec.scanId,
              scan_id_string: dec.scanId,
              patientId: dec.patientId,
              status: dec.status,
              doctorStatus: dec.status,
              doctorNotes: dec.notes,
              isSignedOff: true,
              signedOffAt: dec.signedOffAt,
              signedOffBy: dec.signedOffBy,
              prediction: dec.prediction || 'CN',
              riskScore: dec.riskScore || 18,
              date: dec.signedOffAt ? dec.signedOffAt.split(' · ')[0] : new Date().toISOString().split('T')[0],
              uploadDate: dec.signedOffAt ? dec.signedOffAt.split(' · ')[0] : new Date().toISOString().split('T')[0],
            });
          }
        }
      });

      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(mergedScans));
        } catch (e) {}
      }

      return { ...state, scans: mergedScans };
    }

    case 'DELETE_SCAN': {
      const targetScanId = action.payload;
      const updatedScans = (state.scans || []).filter(
        (s) => (s.scanId || s.scan_id_string || s.id) !== targetScanId
      );

      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(updatedScans));
        } catch (e) {}
      }

      return {
        ...state,
        scans: updatedScans,
      };
    }

    case 'UPDATE_SCAN_RESULT': {
      const updated = action.payload;
      const targetScanId = updated.scanId || updated.scan_id_string || updated.id || updated.scan_id;
      if (!targetScanId) return state;
      const currentScans = Array.isArray(state.scans) ? state.scans : [];
      let found = false;
      const newScans = currentScans.map((s) => {
        const sId = s.scanId || s.scan_id_string || s.id;
        if (sId === targetScanId) {
          found = true;
          return {
            ...s,
            ...updated,
            scanId: targetScanId,
            scan_id_string: targetScanId,
            prediction: updated.prediction || s.prediction,
            riskScore: updated.riskScore ?? updated.risk_score ?? s.riskScore,
          };
        }
        return s;
      });
      if (!found) {
        newScans.unshift({
          scanId: targetScanId,
          scan_id_string: targetScanId,
          ...updated,
        });
      }
      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(newScans));
        } catch (e) {}
      }
      return {
        ...state,
        scans: newScans,
      };
    }

    case 'ADD_SCAN': {
      const newScan = action.payload;
      const scanId = newScan.scanId || newScan.scan_id_string || newScan.id;
      const currentScans = Array.isArray(state.scans) ? state.scans : [];
      const targetPatientId = newScan.patientId || newScan.patient_id;
      const targetPatientName = (newScan.patientName || newScan.patient || '').trim().toLowerCase();

      // Add new scan at the top, scoped to current account
      const updatedScans = [
        newScan,
        ...currentScans.filter((s) => {
          const sId = s.scanId || s.scan_id_string || s.id;
          return sId !== scanId;
        }),
      ];

      const updatedPatients = (state.patients || []).map((pat) => {
        const pId = pat.id || pat._id;
        const pName = (pat.full_name || pat.name || '').toLowerCase();
        if (pId === targetPatientId || (targetPatientName && pName === targetPatientName)) {
          const currentCount = pat.scansCount || pat.scan_count || 0;
          return {
            ...pat,
            scan_count: currentCount + 1,
            scansCount: currentCount + 1,
            lastScanDate: newScan.date || newScan.uploadDate || new Date().toISOString().split('T')[0],
            condition: newScan.prediction || pat.condition,
            diagnosis: newScan.prediction || pat.diagnosis,
            riskScore: newScan.riskScore || pat.riskScore,
          };
        }
        return pat;
      });

      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(updatedScans));
          localStorage.setItem(`na_patients_${uKey}`, JSON.stringify(updatedPatients));
        } catch (e) {}
      }

      return {
        ...state,
        scans: updatedScans,
        patients: updatedPatients,
        activeScanId: scanId,
      };
    }

    case 'UPDATE_SCAN_DECISION': {
      const { scanId, patientId, status, notes, signedOffAt, signedOffBy, prediction, riskScore } = action.payload;
      const currentScans = Array.isArray(state.scans) ? state.scans : [];
      let found = false;

      // Persist decision permanently
      saveStoredDecision(state.auth?.user, scanId, {
        scanId,
        patientId,
        status,
        notes,
        signedOffAt,
        signedOffBy,
        isSignedOff: true,
        prediction,
        riskScore,
      });

      const updatedScans = currentScans.map((s) => {
        const sId = s.scanId || s.scan_id_string || s.id;
        if (sId === scanId || (patientId && (s.patientId === patientId || s.patient_id === patientId))) {
          found = true;
          return {
            ...s,
            doctorStatus: status,
            status: status,
            doctorNotes: notes,
            isSignedOff: true,
            signedOffAt: signedOffAt,
            signedOffBy: signedOffBy,
          };
        }
        return s;
      });

      if (!found) {
        updatedScans.unshift({
          scanId,
          scan_id_string: scanId,
          patientId,
          status,
          doctorStatus: status,
          doctorNotes: notes,
          isSignedOff: true,
          signedOffAt,
          signedOffBy,
          prediction: prediction || 'CN',
          riskScore: riskScore || 18,
          date: new Date().toISOString().split('T')[0],
          uploadDate: new Date().toISOString().split('T')[0],
        });
      }

      const updatedPatients = (state.patients || []).map((pat) => {
        const pId = pat.id || pat._id;
        if (pId === patientId || (pat.lastScanId && pat.lastScanId === scanId)) {
          return {
            ...pat,
            isSignedOff: true,
            status,
            doctorStatus: status,
            doctorNotes: notes,
            reviewed_at: signedOffAt,
            reviewed_by: signedOffBy,
          };
        }
        return pat;
      });

      const uKey = getUserStorageKey(state.auth?.user);
      if (uKey !== 'guest') {
        try {
          localStorage.setItem(`na_scans_${uKey}`, JSON.stringify(updatedScans));
          localStorage.setItem(`na_patients_${uKey}`, JSON.stringify(updatedPatients));
        } catch (e) {}
      }

      return {
        ...state,
        scans: updatedScans,
        patients: updatedPatients,
      };
    }

    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: { ...state.settings, ...action.payload },
      };

    default:
      return state;
  }
}

const AppContext = createContext();

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Validate stored token on mount
  useEffect(() => {
    const token = localStorage.getItem('na_token');
    const user = JSON.parse(localStorage.getItem('na_user') || 'null');

    if (token && user) {
      authAPI
        .me()
        .then((res) => {
          const freshUser = res.data?.user || res.data || user;
          localStorage.setItem('na_user', JSON.stringify(freshUser));
          dispatch({ type: 'SET_AUTH', payload: { token, user: freshUser } });
        })
        .catch(() => {
          // Token expired or invalid
          dispatch({ type: 'LOGOUT' });
        })
        .finally(() => {
          dispatch({ type: 'AUTH_LOADED' });
        });
    } else {
      dispatch({ type: 'AUTH_LOADED' });
    }
  }, []);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
