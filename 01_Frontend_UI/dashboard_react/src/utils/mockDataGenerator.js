/**
 * mockDataGenerator.js
 * Deterministic seeded-random data generator for NeuroAssist.
 * Given a string key (scan_id, patient_id), produces unique but reproducible
 * clinical values — prediction, confidence, biomarkers, Grad-CAM regions, etc.
 */

// Simple seeded PRNG (Mulberry32) — deterministic from a 32-bit seed
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Convert any string to a 32-bit integer hash
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash + char) | 0;
  }
  return Math.abs(hash);
}

/**
 * Generate a seeded random number generator from a string key.
 */
function seededRng(key) {
  return mulberry32(hashString(String(key || 'default')));
}

/**
 * Pick a random element from an array using the given rng.
 */
function pick(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

/**
 * Random float in [min, max] using seeded rng.
 */
function randFloat(rng, min, max) {
  return min + rng() * (max - min);
}

/**
 * Random int in [min, max] inclusive using seeded rng.
 */
function randInt(rng, min, max) {
  return Math.floor(randFloat(rng, min, max + 1));
}

// ─── Exported Generators ────────────────────────────────────────────

/**
 * Generate mock scan data for a given scanId string or file identifier.
 * Returns deterministic clinical values that vary per scan, but are 100% reproducible for the same file.
 */
export function generateScanData(scanId, fileName = '', targetCondition = null) {
  const rng = seededRng(scanId);

  // If target condition is explicitly provided, enforce it
  let prediction;
  if (targetCondition && ['CN', 'MCI', 'AD'].includes(targetCondition.toUpperCase())) {
    prediction = targetCondition.toUpperCase();
  } else {
    // Check for condition keywords in filename or scanId
    const searchStr = `${scanId} ${fileName}`.toLowerCase();
    if (
      searchStr.includes('_cn') ||
      searchStr.includes('cn_') ||
      searchStr.includes('-cn') ||
      searchStr.includes('cn.') ||
      searchStr.includes('normal') ||
      searchStr.includes('control') ||
      searchStr.includes('cognitively_normal')
    ) {
      prediction = 'CN';
    } else if (
      searchStr.includes('_ad') ||
      searchStr.includes('ad_') ||
      searchStr.includes('-ad') ||
      searchStr.includes('ad.') ||
      searchStr.includes('alzheimer') ||
      searchStr.includes('dementia')
    ) {
      prediction = 'AD';
    } else if (
      searchStr.includes('_mci') ||
      searchStr.includes('mci_') ||
      searchStr.includes('-mci') ||
      searchStr.includes('mci.') ||
      searchStr.includes('impairment')
    ) {
      prediction = 'MCI';
    } else {
      // Deterministic weighted prediction class from file signature seed
      const roll = rng();
      prediction = roll < 0.35 ? 'CN' : roll < 0.70 ? 'MCI' : 'AD';
    }
  }

  // Confidence varies by class
  const confidence =
    prediction === 'AD'
      ? parseFloat(randFloat(rng, 78, 96).toFixed(1))
      : prediction === 'MCI'
      ? parseFloat(randFloat(rng, 62, 88).toFixed(1))
      : parseFloat(randFloat(rng, 70, 95).toFixed(1));

  // Softmax probabilities (sum to strictly 100%)
  let pCN, pMCI, pAD;
  if (prediction === 'AD') {
    pAD = parseFloat(randFloat(rng, 72, 88).toFixed(1));
    pMCI = parseFloat(randFloat(rng, 8, 18).toFixed(1));
    pCN = parseFloat((100 - pAD - pMCI).toFixed(1));
  } else if (prediction === 'MCI') {
    pMCI = parseFloat(randFloat(rng, 62, 78).toFixed(1));
    pCN = parseFloat(randFloat(rng, 14, 24).toFixed(1));
    pAD = parseFloat((100 - pMCI - pCN).toFixed(1));
  } else {
    pCN = parseFloat(randFloat(rng, 74, 90).toFixed(1));
    pMCI = parseFloat(randFloat(rng, 7, 18).toFixed(1));
    pAD = parseFloat((100 - pCN - pMCI).toFixed(1));
  }
  const normCN = parseFloat(pCN.toFixed(1));
  const normMCI = parseFloat(pMCI.toFixed(1));
  const normAD = parseFloat((100.0 - normCN - normMCI).toFixed(1));

  const probabilities = { CN: normCN, MCI: normMCI, AD: normAD };

  // Risk score correlated with prediction
  const riskScore =
    prediction === 'AD'
      ? randInt(rng, 72, 96)
      : prediction === 'MCI'
      ? randInt(rng, 38, 71)
      : randInt(rng, 8, 37);

  // Biomarkers with severity
  const hippoVol = parseFloat(randFloat(rng, 1.8, 4.2).toFixed(2));
  const hippoDevPct = prediction === 'AD' ? randInt(rng, -35, -18) : prediction === 'MCI' ? randInt(rng, -17, -5) : randInt(rng, -4, 5);
  const hippoSeverity = hippoDevPct <= -20 ? 'Significant Atrophy' : hippoDevPct <= -10 ? 'Borderline Atrophy' : 'Within Normal';
  const hippoStatus = hippoDevPct <= -20 ? 'high' : hippoDevPct <= -10 ? 'medium' : 'low';
  const hippoSeverityPct = Math.min(100, Math.max(0, Math.abs(hippoDevPct) * 3));

  const ventVol = parseFloat(randFloat(rng, 28, 58).toFixed(1));
  const ventDevPct = prediction === 'AD' ? randInt(rng, 20, 42) : prediction === 'MCI' ? randInt(rng, 8, 22) : randInt(rng, -3, 8);
  const ventSeverity = ventDevPct >= 25 ? 'Marked Dilation' : ventDevPct >= 12 ? 'Moderate Dilation' : 'Normal Range';
  const ventStatus = ventDevPct >= 25 ? 'high' : ventDevPct >= 12 ? 'medium' : 'low';
  const ventSeverityPct = Math.min(100, Math.max(0, ventDevPct * 2.5));

  const entThick = parseFloat(randFloat(rng, 1.6, 3.2).toFixed(2));
  const entDevPct = prediction === 'AD' ? randInt(rng, -28, -15) : prediction === 'MCI' ? randInt(rng, -14, -5) : randInt(rng, -4, 3);
  const entSeverity = entDevPct <= -18 ? 'Early Degeneration' : entDevPct <= -8 ? 'Focal Thinning' : 'Normal Thickness';
  const entStatus = entDevPct <= -18 ? 'high' : entDevPct <= -8 ? 'medium' : 'low';
  const entSeverityPct = Math.min(100, Math.max(0, Math.abs(entDevPct) * 3.5));

  const biomarkers = {
    hippocampus: {
      value: `${hippoVol} cm³`,
      deviation: `${hippoDevPct > 0 ? '+' : ''}${hippoDevPct}% vs Norm`,
      severity: hippoSeverity,
      status: hippoStatus,
      severityPct: hippoSeverityPct,
    },
    ventricles: {
      value: `${ventVol} mL`,
      deviation: `${ventDevPct > 0 ? '+' : ''}${ventDevPct}% Volume ${ventDevPct >= 0 ? 'Expansion' : 'Contraction'}`,
      severity: ventSeverity,
      status: ventStatus,
      severityPct: ventSeverityPct,
    },
    entorhinalThickness: {
      value: `${entThick} mm`,
      deviation: `${entDevPct > 0 ? '+' : ''}${entDevPct}% ${entDevPct < 0 ? 'Thinning' : 'Thickness'}`,
      severity: entSeverity,
      status: entStatus,
      severityPct: entSeverityPct,
    },
  };

  // Grad-CAM regions with varied attention
  const allRegions = [
    { name: 'Hippocampus (CA1 Subfield)', baseAttention: prediction === 'AD' ? 88 : prediction === 'MCI' ? 65 : 30 },
    { name: 'Right Entorhinal Cortex', baseAttention: prediction === 'AD' ? 78 : prediction === 'MCI' ? 55 : 22 },
    { name: 'Lateral Ventricles', baseAttention: prediction === 'AD' ? 68 : prediction === 'MCI' ? 42 : 15 },
    { name: 'Medial Temporal Lobe', baseAttention: prediction === 'AD' ? 72 : prediction === 'MCI' ? 50 : 18 },
    { name: 'Posterior Cingulate', baseAttention: prediction === 'AD' ? 58 : prediction === 'MCI' ? 35 : 12 },
    { name: 'Precuneus Region', baseAttention: prediction === 'AD' ? 52 : prediction === 'MCI' ? 28 : 10 },
  ];

  const gradCamRegions = allRegions
    .slice(0, prediction === 'AD' ? 4 : prediction === 'MCI' ? 3 : 2)
    .map((r) => ({
      name: r.name,
      attention: Math.min(99, Math.max(5, r.baseAttention + randInt(rng, -8, 8))),
      note: pick(rng, [
        'Volumetric reduction detected',
        'Cortical thinning gradient',
        'Localized signal anomaly',
        'Asymmetric atrophy pattern',
        'Structural deviation from norm',
      ]),
    }));

  // Processing time
  const processingTime = `${randFloat(rng, 0.8, 2.4).toFixed(2)}s`;

  return {
    prediction,
    confidence,
    probabilities,
    riskScore,
    biomarkers,
    gradCamRegions,
    processingTime,
    riskLevel:
      riskScore >= 72
        ? 'High Neurological Risk'
        : riskScore >= 36
        ? 'Moderate Impairment'
        : 'Low Risk (Normal)',
  };
}

/**
 * Generate mock patient profile data from patientId string.
 */
export function generatePatientData(patientId) {
  const rng = seededRng(patientId);

  const roll = rng();
  const condition = roll < 0.35 ? 'CN' : roll < 0.70 ? 'MCI' : 'AD';

  const riskScore =
    condition === 'AD'
      ? randInt(rng, 72, 95)
      : condition === 'MCI'
      ? randInt(rng, 38, 70)
      : randInt(rng, 8, 35);

  const mmseScore =
    condition === 'AD'
      ? randInt(rng, 14, 22)
      : condition === 'MCI'
      ? randInt(rng, 21, 26)
      : randInt(rng, 26, 30);

  const stages = {
    CN: 'Cognitively Normal',
    MCI: 'Mild Cognitive Impairment',
    AD: "Alzheimer's Disease",
  };

  // Longitudinal history: 3-5 timepoints showing trajectory
  const numPoints = randInt(rng, 3, 5);
  const history = [];
  const startYear = 2023;
  let prevRisk = randInt(rng, 15, 35);
  let prevMmse = randInt(rng, 27, 30);

  for (let i = 0; i < numPoints; i++) {
    const month = randInt(rng, 1, 12);
    const year = startYear + i;
    const dateStr = `${year}-${String(month).padStart(2, '0')}`;

    // Gradual progression
    const riskDelta = condition === 'AD' ? randInt(rng, 8, 18) : condition === 'MCI' ? randInt(rng, 4, 12) : randInt(rng, -2, 4);
    prevRisk = Math.min(98, Math.max(5, prevRisk + riskDelta));

    const mmseDelta = condition === 'AD' ? randInt(rng, -4, -1) : condition === 'MCI' ? randInt(rng, -2, 0) : randInt(rng, -1, 1);
    prevMmse = Math.min(30, Math.max(10, prevMmse + mmseDelta));

    const pointLabel = prevRisk >= 72 ? 'AD' : prevRisk >= 36 ? 'MCI' : 'CN';

    history.push({
      date: dateStr,
      riskScore: prevRisk,
      mmse: prevMmse,
      label: pointLabel,
    });
  }

  return {
    condition,
    riskScore,
    mmseScore,
    stage: stages[condition],
    history,
  };
}

const mockDataGenerator = { generateScanData, generatePatientData };
export default mockDataGenerator;
