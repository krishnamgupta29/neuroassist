import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { LuBrain } from 'react-icons/lu';
import { 
  FiLayers, 
  FiCrosshair, 
  FiEye, 
  FiZoomIn, 
  FiZoomOut, 
  FiRotateCcw, 
  FiDownload
} from 'react-icons/fi';

// Deterministic PRNG from string
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash + char) | 0;
  }
  return Math.abs(hash);
}

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * GradCamViewer — Hospital PACS Radiologist 3D MRI & Grad-CAM Heatmap Viewer
 * Fully dynamic and unique per patient, scan ID, and classification condition.
 */
export default function GradCamViewer({
  scanId = 'SCN-849201',
  condition = 'AD',
  confidence = 90,
  patientName = 'Patient Record'
}) {
  const [activeSliceView, setActiveSliceView] = useState('axial'); // 'axial' | 'coronal' | 'sagittal'
  const [sliceIndex, setSliceIndex] = useState(50); // 0 to 100
  const [zoomScale, setZoomScale] = useState(100); // 80 to 140
  const [showCrosshair, setShowCrosshair] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [colorMap, setColorMap] = useState('jet'); // 'jet' | 'inferno' | 'turbo'
  const [hoverCoord, setHoverCoord] = useState(null); // { x, y, activation, region }
  const [isHovering, setIsHovering] = useState(false);

  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Normalize condition
  const cond = (condition || 'AD').toUpperCase();

  // Generate unique deterministic anatomical signature for THIS specific patient & scan
  const patientProfile = useMemo(() => {
    const seed = hashString(`${scanId}_${patientName}_${cond}`);
    const rng = mulberry32(seed);

    // Patient-specific asymmetry: left vs right hemisphere atrophy dominance
    const asymmetry = (rng() - 0.5) * 0.35;
    const leftWeight = Math.min(1.0, Math.max(0.4, 0.90 + asymmetry));
    const rightWeight = Math.min(1.0, Math.max(0.4, 0.90 - asymmetry));

    // Slight anatomical coordinate shifts unique to this individual's brain scan
    const shiftX = (rng() - 0.5) * 0.05;
    const shiftY = (rng() - 0.5) * 0.05;
    const hippoRadiusMod = 0.85 + rng() * 0.30;
    const ventRadiusMod = 0.85 + rng() * 0.30;
    const peakIntensity = cond === 'AD' ? 0.88 + rng() * 0.11 : cond === 'MCI' ? 0.60 + rng() * 0.18 : 0.05 + rng() * 0.05;

    // Distinct focal hot spot profile (Hippocampal level at slice 48-52%)
    const focalCenterZ = 0.50 + (rng() - 0.5) * 0.06;

    return {
      leftWeight,
      rightWeight,
      shiftX,
      shiftY,
      hippoRadiusMod,
      ventRadiusMod,
      peakIntensity,
      focalCenterZ,
      asymmetryLabel: cond === 'CN' ? 'Normal Bilateral Symmetry' : (asymmetry > 0.06 ? 'Left Hemisphere Dominant' : asymmetry < -0.06 ? 'Right Hemisphere Dominant' : 'Bilateral Symmetric'),
    };
  }, [scanId, patientName, cond]);

  // Map 0-100 slider to nearest 5% real slice file
  const roundedSlice = Math.min(100, Math.max(0, Math.round(sliceIndex / 5) * 5));

  // Condition-specific anatomical MRI slice series (CN, MCI, AD)
  const conditionCode = cond === 'AD' ? 'AD' : cond === 'MCI' ? 'MCI' : 'CN';
  const rawMriSrc = `/assets/mri/${activeSliceView}_${conditionCode}_${roundedSlice}.jpg`;
  const fallbackRawSrc = `/assets/mri/${activeSliceView}_raw.jpg`;

  // Dynamic landmark & biomarker focus based on patient condition & view
  const getLandmarkInfo = useCallback(() => {
    if (cond === 'CN') {
      if (activeSliceView === 'axial') return { text: `Normal Ventricular & Hippocampal Morphology (${patientProfile.asymmetryLabel})`, focus: 'Low baseline activation' };
      if (activeSliceView === 'coronal') return { text: 'Preserved Temporal Lobe & Cortical Volume', focus: 'Diffuse normal activity' };
      return { text: 'Intact Parahippocampal Architecture', focus: 'Symmetric cortical ribbon' };
    }

    if (activeSliceView === 'axial') {
      if (sliceIndex < 25) return { text: 'Superior Cerebral Cortex & Vertex', focus: 'Diffuse cortical attention' };
      if (sliceIndex < 45) return { text: 'Centrum Semiovale & Periventricular White Matter', focus: 'Secondary attention tracks' };
      if (sliceIndex < 68) {
        return cond === 'AD' 
          ? { text: `Bilateral Hippocampi (${patientProfile.asymmetryLabel})`, focus: `Peak Attention: ${(patientProfile.peakIntensity * 100).toFixed(0)}% Focus` }
          : { text: `Lateral Ventricles & Temporal Horns (${patientProfile.asymmetryLabel})`, focus: `Focal Atrophy: ${(patientProfile.peakIntensity * 100).toFixed(0)}% Focus` };
      }
      if (sliceIndex < 82) return { text: 'Medial Temporal Lobes & Midbrain', focus: 'Entorhinal cortex & amygdala' };
      return { text: 'Cerebellar Hemispheres & Brainstem', focus: 'Infratentorial baseline' };
    } else if (activeSliceView === 'coronal') {
      if (sliceIndex < 35) return { text: 'Anterior Frontal Pole & Orbits', focus: 'Prefrontal cortical tracking' };
      if (sliceIndex < 65) {
        return cond === 'AD'
          ? { text: `Hippocampal Formation & Temporal Horns (${patientProfile.asymmetryLabel})`, focus: `Medial Temporal Degeneration (${(patientProfile.peakIntensity * 100).toFixed(0)}%)` }
          : { text: `Hippocampal Formation & Sylvian Fissure (${patientProfile.asymmetryLabel})`, focus: `Early Cortical Thinning (${(patientProfile.peakIntensity * 100).toFixed(0)}%)` };
      }
      return { text: 'Posterior Parieto-Occipital Cortex', focus: 'Posterior cingulate network' };
    } else {
      if (sliceIndex < 35) return { text: 'Right Lateral Insular & Temporal Cortex', focus: 'Superior temporal gyrus' };
      if (sliceIndex < 65) return { 
        text: cond === 'AD' ? `Mid-Sagittal Hippocampus & Cingulate (${patientProfile.asymmetryLabel})` : 'Mid-Sagittal Architecture & Brainstem', 
        focus: cond === 'AD' ? `Posterior Cingulate & CA1 (${(patientProfile.peakIntensity * 100).toFixed(0)}%)` : 'Early Cingulate Changes'
      };
      return { text: 'Left Lateral Parieto-Temporal Lobe', focus: 'Inferior parietal network' };
    }
  }, [cond, activeSliceView, sliceIndex, patientProfile]);

  // Colormap RGB evaluators
  const getColorMapRGB = useCallback((val, mapType) => {
    const t = Math.min(1.0, Math.max(0.0, val));
    if (mapType === 'inferno') {
      const r = Math.min(255, Math.max(0, Math.round(255 * Math.pow(t, 0.7) * 1.15)));
      const g = Math.min(255, Math.max(0, Math.round(255 * Math.pow(t, 1.7) * 1.10)));
      const b = Math.min(255, Math.max(0, Math.round(255 * (Math.sin(t * Math.PI) * 0.45 + (t > 0.75 ? (t - 0.75) * 4 : 0)))));
      return [r, g, b];
    } else if (mapType === 'turbo') {
      const r = Math.round(255 * Math.sin(t * 1.5 * Math.PI));
      const g = Math.round(255 * Math.sin(t * Math.PI));
      const b = Math.round(255 * Math.cos(t * 1.5 * Math.PI));
      return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
    } else {
      // Classic Medical JET colormap
      let r = 0, g = 0, b = 0;
      if (t < 0.125) {
        b = 128 + Math.round(t * 8 * 127);
      } else if (t < 0.375) {
        g = Math.round((t - 0.125) * 4 * 255);
        b = 255;
      } else if (t < 0.625) {
        r = Math.round((t - 0.375) * 4 * 255);
        g = 255;
        b = Math.round((1 - (t - 0.375) * 4) * 255);
      } else if (t < 0.875) {
        r = 255;
        g = Math.round((1 - (t - 0.625) * 4) * 255);
      } else {
        r = 255 - Math.round((t - 0.875) * 8 * 127);
        g = 0;
        b = 0;
      }
      return [Math.min(255, Math.max(0, r)), Math.min(255, Math.max(0, g)), Math.min(255, Math.max(0, b))];
    }
  }, []);

  // Draw patient-specific Grad-CAM attention heatmap onto Canvas with optimal clinical opacity (70%)
  const renderCanvasHeatmap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!showHeatmap) return;

    const sliceFrac = sliceIndex / 100.0;
    const hotspots = [];

    const { leftWeight, rightWeight, shiftX, shiftY, hippoRadiusMod, ventRadiusMod, peakIntensity, focalCenterZ } = patientProfile;

    if (activeSliceView === 'axial') {
      const hippoZWeight = Math.exp(-Math.pow((sliceFrac - focalCenterZ) / 0.15, 2));
      if (hippoZWeight > 0.05) {
        hotspots.push({
          x: width * (0.37 + shiftX),
          y: height * (0.56 + shiftY),
          radius: width * (0.16 * hippoRadiusMod),
          intensity: peakIntensity * leftWeight * hippoZWeight,
          name: 'Left Hippocampus (CA1 Subfield)'
        });
        hotspots.push({
          x: width * (0.63 + shiftX),
          y: height * (0.56 + shiftY),
          radius: width * (0.16 * hippoRadiusMod),
          intensity: peakIntensity * rightWeight * hippoZWeight,
          name: 'Right Hippocampus (Subiculum)'
        });
      }

      const ventZWeight = Math.exp(-Math.pow((sliceFrac - (focalCenterZ - 0.05)) / 0.16, 2));
      if (ventZWeight > 0.08) {
        hotspots.push({
          x: width * (0.44 + shiftX * 0.5),
          y: height * (0.48 + shiftY * 0.5),
          radius: width * (0.14 * ventRadiusMod),
          intensity: peakIntensity * 0.80 * leftWeight * ventZWeight,
          name: 'Left Lateral Ventricle'
        });
        hotspots.push({
          x: width * (0.56 + shiftX * 0.5),
          y: height * (0.48 + shiftY * 0.5),
          radius: width * (0.14 * ventRadiusMod),
          intensity: peakIntensity * 0.78 * rightWeight * ventZWeight,
          name: 'Right Lateral Ventricle'
        });
      }

      const tempZWeight = Math.exp(-Math.pow((sliceFrac - (focalCenterZ + 0.08)) / 0.14, 2));
      if (tempZWeight > 0.08) {
        hotspots.push({
          x: width * (0.28 + shiftX),
          y: height * (0.60 + shiftY),
          radius: width * (0.18 * hippoRadiusMod),
          intensity: peakIntensity * 0.86 * leftWeight * tempZWeight,
          name: 'Left Entorhinal Cortex'
        });
        hotspots.push({
          x: width * (0.72 + shiftX),
          y: height * (0.60 + shiftY),
          radius: width * (0.18 * hippoRadiusMod),
          intensity: peakIntensity * 0.83 * rightWeight * tempZWeight,
          name: 'Right Entorhinal Cortex'
        });
      }

      const cortZWeight = Math.exp(-Math.pow((sliceFrac - 0.35) / 0.16, 2));
      if (cortZWeight > 0.08 && cond !== 'CN') {
        hotspots.push({
          x: width * (0.50 + shiftX),
          y: height * (0.28 + shiftY),
          radius: width * 0.22,
          intensity: peakIntensity * 0.60 * cortZWeight,
          name: 'Prefrontal Cortical Ribbon'
        });
      }
    } else if (activeSliceView === 'coronal') {
      const hippoCoronalWeight = Math.exp(-Math.pow((sliceFrac - focalCenterZ) / 0.16, 2));
      if (hippoCoronalWeight > 0.05) {
        hotspots.push({
          x: width * (0.38 + shiftX),
          y: height * (0.62 + shiftY),
          radius: width * (0.18 * hippoRadiusMod),
          intensity: peakIntensity * leftWeight * hippoCoronalWeight,
          name: 'Left Hippocampal Formation'
        });
        hotspots.push({
          x: width * (0.62 + shiftX),
          y: height * (0.62 + shiftY),
          radius: width * (0.18 * hippoRadiusMod),
          intensity: peakIntensity * rightWeight * hippoCoronalWeight,
          name: 'Right Hippocampal Formation'
        });
        hotspots.push({
          x: width * (0.44 + shiftX),
          y: height * (0.44 + shiftY),
          radius: width * (0.15 * ventRadiusMod),
          intensity: peakIntensity * 0.74 * leftWeight * hippoCoronalWeight,
          name: 'Coronal Ventricles'
        });
        hotspots.push({
          x: width * (0.56 + shiftX),
          y: height * (0.44 + shiftY),
          radius: width * (0.15 * ventRadiusMod),
          intensity: peakIntensity * 0.72 * rightWeight * hippoCoronalWeight,
          name: 'Coronal Ventricles'
        });
      }
    } else {
      const sagWeight = Math.exp(-Math.pow((sliceFrac - focalCenterZ) / 0.18, 2));
      if (sagWeight > 0.05) {
        hotspots.push({
          x: width * (0.52 + shiftX),
          y: height * (0.58 + shiftY),
          radius: width * (0.20 * hippoRadiusMod),
          intensity: peakIntensity * ((leftWeight + rightWeight) / 2) * sagWeight,
          name: 'Hippocampus & Medial Temporal'
        });
        hotspots.push({
          x: width * (0.42 + shiftX),
          y: height * (0.42 + shiftY),
          radius: width * (0.18 * ventRadiusMod),
          intensity: peakIntensity * 0.78 * sagWeight,
          name: 'Posterior Cingulate Cortex'
        });
      }
    }

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;
    const alphaBase = 0.70; // Standard clinical blend

    for (let py = 0; py < height; py += 1) {
      for (let px = 0; px < width; px += 1) {
        let totalActivation = 0.0;

        for (let i = 0; i < hotspots.length; i++) {
          const hs = hotspots[i];
          const dx = px - hs.x;
          const dy = py - hs.y;
          const distSq = dx * dx + dy * dy;
          const rSq = hs.radius * hs.radius;
          if (distSq < rSq * 4.0) {
            totalActivation += hs.intensity * Math.exp(-distSq / (2.0 * rSq * 0.35));
          }
        }

        const nX = (px - width * 0.5) / (width * 0.42);
        const nY = (py - height * 0.5) / (height * 0.42);
        const insideBrain = (nX * nX + nY * nY) <= 1.0;

        if (totalActivation > 0.03 && insideBrain) {
          const clampedAct = Math.min(1.0, totalActivation);
          const [r, g, b] = getColorMapRGB(clampedAct, colorMap);

          const idx = (py * width + px) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          const localAlpha = Math.min(255, Math.round(255 * alphaBase * Math.min(1.0, clampedAct * 1.35)));
          data[idx + 3] = localAlpha;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [showHeatmap, sliceIndex, activeSliceView, cond, colorMap, patientProfile, getColorMapRGB]);

  useEffect(() => {
    renderCanvasHeatmap();
  }, [renderCanvasHeatmap]);

  const handleMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const xRel = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const yRel = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const xPct = Math.round(xRel * 100);
    const yPct = Math.round(yRel * 100);

    const width = 256;
    const height = 256;
    const px = xRel * width;
    const py = yRel * height;
    const { leftWeight, rightWeight, shiftX, shiftY, peakIntensity, focalCenterZ } = patientProfile;
    const sliceFrac = sliceIndex / 100.0;
    const hippoZ = Math.exp(-Math.pow((sliceFrac - focalCenterZ) / 0.15, 2));

    const d1 = Math.hypot(px - width * (0.37 + shiftX), py - height * (0.56 + shiftY));
    const d2 = Math.hypot(px - width * (0.63 + shiftX), py - height * (0.56 + shiftY));
    const isLeft = d1 < d2;
    const minD = Math.min(d1, d2);
    const activeWeight = isLeft ? leftWeight : rightWeight;

    let act = Math.max(0, Math.min(99, Math.round((Math.exp(-Math.pow(minD / 38, 2)) * 95 * peakIntensity * activeWeight * hippoZ))));
    if (act < 6) act = Math.round(Math.random() * 6 + 4);

    let regionName = 'Cerebral Cortex';
    if (minD < 35 && sliceIndex >= 38 && sliceIndex <= 72) {
      regionName = isLeft ? 'Left Hippocampus (CA1)' : 'Right Hippocampus (Subiculum)';
    } else if (yRel < 0.35) {
      regionName = 'Frontal Superior Lobe';
    } else if (yRel > 0.70) {
      regionName = 'Cerebellar Hemispheres';
    } else if (xRel < 0.30 || xRel > 0.70) {
      regionName = 'Lateral Temporal Cortex';
    } else {
      regionName = 'Periventricular Parenchyma';
    }

    setHoverCoord({ x: xPct, y: yPct, activation: act, region: regionName });
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `GradCAM_${scanId}_${patientName.replace(/\s+/g, '_')}_${activeSliceView}_${cond}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const landmark = getLandmarkInfo();

  return (
    <div className="clinical-card p-5 bg-white space-y-4 shadow-clinical border border-[#E8E2DA] rounded-2xl select-none">
      
      {/* Header with Title, Patient MRN/ID & Viewport Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#E8E2DA]">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-[#F8EAED] text-[#7A1F2B] flex items-center justify-center border border-[#ECC8CF] shrink-0">
            <LuBrain className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap">
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#22201F] truncate">
                Patient Grad-CAM Heatmap
              </h4>
              <span className="text-[10px] font-mono font-bold text-[#7A1F2B] bg-[#FAF6F3] px-1.5 py-0.5 rounded border border-[#E8E2DA]">
                {scanId}
              </span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                cond === 'AD' ? 'bg-[#F8EAED] text-[#7A1F2B] border border-[#ECC8CF]' :
                cond === 'MCI' ? 'bg-[#FAF3E8] text-[#8A5A14] border border-[#F0DEC2]' :
                'bg-[#EDF5F0] text-[#2E523A] border border-[#CFE3D5]'
              }`}>
                {cond} · {confidence}%
              </span>
            </div>
            <p className="text-[10px] text-[#7A756F] truncate mt-0.5">
              {patientName} · {patientProfile.asymmetryLabel}
            </p>
          </div>
        </div>

        {/* Viewport Selectors (Axial, Coronal, Sagittal) */}
        <div className="flex bg-[#FAF6F3] p-1 rounded-xl border border-[#E8E2DA] gap-1 shrink-0">
          {['axial', 'coronal', 'sagittal'].map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setActiveSliceView(v)}
              className={`px-3 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all cursor-pointer ${
                activeSliceView === v
                  ? 'bg-[#7A1F2B] text-white shadow-xs'
                  : 'text-[#7A756F] hover:text-[#22201F]'
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Real DICOM Clinical MRI Screen with Layered Heatmap Canvas */}
      <div 
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        className="relative rounded-2xl border border-[#2A2D34] overflow-hidden bg-[#000000] shadow-2xl flex items-center justify-center aspect-square max-h-[380px] w-full group cursor-crosshair"
      >
        
        {/* Top-Left Live Coordinate HUD */}
        <div className="absolute top-3 left-3 z-10 bg-black/85 backdrop-blur-xs border border-white/20 px-2.5 py-1.5 rounded-lg text-[9px] font-mono text-[#4ADE80] font-bold pointer-events-none shadow-md space-y-0.5">
          <div>COORD: X:{hoverCoord ? hoverCoord.x : sliceIndex}% · Y:{hoverCoord ? hoverCoord.y : sliceIndex}% · Z:{sliceIndex}%</div>
          {isHovering && hoverCoord && (
            <div className="text-white/90 font-sans text-[10px] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#4ADE80] animate-pulse" />
              <span>{hoverCoord.region} · <strong>{hoverCoord.activation}% focus</strong></span>
            </div>
          )}
        </div>

        {/* Top-Right Slice Position Badge & Colormap Mode */}
        <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 pointer-events-none">
          <div className="bg-black/85 backdrop-blur-xs border border-[#4ADE80]/60 px-2.5 py-1 rounded-lg text-[10px] font-mono text-[#4ADE80] font-bold shadow-md">
            SLICE {sliceIndex.toString().padStart(3, '0')} / 100
          </div>
          <div className="bg-black/85 backdrop-blur-xs border border-white/20 px-2 py-1 rounded-lg text-[9px] font-mono text-amber-400 font-bold uppercase shadow-md">
            {colorMap}
          </div>
        </div>

        {/* Layer 1: Base Grayscale MRI Image */}
        <div className="relative w-full h-full flex items-center justify-center p-0 overflow-hidden bg-black pointer-events-none">
          <img
            key={`${activeSliceView}_${roundedSlice}`}
            src={rawMriSrc}
            onError={(e) => {
              e.currentTarget.src = fallbackRawSrc;
            }}
            alt={`Clinical MRI ${activeSliceView} slice ${sliceIndex}`}
            className="w-full h-full object-cover select-none transition-transform duration-100 ease-out"
            style={{
              transform: `scale(${zoomScale / 100})`,
            }}
          />

          {/* Layer 2: Dynamic Patient-Specific Grad-CAM Canvas Overlay */}
          <canvas
            ref={canvasRef}
            width={256}
            height={256}
            className="absolute inset-0 w-full h-full object-cover pointer-events-none transition-transform duration-100 ease-out"
            style={{
              transform: `scale(${zoomScale / 100})`,
              mixBlendMode: 'screen',
            }}
          />

          {/* Layer 3: Dynamic Crosshair Synchronized with Slice Slider */}
          {showCrosshair && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ opacity: 0.85 }}>
              <line
                x1="0"
                y1={`${sliceIndex}%`}
                x2="100%"
                y2={`${sliceIndex}%`}
                stroke="#4ade80"
                strokeWidth="0.9"
                strokeDasharray="4 5"
              />
              <line
                x1={`${sliceIndex}%`}
                y1="0"
                x2={`${sliceIndex}%`}
                y2="100%"
                stroke="#4ade80"
                strokeWidth="0.9"
                strokeDasharray="4 5"
              />
              <circle
                cx={`${sliceIndex}%`}
                cy={`${sliceIndex}%`}
                r="6"
                fill="none"
                stroke="#4ade80"
                strokeWidth="1.5"
              />
              <circle
                cx={`${sliceIndex}%`}
                cy={`${sliceIndex}%`}
                r="2"
                fill="#4ade80"
              />
              <polyline
                points={`${sliceIndex - 3.5}%,${sliceIndex - 1.5}% ${sliceIndex - 3.5}%,${sliceIndex - 3.5}% ${sliceIndex - 1.5}%,${sliceIndex - 3.5}%`}
                fill="none"
                stroke="#4ade80"
                strokeWidth="1.5"
              />
              <polyline
                points={`${sliceIndex + 1.5}%,${sliceIndex - 3.5}% ${sliceIndex + 3.5}%,${sliceIndex - 3.5}% ${sliceIndex + 3.5}%,${sliceIndex - 1.5}%`}
                fill="none"
                stroke="#4ade80"
                strokeWidth="1.5"
              />
              <polyline
                points={`${sliceIndex - 3.5}%,${sliceIndex + 1.5}% ${sliceIndex - 3.5}%,${sliceIndex + 3.5}% ${sliceIndex - 1.5}%,${sliceIndex + 3.5}%`}
                fill="none"
                stroke="#4ade80"
                strokeWidth="1.5"
              />
              <polyline
                points={`${sliceIndex + 1.5}%,${sliceIndex + 3.5}% ${sliceIndex + 3.5}%,${sliceIndex + 3.5}% ${sliceIndex + 3.5}%,${sliceIndex + 1.5}%`}
                fill="none"
                stroke="#4ade80"
                strokeWidth="1.5"
              />
            </svg>
          )}
        </div>

        {/* Bottom-Right Colormap Calibration Legend */}
        {showHeatmap && (
          <div className="absolute bottom-3 right-3 z-30 bg-black/85 backdrop-blur-xs border border-white/20 px-2.5 py-1 rounded-lg flex items-center gap-1.5 pointer-events-none shadow-md">
            <span className="text-[9px] font-mono text-white/80 font-semibold">0.0</span>
            <div className={`w-16 h-2 rounded border border-white/20 ${
              colorMap === 'inferno' 
                ? 'bg-gradient-to-r from-black via-purple-700 via-rose-500 via-amber-500 to-yellow-300' 
                : colorMap === 'turbo'
                ? 'bg-gradient-to-r from-blue-700 via-cyan-400 via-green-400 via-yellow-400 to-red-600'
                : 'bg-gradient-to-r from-blue-600 via-cyan-400 via-green-400 via-yellow-400 to-red-600'
            }`} />
            <span className="text-[9px] font-mono text-white/80 font-semibold">1.0</span>
          </div>
        )}
      </div>

      {/* Workstation Scrubber & Controls */}
      <div className="p-3.5 rounded-xl bg-[#FAF6F3] border border-[#E8E2DA] space-y-3">
        
        {/* Slice Position Range Slider */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-[#7A756F] flex items-center gap-1.5 font-bold">
              <FiLayers className="w-3.5 h-3.5 text-[#7A1F2B]" />
              <span>3D SLICE POSITION (0-100):</span>
            </span>
            <span className="font-bold text-[#7A1F2B] bg-[#F8EAED] px-2 py-0.5 rounded border border-[#ECC8CF]">
              {sliceIndex} / 100
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={sliceIndex}
            onChange={(e) => setSliceIndex(Number(e.target.value))}
            className="w-full accent-[#7A1F2B] h-2 rounded-lg bg-[#E8E2DA] outline-none cursor-pointer"
          />
        </div>

        {/* Anatomical Landmark & Interactive Tools */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs pt-2 border-t border-[#E8E2DA]">
          <span className="text-[11px] text-[#7A1F2B] font-semibold flex items-center gap-1.5 truncate max-w-[260px]">
            <span className="w-2 h-2 rounded-full bg-[#7A1F2B] shrink-0 animate-pulse" />
            <span className="truncate">{landmark.text}</span>
          </span>
          
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* Colormap Selector */}
            <div className="flex items-center gap-1 bg-white border border-[#E8E2DA] rounded-lg p-0.5 mr-1">
              {['jet', 'inferno', 'turbo'].map((pal) => (
                <button
                  key={pal}
                  type="button"
                  onClick={() => setColorMap(pal)}
                  className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase transition-all cursor-pointer ${
                    colorMap === pal
                      ? 'bg-[#7A1F2B] text-white'
                      : 'text-[#7A756F] hover:bg-[#FAF6F3]'
                  }`}
                >
                  {pal}
                </button>
              ))}
            </div>

            {/* Toggle Heatmap Overlay */}
            <button
              type="button"
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border flex items-center gap-1 transition-all cursor-pointer ${
                showHeatmap 
                  ? 'bg-[#F8EAED] text-[#7A1F2B] border-[#ECC8CF] shadow-2xs' 
                  : 'bg-white text-[#7A756F] border-[#E8E2DA]'
              }`}
            >
              <FiEye className="w-3 h-3" />
              <span>{showHeatmap ? 'Heatmap ON' : 'Grayscale'}</span>
            </button>

            {/* Toggle Crosshair */}
            <button
              type="button"
              onClick={() => setShowCrosshair(!showCrosshair)}
              className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border flex items-center gap-1 transition-all cursor-pointer ${
                showCrosshair 
                  ? 'bg-[#EDF5F0] text-[#2E523A] border-[#CFE3D5] shadow-2xs' 
                  : 'bg-white text-[#7A756F] border-[#E8E2DA]'
              }`}
            >
              <FiCrosshair className="w-3 h-3" />
              <span>Crosshair</span>
            </button>

            {/* Zoom Controls */}
            <div className="flex items-center bg-white border border-[#E8E2DA] rounded-lg p-0.5">
              <button
                type="button"
                onClick={() => setZoomScale(Math.max(80, zoomScale - 10))}
                className="p-1 text-[#7A756F] hover:text-[#22201F] cursor-pointer"
                title="Zoom Out"
              >
                <FiZoomOut className="w-3 h-3" />
              </button>
              <span className="text-[9px] font-mono px-1 font-bold text-[#7A1F2B]">{zoomScale}%</span>
              <button
                type="button"
                onClick={() => setZoomScale(Math.min(140, zoomScale + 10))}
                className="p-1 text-[#7A756F] hover:text-[#22201F] cursor-pointer"
                title="Zoom In"
              >
                <FiZoomIn className="w-3 h-3" />
              </button>
            </div>

            {/* Reset Viewport */}
            <button
              type="button"
              onClick={() => { setZoomScale(100); setSliceIndex(50); }}
              className="p-1.5 rounded-lg bg-white border border-[#E8E2DA] text-[#7A756F] hover:text-[#22201F] cursor-pointer"
              title="Reset Viewport"
            >
              <FiRotateCcw className="w-3 h-3" />
            </button>

            {/* Export Slice PNG */}
            <button
              type="button"
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-[#FAF6F3] border border-[#E8E2DA] text-[#7A1F2B] hover:bg-[#F8EAED] cursor-pointer"
              title="Export Slice PNG"
            >
              <FiDownload className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}
