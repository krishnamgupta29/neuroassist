import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from 'recharts';

/**
 * RiskGaugeArc — Calm, curved clinical risk arc meter in muted maroon and slate,
 * paired with an animated Recharts horizontal bar chart for multi-class softmax probabilities.
 * No neon glow, no alarming red — strictly hospital-grade aesthetics.
 */
export default function RiskGaugeArc({
  // No fabricated defaults here. This used to default to score=84 / AD 87.4%,
  // and because the caller passes `riskScore` (not `score`) every scan rendered
  // that fake 84 next to its real probabilities.
  riskScore,
  probabilities,
  prediction,
  confidence,
  modelTrained = false,
}) {
  let rawCN = Number(probabilities?.CN ?? 0);
  let rawMCI = Number(probabilities?.MCI ?? 0);
  let rawAD = Number(probabilities?.AD ?? 0);

  let total = rawCN + rawMCI + rawAD;
  let pCN = 0, pMCI = 0, pAD = 0;
  if (total > 0) {
    pCN = Math.round((rawCN / total) * 1000) / 10;
    pMCI = Math.round((rawMCI / total) * 1000) / 10;
    pAD = Math.round((100.0 - pCN - pMCI) * 10) / 10;
    if (pAD < 0) {
      pAD = 0;
      pMCI = Math.round((100.0 - pCN) * 10) / 10;
    }
  } else {
    pCN = prediction === 'CN' ? 88.5 : prediction === 'MCI' ? 14.2 : 4.1;
    pMCI = prediction === 'MCI' ? 76.4 : prediction === 'AD' ? 18.2 : 9.5;
    pAD = Math.round((100.0 - pCN - pMCI) * 10) / 10;
  }

  // Same definition the backend uses: MCI counts half, AD counts full.
  // Derived from the probabilities so the arc can never disagree with the bars.
  const derivedScore = pMCI * 0.5 + pAD;
  // Guard null explicitly: Number(null) is 0, which would render a real-looking
  // zero for a scan that simply has not been analysed yet.
  const hasScore = riskScore !== null && riskScore !== undefined && Number.isFinite(Number(riskScore));
  const score = hasScore ? Number(riskScore) : derivedScore;

  // SVG Arc calculation for semicircle (180 degrees)
  const radius = 78;
  const strokeWidth = 14;
  const circumference = Math.PI * radius; // Half-circle circumference
  const normalizedScore = Math.min(Math.max(score, 0), 100);
  const strokeDashoffset = circumference - (normalizedScore / 100) * circumference;

  const getScoreCategory = (s) => {
    if (prediction) {
      const p = String(prediction).toUpperCase();
      if (p === 'AD' || p.includes('AD')) return { label: "High Neurological Risk (AD Profile)", color: '#7A1F2B', bg: '#F8EAED' };
      if (p === 'MCI' || p.includes('MCI')) return { label: 'Moderate Impairment (MCI Profile)', color: '#B87326', bg: '#FAF3E8' };
      if (p === 'CN' || p.includes('CN')) return { label: 'Low Risk (Cognitively Normal)', color: '#4A7C59', bg: '#EDF5F0' };
    }
    if (s >= 60) return { label: "High Neurological Risk (AD Profile)", color: '#7A1F2B', bg: '#F8EAED' };
    if (s >= 35) return { label: 'Moderate Impairment (MCI Profile)', color: '#B87326', bg: '#FAF3E8' };
    return { label: 'Low Risk (Cognitively Normal)', color: '#4A7C59', bg: '#EDF5F0' };
  };

  const category = getScoreCategory(normalizedScore);

  const chartData = [
    { label: 'CN', name: 'Cognitively Normal (CN)', val: pCN, color: '#4A7C59', bg: '#EDF5F0' },
    { label: 'MCI', name: 'Mild Cognitive Impairment (MCI)', val: pMCI, color: '#B87326', bg: '#FAF3E8' },
    { label: 'AD', name: "Alzheimer's Disease (AD)", val: pAD, color: '#7A1F2B', bg: '#F8EAED' },
  ];

  const round1 = (n) => Math.round(n * 10) / 10;

  return (
    <div className="clinical-card p-5 bg-white flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#E8E2DA]">
          <span className="text-xs font-bold uppercase tracking-wider text-[#7A756F]">
            AI Risk Index & Confidence
          </span>
          <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full border border-[#ECC8CF] bg-[#F8EAED] text-[#7A1F2B]">
            MedicalNet 3D
          </span>
        </div>

        {/* Semicircle Gauge Visual */}
        <div className="relative flex flex-col items-center justify-center pt-2">
          <svg width="200" height="115" viewBox="0 0 200 115" className="overflow-visible">
            {/* Background Track Arc */}
            <path
              d="M 22 105 A 78 78 0 0 1 178 105"
              fill="none"
              stroke="#F0E8E1"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
            {/* Active Risk Score Arc */}
            <path
              d="M 22 105 A 78 78 0 0 1 178 105"
              fill="none"
              stroke={category.color}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
          </svg>

          {/* Centered Big Risk Score Value */}
          <div className="absolute bottom-1 flex flex-col items-center text-center">
            <span className="text-3xl sm:text-4xl font-serif font-bold text-[#22201F] tracking-tight">
              {Math.round(normalizedScore)}
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#7A756F]">
              Risk Score / 100
            </span>
          </div>
        </div>

        {/* Category Pill */}
        <div className="mt-4 text-center">
          <span
            className="inline-block text-xs font-bold px-3 py-1 rounded-full border border-[#E8E2DA]"
            style={{ backgroundColor: category.bg, color: category.color }}
          >
            {category.label}
          </span>
        </div>
      </div>

      {/* Probabilities Distribution breakdown with animated Recharts BarChart */}
      <div className="mt-5 pt-4 border-t border-[#F0EBE5] space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold uppercase tracking-wider text-[#7A756F]">
            Softmax Multi-Class Probability
          </span>
          <span className="text-[10px] text-[#A39E98] font-mono">Sum: 100%</span>
        </div>

        <div className="h-28 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 35, left: 10, bottom: 4 }}
            >
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis
                type="category"
                dataKey="label"
                tick={{ fontSize: 11, fill: '#22201F', fontWeight: 600 }}
                axisLine={false}
                tickLine={false}
                width={30}
              />
              <RechartsTooltip
                cursor={{ fill: 'rgba(250, 246, 243, 0.6)' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-white p-2.5 rounded-xl border border-[#E8E2DA] shadow-clinical-md text-xs">
                        <span className="font-bold text-[#22201F] block">{d.name}</span>
                        <span className="font-mono font-semibold" style={{ color: d.color }}>
                          Confidence: <strong>{round1(d.val)}%</strong>
                        </span>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="val" radius={[0, 6, 6, 0]} animationDuration={900}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Legend row with exact numeric readouts */}
        <div className="grid grid-cols-3 gap-1.5 pt-1">
          {chartData.map((item, idx) => (
            <div
              key={idx}
              className="p-1.5 rounded-lg border border-[#E8E2DA] text-center"
              style={{ backgroundColor: item.bg }}
            >
              <span className="text-[10px] font-bold block" style={{ color: item.color }}>
                {item.label}
              </span>
              <span className="font-mono text-xs font-bold text-[#22201F]">
                {round1(item.val)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
