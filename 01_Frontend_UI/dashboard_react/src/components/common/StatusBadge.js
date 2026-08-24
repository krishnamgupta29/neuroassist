import React from 'react';

/**
 * StatusBadge — Desaturated, hospital-grade condition pills.
 * CN: Soft sage green | MCI: Soft amber | AD: Muted clinical maroon
 */
export default function StatusBadge({ status, size = 'sm', showDot = true, short = false, format }) {
  const norm = (status || '').toUpperCase();

  const isSmall = size === 'xs' || size === 'sm';
  const sizeClasses = size === 'xs'
    ? 'px-2 py-0.5 text-[11px] font-semibold'
    : isSmall
    ? 'px-2.5 py-0.5 text-xs font-semibold'
    : 'px-3.5 py-1 text-sm font-semibold';

  const isShort = short || format === 'short';

  if (norm === 'CN' || norm.includes('NORMAL')) {
    return (
      <span 
        title="CN · Cognitively Normal"
        className={`inline-flex items-center gap-1.5 rounded-full bg-[#EDF5F0] text-[#2E523A] border border-[#CFE3D5] whitespace-nowrap ${sizeClasses}`}
      >
        {showDot && <span className="w-1.5 h-1.5 rounded-full bg-[#4A7C59] flex-shrink-0" />}
        <span>{isShort ? 'CN' : 'CN · Cognitively Normal'}</span>
      </span>
    );
  }

  if (norm === 'MCI' || norm.includes('IMPAIRMENT')) {
    return (
      <span 
        title="MCI · Mild Cognitive Impairment"
        className={`inline-flex items-center gap-1.5 rounded-full bg-[#FAF3E8] text-[#8A5A14] border border-[#F0DEC2] whitespace-nowrap ${sizeClasses}`}
      >
        {showDot && <span className="w-1.5 h-1.5 rounded-full bg-[#B87326] flex-shrink-0" />}
        <span>{isShort ? 'MCI' : 'MCI · Mild Impairment'}</span>
      </span>
    );
  }

  if (norm === 'AD' || norm.includes('ALZHEIMER')) {
    return (
      <span 
        title="AD · Alzheimer's Disease"
        className={`inline-flex items-center gap-1.5 rounded-full bg-[#F8EAED] text-[#7A1F2B] border border-[#ECC8CF] whitespace-nowrap ${sizeClasses}`}
      >
        {showDot && <span className="w-1.5 h-1.5 rounded-full bg-[#7A1F2B] flex-shrink-0" />}
        <span>{isShort ? 'AD' : "AD · Alzheimer's Disease"}</span>
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full bg-[#F4F7FA] text-[#5B7C99] border border-[#CFDEEB] whitespace-nowrap ${sizeClasses}`}>
      {showDot && <span className="w-1.5 h-1.5 rounded-full bg-[#5B7C99] flex-shrink-0" />}
      <span>{status || 'Unknown'}</span>
    </span>
  );
}

