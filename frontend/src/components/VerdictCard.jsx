import React from 'react';

export default function VerdictCard({ result, originalQuery }) {
  if (!result) return null;

  const rawVerdict = (result.verdict || 'UNCERTAIN').toUpperCase();
  const confidence = Math.min(Math.max(result.confidence || 0.5, 0), 1);
  const confPercent = Math.round(confidence * 100);

  // Determine styling based on verdict
  let verdictClass = 'uncertain';
  let verdictIcon = '?';
  let verdictTitle = 'UNCERTAIN / UNVERIFIED';

  if (rawVerdict === 'FAKE') {
    verdictClass = 'fake';
    verdictIcon = '✕';
    verdictTitle = 'FAKE NEWS';
  } else if (rawVerdict === 'REAL') {
    verdictClass = 'real';
    verdictIcon = '✓';
    verdictTitle = 'AUTHENTIC NEWS';
  }

  // Radial Gauge Math (r = 30 => Circumference = 2 * PI * 30 ≈ 188.5)
  const CIRCUMFERENCE = 188.5;
  const strokeDashoffset = CIRCUMFERENCE - confidence * CIRCUMFERENCE;

  // Confidence subtext description
  let certaintyDesc = 'Safety Grounded';
  if (confidence >= 0.85) {
    certaintyDesc = 'High Certainty';
  } else if (confidence >= 0.70) {
    certaintyDesc = 'Moderate Certainty';
  }

  const entities = result.entities || [];
  const latency = (result.latency_seconds || 0).toFixed(2);
  const claimText = result.claim || originalQuery;

  return (
    <div className="result-card">
      {/* Header Top Grid */}
      <div className="result-header-grid">
        {/* Big Verdict Badge */}
        <div className={`verdict-badge ${verdictClass}`}>
          <span className="verdict-icon">{verdictIcon}</span>
          <span>{verdictTitle}</span>
        </div>

        {/* Radial Confidence Gauge */}
        <div className="confidence-gauge-box">
          <div className="radial-gauge">
            <svg viewBox="0 0 72 72">
              <circle className="gauge-bg" cx="36" cy="36" r="30"></circle>
              <circle
                className={`gauge-fill ${verdictClass}`}
                cx="36"
                cy="36"
                r="30"
                style={{
                  strokeDasharray: CIRCUMFERENCE,
                  strokeDashoffset: strokeDashoffset,
                }}
              ></circle>
            </svg>
            <div className="gauge-value">{confPercent}%</div>
          </div>
          <div className="confidence-meta">
            <span className="confidence-label">Confidence Score</span>
            <span className="confidence-subtext">{certaintyDesc}</span>
          </div>
        </div>

        {/* Entities & Telemetry Ribbon */}
        <div className="entity-ribbon">
          <span className="entity-ribbon-title">Recognized Entities &amp; Telemetry</span>
          <div className="entity-tags-list">
            {entities.slice(0, 4).map((ent, i) => (
              <span key={i} className="entity-pill">
                {ent}
              </span>
            ))}
            <span className="entity-pill date">⚡ {latency}s Latency</span>
          </div>
        </div>
      </div>

      {/* Extracted Core Claim Statement */}
      <div className="claim-box">
        <div className="claim-box-label">Grounded Core Claim</div>
        <div className="claim-box-text">"{claimText}"</div>
      </div>
    </div>
  );
}
