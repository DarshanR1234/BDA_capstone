import React from 'react';

export default function ReasoningBox({ explanation }) {
  if (!explanation) return null;

  return (
    <div className="explanation-box">
      <div className="explanation-header">
        <span>🧠</span>
        <span>Grounded AI Evidence Reasoning</span>
      </div>
      <div className="explanation-text">{explanation}</div>
    </div>
  );
}
