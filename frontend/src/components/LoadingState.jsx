import React, { useState, useEffect } from 'react';

const LOADING_STEPS = [
  "Step 1: Extracting core claims & named entities via Gemini LLM...",
  "Step 2: Dual Hybrid search: BM25/TF-IDF lexical + dense semantic embeddings...",
  "Step 3: Multi-signal reranking across 67,587 Hindi & English sources...",
  "Step 4: Evidence-grounded reasoning & synthesis...",
];

export default function LoadingState() {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStepIdx((prev) => (prev + 1) % LOADING_STEPS.length);
    }, 1200);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="loading-container" style={{ display: 'block' }}>
      <div className="spinner"></div>
      <div className="loading-title">Cross-examining 67,587 articles in hybrid vector space...</div>
      <div className="loading-step">{LOADING_STEPS[stepIdx]}</div>
    </div>
  );
}
