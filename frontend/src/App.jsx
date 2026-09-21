import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import SearchBar from './components/SearchBar';
import SampleChips from './components/SampleChips';
import LoadingState from './components/LoadingState';
import VerdictCard from './components/VerdictCard';
import ReasoningBox from './components/ReasoningBox';
import EvidenceGrid from './components/EvidenceGrid';
import Footer from './components/Footer';

const API_BASE = window.location.origin.includes(':8000') ? '' : 'http://127.0.0.1:8000';

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('veritas_theme') || 'light';
  });
  const [systemStatus, setSystemStatus] = useState('RAG Engine Online (67,587 Articles)');
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const resultRef = useRef(null);

  // Sync theme with document element and localStorage
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('veritas_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // Health check on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(`RAG Engine Online (${data.dataset_rows.toLocaleString()} Articles)`);
        }
      } catch (err) {
        console.warn('Backend ping check failed:', err);
      }
    }
    checkHealth();
  }, []);

  const handleVerify = async (textToVerify) => {
    if (!textToVerify.trim()) return;

    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);
    setSubmittedQuery(textToVerify);

    try {
      const res = await fetch(`${API_BASE}/api/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: textToVerify }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);

      setTimeout(() => {
        if (resultRef.current) {
          resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err) {
      console.error('Verification failed:', err);
      setErrorMsg(err.message || 'Verification failed. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectSample = (sampleQuery) => {
    setQuery(sampleQuery);
    handleVerify(sampleQuery);
  };

  return (
    <>
      <Header
        theme={theme}
        toggleTheme={toggleTheme}
        systemStatus={systemStatus}
      />

      <main className="main-content">
        <div className="container">
          <Hero />

          <SearchBar
            query={query}
            setQuery={setQuery}
            onVerify={handleVerify}
            isLoading={isLoading}
          />

          <SampleChips
            onSelectSample={handleSelectSample}
            isLoading={isLoading}
          />

          {isLoading && <LoadingState />}

          {errorMsg && (
            <div
              style={{
                background: 'var(--verdict-fake-bg)',
                border: '1px solid var(--verdict-fake-border)',
                color: 'var(--verdict-fake-text)',
                padding: '16px 20px',
                borderRadius: 'var(--radius-md)',
                marginBottom: '24px',
                textAlign: 'center',
                fontWeight: 600,
              }}
            >
              ⚠️ {errorMsg}
            </div>
          )}

          {result && (
            <section
              ref={resultRef}
              className="result-section"
              style={{ display: 'block' }}
            >
              <VerdictCard
                result={result}
                originalQuery={submittedQuery}
              />

              <ReasoningBox
                explanation={result.explanation}
              />

              <EvidenceGrid
                evidenceList={result.evidence}
              />
            </section>
          )}
        </div>
      </main>

      <Footer />
    </>
  );
}
