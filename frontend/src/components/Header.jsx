import React from 'react';

export default function Header({ theme, toggleTheme, systemStatus }) {
  return (
    <header className="site-header">
      <div className="container header-inner">
        <div className="brand">
          <div className="brand-icon">V</div>
          <div>
            <span className="brand-title">VERITAS AI</span>
            <span className="brand-subtitle">Big Data Fact Checker</span>
          </div>
        </div>
        <div className="header-actions">
          <div className="status-pill" title="Hybrid RAG indexes loaded in memory">
            <span className="status-dot"></span>
            <span>{systemStatus}</span>
          </div>
          <button 
            className="theme-toggle-btn" 
            onClick={toggleTheme} 
            aria-label="Toggle Day and Night Theme"
            title={theme === 'light' ? 'Switch to Night Mode' : 'Switch to Day Mode'}
          >
            {theme === 'light' ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </header>
  );
}
