import React from 'react';

export default function SearchBar({ query, setQuery, onVerify, isLoading }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onVerify(query.trim());
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  return (
    <form onSubmit={handleSubmit} className="search-container">
      <span className="search-icon">🔍</span>
      <input
        type="text"
        className="search-input"
        placeholder="Enter any claim, question, or news statement to verify..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoComplete="off"
        spellCheck="false"
      />
      <div className="search-actions">
        {query.trim().length > 0 && (
          <button
            type="button"
            className="clear-btn"
            style={{ display: 'inline-block' }}
            onClick={handleClear}
            title="Clear input"
            aria-label="Clear input"
          >
            ✕
          </button>
        )}
        <button
          type="submit"
          className="verify-btn"
          disabled={isLoading || !query.trim()}
        >
          <span className="btn-text">{isLoading ? 'Verifying...' : 'Verify Claim'}</span>
          <span className="btn-icon">⚡</span>
        </button>
      </div>
    </form>
  );
}
