import React from 'react';

export default function EvidenceGrid({ evidenceList }) {
  if (!evidenceList || evidenceList.length === 0) {
    return (
      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>
        No direct corroborating evidence articles found in the repository.
      </div>
    );
  }

  return (
    <>
      <div className="evidence-header">
        <h2 className="evidence-title">
          <span>📑</span>
          <span>Retrieved Evidence Articles (Top {Math.min(evidenceList.length, 3)} of 67,587)</span>
        </h2>
        <span className="evidence-meta">Ranked by Hybrid Semantic + Lexical Reranker</span>
      </div>

      <div className="evidence-grid">
        {evidenceList.slice(0, 3).map((item, idx) => {
          const isFactCheck =
            item.evidence_type === 'FACT_CHECK' ||
            (item.title && item.title.includes('सच'));
          const badgeTypeClass = isFactCheck ? 'fact-check' : 'news';
          const badgeLabel = isFactCheck ? 'Fact Check Report' : 'News Evidence';

          const scorePct = Math.min(
            Math.round((item.evidence_score || 0.5) * 100),
            100
          );
          const cleanSnippet = item.content_snippet
            ? item.content_snippet.replace(/\s+/g, ' ').trim()
            : 'Snippet unavailable.';

          return (
            <div key={idx} className="evidence-card">
              <div>
                <div className="evidence-card-header">
                  <span className="evidence-rank">RANK #{idx + 1}</span>
                  <span className={`evidence-badge ${badgeTypeClass}`}>
                    {badgeLabel}
                  </span>
                </div>
                <div className="evidence-card-title">
                  {item.title || 'Untitled Document'}
                </div>
                <div className="evidence-card-snippet">{cleanSnippet}</div>
              </div>

              <div className="similarity-box">
                <div className="similarity-header">
                  <span>Hybrid Match Relevance</span>
                  <span>
                    {scorePct}% (ID: #{item.article_id || 'N/A'})
                  </span>
                </div>
                <div className="similarity-bar-bg">
                  <div
                    className="similarity-bar-fill"
                    style={{ width: `${scorePct}%` }}
                  ></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
