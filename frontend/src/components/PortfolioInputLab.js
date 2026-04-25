"use client";
import { useState } from 'react';
import DataFeedTab from './DataFeedTab';

const tabStyle = (active) => ({
  flex: 1,
  padding: '16px',
  background: 'transparent',
  border: 'none',
  color: active ? '#fff' : 'var(--text-secondary)',
  borderBottom: active ? '2px solid var(--accent-color)' : '2px solid transparent',
  cursor: 'pointer',
  fontFamily: 'Outfit, sans-serif',
  fontSize: '1rem',
  fontWeight: 600,
  transition: 'all 0.3s',
  whiteSpace: 'nowrap',
});

const inputStyle = {
  width: '100%',
  padding: '12px',
  borderRadius: '8px',
  background: 'rgba(0,0,0,0.3)',
  border: '1px solid var(--border-color)',
  color: '#fff',
  fontFamily: 'Inter, sans-serif',
  fontSize: '0.95rem',
  outline: 'none',
};

export default function PortfolioInputLab({ onSubmit, loading }) {
  const [activeTab, setActiveTab]   = useState('portfolio');
  const [userId, setUserId]         = useState('user_42');
  const [interests, setInterests]   = useState('tech, macro economy, global markets');
  const [jsonInput, setJsonInput]   = useState(
    '{\n  "EQUITIES": {\n    "EQUITIES0": {\n      "summary": {\n        "investment": {\n          "holdings": {\n            "holding": [\n              {"issuerName": "RELIANCE INDUSTRIES", "description": "Oil & Energy"}\n            ]\n          }\n        }\n      }\n    }\n  }\n}'
  );
  const [clickedNews, setClickedNews] = useState('');
  const [useLatest, setUseLatest]   = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (activeTab === 'portfolio') {
      try {
        const parsed = JSON.parse(jsonInput);
        onSubmit({ 
          user_id: userId, 
          portfolio: parsed, 
          interests, 
          categories: [], 
          use_latest: useLatest 
        }, 'portfolio');
      } catch {
        alert('Invalid JSON in portfolio field.');
      }
    } else {
      if (!clickedNews.trim()) return alert('Please paste a news article.');
      onSubmit({ 
        user_id: userId, 
        clicked_news: clickedNews, 
        interests, 
        categories: [], 
        use_latest: useLatest 
      }, 'news');
    }
  };

  return (
    <div
      className="glass-panel"
      style={{ maxWidth: '860px', margin: '0 auto', width: '100%', overflow: 'hidden' }}
    >
      {/* ── Tab Bar ────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border-color)',
        background: 'rgba(0,0,0,0.2)',
        overflowX: 'auto',
      }}>
        {[
          { key: 'portfolio', label: '💼 Portfolio Engine' },
          { key: 'news',      label: '📰 Context Engine'  },
          { key: 'feed',      label: '⬆ Data Feed'        },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            style={tabStyle(activeTab === key)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab Content ────────────────────────────────────────────────── */}
      <div style={{ padding: '2rem' }}>

        {/* Data Feed tab — standalone, no shared form */}
        {activeTab === 'feed' ? (
          <DataFeedTab />
        ) : (

          /* Portfolio + Context Engine tabs share the common form */
          <form onSubmit={handleSubmit}>

            {/* User ID + Interests */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: '140px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  User ID
                </label>
                <input
                  type="text"
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <div style={{ flex: 2, minWidth: '200px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Core Interests
                </label>
                <input
                  type="text"
                  value={interests}
                  onChange={e => setInterests(e.target.value)}
                  style={inputStyle}
                />
              </div>
            </div>

            {/* Portfolio — AA JSON textarea */}
            {activeTab === 'portfolio' && (
              <div>
                <label style={{
                  display: 'flex', justifyContent: 'space-between',
                  marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)',
                }}>
                  <span>Account Aggregator JSON</span>
                  <span style={{ color: '#a5b4fc', fontSize: '0.8rem' }}>
                    EQUITIES · MUTUALFUNDS · SIP · REIT · INVIT · DEPOSIT_V2 · INSURANCE_POLICIES
                  </span>
                </label>
                <textarea
                  value={jsonInput}
                  onChange={e => setJsonInput(e.target.value)}
                  style={{
                    ...inputStyle,
                    height: '200px',
                    color: '#a5b4fc',
                    fontFamily: 'monospace',
                    resize: 'vertical',
                    lineHeight: '1.5',
                  }}
                />
              </div>
            )}

            {/* Context — clicked news textarea */}
            {activeTab === 'news' && (
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Clicked News Article
                </label>
                <textarea
                  value={clickedNews}
                  onChange={e => setClickedNews(e.target.value)}
                  placeholder="Paste the full text of the article the user just read..."
                  style={{ ...inputStyle, height: '200px', resize: 'vertical', lineHeight: '1.5' }}
                />
              </div>
            )}

            {/* Submit + Mode Toggle */}
            <div style={{ marginTop: '2.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '2rem', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                <input 
                  type="checkbox" 
                  id="useLatest" 
                  checked={useLatest}
                  onChange={e => setUseLatest(e.target.checked)}
                  style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: '#4ade80' }}
                />
                <label htmlFor="useLatest" style={{ fontSize: '0.88rem', color: '#fff', cursor: 'pointer', lineHeight: '1.4' }}>
                  Prioritize <span style={{ color: '#4ade80', fontWeight: 700 }}>Recency</span><br/>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                    Personalize only the news from the <strong>latest hourly ingest</strong> (last 100 items).
                  </span>
                </label>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="btn-primary"
                style={{ minWidth: '220px' }}
              >
                {loading
                  ? <><span className="spinner" style={{ width: '16px', height: '16px', marginRight: '8px', borderWidth: '2px' }} />Processing...</>
                  : '✦ Generate Recommendations'
                }
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
