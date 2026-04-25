"use client";

import { useState, useEffect } from 'react';
import NeuralHero from '../components/NeuralHero';
import PortfolioInputLab from '../components/PortfolioInputLab';
import NewsBentoGrid from '../components/NewsBentoGrid';
import InsightModal from '../components/InsightModal';

export default function Home() {
  const [recommendations, setRecommendations] = useState([]);
  const [latestNews, setLatestNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchingLatest, setFetchingLatest] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState(null);

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchLatest = async () => {
      setFetchingLatest(true);
      try {
        const res = await fetch(`${API}/api/v1/news/latest?limit=12`);
        const data = await res.json();
        if (data && data.recommendations) {
          setLatestNews(data.recommendations);
        }
      } catch (err) {
        console.error('Failed to fetch latest news:', err);
      } finally {
        setFetchingLatest(false);
      }
    };
    fetchLatest();
  }, [API]);

  const handleFetchRecommendations = async (payload, type) => {
    setLoading(true);
    setRecommendations([]);
    try {
      const endpoint = type === 'portfolio'
        ? `${API}/api/v1/recommend/portfolio`
        : `${API}/api/v1/recommend`;
        
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      const data = await res.json();
      if (data && data.recommendations) {
        setRecommendations(data.recommendations);
      }
    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: '2rem 1rem', maxWidth: '1400px', margin: '0 auto' }}>
      <NeuralHero />
      
      <div style={{ marginTop: '3rem', display: 'flex', flexDirection: 'column', gap: '4rem' }}>
        <PortfolioInputLab onSubmit={handleFetchRecommendations} loading={loading} />
        
        {loading && (
          <div style={{ textAlign: 'center', margin: '4rem 0' }}>
            <div className="spinner" style={{ width: '40px', height: '40px', borderWidth: '4px' }}></div>
            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Processing neural embeddings...</p>
          </div>
        )}

        {!loading && recommendations.length > 0 && (
          <div>
            <h2 style={{ marginBottom: '2rem', fontSize: '1.5rem', color: '#818cf8', display: 'flex', alignItems: 'center', gap: '12px' }}>
               <span style={{ width: '12px', height: '12px', background: '#818cf8', borderRadius: '50%' }}></span>
               Personalized Recommendations
            </h2>
            <NewsBentoGrid items={recommendations} onSelectArticle={setSelectedArticle} />
          </div>
        )}

        {!loading && recommendations.length === 0 && !fetchingLatest && latestNews.length > 0 && (
          <div>
            <h2 style={{ marginBottom: '2rem', fontSize: '1.5rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '12px' }}>
               <span style={{ width: '12px', height: '12px', background: '#4ade80', borderRadius: '50%', boxShadow: '0 0 10px #4ade80' }}></span>
               Latest Market Intelligence
            </h2>
            <NewsBentoGrid items={latestNews} onSelectArticle={setSelectedArticle} />
          </div>
        )}

        {fetchingLatest && latestNews.length === 0 && (
          <div style={{ textAlign: 'center', margin: '4rem 0' }}>
            <div className="spinner" style={{ width: '30px', height: '30px', borderWidth: '3px' }}></div>
            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Fetching latest news...</p>
          </div>
        )}
      </div>

      {selectedArticle && (
        <InsightModal article={selectedArticle} onClose={() => setSelectedArticle(null)} />
      )}
    </main>
  );
}
