"use client";

import { useState } from 'react';
import NeuralHero from '../components/NeuralHero';
import PortfolioInputLab from '../components/PortfolioInputLab';
import NewsBentoGrid from '../components/NewsBentoGrid';
import InsightModal from '../components/InsightModal';

export default function Home() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState(null);

  const handleFetchRecommendations = async (payload, type) => {
    setLoading(true);
    setRecommendations([]);
    try {
      const endpoint = type === 'portfolio' 
        ? 'http://localhost:8000/api/v1/recommend/portfolio' 
        : 'http://localhost:8000/api/v1/recommend';
        
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
          <NewsBentoGrid items={recommendations} onSelectArticle={setSelectedArticle} />
        )}
      </div>

      {selectedArticle && (
        <InsightModal article={selectedArticle} onClose={() => setSelectedArticle(null)} />
      )}
    </main>
  );
}
