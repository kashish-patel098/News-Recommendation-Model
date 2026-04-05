export default function NewsBentoGrid({ items, onSelectArticle }) {
  if (!items || items.length === 0) return null;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
      gap: '1.5rem',
      padding: '1rem'
    }}>
      {items.map((item, idx) => {
        // First item is larger (bento style logic)
        const isFeatured = idx === 0;
        
        return (
          <div 
            key={item.article_id} 
            className="glass-panel"
            onClick={() => onSelectArticle(item)}
            style={{
              gridColumn: isFeatured ? '1 / -1' : 'auto',
              padding: '1.5rem',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            {isFeatured && (
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'linear-gradient(90deg, #6366f1, #a855f7)' }} />
            )}
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {item.category && item.category.slice(0, 3).map(c => (
                  <span key={c} style={{ fontSize: '0.7rem', padding: '4px 8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', color: '#cbd5e1' }}>
                    {c}
                  </span>
                ))}
              </div>
              <div style={{ 
                background: 'rgba(99, 102, 241, 0.1)', 
                color: '#818cf8', 
                padding: '4px 8px', 
                borderRadius: '6px', 
                fontSize: '0.8rem', 
                fontWeight: 'bold',
                border: '1px solid rgba(99, 102, 241, 0.2)'
              }}>
                {(item.score * 100).toFixed(1)}% Match
              </div>
            </div>
            
            <h3 style={{ fontSize: isFeatured ? '2rem' : '1.25rem', marginBottom: '1rem', lineHeight: 1.3, color: '#f8fafc' }}>
              {item.title}
            </h3>
            
            <p style={{ color: 'var(--text-secondary)', fontSize: isFeatured ? '1.1rem' : '0.95rem', lineHeight: 1.5, flex: 1, display: '-webkit-box', WebkitLineClamp: isFeatured ? 4 : 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
              {item.summary}
            </p>
            
            {isFeatured && item.full_article && item.full_article.ai_image_prompt && (
               <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', borderLeft: '3px solid #8b5cf6' }}>
                 <p style={{ color: '#a855f7', fontSize: '0.8rem', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '1px' }}>AI Image Generation Context</p>
                 <p style={{ color: '#94a3b8', fontSize: '0.9rem', fontStyle: 'italic' }}>{item.full_article.ai_image_prompt}</p>
               </div>
            )}
          </div>
        )
      })}
    </div>
  );
}
