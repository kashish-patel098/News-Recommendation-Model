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
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {item.category && item.category.slice(0, 3).map(c => (
                    <span key={c} style={{ fontSize: '0.7rem', padding: '4px 8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', color: '#cbd5e1' }}>
                      {c}
                    </span>
                  ))}
                </div>
                {(() => {
                  const ts = item.timestamp || (item.full_article && item.full_article.published_time_unix);
                  let displayDate = "";
                  if (ts) {
                    const parsed = new Date(isNaN(ts) ? ts : Number(ts));
                    if (!isNaN(parsed)) {
                      displayDate = parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
                    }
                  }
                  if (!displayDate && item.full_article?.published_time) {
                    displayDate = item.full_article.published_time.split(' ')[0]; // Show date part only if it's a long string
                  }
                  
                  return displayDate ? (
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      📅 {displayDate}
                    </span>
                  ) : null;
                })()}
              </div>
              <div style={{ 
                background: 'rgba(99, 102, 241, 0.1)', 
                color: '#818cf8', 
                padding: '4px 8px', 
                borderRadius: '6px', 
                fontSize: '0.8rem', 
                fontWeight: 'bold',
                border: '1px solid rgba(99, 102, 241, 0.2)',
                whiteSpace: 'nowrap'
              }}>
                {(item.score * 100).toFixed(1)}% Match
              </div>
            </div>
            
            <h3 style={{ fontSize: isFeatured ? '2rem' : '1.25rem', marginBottom: '1rem', lineHeight: 1.3, color: '#f8fafc', fontWeight: 700 }}>
              {item.title || (item.full_article && item.full_article.title) || 'Untitled Article'}
            </h3>
            
            <p style={{ color: 'var(--text-secondary)', fontSize: isFeatured ? '1.1rem' : '0.95rem', lineHeight: 1.5, flex: 1, display: '-webkit-box', WebkitLineClamp: isFeatured ? 4 : 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
              {item.summary || (item.full_article && item.full_article.introductory_paragraph) || 'No summary available for this article.'}
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
