import { useEffect } from 'react';

// Custom Parsers for Pseudo-JSON Output
function parseMarketPerception(str) {
  if (!str) return [];
  try { 
    const parsed = JSON.parse(str); 
    if (Array.isArray(parsed)) return parsed;
  } catch(e) {}
  
  const results = [];
  const blocks = str.split('tag=');
  for (let i=1; i<blocks.length; i++) {
    const block = blocks[i];
    const tagM = block.match(/^(.*?),\s*text=/);
    const textM = block.match(/text=(.*?)(?:\}|\])/);
    if (tagM || textM) {
       results.push({ tag: tagM ? tagM[1] : 'Insight', text: textM ? textM[1] : block.replace('}', '') });
    }
  }
  return results;
}

function parseEconomicImpact(str) {
  if (!str) return null;
  try { return JSON.parse(str); } catch(e) {}
  
  const scoreM = str.match(/score=(.*?),\s*reason=/);
  const reasonM = str.match(/reason=(.*?)(?:\}|\])/);
  if (scoreM || reasonM) {
    return { score: scoreM ? scoreM[1] : '-', reason: reasonM ? reasonM[1] : str.replace('}', '') };
  }
  return { reason: str.replace(/[\{\}]/g, '') };
}

function parseImpactMatrix(str) {
  if (!str) return [];
  try { 
    const parsed = JSON.parse(str); 
    if (Array.isArray(parsed)) return parsed;
  } catch(e) {}
  
  const results = [];
  const blocks = str.split('type=');
  for (let i = 1; i < blocks.length; i++) {
     const block = blocks[i];
     const typeMatch = block.match(/^(.*?),\s*entity=\[/);
     const type = typeMatch ? typeMatch[1] : 'Entity';
     
     const entities = [];
     const entBlocks = block.split('name=');
     for (let j = 1; j < entBlocks.length; j++) {
        const e = entBlocks[j];
        const nameM = e.match(/^(.*?),\s*score=/);
        const scoreM = e.match(/score=(.*?),\s*reason=/);
        const reasonM = e.match(/reason=(.*?),\s*perception_line=/);
        const percM = e.match(/perception_line=(.*?)(?:\}|\])/);
        
        if (nameM || scoreM) {
           entities.push({
              name: nameM ? nameM[1] : '-',
              score: scoreM ? scoreM[1] : '-',
              reason: reasonM ? reasonM[1] : 'No specific reason provided.',
              perception_line: percM ? percM[1] : ''
           });
        }
     }
     if (entities.length > 0) results.push({ type, entity: entities });
  }
  return results;
}

export default function InsightModal({ article, onClose }) {
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = 'auto'; };
  }, []);

  if (!article) return null;

  const fa = article.full_article || {};
  const eco = parseEconomicImpact(fa.economyimpact);
  const market = parseMarketPerception(fa.perception_lines);
  const matrix = parseImpactMatrix(fa.impact_matrix);

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000, padding: '2rem'
    }} onClick={onClose}>
      
      <div className="glass-panel animate-in" style={{
        maxWidth: '1000px', width: '100%', maxHeight: '90vh', overflowY: 'auto',
        background: '#0b0f19', padding: '0', position: 'relative',
        boxShadow: '0 25px 50px -12px rgba(99, 102, 241, 0.4)'
      }} onClick={e => e.stopPropagation()}>
        
        {/* Header Ribbon */}
        <div style={{ padding: '2.5rem 3rem', background: 'linear-gradient(to right, rgba(99, 102, 241, 0.15), rgba(0,0,0,0.4))', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          <button onClick={onClose} style={{
            position: 'absolute', top: '2rem', right: '2rem',
            background: 'rgba(255,255,255,0.08)', border: 'none', color: '#fff',
            width: '40px', height: '40px', borderRadius: '50%', cursor: 'pointer',
            fontSize: '1.2rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.2s'
          }} onMouseOver={e => e.currentTarget.style.background='rgba(255,255,255,0.15)'} onMouseOut={e => e.currentTarget.style.background='rgba(255,255,255,0.08)'}>✕</button>
          
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.5rem', alignItems: 'center' }}>
             <span style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', color: '#fff', padding: '6px 14px', borderRadius: '30px', fontSize: '0.85rem', fontWeight: 'bold', boxShadow: '0 4px 15px rgba(99, 102, 241, 0.3)' }}>
               Relevancy Match: {(article.score * 100).toFixed(0)}%
             </span>
             {(() => {
               const ts = article.timestamp || (fa && fa.published_time_unix);
               let dateStr = "";
               if (ts) {
                 const parsed = new Date(isNaN(ts) ? ts : Number(ts));
                 if (!isNaN(parsed)) dateStr = parsed.toLocaleString();
               }
               if (!dateStr && fa?.published_time) dateStr = fa.published_time;
               
               return dateStr ? (
                 <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', letterSpacing: '0.5px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                   📅 {dateStr}
                 </span>
               ) : null;
             })()}
             {/* Portfolio Tags / Categories */}
             {article.category && article.category.length > 0 && (
               <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginLeft: 'auto' }}>
                 {article.category.map((cat, idx) => (
                   <span key={idx} style={{ background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.4)', color: '#7dd3fc', padding: '4px 12px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: '600', letterSpacing: '0.5px' }}>
                     {cat}
                   </span>
                 ))}
               </div>
             )}
          </div>
          
          <h2 style={{ fontSize: '2.5rem', color: '#f8fafc', lineHeight: 1.3, marginBottom: '1rem', fontWeight: 800 }}>
            {article.title || fa.title || 'Untitled Article'}
          </h2>
          <div style={{ color: '#a5b4fc', fontSize: '1.15rem', marginTop: '1.5rem', fontStyle: 'italic', borderLeft: '4px solid #818cf8', paddingLeft: '1.5rem', lineHeight: 1.6 }}
               dangerouslySetInnerHTML={{ __html: fa.introductory_paragraph || article.summary }} />
        </div>
        
        {/* Full Content Body */}
        <div style={{ padding: '3rem', display: 'flex', flexDirection: 'column', gap: '3rem' }}>
          
          {fa.descriptive_paragraph && (
            <div>
              <h4 style={{ color: '#f8fafc', fontSize: '1.4rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ width: '8px', height: '8px', background: '#fff', borderRadius: '50%', display: 'inline-block' }}></span>
                Context & Analysis
              </h4>
              <div className="markdown-body" style={{ color: '#cbd5e1', fontSize: '1.05rem', lineHeight: 1.8 }}
                   dangerouslySetInnerHTML={{ __html: fa.descriptive_paragraph }} />
            </div>
          )}

          {fa.historical_context && (
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <h4 style={{ color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '2px', fontSize: '0.85rem', marginBottom: '1rem' }}>Historical Perspective</h4>
              <div className="markdown-body" style={{ color: '#94a3b8', fontSize: '0.95rem' }} 
                   dangerouslySetInnerHTML={{ __html: fa.historical_context }} />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '2rem' }}>
            {eco && (
              <div style={{ background: 'linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02))', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '2rem', borderRadius: '16px' }}>
                <h4 style={{ color: '#34d399', fontSize: '1.2rem', marginBottom: '1.2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                   Economic Impact
                   <span style={{ background: '#10b981', color: '#fff', padding: '4px 12px', borderRadius: '20px', fontSize: '0.85rem', fontWeight: 'bold' }}>
                     Score: {eco.score}
                   </span>
                </h4>
                <div className="markdown-body" style={{ color: '#a7f3d0', fontSize: '1rem', lineHeight: 1.7 }}
                     dangerouslySetInnerHTML={{ __html: eco.reason }} />
              </div>
            )}
            
            {market && market.length > 0 && (
              <div style={{ background: 'linear-gradient(145deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.02))', border: '1px solid rgba(245, 158, 11, 0.2)', padding: '2rem', borderRadius: '16px' }}>
                <h4 style={{ color: '#fbbf24', fontSize: '1.2rem', marginBottom: '1.5rem' }}>Market Perception</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {market.map((m, i) => (
                     <div key={i} style={{ borderLeft: '3px solid rgba(245, 158, 11, 0.5)', paddingLeft: '1.2rem' }}>
                        <div style={{ color: '#fcd34d', fontSize: '0.8rem', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '1px' }}>{m.tag.replace(/_/g, ' ')}</div>
                        <div style={{ color: '#fde68a', fontSize: '0.95rem', lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: m.text }} />
                     </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {matrix && matrix.length > 0 && (
             <div style={{ paddingTop: '1rem' }}>
                <h4 style={{ color: '#f8fafc', fontSize: '1.4rem', marginBottom: '2rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
                  Impact Ecosystem Matrix
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem' }}>
                  {matrix.map((typeBlock, idx) => (
                     <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '16px', overflow: 'hidden' }}>
                        
                        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.2rem 2rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                           <h5 style={{ color: '#c7d2fe', fontSize: '1.1rem', textTransform: 'uppercase', letterSpacing: '2px', margin: 0 }}>
                             {typeBlock.type} Level Impacts
                           </h5>
                        </div>

                        <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                           {typeBlock.entity.map((ent, eIdx) => (
                              <div key={eIdx} style={{ position: 'relative' }}>
                                 {eIdx !== 0 && <hr style={{ border: 'none', borderTop: '1px dashed rgba(255,255,255,0.1)', position: 'absolute', top: '-1.25rem', left: 0, right: 0 }} />}
                                 
                                 <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.2rem' }}>
                                    <span style={{ color: '#fff', fontSize: '1.25rem', fontWeight: '700' }}>{ent.name}</span>
                                    {ent.score !== '-' && (
                                       <span style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', padding: '4px 12px', borderRadius: '6px', fontSize: '0.85rem', border: '1px solid rgba(99, 102, 241, 0.3)', fontWeight: 'bold' }}>
                                         Score: {ent.score}
                                       </span>
                                    )}
                                 </div>
                                 
                                 <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1.5rem', background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.03)' }}>
                                    <div>
                                       <div style={{ color: '#64748b', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px', fontWeight: 'bold' }}>Reason</div>
                                       <div style={{ color: '#e2e8f0', fontSize: '0.95rem', lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: ent.reason }} />
                                    </div>
                                    <div>
                                       <div style={{ color: '#64748b', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px', fontWeight: 'bold' }}>Investor Perception</div>
                                       <div style={{ color: '#e2e8f0', fontSize: '0.95rem', lineHeight: 1.6, fontStyle: 'italic' }} dangerouslySetInnerHTML={{ __html: ent.perception_line }} />
                                    </div>
                                 </div>
                              </div>
                           ))}
                        </div>
                     </div>
                  ))}
                </div>
             </div>
          )}

        </div>
      </div>
    </div>
  );
}
