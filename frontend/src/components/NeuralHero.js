export default function NeuralHero() {
  return (
    <section style={{
      textAlign: 'center',
      padding: '4rem 1rem 2rem',
      position: 'relative'
    }}>
      <div className="animate-float" style={{
        display: 'inline-flex',
        alignItems: 'center',
        background: 'rgba(99, 102, 241, 0.1)',
        border: '1px solid rgba(99, 102, 241, 0.3)',
        padding: '6px 16px',
        borderRadius: '30px',
        color: '#a5b4fc',
        fontSize: '0.85rem',
        fontWeight: '600',
        letterSpacing: '1px',
        marginBottom: '1.5rem',
        textTransform: 'uppercase'
      }}>
        <span style={{ width: '8px', height: '8px', background: '#818cf8', borderRadius: '50%', marginRight: '8px', boxShadow: '0 0 10px #818cf8' }}></span>
        Neural Re-Ranking Engine Live
      </div>
      
      <h1 style={{
        fontSize: 'clamp(2.5rem, 5vw, 4.5rem)',
        lineHeight: 1.1,
        marginBottom: '1.5rem',
        background: 'linear-gradient(to right, #ffffff, #a5b4fc)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent'
      }}>
        Intelligence. <br/>Personalized.
      </h1>
      
      <p style={{
        color: 'var(--text-secondary)',
        fontSize: '1.1rem',
        maxWidth: '600px',
        margin: '0 auto',
        lineHeight: 1.6
      }}>
        Experience next-generation financial news curation. Powered by BGE-m3 multi-lingual vector representations and deep predictive neural networks.
      </p>
    </section>
  );
}
