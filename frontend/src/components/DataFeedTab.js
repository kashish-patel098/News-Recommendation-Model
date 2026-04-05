"use client";
import { useState, useRef, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_BASE_V1 = `${API_BASE}/api/v1`;

const styles = {
  dropZone: (dragging, hasFile) => ({
    border: `2px dashed ${dragging ? '#6366f1' : hasFile ? '#22c55e' : 'rgba(255,255,255,0.15)'}`,
    borderRadius: '16px',
    padding: '2.5rem 2rem',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    background: dragging
      ? 'rgba(99, 102, 241, 0.1)'
      : hasFile
      ? 'rgba(34, 197, 94, 0.06)'
      : 'rgba(0,0,0,0.2)',
    position: 'relative',
  }),

  progressBar: (pct) => ({
    width: `${pct}%`,
    height: '100%',
    background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
    borderRadius: '4px',
    transition: 'width 0.5s ease',
  }),

  progressTrack: {
    width: '100%',
    height: '8px',
    background: 'rgba(255,255,255,0.08)',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '0.75rem',
  },

  pill: (color) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 12px',
    borderRadius: '100px',
    fontSize: '0.78rem',
    fontWeight: 600,
    background: color === 'green'
      ? 'rgba(34,197,94,0.15)'
      : color === 'blue'
      ? 'rgba(99,102,241,0.15)'
      : color === 'orange'
      ? 'rgba(251,146,60,0.15)'
      : 'rgba(239,68,68,0.15)',
    color: color === 'green'
      ? '#4ade80'
      : color === 'blue'
      ? '#a5b4fc'
      : color === 'orange'
      ? '#fb923c'
      : '#f87171',
    border: `1px solid ${color === 'green' ? 'rgba(74,222,128,0.25)' : color === 'blue' ? 'rgba(165,180,252,0.25)' : color === 'orange' ? 'rgba(251,146,60,0.25)' : 'rgba(248,113,113,0.25)'}`,
  }),
};

function StatusPill({ status }) {
  const map = {
    queued:  { color: 'orange', icon: '⏳', label: 'Queued' },
    running: { color: 'blue',   icon: '⚙️',  label: 'Running' },
    done:    { color: 'green',  icon: '✅',  label: 'Complete' },
    error:   { color: 'red',    icon: '❌',  label: 'Error' },
  };
  const s = map[status] || map.queued;
  return <span style={styles.pill(s.color)}>{s.icon} {s.label}</span>;
}

export default function DataFeedTab() {
  const [dragging, setDragging]     = useState(false);
  const [file, setFile]             = useState(null);
  const [uploading, setUploading]   = useState(false);
  const [job, setJob]               = useState(null);
  const [pollTimer, setPollTimer]   = useState(null);
  const [error, setError]           = useState('');
  const inputRef = useRef(null);

  // -- Drag & Drop -----------------------------------------------------------
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragging(false), []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith('.csv')) {
      setFile(dropped);
      setError('');
      setJob(null);
    } else {
      setError('Only .csv files are accepted.');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError('');
      setJob(null);
    }
  };

  // -- Polling ---------------------------------------------------------------
  const startPolling = (jobId) => {
    const timer = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE_V1}/ingest/status/${jobId}`);
        const data = await res.json();
        setJob(data);
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(timer);
          setPollTimer(null);
        }
      } catch {
        clearInterval(timer);
      }
    }, 1500);
    setPollTimer(timer);
  };

  // -- Upload ----------------------------------------------------------------
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    setJob(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res  = await fetch(`${API_BASE_V1}/ingest/csv`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setJob({ ...data, status: 'queued', ingested: 0, skipped: 0 });
      startPolling(data.job_id);
    } catch (err) {
      setError(err.message || 'Upload failed. Is the API server running?');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    if (pollTimer) clearInterval(pollTimer);
    setFile(null);
    setJob(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  // -- Derived state ---------------------------------------------------------
  const progress = job
    ? job.status === 'done'
      ? 100
      : job.total > 0
      ? Math.round(((job.ingested + job.skipped) / job.total) * 100)
      : 0
    : 0;

  const isRunning = job && (job.status === 'queued' || job.status === 'running');

  // -- Render ----------------------------------------------------------------
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* Description */}
      <div style={{
        padding: '1rem 1.25rem',
        background: 'rgba(99,102,241,0.08)',
        borderRadius: '12px',
        border: '1px solid rgba(99,102,241,0.2)',
        fontSize: '0.88rem',
        color: 'var(--text-secondary)',
        lineHeight: '1.6',
      }}>
        Upload a CSV file in the same format as <code style={{ color: '#a5b4fc' }}>news_dataset.csv</code>.
        Required columns: <code style={{ color: '#a5b4fc' }}>id</code>, <code style={{ color: '#a5b4fc' }}>title</code>.
        Optional: <code style={{ color: '#a5b4fc' }}>introductory_paragraph, tags, published_time_unix, descriptive_paragraph</code>, and more.
        Articles already indexed are automatically skipped.
      </div>

      {/* Drop Zone */}
      <div
        style={styles.dropZone(dragging, !!file)}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !file && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />

        {!file ? (
          <>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>📂</div>
            <p style={{ color: '#fff', fontWeight: 600, fontSize: '1rem', marginBottom: '0.35rem' }}>
              Drop your CSV here
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              or click to browse — <strong>.csv</strong> files only
            </p>
          </>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '2rem' }}>📄</span>
            <div style={{ textAlign: 'left' }}>
              <p style={{ color: '#4ade80', fontWeight: 700, fontSize: '1rem' }}>{file.name}</p>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); handleReset(); }}
              style={{
                background: 'rgba(239,68,68,0.15)',
                border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: '8px',
                color: '#f87171',
                padding: '6px 14px',
                cursor: 'pointer',
                fontSize: '0.82rem',
                fontWeight: 600,
              }}
            >
              Remove
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: '0.85rem 1rem',
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: '10px',
          color: '#f87171',
          fontSize: '0.88rem',
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Job Status */}
      {job && (
        <div style={{
          padding: '1.25rem',
          background: 'rgba(0,0,0,0.25)',
          borderRadius: '14px',
          border: '1px solid rgba(255,255,255,0.07)',
        }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <StatusPill status={job.status} />
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                Job <code style={{ color: '#a5b4fc' }}>{job.job_id}</code>
              </span>
              {job.filename && (
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  · {job.filename}
                </span>
              )}
            </div>
            {isRunning && <span className="spinner" style={{ width: '18px', height: '18px', borderWidth: '3px' }} />}
          </div>

          {/* Progress bar */}
          <div style={styles.progressTrack}>
            <div style={styles.progressBar(progress)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            {progress}% complete
          </p>

          {/* Stats */}
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
            {[
              { label: 'Total rows',  value: job.total    ?? '—', color: '#94a3b8' },
              { label: 'Ingested',    value: job.ingested ?? 0,   color: '#4ade80' },
              { label: 'Skipped',     value: job.skipped  ?? 0,   color: '#fb923c' },
              { label: 'Errors',      value: job.errors   ?? 0,   color: '#f87171' },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '2px' }}>{label}</p>
                <p style={{ fontSize: '1.25rem', fontWeight: 700, color, fontFamily: 'Outfit, sans-serif' }}>
                  {value}
                </p>
              </div>
            ))}
          </div>

          {/* Message */}
          {job.message && (
            <p style={{
              marginTop: '0.85rem',
              fontSize: '0.85rem',
              color: job.status === 'error' ? '#f87171' : 'var(--text-secondary)',
              fontStyle: 'italic',
            }}>
              {job.message}
            </p>
          )}
        </div>
      )}

      {/* Upload Button */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
        {job?.status === 'done' && (
          <button
            onClick={handleReset}
            style={{
              padding: '12px 22px',
              borderRadius: '12px',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.1)',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '0.9rem',
            }}
          >
            Upload Another
          </button>
        )}
        <button
          onClick={handleUpload}
          disabled={!file || uploading || isRunning}
          className="btn-primary"
          style={{ minWidth: '180px' }}
        >
          {uploading || isRunning ? (
            <><span className="spinner" style={{ width: '16px', height: '16px', marginRight: '8px', borderWidth: '2px' }} />Ingesting...</>
          ) : (
            '⬆ Upload & Ingest CSV'
          )}
        </button>
      </div>
    </div>
  );
}
