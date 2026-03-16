import React, { useState } from 'react'
import { ImageOff } from 'lucide-react'

// Images are served from FastAPI static mount at /static/eda/{domain}/{file}
// e.g. FastAPI: app.mount("/static", StaticFiles(directory="notebooks/eda"), name="static")

function EDAImage({ path, caption }) {
  const [err, setErr] = useState(false)
  const src = `public/eda/${path}`

  if (err) {
    return (
      <div style={{
        background: 'var(--bg-elevated)',
        border: '1px dashed var(--border)',
        borderRadius: 'var(--radius-md)',
        padding: 24,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        color: 'var(--text-muted)',
        minHeight: 160,
      }}>
        <ImageOff size={24} />
        <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', textAlign: 'center' }}>{caption}</div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>Image not available</div>
      </div>
    )
  }

  return (
    <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}>
      <img
        src={src}
        alt={caption}
        onError={() => setErr(true)}
        style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-md) var(--radius-md) 0 0' }}
      />
      {caption && (
        <div style={{ padding: '8px 12px', fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
          {caption}
        </div>
      )}
    </div>
  )
}

export default function EDAGallery({ images }) {
  // images: [{path, caption}] — path relative to /static/eda/
  return (
    <div>
      <div style={{
        padding: '10px 14px',
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        fontSize: 12,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
        marginBottom: 20,
      }}>
        ℹ Images served from FastAPI static mount at{' '}
        <span style={{ color: 'var(--cyan-dim)' }}>/static/eda/</span>.
        Mount <span style={{ color: 'var(--cyan-dim)' }}>frontend/public/eda</span> as a StaticFiles directory in your FastAPI app.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {images.map(({ path, caption }) => (
          <EDAImage key={path} path={path} caption={caption} />
        ))}
      </div>
    </div>
  )
}