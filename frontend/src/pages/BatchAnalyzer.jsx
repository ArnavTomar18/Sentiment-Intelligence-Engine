import React, { useState } from 'react'
import { Zap, Download } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import { analyzeBatch } from '../api/client'

const EXAMPLE = `hotel: Room was clean but staff was rude
news: Scientists cure cancer overnight — shocking discovery
app: App crashes every time I open it
fashion: Size runs small but quality is great
ott: A gripping thriller with stunning visuals`

function getLabelColor(label) {
  if (!label) return 'var(--text-muted)'
  const l = label.toString().toLowerCase()
  if (l.includes('positive') || l === 'real' || l.includes('recommend') || l === 'low') return 'var(--emerald)'
  if (l.includes('negative') || l === 'fake' || l.includes('not') || l === 'high' || l === 'churn') return 'var(--rose)'
  return 'var(--cyan)'
}

export default function BatchAnalyzer() {
  const [input, setInput]   = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const [warnings, setWarnings] = useState([])

  async function run() {
    const lines = input.split('\n').map(l => l.trim()).filter(Boolean)
    if (!lines.length) return

    const payload = []
    const warns   = []
    for (const line of lines) {
      if (!line.includes(':')) { warns.push(`Skipped (no colon): "${line}"`); continue }
      const [domain, ...rest] = line.split(':')
      const text = rest.join(':').trim()
      if (!text) { warns.push(`Skipped (empty text): "${line}"`); continue }
      payload.push({ domain: domain.trim().toLowerCase(), text })
    }
    setWarnings(warns)
    if (!payload.length) return

    setLoading(true); setError(null); setResults([])
    try { setResults(await analyzeBatch(payload)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  function downloadCSV() {
    if (!results.length) return
    const keys = Object.keys(results[0])
    const header = keys.join(',')
    const rows = results.map(r => keys.map(k => `"${String(r[k] ?? '').replace(/"/g, '""')}"`).join(','))
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'batch_results.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  const lineCount = input.split('\n').filter(l => l.trim()).length

  return (
    <>
      <Topbar title="Batch Analyzer" subtitle="Analyze multiple texts at once" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / Batch Analyzer</div>
          <h1 className="page-title">Batch <span>Analyzer</span></h1>
          <p className="page-desc">Analyze multiple reviews across different domains in a single request.</p>
        </div>

        <div className="card fade-up fade-up-1" style={{ marginBottom: 20 }}>
          <div className="card-title">Format Guide</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
            One review per line as <span style={{ color: 'var(--cyan)' }}>domain: your text</span>.
            Valid domains: <span style={{ color: 'var(--amber)' }}>hotel, news, app, fashion, ott</span>
          </div>
          <pre style={{
            background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
            padding: '12px 16px', fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)',
            overflowX: 'auto', lineHeight: 1.8,
          }}>
            {EXAMPLE}
          </pre>
          <button onClick={() => setInput(EXAMPLE)} style={{
            marginTop: 10, padding: '5px 14px', background: 'var(--bg-elevated)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            color: 'var(--cyan-dim)', fontFamily: 'var(--font-mono)', fontSize: 12, cursor: 'pointer',
          }}>
            Load example ↑
          </button>
        </div>

        <div className="card fade-up fade-up-2" style={{ marginBottom: 20 }}>
          <div className="card-title">Input</div>
          <div className="input-group">
            <label className="input-label">Reviews (one per line, domain: text format)</label>
            <textarea
              className="textarea"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={EXAMPLE}
              style={{ minHeight: 200, fontFamily: 'var(--font-mono)', fontSize: 13 }}
              rows={10}
            />
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 6 }}>
              {lineCount} {lineCount === 1 ? 'review' : 'reviews'} queued
            </div>
          </div>
          <button className="btn-predict" disabled={!input.trim() || loading} onClick={run}>
            {loading ? <><div className="spinner" /> Analyzing {lineCount} items…</> : <><Zap size={16} /> Analyze All</>}
          </button>
          {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}
          {warnings.map((w, i) => (
            <div key={i} style={{ marginTop: 8, fontSize: 12, color: 'var(--amber)', fontFamily: 'var(--font-mono)', padding: '5px 10px', background: 'var(--amber-glow)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(245,158,11,0.2)' }}>
              ⚠ {w}
            </div>
          ))}
        </div>

        {results.length > 0 && (
          <div className="card fade-up">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div className="card-title" style={{ marginBottom: 0 }}>
                Results — {results.length} {results.length === 1 ? 'prediction' : 'predictions'}
              </div>
              <button onClick={downloadCSV} style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px',
                background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', color: 'var(--cyan)', fontFamily: 'var(--font-mono)',
                fontSize: 12, cursor: 'pointer',
              }}>
                <Download size={13} /> Export CSV
              </button>
            </div>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Domain</th>
                    <th>Text</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Task</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => {
                    const label = r.label || r.prediction || r.sentiment || '—'
                    const conf  = r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '—'
                    const text  = r.text || r.input || '—'
                    return (
                      <tr key={i}>
                        <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                        <td><span className="tag cyan">{r.domain || '—'}</span></td>
                        <td style={{ maxWidth: 220 }}>{text.length > 70 ? text.slice(0, 70) + '…' : text}</td>
                        <td style={{ color: getLabelColor(label), fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{label}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>{conf}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>{r.task || '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </>
  )
}