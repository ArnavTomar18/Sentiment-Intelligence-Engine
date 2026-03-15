import React, { useState } from 'react'
import { SendHorizonal, CheckCircle, AlertTriangle } from 'lucide-react'

const LABEL_MAP = {
  'fake': 'Fake', 'real': 'Real',
  'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral',
  'high churn risk': 'High Churn Risk', 'low churn risk': 'Low Churn Risk',
  'high viral': 'High Viral', 'low viral': 'Low Viral',
  'recommended': 'Recommended', 'not recommended': 'Not Recommended',
  'bug': 'Bug Report', 'feature': 'Feature Request', 'praise': 'Praise',
}

const TASK_MAP = {
  fake_detection: { '0': 'Fake',           '1': 'Real' },
  sentiment:      { '0': 'Negative',        '1': 'Positive' },
  churn:          { '0': 'Low Churn Risk',  '1': 'High Churn Risk' },
  viral:          { '0': 'Low Viral',       '1': 'High Viral' },
  recommend:      { '0': 'Not Recommended', '1': 'Recommended' },
  feedback:       { '0': 'Bug Report',      '1': 'Feature Request', '2': 'Praise' },
}

function resolveLabel(raw, task) {
  if (!raw && raw !== 0) return '—'
  const s = String(raw).trim()
  const mapped = LABEL_MAP[s.toLowerCase()]
  if (mapped) return mapped
  if (task && TASK_MAP[task] && TASK_MAP[task][s] !== undefined) return TASK_MAP[task][s]
  // Infer task from model name prefix (e.g. "sentiment/logistic" → "sentiment")
  if (s === '0' || s === '1') return s === '1' ? 'Positive / Real / Yes' : 'Negative / Fake / No'
  return s
}

function inferTask(modelName) {
  if (!modelName) return null
  const m = modelName.toLowerCase()
  if (m.includes('sentiment')) return 'sentiment'
  if (m.includes('churn'))     return 'churn'
  if (m.includes('viral'))     return 'viral'
  if (m.includes('recommend')) return 'recommend'
  if (m.includes('feedback'))  return 'feedback'
  if (m.includes('news') || m.includes('logistic') || m.includes('svc') || m.includes('xgboost')) return 'fake_news_detection'
  return null
}

function getLabelVariant(label) {
  if (!label) return 'default'
  const l = label.toString().toLowerCase()
  if (l.includes('positive') || l === 'real' || (l.includes('recommend') && !l.includes('not')) ||
      l.includes('praise') || l === 'low churn risk' || l === 'low viral')
    return 'positive'
  if (l.includes('negative') || l === 'fake' || l.includes('not recommend') ||
      l.includes('bug') || l.includes('high churn') || l.includes('high viral'))
    return 'negative'
  return 'neutral'
}

const VARIANT_STYLE = {
  positive: { background: 'rgba(16,185,129,0.1)', color: 'var(--emerald)', border: '1px solid rgba(16,185,129,0.2)' },
  negative: { background: 'rgba(244,63,94,0.1)',  color: 'var(--rose)',    border: '1px solid rgba(244,63,94,0.2)' },
  neutral:  { background: 'var(--cyan-glow)',      color: 'var(--cyan)',    border: '1px solid rgba(0,212,255,0.2)' },
  default:  { background: 'var(--amber-glow)',     color: 'var(--amber)',   border: '1px solid rgba(245,158,11,0.2)' },
}

// sections: [{ title, apiFn, models: [{name, key}] }]
export default function ComparePanel({ sections, placeholder }) {
  const [text, setText] = useState('')
  const [results, setResults] = useState({})   // { sectionTitle: [{model, label, confidence}] }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [ran, setRan] = useState(false)

  async function run() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResults({})
    setRan(false)
    try {
      const out = {}
      for (const section of sections) {
        try {
          const data = await section.apiFn(text)
          // Backend returns array: [{model, label, confidence}] or similar
          out[section.title] = Array.isArray(data) ? data : data.results || []
        } catch (e) {
          out[section.title] = [{ model: 'Error', label: e.message, confidence: null }]
        }
      }
      setResults(out)
      setRan(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Input Text</div>
        <div className="input-group">
          <textarea
            className="textarea"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder={placeholder || 'Enter text to compare all models…'}
            rows={5}
          />
        </div>
        <button className="btn-predict" disabled={!text.trim() || loading} onClick={run}>
          {loading
            ? <><div className="spinner" /> Running all models…</>
            : <><SendHorizonal size={16} /> Compare All Models</>}
        </button>
        {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}
      </div>

      {ran && Object.entries(results).map(([sectionTitle, rows]) => {
        const labels = rows.map(r => resolveLabel(r.label || r.prediction || '', r.task || inferTask(r.model || '')))
        const allAgree = new Set(labels).size <= 1 && labels.length > 0
        return (
          <div key={sectionTitle} className="card fade-up" style={{ marginBottom: 16 }}>
            <div className="card-title">{sectionTitle}</div>

            {/* Columns */}
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(rows.length, 3)}, 1fr)`, gap: 12, marginBottom: 16 }}>
              {rows.map((row, i) => {
                const rawLabel = row.label || row.prediction || '—'
                const task  = row.task || inferTask(row.model || '')
                const label = resolveLabel(rawLabel, task)
                const conf  = row.confidence ?? row.confidence_str ?? null
                const v     = getLabelVariant(label)
                return (
                  <div key={i} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 14 }}>
                    <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      {row.model || row.model_name || `Model ${i+1}`}
                    </div>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 20, fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600, ...VARIANT_STYLE[v], marginBottom: conf ? 8 : 0 }}>
                      <span>●</span> {label}
                    </div>
                    {conf != null && (
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                        {typeof conf === 'string' ? conf : `${(conf * 100).toFixed(1)}%`}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {/* Agreement banner */}
            {labels.length > 1 && (
              allAgree
                ? <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 'var(--radius-sm)', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)', color: 'var(--emerald)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
                    <CheckCircle size={14} /> All models agree
                  </div>
                : <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 'var(--radius-sm)', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', color: 'var(--amber)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
                    <AlertTriangle size={14} /> Models disagree — review confidence scores
                  </div>
            )}

            {/* Summary table */}
            {rows.length > 0 && rows[0].model && (
              <div className="table-wrapper" style={{ marginTop: 16 }}>
                <table>
                  <thead><tr>
                    <th>Model</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr></thead>
                  <tbody>
                    {rows.map((row, i) => {
                      const rawL = row.label || row.prediction || '—'
                      const tsk  = row.task || inferTask(row.model || '')
                      const lbl  = resolveLabel(rawL, tsk)
                      const conf = row.confidence ?? row.confidence_str ?? '—'
                      return (
                        <tr key={i}>
                          <td>{row.model || row.model_name || `Model ${i+1}`}</td>
                          <td className="highlight">{lbl}</td>
                          <td>{typeof conf === 'number' ? `${(conf*100).toFixed(1)}%` : conf}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}