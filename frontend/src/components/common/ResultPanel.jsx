import React from 'react'
import { BrainCircuit, AlertCircle, CheckCircle2, Clock } from 'lucide-react'

// ── Label resolution (preserves your existing LABEL_MAP + TASK_MAP) ──────────
const LABEL_MAP = {
  fake: 'Fake', real: 'Real',
  positive: 'Positive', negative: 'Negative', neutral: 'Neutral',
  'high churn risk': 'High Churn Risk', 'low churn risk': 'Low Churn Risk',
  'high viral': 'High Viral', 'low viral': 'Low Viral',
  recommended: 'Recommended', 'not recommended': 'Not Recommended',
  bug: 'Bug Report', feature: 'Feature Request', praise: 'Praise',
  'bug report': 'Bug Report', 'feature request': 'Feature Request',
}

const TASK_MAP = {
  fake_detection: { '0': 'Fake',            '1': 'Real'             },
  sentiment:      { '0': 'Negative',         '1': 'Positive'         },
  churn:          { '0': 'Low Churn Risk',   '1': 'High Churn Risk'  },
  viral:          { '0': 'Low Viral',        '1': 'High Viral'       },
  recommend:      { '0': 'Not Recommended',  '1': 'Recommended'      },
  feedback:       { '0': 'Bug Report',       '1': 'Feature Request', '2': 'Praise' },
}

function resolveLabel(raw, task) {
  if (raw == null) return '—'
  const s = String(raw).trim()
  const mapped = LABEL_MAP[s.toLowerCase()]
  if (mapped) return mapped
  if (task && TASK_MAP[task]?.[s] !== undefined) return TASK_MAP[task][s]
  if (s === '0') return 'Negative / Fake / No'
  if (s === '1') return 'Positive / Real / Yes'
  return s
}

function getSentimentClass(label) {
  const l = (label || '').toLowerCase()
  if (
    l.includes('positive') || l === 'real' ||
    (l.includes('recommend') && !l.includes('not')) ||
    l.includes('praise') || l === 'low churn risk' || l === 'low viral'
  ) return 'positive'
  if (
    l.includes('negative') || l === 'fake' ||
    l.includes('not recommend') || l.includes('bug') ||
    l.includes('high churn') || l.includes('high viral')
  ) return 'negative'
  if (l.includes('neutral') || l.includes('feature')) return 'neutral'
  return 'default'
}

// ── Probability row ──────────────────────────────────────────────────────────
function ProbRow({ label, prob }) {
  const pct = (prob * 100).toFixed(1)
  return (
    <div className="prob-row">
      <span className="prob-label" title={label}>{label}</span>
      <div className="prob-bar-wrap">
        <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="prob-val">{pct}%</span>
    </div>
  )
}

// ── Meta item ────────────────────────────────────────────────────────────────
function MetaItem({ label, value }) {
  if (value == null) return null
  return (
    <div className="meta-item">
      <div className="meta-key">{label}</div>
      <div className="meta-val">{value}</div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
// Props:
//   result     object | null  – API response object
//   loading    boolean        – show loading state
//   error      string | null  – show error banner
//   taskLabel  string         – card header (e.g. "Sentiment Prediction")
export default function ResultPanel({ result, loading, error, taskLabel = 'Prediction' }) {
  // ── Loading ──
  if (loading) {
    return (
      <div className="result-panel">
        <div className="result-empty">
          <div
            className="result-empty-icon"
            style={{
              background: 'var(--accent-light)',
              color: 'var(--accent)',
              animation: 'spin 1.2s linear infinite',
            }}
          >
            <BrainCircuit size={22} />
          </div>
          <p style={{ color: 'var(--accent)' }}>Running inference…</p>
        </div>
      </div>
    )
  }

  // ── Error ──
  if (error) {
    return (
      <div className="result-panel">
        <div className="card-title"><span className="dot" />{taskLabel} Result</div>
        <div className="error-banner">
          <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
          {error}
        </div>
      </div>
    )
  }

  // ── Empty ──
  if (!result) {
    return (
      <div className="result-panel">
        <div className="result-empty">
          <div className="result-empty-icon">
            <Clock size={22} />
          </div>
          <p>Submit text to see prediction</p>
        </div>
      </div>
    )
  }

  // ── Resolve label + variant ──
  const rawLabel   = result.label ?? result.prediction ?? result.sentiment ?? result.category ?? '—'
  const task       = result.task || ''
  const label      = resolveLabel(rawLabel, task)
  const variant    = getSentimentClass(label)
  const confidence = result.confidence ?? result.probability ?? null
  const allProbs   = result.all_probabilities ?? result.probabilities ?? null
  const confPct    = confidence != null ? (confidence * 100).toFixed(1) : null

  // Sort probabilities descending
  const sortedProbs = allProbs
    ? Object.entries(allProbs).sort((a, b) => b[1] - a[1])
    : null

  return (
    <div className="result-panel fade-up">
      <div className="card-title"><span className="dot" />{taskLabel} Result</div>

      {/* Sentiment badge */}
      <div className={`sentiment-badge ${variant}`}>
        <span className="sentiment-dot" />
        {label}
        {variant === 'positive' && <CheckCircle2 size={15} />}
      </div>

      {/* Confidence bar */}
      {confPct != null && (
        <div className="confidence-wrap">
          <div className="confidence-row">
            <span>Confidence</span>
            <span className="confidence-pct">{confPct}%</span>
          </div>
          <div className="conf-bar">
            <div className="conf-fill" style={{ width: `${confPct}%` }} />
          </div>
        </div>
      )}

      {/* Meta grid – all the extra fields your backend returns */}
      <div className="result-meta-grid">
        <MetaItem label="Model"  value={result.model}  />
        <MetaItem label="Domain" value={result.domain} />
        <MetaItem label="Task"   value={result.task}   />
        {confPct != null && (
          <MetaItem
            label="Score"
            value={<span style={{ color: 'var(--accent)' }}>{confPct}%</span>}
          />
        )}
        {result.rating != null && (
          <MetaItem label="Rating" value={`${result.rating} ★`} />
        )}
        {result.churn_risk != null && (
          <MetaItem label="Churn Risk"  value={resolveLabel(result.churn_risk,        'churn'   )} />
        )}
        {result.viral_probability != null && (
          <MetaItem label="Viral"       value={resolveLabel(result.viral_probability, 'viral'   )} />
        )}
        {result.genre_prediction != null && (
          <MetaItem label="Genre"       value={result.genre_prediction} />
        )}
        {result.recommend != null && (
          <MetaItem label="Recommend"   value={resolveLabel(result.recommend,         'recommend')} />
        )}
        {result.feedback_type != null && (
          <MetaItem label="Feedback"    value={resolveLabel(result.feedback_type,     'feedback' )} />
        )}
      </div>

      {/* All class probabilities */}
      {sortedProbs && sortedProbs.length > 0 && (
        <>
          <div className="section-divider" />
          <div
            style={{
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: 'var(--text-4)',
              marginBottom: 10,
            }}
          >
            Class Probabilities
          </div>
          <div className="probs-list">
            {sortedProbs.map(([cls, prob]) => (
              <ProbRow key={cls} label={cls} prob={prob} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}