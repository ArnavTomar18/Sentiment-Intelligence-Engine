import React, { useState } from 'react'
import { SendHorizonal, Download, Play } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import { predictAppRecommend, compareAppModels } from '../api/client'

const TABS = [
  { key: 'single',  label: 'Single Prediction', icon: '🔍' },
  { key: 'compare', label: 'Compare All Models', icon: '⚖️' },
  { key: 'batch',   label: 'Batch Predictions',  icon: '📋' },
]
const MODELS = ['svc', 'xgboost', 'lightgbm']
const MODEL_DESC = {
  svc:      'Fast linear classifier. No probability output — uses decision score.',
  xgboost:  'Gradient boosted trees. High accuracy, outputs class probabilities.',
  lightgbm: 'Fast boosted trees. Best balance of speed and accuracy.',
}

function getLabelVariant(label) {
  if (!label) return 'default'
  const l = label.toString().toLowerCase()
  if (l.includes('not') || l === '0' || l === 'false') return 'negative'
  return 'positive'
}

const VARIANT_STYLE = {
  positive: { background: 'rgba(16,185,129,0.1)', color: 'var(--emerald)', border: '1px solid rgba(16,185,129,0.2)' },
  negative: { background: 'rgba(244,63,94,0.1)',  color: 'var(--rose)',    border: '1px solid rgba(244,63,94,0.2)' },
}

export default function AppRecommender() {
  const [tab, setTab]   = useState('single')

  // Single
  const [sModel, setSModel]   = useState('')
  const [sText, setSText]     = useState('')
  const [sResult, setSResult] = useState(null)
  const [sLoad, setSLoad]     = useState(false)
  const [sErr, setSErr]       = useState(null)

  // Compare
  const [cText, setCText]   = useState('')
  const [cResult, setCResult] = useState(null)
  const [cLoad, setCLoad]   = useState(false)
  const [cErr, setCErr]     = useState(null)

  // Batch
  const [bModel, setBModel]   = useState('')
  const [bText, setBText]     = useState('')
  const [bResult, setBResult] = useState([])
  const [bLoad, setBLoad]     = useState(false)
  const [bErr, setBErr]       = useState(null)

  async function runSingle() {
    if (!sText.trim()) return
    setSLoad(true); setSErr(null); setSResult(null)
    try { setSResult(await predictAppRecommend(sText, sModel || undefined)) }
    catch (e) { setSErr(e.message) }
    finally { setSLoad(false) }
  }

  async function runCompare() {
    if (!cText.trim()) return
    setCLoad(true); setCErr(null); setCResult(null)
    try { setCResult(await compareAppModels(cText)) }
    catch (e) { setCErr(e.message) }
    finally { setCLoad(false) }
  }

  async function runBatch() {
    const lines = bText.split('\n').map(l => l.trim()).filter(Boolean)
    if (!lines.length) return
    setBLoad(true); setBErr(null); setBResult([])
    try {
      const rows = []
      for (const line of lines) {
        try {
          const r = await predictAppRecommend(line, bModel || undefined)
          rows.push({ review: line, label: r.label || r.prediction || '—', confidence: r.confidence ?? null })
        } catch {
          rows.push({ review: line, label: 'Error', confidence: null })
        }
      }
      setBResult(rows)
    } catch (e) { setBErr(e.message) }
    finally { setBLoad(false) }
  }

  function downloadCSV() {
    const header = 'Review,Prediction,Confidence\n'
    const rows = bResult.map(r =>
      `"${r.review.replace(/"/g, '""')}","${r.label}","${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—'}"`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'app_recommendations.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <Topbar title="App Recommender" subtitle="SVC · XGBoost · LightGBM" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / App Recommender</div>
          <h1 className="page-title">App <span>Recommendation</span> Predictor</h1>
          <p className="page-desc">Predict whether an app should be recommended from review text. Supports single, comparison, and batch modes.</p>
        </div>

        <TabBar tabs={TABS} active={tab} onChange={setTab} />

        {/* ── Single ─────────────────────────────────── */}
        {tab === 'single' && (
          <div className="fade-up">
            <div style={{ marginBottom: 20 }}>
              <label className="input-label">Model</label>
              <select className="select" style={{ maxWidth: 280 }} value={sModel} onChange={e => setSModel(e.target.value)}>
                <option value="">Auto (best model)</option>
                {MODELS.map(m => <option key={m} value={m}>{m.toUpperCase()}</option>)}
              </select>
              {sModel && MODEL_DESC[sModel] && (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 8, padding: '6px 10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                  ℹ {MODEL_DESC[sModel]}
                </div>
              )}
            </div>
            <div className="predict-layout">
              <div className="card">
                <div className="card-title">Input</div>
                <div className="input-group">
                  <label className="input-label">App Review</label>
                  <textarea className="textarea" value={sText} onChange={e => setSText(e.target.value)}
                    placeholder="e.g. The app is really smooth and the new update fixed all the previous bugs. Highly recommend!"
                    rows={6} />
                </div>
                <button className="btn-predict" disabled={!sText.trim() || sLoad} onClick={runSingle}>
                  {sLoad ? <><div className="spinner" /> Predicting…</> : <><SendHorizonal size={16} /> Predict Recommendation</>}
                </button>
              </div>
              <ResultPanel result={sResult} loading={sLoad} error={sErr} taskLabel="Recommendation Prediction" />
            </div>
          </div>
        )}

        {/* ── Compare ────────────────────────────────── */}
        {tab === 'compare' && (
          <div className="fade-up">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Input</div>
              <div className="input-group">
                <label className="input-label">App Review</label>
                <textarea className="textarea" value={cText} onChange={e => setCText(e.target.value)}
                  placeholder="e.g. Great app but it drains battery too fast. Would not recommend until they fix it."
                  rows={5} />
              </div>
              <button className="btn-predict" disabled={!cText.trim() || cLoad} onClick={runCompare}>
                {cLoad ? <><div className="spinner" /> Running all models…</> : <><SendHorizonal size={16} /> Compare All Models</>}
              </button>
              {cErr && <div className="error-banner" style={{ marginTop: 12 }}>{cErr}</div>}
            </div>

            {cResult && (() => {
              const rows = Array.isArray(cResult) ? cResult : cResult.results || []
              const labels = rows.map(r => r.label || r.prediction || '')
              const allAgree = new Set(labels).size <= 1
              return (
                <div className="card fade-up">
                  <div className="card-title">Results — All Models</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
                    {rows.map((row, i) => {
                      const label = row.label || row.prediction || '—'
                      const v = getLabelVariant(label)
                      return (
                        <div key={i} style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 14 }}>
                          <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>{row.model || `Model ${i+1}`}</div>
                          <div style={{ ...VARIANT_STYLE[v], display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 20, fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                            {v === 'positive' ? '✅' : '❌'} {label}
                          </div>
                          {row.confidence != null && (
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 6 }}>
                              {typeof row.confidence === 'number' ? `${(row.confidence * 100).toFixed(1)}%` : row.confidence}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                  <div style={{ padding: '8px 12px', borderRadius: 'var(--radius-sm)', fontSize: 13, fontFamily: 'var(--font-mono)',
                    ...(allAgree
                      ? { background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)', color: 'var(--emerald)' }
                      : { background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', color: 'var(--amber)' })
                  }}>
                    {allAgree ? '✅ All models agree!' : '⚠️ Models disagree — consider the majority vote or higher-confidence model'}
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* ── Batch ──────────────────────────────────── */}
        {tab === 'batch' && (
          <div className="fade-up">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Batch Configuration</div>
              <div className="input-group">
                <label className="input-label">Model for batch run</label>
                <select className="select" style={{ maxWidth: 280 }} value={bModel} onChange={e => setBModel(e.target.value)}>
                  <option value="">Auto (best model)</option>
                  {MODELS.map(m => <option key={m} value={m}>{m.toUpperCase()}</option>)}
                </select>
              </div>
              <div className="input-group">
                <label className="input-label">Reviews (one per line)</label>
                <textarea className="textarea" value={bText} onChange={e => setBText(e.target.value)}
                  placeholder={`App crashes every time I try to upload a photo.\nThis is the best budgeting app I've ever used!\nOK app but nothing special, has too many ads.`}
                  style={{ minHeight: 180 }} rows={8} />
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 6 }}>
                  {bText.split('\n').filter(l => l.trim()).length} reviews queued
                </div>
              </div>
              <button className="btn-predict" disabled={!bText.trim() || bLoad} onClick={runBatch}>
                {bLoad ? <><div className="spinner" /> Processing batch…</> : <><Play size={16} /> Run Batch</>}
              </button>
              {bErr && <div className="error-banner" style={{ marginTop: 12 }}>{bErr}</div>}
            </div>

            {bResult.length > 0 && (
              <div className="card fade-up">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                  <div className="card-title" style={{ marginBottom: 0 }}>Results — {bResult.length} predictions</div>
                  <button onClick={downloadCSV} style={{
                    display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px',
                    background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', color: 'var(--cyan)', fontFamily: 'var(--font-mono)',
                    fontSize: 12, cursor: 'pointer', transition: 'border-color 0.15s'
                  }}>
                    <Download size={13} /> Export CSV
                  </button>
                </div>
                <div className="table-wrapper">
                  <table>
                    <thead><tr><th>#</th><th>Review</th><th>Prediction</th><th>Confidence</th></tr></thead>
                    <tbody>
                      {bResult.map((row, i) => {
                        const v = getLabelVariant(row.label)
                        return (
                          <tr key={i}>
                            <td style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                            <td>{row.review.length > 80 ? row.review.slice(0, 80) + '…' : row.review}</td>
                            <td>
                              <span style={{ ...VARIANT_STYLE[v], padding: '2px 10px', borderRadius: 20, fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                                {row.label}
                              </span>
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
                              {row.confidence != null ? `${(row.confidence * 100).toFixed(1)}%` : '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}