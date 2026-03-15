import React, { useState } from 'react'
import { SendHorizonal } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import ComparePanel from '../components/common/ComparePanel'
import EDAGallery from '../components/common/EDAGallery'
import { predictFashionSentiment, predictFashionRating, compareFashionModels } from '../api/client'

const TASK_TABS = [
  { key: 'sentiment', label: 'Sentiment Analysis' },
  { key: 'rating',    label: 'Rating Prediction' },
]
const TABS = [
  { key: 'analyze', label: 'Analyze',    icon: '🔍' },
  { key: 'compare', label: 'Compare All',icon: '⚖️' },
  { key: 'eda',     label: 'EDA',        icon: '📊' },
]
const MODELS = {
  sentiment: ['logistic', 'xgboost', 'lightgbm'],
  rating:    ['xgboost', 'svr'],           // ← ridge removed
}
const PLACEHOLDERS = {
  sentiment: 'e.g. "The fabric quality is amazing but sizing runs small. Love the colour!"',
  rating:    'e.g. "Absolutely gorgeous dress, perfect fit, arrived on time. Will buy again."',
}
const API_FNS = { sentiment: predictFashionSentiment, rating: predictFashionRating }
const EDA_IMGS = [
  { path: 'fashion/rating_distribution.png', caption: 'Rating Distribution' },
  { path: 'fashion/top_items.png',           caption: 'Top Items Reviewed' },
  { path: 'fashion/aspect_counts.png',       caption: 'Aspect Mention Count' },
  { path: 'fashion/wordcloud.png',           caption: 'Word Cloud' },
]

export default function Fashion() {
  const [tab, setTab]         = useState('analyze')
  const [activeTask, setTask] = useState('sentiment')
  const [text, setText]       = useState('')
  const [model, setModel]     = useState('')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  async function predict() {
    if (!text.trim()) return
    setLoading(true); setError(null); setResult(null)
    try { setResult(await API_FNS[activeTask](text, model || undefined)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <>
      <Topbar title="Fashion Reviews" subtitle="5 models · 2 tasks" />  {/* ← 4→5 */}
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / Fashion</div>
          <h1 className="page-title">Fashion <span>Review</span> Analyzer</h1>
          <p className="page-desc">Sentiment Analysis · Rating Prediction — Logistic · XGBoost · LightGBM · XGBoost · SVR</p>
          {/* ↑ Ridge replaced with XGBoost + SVR */}
        </div>

        <TabBar tabs={TABS} active={tab} onChange={setTab} />

        {tab === 'analyze' && (
          <div className="fade-up">
            <div className="task-tabs">
              {TASK_TABS.map(({ key, label }) => (
                <button key={key} className={`task-tab${activeTask === key ? ' active' : ''}`}
                  onClick={() => { setTask(key); setResult(null); setError(null) }}>
                  {label}
                </button>
              ))}
            </div>
            {activeTask === 'rating' && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: 16,
                padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                ℹ Rating uses XGBoost or SVR regression — predicts stars 1–5
                {/* ↑ Ridge Regression → XGBoost or SVR */}
              </div>
            )}
            <div className="predict-layout">
              <div className="card">
                <div className="card-title">Input</div>
                <div className="input-group">
                  <label className="input-label">Review Text</label>
                  <textarea className="textarea" value={text} onChange={e => setText(e.target.value)}
                    placeholder={PLACEHOLDERS[activeTask]} rows={6} />
                </div>
                <div className="input-group">
                  <label className="input-label">Model</label>
                  <select className="select" value={model} onChange={e => setModel(e.target.value)}>
                    <option value="">Auto (best model)</option>
                    {MODELS[activeTask].map(m => <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>)}
                  </select>
                </div>
                <button className="btn-predict" disabled={!text.trim() || loading} onClick={predict}>
                  {loading ? <><div className="spinner" /> Analysing…</> : <><SendHorizonal size={16} /> Run Prediction</>}
                </button>
              </div>
              <ResultPanel result={result} loading={loading} error={error} taskLabel={TASK_TABS.find(t => t.key === activeTask)?.label} />
            </div>
          </div>
        )}

        {tab === 'compare' && (
          <div className="fade-up">
            <ComparePanel
              sections={[{ title: '👗 Fashion — Sentiment Models + Rating', apiFn: compareFashionModels }]}
              placeholder="e.g. Beautiful dress, but the stitching came apart after one wash…"
            />
          </div>
        )}

        {tab === 'eda' && (
          <div className="fade-up">
            <div className="page-header" style={{ marginBottom: 20 }}>
              <h2 className="page-title" style={{ fontSize: 20 }}>Fashion Dataset — <span>Exploratory Analysis</span></h2>
            </div>
            <EDAGallery images={EDA_IMGS} />
          </div>
        )}
      </div>
    </>
  )
}