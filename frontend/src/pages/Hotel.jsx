import React, { useState } from 'react'
import { SendHorizonal } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import ComparePanel from '../components/common/ComparePanel'
import EDAGallery from '../components/common/EDAGallery'
import { predictHotelSentiment, predictHotelRating, predictHotelChurn, compareHotelModels } from '../api/client'

const TASK_TABS = [
  { key: 'sentiment', label: 'Sentiment' },
  { key: 'rating',    label: 'Rating Prediction' },
  { key: 'churn',     label: 'Churn Risk' },
]
const TABS = [
  { key: 'analyze', label: 'Analyze',    icon: '🔍' },
  { key: 'compare', label: 'Compare All',icon: '⚖️' },
  { key: 'eda',     label: 'EDA',        icon: '📊' },
]
const MODELS = {
  sentiment: ['logistic', 'xgboost', 'lightgbm'],
  rating:    ['ridge'],
  churn:     ['svc', 'xgboost', 'lightgbm'],
}
const HINTS = {
  sentiment: 'Classify the review as Positive, Negative, or Neutral.',
  rating:    'Predict the numeric star rating (1–5) using Ridge regression.',
  churn:     'Estimate whether this guest is at high risk of churning.',
}
const PLACEHOLDERS = {
  sentiment: 'e.g. "Room was spotless and staff were incredibly friendly!"',
  rating:    'e.g. "Disappointing stay. Noisy corridors, cold water, rude receptionist."',
  churn:     'e.g. "I used to stay here every month but last time was terrible. Probably won\'t return."',
}
const API_FNS = { sentiment: predictHotelSentiment, rating: predictHotelRating, churn: predictHotelChurn }
const EDA_IMGS = [
  { path: 'hotel/rating_distribution.png', caption: 'Rating Distribution' },
  { path: 'hotel/sentiment_split.png',     caption: 'Sentiment Split' },
  { path: 'hotel/review_length.png',       caption: 'Review Length Distribution' },
  { path: 'hotel/top_keywords.png',        caption: 'Top Keywords' },
  { path: 'hotel/length_by_rating.png',    caption: 'Review Length by Rating' },
  { path: 'hotel/wordcloud.png',           caption: 'Word Cloud' },
]

export default function Hotel() {
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
      <Topbar title="Hotel Reviews" subtitle="7 models · 3 tasks" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / Hotel</div>
          <h1 className="page-title">Hotel <span>Review</span> Analyzer</h1>
          <p className="page-desc">Sentiment · Churn Risk · Predicted Rating — Logistic · SVC · XGBoost · LightGBM · Ridge</p>
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
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: 16,
              padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              ℹ {HINTS[activeTask]}
            </div>
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
              sections={[{ title: '🏨 Hotel — All Models (Sentiment · Churn · Rating)', apiFn: compareHotelModels }]}
              placeholder="e.g. The breakfast was excellent but the room smelled musty and the AC was broken…"
            />
          </div>
        )}

        {tab === 'eda' && (
          <div className="fade-up">
            <div className="page-header" style={{ marginBottom: 20 }}>
              <h2 className="page-title" style={{ fontSize: 20 }}>Hotel Dataset — <span>Exploratory Analysis</span></h2>
            </div>
            <EDAGallery images={EDA_IMGS} />
          </div>
        )}
      </div>
    </>
  )
}