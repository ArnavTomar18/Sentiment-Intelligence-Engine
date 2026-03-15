import React, { useState } from 'react'
import { SendHorizonal } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import ComparePanel from '../components/common/ComparePanel'
import EDAGallery from '../components/common/EDAGallery'
import { predictOttSentiment, predictOttViral, predictOttRecommend, compareOttModels } from '../api/client'

const TASK_TABS = [
  { key: 'sentiment',   label: 'Sentiment' },
  { key: 'viral',       label: 'Viral Probability' },
  { key: 'recommender', label: 'Recommendation' },
]
const TABS = [
  { key: 'analyze', label: 'Analyze',    icon: '🔍' },
  { key: 'compare', label: 'Compare All',icon: '⚖️' },
  { key: 'eda',     label: 'EDA',        icon: '📊' },
]
const MODELS = {
  sentiment:   ['logistic', 'xgboost', 'lightgbm'],
  viral:       ['svc', 'xgboost', 'lightgbm'],
  recommender: ['logistic', 'xgboost'],
}
const PLACEHOLDERS = {
  sentiment:   'e.g. "A gripping thriller with outstanding performances and brilliant cinematography…"',
  viral:       'e.g. "Mind-blowing plot twists and an unforgettable finale. Everyone is talking about it!"',
  recommender: 'e.g. "An emotional family drama with stunning visuals and an Oscar-worthy script…"',
}
const API_FNS = { sentiment: predictOttSentiment, viral: predictOttViral, recommender: predictOttRecommend }
const EDA_IMGS = [
  { path: 'ott/content_type.png',          caption: 'Content Type Distribution' },
  { path: 'ott/platform_distribution.png', caption: 'Platform Distribution' },
  { path: 'ott/top_genres.png',            caption: 'Top Genres' },
  { path: 'ott/release_year_trend.png',    caption: 'Content by Release Year' },
  { path: 'ott/wordcloud.png',             caption: 'Description Word Cloud' },
]

export default function OTT() {
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
      <Topbar title="OTT Content Analyzer" subtitle="8 models · 3 tasks" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / OTT</div>
          <h1 className="page-title">OTT Content <span>Analyzer</span></h1>
          <p className="page-desc">Sentiment · Viral Probability · Recommendation Score — Logistic · XGBoost · LightGBM · SVC</p>
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
            <div className="predict-layout">
              <div className="card">
                <div className="card-title">Input</div>
                <div className="input-group">
                  <label className="input-label">Content Description / Review</label>
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
              sections={[{ title: '🎬 OTT — All Models (Sentiment · Viral · Recommendation)', apiFn: compareOttModels }]}
              placeholder="e.g. An emotional family drama with stunning visuals and an Oscar-worthy script…"
            />
          </div>
        )}

        {tab === 'eda' && (
          <div className="fade-up">
            <div className="page-header" style={{ marginBottom: 20 }}>
              <h2 className="page-title" style={{ fontSize: 20 }}>OTT Dataset — <span>Exploratory Analysis</span></h2>
            </div>
            <EDAGallery images={EDA_IMGS} />
          </div>
        )}
      </div>
    </>
  )
}