import React, { useState } from 'react'
import { SendHorizonal } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import ComparePanel from '../components/common/ComparePanel'
import EDAGallery from '../components/common/EDAGallery'
import { predictAppFeedback, predictAppRecommend, compareAppModels } from '../api/client'

const TASK_TABS = [
  { key: 'feedback',    label: 'Feedback Classification' },
  { key: 'recommender', label: 'Recommendation Prediction' },
]
const TABS = [
  { key: 'analyze', label: 'Analyze',    icon: '🔍' },
  { key: 'compare', label: 'Compare All',icon: '⚖️' },
  { key: 'eda',     label: 'EDA',        icon: '📊' },
]
const MODELS = {
  feedback:    ['logistic', 'xgboost', 'lightgbm'],
  recommender: ['svc', 'xgboost', 'lightgbm'],
}
const PLACEHOLDERS = {
  feedback:    'e.g. "App crashes frequently and the UI is confusing. Very disappointed."',
  recommender: 'e.g. "Great experience overall, smooth interface and helpful customer support!"',
}
const API_FNS = { feedback: predictAppFeedback, recommender: predictAppRecommend }
const EDA_IMGS = [
  { path: 'app/rating_distribution.png',   caption: 'Rating Distribution' },
  { path: 'app/top_apps.png',              caption: 'Top Apps Reviewed' },
  { path: 'app/feedback_distribution.png', caption: 'Feedback Type Distribution' },
  { path: 'app/wordcloud.png',             caption: 'Word Cloud' },
]

export default function AppReviews() {
  const [tab, setTab]         = useState('analyze')
  const [activeTask, setTask] = useState('feedback')
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
      <Topbar title="App Reviews" subtitle="6 models · 2 tasks" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / App Reviews</div>
          <h1 className="page-title">App <span>Review</span> Analyzer</h1>
          <p className="page-desc">Feedback Classification (Bug/Feature/Praise) · Recommendation Prediction — Logistic · XGBoost · LightGBM · SVC</p>
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
              sections={[{ title: '📱 App Reviews — Feedback + Recommendation (All Models)', apiFn: compareAppModels }]}
              placeholder="e.g. Love the new UI but it keeps crashing on my phone…"
            />
          </div>
        )}

        {tab === 'eda' && (
          <div className="fade-up">
            <div className="page-header" style={{ marginBottom: 20 }}>
              <h2 className="page-title" style={{ fontSize: 20 }}>App Reviews Dataset — <span>Exploratory Analysis</span></h2>
            </div>
            <EDAGallery images={EDA_IMGS} />
          </div>
        )}
      </div>
    </>
  )
}