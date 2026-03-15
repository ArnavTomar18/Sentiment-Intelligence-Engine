import React, { useState } from 'react'
import { SendHorizonal, ShieldCheck, ShieldAlert } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import ResultPanel from '../components/common/ResultPanel'
import TabBar from '../components/common/TabBar'
import ComparePanel from '../components/common/ComparePanel'
import EDAGallery from '../components/common/EDAGallery'
import { predictNews, compareNewsModels } from '../api/client'

const MODELS   = ['logistic', 'svc', 'xgboost']
const TABS     = [
  { key: 'analyze', label: 'Analyze',         icon: '🔍' },
  { key: 'compare', label: 'Compare All',      icon: '⚖️' },
  { key: 'eda',     label: 'EDA',              icon: '📊' },
]
const EDA_IMGS = [
  { path: 'news/label_distribution.png',   caption: 'Fake vs Real Distribution' },
  { path: 'news/subject_distribution.png', caption: 'Subject Distribution' },
  { path: 'news/article_length.png',       caption: 'Article Length Distribution' },
  { path: 'news/wordcloud_fake.png',       caption: 'Fake News Word Cloud' },
  { path: 'news/wordcloud_real.png',       caption: 'Real News Word Cloud' },
]
const REAL_EX = `Scientists at Johns Hopkins University have developed a new vaccine candidate that showed 94% efficacy in phase 3 clinical trials. The study, published in The Lancet, enrolled 45,000 participants across 12 countries.`
const FAKE_EX = `A shocking report claims that scientists discovered a miracle cure that can eliminate cancer in just 24 hours using a simple herbal drink. The story quickly spread across social media, attracting millions of views and shares. However, medical experts warn that no verified research supports this claim. Authorities say the article was created by an unverified website to gain traffic and advertising revenue. Readers are advised to check credible sources before believing or sharing such sensational health news online.`

export default function News() {
  const [tab, setTab]     = useState('analyze')
  const [text, setText]   = useState('')
  const [model, setModel] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function predict() {
    if (!text.trim()) return
    setLoading(true); setError(null); setResult(null)
    try { setResult(await predictNews(text, model || undefined)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <>
      <Topbar title="News Classifier" subtitle="3 models · Fake / Real Detection" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / News</div>
          <h1 className="page-title">Fake <span>News</span> Detection</h1>
          <p className="page-desc">Classify news articles as real or fake using TF-IDF + ML classifiers.</p>
        </div>

        <TabBar tabs={TABS} active={tab} onChange={setTab} />

        {tab === 'analyze' && (
          <div className="fade-up">
            <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
              <button className="task-tab" onClick={() => setText(REAL_EX)} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <ShieldCheck size={13} /> Load Real Example
              </button>
              <button className="task-tab" onClick={() => setText(FAKE_EX)} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <ShieldAlert size={13} /> Load Fake Example
              </button>
            </div>
            <div className="predict-layout">
              <div className="card">
                <div className="card-title">Input</div>
                <div className="input-group">
                  <label className="input-label">Article Text</label>
                  <textarea className="textarea" placeholder="Paste a news article or headline…" value={text}
                    onChange={e => setText(e.target.value)} rows={8} style={{ minHeight: 180 }} />
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
                    {text.length} chars · {text.trim().split(/\s+/).filter(Boolean).length} words
                  </div>
                </div>
                <div className="input-group">
                  <label className="input-label">Model</label>
                  <select className="select" value={model} onChange={e => setModel(e.target.value)}>
                    <option value="">Auto (best model)</option>
                    {MODELS.map(m => <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>)}
                  </select>
                </div>
                <button className="btn-predict" disabled={text.trim().length < 10 || loading} onClick={predict}>
                  {loading ? <><div className="spinner" /> Analysing…</> : <><SendHorizonal size={16} /> Classify Article</>}
                </button>
              </div>
              <ResultPanel result={result} loading={loading} error={error} taskLabel="Fake News Detection" />
            </div>
          </div>
        )}

        {tab === 'compare' && (
          <div className="fade-up">
            <ComparePanel
              sections={[{ title: '📰 News Classification — All Models', apiFn: compareNewsModels }]}
              placeholder="Paste a news article to run through all 3 models simultaneously…"
            />
          </div>
        )}

        {tab === 'eda' && (
          <div className="fade-up">
            <div className="page-header" style={{ marginBottom: 20 }}>
              <h2 className="page-title" style={{ fontSize: 20 }}>News Dataset — <span>Exploratory Analysis</span></h2>
            </div>
            <EDAGallery images={EDA_IMGS} />
          </div>
        )}
      </div>
    </>
  )
}