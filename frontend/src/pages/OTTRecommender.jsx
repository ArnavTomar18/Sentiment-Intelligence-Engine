import React, { useState, useEffect } from 'react'
import { Search, Tv2, Film, Sliders, BrainCircuit } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import TabBar from '../components/common/TabBar'
import { getOttTitles, getOttSimilar, getOttByPreference, predictOttRecommend } from '../api/client'

const TABS = [
  { key: 'shows',      label: 'Similar TV Shows', icon: '📺' },
  { key: 'movies',     label: 'Similar Movies',   icon: '🎥' },
  { key: 'preference', label: 'Find By Preference',icon: '🔮' },
  { key: 'model',      label: 'Model Score',       icon: '🤖' },
]
const PLATFORMS = ['Any', 'Netflix', 'Amazon Prime', 'Disney+', 'Hulu', 'HBO Max', 'Apple TV+']
const GENRES    = ['action', 'comedy', 'drama', 'thriller', 'horror', 'romance', 'sci-fi', 'documentary', 'animation', 'crime', 'mystery', 'fantasy']

function RecCard({ row }) {
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
      padding: 16,
      transition: 'border-color 0.15s',
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-bright)'}
      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
    >
      <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-display)', color: 'var(--text-primary)', marginBottom: 8 }}>
        🎬 {row.title || '—'}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {row.platform && <span className="tag cyan">🖥 {row.platform}</span>}
        {row.genre && <span className="tag">{(row.genre || '').slice(0, 40)}</span>}
        {row.release_year && <span className="tag">📅 {row.release_year}</span>}
        {row.age_rating && <span className="tag">🔞 {row.age_rating}</span>}
        {row.duration && <span className="tag">⏱ {row.duration}</span>}
      </div>
    </div>
  )
}

export default function OTTRecommender() {
  const [tab, setTab] = useState('shows')

  // TV Shows
  const [shows, setShows]             = useState([])
  const [showSel, setShowSel]         = useState('')
  const [showN, setShowN]             = useState(6)
  const [showRes, setShowRes]         = useState([])
  const [showLoad, setShowLoad]       = useState(false)
  const [showErr, setShowErr]         = useState(null)

  // Movies
  const [movies, setMovies]           = useState([])
  const [movieSel, setMovieSel]       = useState('')
  const [movieN, setMovieN]           = useState(6)
  const [movieRes, setMovieRes]       = useState([])
  const [movieLoad, setMovieLoad]     = useState(false)
  const [movieErr, setMovieErr]       = useState(null)

  // Preference
  const [prefType, setPrefType]       = useState('movie')
  const [prefYear, setPrefYear]       = useState('recent')
  const [prefAge, setPrefAge]         = useState(25)
  const [prefGenres, setPrefGenres]   = useState([])
  const [prefPlatform, setPrefPlatform] = useState('Any')
  const [prefRes, setPrefRes]         = useState([])
  const [prefLoad, setPrefLoad]       = useState(false)
  const [prefErr, setPrefErr]         = useState(null)

  // Model score
  const [mText, setMText]             = useState('')
  const [mModel, setMModel]           = useState('')
  const [mRes, setMRes]               = useState(null)
  const [mLoad, setMLoad]             = useState(false)
  const [mErr, setMErr]               = useState(null)

  useEffect(() => {
    getOttTitles('tv show').then(d => setShows(d.titles || [])).catch(() => {})
    getOttTitles('movie').then(d => setMovies(d.titles || [])).catch(() => {})
  }, [])

  async function findSimilarShows() {
    if (!showSel) return
    setShowLoad(true); setShowErr(null); setShowRes([])
    try { const d = await getOttSimilar(showSel, 'tv show', showN); setShowRes(d.results || d) }
    catch (e) { setShowErr(e.message) }
    finally { setShowLoad(false) }
  }

  async function findSimilarMovies() {
    if (!movieSel) return
    setMovieLoad(true); setMovieErr(null); setMovieRes([])
    try { const d = await getOttSimilar(movieSel, 'movie', movieN); setMovieRes(d.results || d) }
    catch (e) { setMovieErr(e.message) }
    finally { setMovieLoad(false) }
  }

  async function findByPreference() {
    if (!prefGenres.length) return
    setPrefLoad(true); setPrefErr(null); setPrefRes([])
    try {
      const d = await getOttByPreference({ content_type: prefType, year_preference: prefYear, age: prefAge, genres: prefGenres, platform: prefPlatform === 'Any' ? '' : prefPlatform })
      setPrefRes(d.results || d)
    } catch (e) { setPrefErr(e.message) }
    finally { setPrefLoad(false) }
  }

  async function runModelScore() {
    if (!mText.trim()) return
    setMLoad(true); setMErr(null); setMRes(null)
    try { setMRes(await predictOttRecommend(mText, mModel || undefined)) }
    catch (e) { setMErr(e.message) }
    finally { setMLoad(false) }
  }

  function toggleGenre(g) {
    setPrefGenres(prev => prev.includes(g) ? prev.filter(x => x !== g) : prev.length < 3 ? [...prev, g] : prev)
  }

  return (
    <>
      <Topbar title="OTT Recommender" subtitle="TF-IDF · Cosine Similarity" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / OTT Recommender</div>
          <h1 className="page-title">OTT Content <span>Recommender</span></h1>
          <p className="page-desc">Find similar shows and movies, or get personalised picks by preference. Powered by TF-IDF cosine similarity.</p>
        </div>

        <TabBar tabs={TABS} active={tab} onChange={setTab} />

        {/* ── TV Shows ─────────────────────────────── */}
        {tab === 'shows' && (
          <div className="fade-up">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Find Similar TV Shows</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'end', marginBottom: 16 }}>
                <div className="input-group" style={{ marginBottom: 0 }}>
                  <label className="input-label">Select a TV Show you love</label>
                  <select className="select" value={showSel} onChange={e => setShowSel(e.target.value)}>
                    <option value="">— choose a title —</option>
                    {shows.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
                <div className="input-group" style={{ marginBottom: 0 }}>
                  <label className="input-label">How many?</label>
                  <select className="select" value={showN} onChange={e => setShowN(Number(e.target.value))}>
                    {[3,4,5,6,8,10,12,15].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>
              </div>
              <button className="btn-predict" disabled={!showSel || showLoad} onClick={findSimilarShows}>
                {showLoad ? <><div className="spinner" /> Finding…</> : <><Search size={16} /> Find Similar Shows</>}
              </button>
              {showErr && <div className="error-banner" style={{ marginTop: 12 }}>{showErr}</div>}
            </div>
            {showRes.length > 0 && (
              <div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
                  Top {showRes.length} shows similar to <span style={{ color: 'var(--cyan)' }}>{showSel}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                  {showRes.map((row, i) => <RecCard key={i} row={row} />)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Movies ───────────────────────────────── */}
        {tab === 'movies' && (
          <div className="fade-up">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Find Similar Movies</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'end', marginBottom: 16 }}>
                <div className="input-group" style={{ marginBottom: 0 }}>
                  <label className="input-label">Select a Movie you love</label>
                  <select className="select" value={movieSel} onChange={e => setMovieSel(e.target.value)}>
                    <option value="">— choose a title —</option>
                    {movies.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
                <div className="input-group" style={{ marginBottom: 0 }}>
                  <label className="input-label">How many?</label>
                  <select className="select" value={movieN} onChange={e => setMovieN(Number(e.target.value))}>
                    {[3,4,5,6,8,10,12,15].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>
              </div>
              <button className="btn-predict" disabled={!movieSel || movieLoad} onClick={findSimilarMovies}>
                {movieLoad ? <><div className="spinner" /> Finding…</> : <><Search size={16} /> Find Similar Movies</>}
              </button>
              {movieErr && <div className="error-banner" style={{ marginTop: 12 }}>{movieErr}</div>}
            </div>
            {movieRes.length > 0 && (
              <div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
                  Top {movieRes.length} movies similar to <span style={{ color: 'var(--cyan)' }}>{movieSel}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                  {movieRes.map((row, i) => <RecCard key={i} row={row} />)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── By Preference ────────────────────────── */}
        {tab === 'preference' && (
          <div className="fade-up">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Find by Your Preferences</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                <div>
                  <div className="input-group">
                    <label className="input-label">What do you want to watch?</label>
                    <div className="task-tabs" style={{ marginBottom: 0 }}>
                      {['movie', 'tv show'].map(t => (
                        <button key={t} className={`task-tab${prefType === t ? ' active' : ''}`} onClick={() => setPrefType(t)}>
                          {t === 'movie' ? '🎥 Movie' : '📺 TV Show'}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="input-group">
                    <label className="input-label">Era Preference</label>
                    <div className="task-tabs" style={{ marginBottom: 0 }}>
                      <button className={`task-tab${prefYear === 'recent' ? ' active' : ''}`} onClick={() => setPrefYear('recent')}>Recent (2020+)</button>
                      <button className={`task-tab${prefYear === 'old' ? ' active' : ''}`} onClick={() => setPrefYear('old')}>Classic (Pre-2020)</button>
                    </div>
                  </div>
                  <div className="input-group">
                    <label className="input-label">Your Age</label>
                    <input type="number" min={5} max={100} value={prefAge} onChange={e => setPrefAge(Number(e.target.value))}
                      style={{ width: 100, background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 14, padding: '8px 12px' }} />
                  </div>
                  <div className="input-group">
                    <label className="input-label">Preferred Platform</label>
                    <select className="select" value={prefPlatform} onChange={e => setPrefPlatform(e.target.value)}>
                      {PLATFORMS.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <label className="input-label">Preferred Genres (pick up to 3)</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                    {GENRES.map(g => (
                      <button key={g} onClick={() => toggleGenre(g)}
                        className={`task-tab${prefGenres.includes(g) ? ' active' : ''}`}
                        style={{ padding: '5px 12px', fontSize: 12 }}>
                        {g}
                      </button>
                    ))}
                  </div>
                  {prefGenres.length > 0 && (
                    <div style={{ marginTop: 10, fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--cyan-dim)' }}>
                      Selected: {prefGenres.join(', ')}
                    </div>
                  )}
                </div>
              </div>
              <button className="btn-predict" disabled={!prefGenres.length || prefLoad} onClick={findByPreference}>
                {prefLoad ? <><div className="spinner" /> Finding your picks…</> : <><Sliders size={16} /> Find My Picks</>}
              </button>
              {prefErr && <div className="error-banner" style={{ marginTop: 12 }}>{prefErr}</div>}
            </div>
            {prefRes.length > 0 && (
              <div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
                  🎉 Found <span style={{ color: 'var(--cyan)' }}>{prefRes.length}</span> picks for you
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                  {prefRes.map((row, i) => <RecCard key={i} row={row} />)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Model Score ──────────────────────────── */}
        {tab === 'model' && (
          <div className="fade-up">
            <div style={{ fontSize: 13, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: 16,
              padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
              ℹ Uses the OTT Recommender models (Logistic / XGBoost). Predict whether a description will be recommended.
            </div>
            <div className="predict-layout">
              <div className="card">
                <div className="card-title">Input</div>
                <div className="input-group">
                  <label className="input-label">Content Description / Review</label>
                  <textarea className="textarea" value={mText} onChange={e => setMText(e.target.value)}
                    placeholder="e.g. A heartwarming story about family, loss and redemption with superb acting…"
                    rows={6} />
                </div>
                <div className="input-group">
                  <label className="input-label">Model</label>
                  <select className="select" value={mModel} onChange={e => setMModel(e.target.value)}>
                    <option value="">Auto (best model)</option>
                    <option value="logistic">Logistic Regression</option>
                    <option value="xgboost">XGBoost</option>
                  </select>
                </div>
                <button className="btn-predict" disabled={!mText.trim() || mLoad} onClick={runModelScore}>
                  {mLoad ? <><div className="spinner" /> Predicting…</> : <><BrainCircuit size={16} /> Predict Recommendation</>}
                </button>
              </div>
              {/* Reuse inline result display */}
              <div className="result-panel">
                {mErr && <div className="error-banner">{mErr}</div>}
                {mLoad && <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>Running inference…</div>}
                {mRes && (() => {
                  const label = mRes.label || mRes.prediction || '—'
                  const isPos = !label.toString().toLowerCase().includes('not') && label !== '0'
                  return (
                    <div className="fade-up">
                      <div className="card-title">Recommendation Score</div>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 16px', borderRadius: 24,
                        fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 15, marginBottom: 16,
                        ...(isPos
                          ? { background: 'rgba(16,185,129,0.12)', color: 'var(--emerald)', border: '1px solid rgba(16,185,129,0.25)' }
                          : { background: 'rgba(244,63,94,0.12)',  color: 'var(--rose)',    border: '1px solid rgba(244,63,94,0.25)' })
                      }}>
                        {isPos ? '✅' : '❌'} {label}
                      </div>
                      {mRes.confidence != null && (
                        <div className="confidence-bar-wrap">
                          <div className="confidence-label-row"><span>Confidence</span><span>{(mRes.confidence * 100).toFixed(1)}%</span></div>
                          <div className="confidence-bar"><div className="confidence-fill" style={{ width: `${(mRes.confidence * 100).toFixed(1)}%` }} /></div>
                        </div>
                      )}
                      {mRes.model && <div className="meta-item" style={{ marginTop: 8 }}><div className="meta-key">Model</div><div className="meta-val">{mRes.model}</div></div>}
                    </div>
                  )
                })()}
                {!mRes && !mLoad && !mErr && (
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', gap: 8 }}>
                    <BrainCircuit size={28} />
                    <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)' }}>Submit a description to score it</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}