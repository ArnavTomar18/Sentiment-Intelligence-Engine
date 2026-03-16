import React, { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import { Trophy, TrendingUp } from 'lucide-react'
import Topbar from '../components/layout/Topbar'
import { getBestModels, getFullComparison } from '../api/client'

function StatImg({ src, caption }) {
  const [err, setErr] = useState(false)
  if (err) return null
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      {caption && <div className="card-title">{caption}</div>}
      <img src={src} alt={caption} onError={() => setErr(true)}
        style={{ width: '100%', borderRadius: 'var(--radius-md)', display: 'block' }} />
    </div>
  )
}

const DOMAIN_COLORS = {
  hotel: 'var(--amber)', news: 'var(--rose)', fashion: 'var(--violet)',
  app: 'var(--cyan)', ott: 'var(--emerald)',
}

export default function ModelComparison() {
  const [best, setBest]         = useState([])
  const [full, setFull]         = useState([])
  const [showFull, setShowFull] = useState(false)
  const [loading, setLoading]   = useState(true)
  const [err, setErr]           = useState(null)

  useEffect(() => {
    Promise.all([getBestModels(), getFullComparison()])
      .then(([b, f]) => { setBest(Array.isArray(b) ? b : b.data || []); setFull(Array.isArray(f) ? f : f.data || []) })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Prepare chart data from best models
  const chartData = best.map(row => ({
    name: `${row.domain || ''} / ${row.task || ''}`.slice(0, 20),
    f1: row.f1 != null ? parseFloat((row.f1 * 100).toFixed(1)) : null,
    accuracy: row.accuracy != null ? parseFloat((row.accuracy * 100).toFixed(1)) : null,
    domain: row.domain || '',
  })).filter(r => r.f1 != null || r.accuracy != null)

  return (
    <>
      <Topbar title="Model Comparison" subtitle="All domains · Best model per task" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / Model Comparison</div>
          <h1 className="page-title">Model <span>Comparison</span> Report</h1>
          <p className="page-desc">All trained models evaluated — best model per task highlighted across all 5 domains.</p>
        </div>

        {/* Report images from static */}
        {/* Report images from backend static */}
        <div className="comparison-grid fade-up fade-up-1">
          <StatImg
            src="https://sentiment-intelligence-engine.onrender.com/api/v1/static/reports/best_models_heatmap.png"
            caption="Best F1 Score per Domain × Task"
          />
          <StatImg
            src="https://sentiment-intelligence-engine.onrender.com/api/v1/static/reports/model_comparison_chart.png"
            caption="Model Comparison — All Domains"
          />
        </div>

        {/* Best models chart */}
        {loading && (
          <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13, textAlign: 'center', padding: 40 }}>
            Loading comparison data…
          </div>
        )}
        {err && <div className="error-banner" style={{ marginBottom: 20 }}>{err}</div>}

        {chartData.length > 0 && (
          <div className="card fade-up fade-up-2" style={{ marginBottom: 20 }}>
            <div className="card-title"><TrendingUp size={14} style={{ marginRight: 4 }} /> F1 Score by Domain / Task</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{ top: 4, right: 16, bottom: 40, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  angle={-35} textAnchor="end" interval={0} />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}
                  formatter={v => [`${v}%`, 'F1']} />
                <Bar dataKey="f1" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={DOMAIN_COLORS[entry.domain] || 'var(--cyan)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Best models table */}
        {best.length > 0 && (
          <div className="card fade-up fade-up-3" style={{ marginBottom: 20 }}>
            <div className="card-title"><Trophy size={14} style={{ marginRight: 4 }} /> Best Model Per Task</div>
            <div className="table-wrapper">
              <table>
                <thead><tr>
                  <th>Domain</th><th>Task</th><th>Best Model</th>
                  <th>F1 Score</th><th>Accuracy</th>
                </tr></thead>
                <tbody>
                  {best.map((row, i) => {
                    const f1  = row.f1  != null ? (row.f1  * 100).toFixed(1) + '%' : '—'
                    const acc = row.accuracy != null ? (row.accuracy * 100).toFixed(1) + '%' : '—'
                    return (
                      <tr key={i}>
                        <td>{row.domain || '—'}</td>
                        <td>{row.task || '—'}</td>
                        <td className="highlight">{row.model || row.best_model || '—'}</td>
                        <td><span className="badge best">{f1}</span></td>
                        <td><span className="badge good">{acc}</span></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Full comparison (collapsible) */}
        {full.length > 0 && (
          <div className="card fade-up">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showFull ? 16 : 0 }}>
              <div className="card-title" style={{ marginBottom: 0 }}>Full Comparison Table ({full.length} rows)</div>
              <button onClick={() => setShowFull(v => !v)}
                style={{ padding: '6px 14px', background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
                  fontSize: 12, cursor: 'pointer' }}>
                {showFull ? 'Collapse ▲' : 'Expand ▼'}
              </button>
            </div>
            {showFull && (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>{Object.keys(full[0] || {}).map(k => <th key={k}>{k}</th>)}</tr>
                  </thead>
                  <tbody>
                    {full.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((v, j) => (
                          <td key={j} style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                            {typeof v === 'number' ? v.toFixed(4) : String(v ?? '—')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}
