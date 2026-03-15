import React from 'react'
import { Link } from 'react-router-dom'
import {
  Smartphone, Shirt, Hotel, Newspaper, Tv2, ArrowRight,
  Cpu, Database, GitBranch, Activity, ExternalLink,
  Github, Linkedin,
} from 'lucide-react'
import Topbar from '../components/layout/Topbar'

// ── Stats (same data as before, just re-skinned) ─────────────────────────────
const STATS = [
  { value: '5',   label: 'Domains',  icon: Database,  color: 'accent'  },
  { value: '18',  label: 'Models',   icon: Cpu,       color: 'violet'  },
  { value: '12',  label: 'Tasks',    icon: GitBranch, color: 'orange'  },
  { value: '99%', label: 'Uptime',   icon: Activity,  color: 'green'   },
]

// ── Domain cards (same routes as before) ─────────────────────────────────────
const DOMAINS = [
  {
    to: '/news',
    label: 'Fake News Detection',
    icon: '📰',
    color: 'accent',
    desc: 'Classify news articles as real or fake using NLP models.',
    tasks: ['Fake / Real', 'NLP'],
    taskColors: ['accent', ''],
  },
  {
    to: '/hotel',
    label: 'Hotel Reviews',
    icon: '🏨',
    color: 'orange',
    desc: 'Detect hotel review sentiment, predict ratings & churn risk.',
    tasks: ['Sentiment', 'Rating', 'Churn'],
    taskColors: ['orange', '', ''],
  },
  {
    to: '/fashion',
    label: 'Fashion Analysis',
    icon: '👗',
    color: 'violet',
    desc: 'Analyse fashion product reviews and predict star ratings.',
    tasks: ['Sentiment', 'Rating'],
    taskColors: ['violet', ''],
  },
  {
    to: '/app-reviews',
    label: 'App Feedback Intel',
    icon: '📱',
    color: 'green',
    desc: 'Classify Play Store feedback and predict user recommendations.',
    tasks: ['Feedback', 'Recommender'],
    taskColors: ['green', ''],
  },
  {
    to: '/ott',
    label: 'OTT Trend Prediction',
    icon: '🎬',
    color: 'cyan',
    desc: 'Analyse OTT reviews, predict viral content & recommendations.',
    tasks: ['Sentiment', 'Viral', 'Recommend'],
    taskColors: ['cyan', '', ''],
  },
]

// ── Quick-access analytics tools ─────────────────────────────────────────────
const TOOLS = [
  { to: '/model-comparison', label: '⚖️  Model Comparison' },
  { to: '/eda-explorer',     label: '📊  EDA Explorer'     },
  { to: '/batch-analyzer',   label: '📦  Batch Analyzer'   },
  { to: '/ott-recommender',  label: '🎬  OTT Recommender'  },
  { to: '/app-recommender',  label: '📱  App Recommender'  },
]

// ── Hero feature pills ────────────────────────────────────────────────────────
const PILLS = [
  'Fake News Detection',
  'Hotel Reviews',
  'Fashion Sentiment',
  'OTT Trends',
  'App Intelligence',
  'Batch Analysis',
]

// ── Component ─────────────────────────────────────────────────────────────────
export default function Home() {
  return (
    <>
      <Topbar title="Sentiment Intelligence Engine" subtitle="" />

      <div className="page-body" style={{ maxWidth: 1200, margin: '0 auto' }}>

        {/* ── Hero ── */}
        <div className="hero fade-up">
          <h1 className="hero-title">
            Sentiment<br />
            <span className="highlight">Intelligence</span> Engine
          </h1>

          <p className="hero-sub">
            AI-powered sentiment analysis across news, hotels, fashion, OTT content,
            and app reviews — 18 trained models, 12 tasks, one unified platform.
          </p>

          <div className="hero-actions">
            <Link to="/news" className="btn btn-primary">
              Start Analysing <ArrowRight size={16} />
            </Link>
            <a
              href="https://github.com/arnavtomar18/Sentiment-Intelligence-Engine"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              <Github size={15} /> View on GitHub
            </a>
          </div>

          {/* Domain pills */}
          <div className="hero-pills">
            {PILLS.map((p) => (
              <span key={p} className="hero-pill">
                <span
                  className="hero-pill-dot"
                  style={{ background: 'var(--accent)' }}
                />
                {p}
              </span>
            ))}
          </div>
        </div>

        {/* ── Stats ── */}
        <div className="stats-grid fade-up fade-up-1">
          {STATS.map(({ value, label, icon: Icon, color }) => (
            <div key={label} className={`stat-card ${color}`}>
              <div className={`stat-icon ${color}`}>
                <Icon size={18} />
              </div>
              <div className="stat-value">{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        <div className="section-divider" />

        {/* ── Domain cards ── */}
        <h2 className="section-title fade-up fade-up-2">Analysis Domains</h2>
        <p
          style={{
            color: 'var(--text-3)',
            fontSize: 14,
            marginBottom: 20,
            marginTop: -8,
          }}
          className="fade-up fade-up-2"
        >
          Five specialised NLP pipelines, each fine-tuned for its domain.
        </p>

        <div className="feature-grid fade-up fade-up-3">
          {DOMAINS.map(({ to, label, icon, color, desc, tasks, taskColors }) => (
            <Link key={to} to={to} className={`feature-card ${color}`}>
              <div className={`feature-icon ${color}`}>{icon}</div>

              <div>
                <div className="feature-name">{label}</div>
                <div className="feature-desc" style={{ marginTop: 4 }}>{desc}</div>
              </div>

              <div className="feature-tags">
                {tasks.map((t, i) => (
                  <span key={t} className={`tag ${taskColors[i] || ''}`}>
                    {t}
                  </span>
                ))}
              </div>

              <div className="feature-arrow">
                <ArrowRight size={16} />
              </div>
            </Link>
          ))}
        </div>

        <div className="section-divider" />

        {/* ── Analytics tools quick-access ── */}
        <div className="card fade-up fade-up-4">
          <div className="card-title">
            <span className="dot" /> Analytics Tools
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {TOOLS.map(({ to, label }) => (
              <Link key={to} to={to} className="btn btn-secondary" style={{ fontSize: 13 }}>
                {label}
              </Link>
            ))}
          </div>
        </div>

        {/* ── Developer card ── */}
        <div className="developer-card fade-up fade-up-5">
          <div className="developer-avatar">AT</div>

          <div className="developer-info">
            <div className="developer-title">About the Developer</div>
            <div className="developer-name">Arnav Tomar</div>
            <div className="developer-bio">
            Intermediate Machine Learning engineer building AI-powered applications with React and NLP pipelines. 
            This project demonstrates multi-domain sentiment analysis across several real-world datasets.
            </div>

            <div className="developer-links">
              <a
                href="https://portfolio-steel-one-88.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                className="dev-link portfolio"
              >
                <ExternalLink size={13} /> Portfolio
              </a>
              <a
                href="https://github.com/arnavtomar18"
                target="_blank"
                rel="noopener noreferrer"
                className="dev-link github"
              >
                <Github size={15} /> GitHub
              </a>
              <a
                href="https://linkedin.com/in/arnavtomar18"
                target="_blank"
                rel="noopener noreferrer"
                className="dev-link linkedin"
              >
                <Linkedin size={15} /> LinkedIn
              </a>
            </div>
          </div>
        </div>

      </div>
    </>
  )
}