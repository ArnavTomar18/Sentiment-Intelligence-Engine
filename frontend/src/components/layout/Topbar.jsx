import React, { useEffect, useState } from 'react'
import { Sun, Moon, Menu, ExternalLink } from 'lucide-react'
import { checkHealth } from '../../api/client'
const [status, setStatus] = useState("checking")

async function checkHealthWithRetry(retries = 9) {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await checkHealth()
      return res
    } catch (e) {
      await new Promise(r => setTimeout(r, 2000))
    }
  }
  throw new Error("API unavailable")
}

// ── Theme helpers ─────────────────────────────────────────────────────────────
function getInitialTheme() {
  const stored = localStorage.getItem('theme')
  if (stored) return stored === 'dark'
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function applyTheme(dark) {
  document.body.setAttribute('data-theme', dark ? 'dark' : 'light')
  localStorage.setItem('theme', dark ? 'dark' : 'light')
}

// ── Topbar ────────────────────────────────────────────────────────────────────
// Props
//   title        string  – page title shown in the bar
//   subtitle     string  – optional subtitle (kept from your existing usage)
//   mobileOpen   boolean – mobile sidebar state (passed from App)
//   setMobileOpen fn     – toggle mobile sidebar
export default function Topbar({ title, subtitle, mobileOpen, setMobileOpen }) {
  const [dark,      setDark]      = useState(getInitialTheme)
  const [apiStatus, setApiStatus] = useState('checking')  // 'checking' | 'online' | 'offline'

  // Apply theme on mount + whenever it changes
  useEffect(() => { applyTheme(dark) }, [dark])

  // Real health-check using your existing client
useEffect(() => {
  setApiStatus('checking')

  checkHealthWithRetry()
    .then(() => setApiStatus('online'))
    .catch(() => setApiStatus('offline'))
}, [])

  const toggleTheme = () => setDark((d) => !d)

  const now = new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })

  return (
    
    <header className="topbar">
      {/* Hamburger – visible on mobile only (CSS handles display:flex/none) */}
      <button
        className="topbar-hamburger"
        onClick={() => setMobileOpen?.(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        <Menu size={18} />
      </button>

      {/* Page title */}
      <div className="topbar-title">
        {title}
        {subtitle && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              fontWeight: 400,
              color: 'var(--text-4)',
              marginLeft: 10,
            }}
          >
            {subtitle}
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="topbar-actions">
        {/* API status chip */}
        <div className={`api-status ${apiStatus}`}>
          <span
            style={{
              background:
                apiStatus === 'online'
                  ? 'var(--green)'
                  : apiStatus === 'offline'
                  ? 'var(--red)'
                  : 'var(--text-4)',
              display: 'inline-block',
              flexShrink: 0,
            }}
          />
          {apiStatus === 'checking'
            ? 'Connecting…'
            : apiStatus === 'online'
            ? `API · ${now}`
            : 'API Offline'}
        </div>

        {/* Theme toggle */}
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label="Toggle theme"
        >
          {dark ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* Portfolio link */}
        <a
          href="https://portfolio-steel-one-88.vercel.app"
          target="_blank"
          rel="noopener noreferrer"
          className="topbar-btn portfolio"
        >
          <ExternalLink size={13} />
          <span>View Developer Portfolio</span>
        </a>
      </div>
    </header>
  )
}
