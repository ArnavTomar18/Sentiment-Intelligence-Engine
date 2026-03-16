import React, { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'

import Home            from './pages/Home'
import News            from './pages/News'
import Hotel           from './pages/Hotel'
import Fashion         from './pages/Fashion'
import AppReviews      from './pages/AppReviews'
import AppRecommender  from './pages/AppRecommender'
import OTT             from './pages/OTT'
// import OTTRecommender  from './pages/OTTRecommender'
import EDAExplorer     from './pages/EDAExplorer'
import ModelComparison from './pages/ModelComparison'
import BatchAnalyzer   from './pages/BatchAnalyzer'

export default function App() {
  // ── Sidebar state ──────────────────────────────────────────────────────────
  const [collapsed,    setCollapsed]    = useState(false)  // desktop collapse
  const [mobileOpen,   setMobileOpen]   = useState(false)  // mobile drawer

  // Apply saved theme on first load (Topbar handles changes after that)
  useEffect(() => {
    const saved = localStorage.getItem('theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    document.body.setAttribute('data-theme', saved ?? (prefersDark ? 'dark' : 'light'))
  }, [])

  // Shared props forwarded to every Topbar inside page components
  const topbarProps = { mobileOpen, setMobileOpen }

  return (
    <div className="app-shell">
      <Sidebar
        collapsed={collapsed}
        setCollapsed={setCollapsed}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      {/* main-content shifts left when sidebar collapses */}
      <div className={`main-content${collapsed ? ' sidebar-collapsed' : ''}`}>
        <Routes>
          <Route path="/"                 element={<Home            {...topbarProps} />} />
          <Route path="/news"             element={<News            {...topbarProps} />} />
          <Route path="/hotel"            element={<Hotel           {...topbarProps} />} />
          <Route path="/fashion"          element={<Fashion         {...topbarProps} />} />
          <Route path="/app-reviews"      element={<AppReviews      {...topbarProps} />} />
          <Route path="/app-recommender"  element={<AppRecommender  {...topbarProps} />} />
          <Route path="/ott"              element={<OTT             {...topbarProps} />} />
          <Route path="/ott-recommender"  element={<OTTRecommender  {...topbarProps} />} />
          <Route path="/eda-explorer"     element={<EDAExplorer     {...topbarProps} />} />
          <Route path="/model-comparison" element={<ModelComparison {...topbarProps} />} />
          <Route path="/batch-analyzer"   element={<BatchAnalyzer   {...topbarProps} />} />
        </Routes>
      </div>
    </div>
  )
}
