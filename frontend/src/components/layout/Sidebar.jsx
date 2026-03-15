import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Smartphone, Shirt, Hotel, Newspaper, Tv2,
  BarChart3, Zap, Bot, BookOpen, PanelLeftClose, PanelLeftOpen,
} from 'lucide-react'

// ── Route config (matches your existing App.jsx routes) ──────────────────────
const NAV = [
  {
    section: 'Overview',
    items: [{ to: '/', label: 'Dashboard', icon: LayoutDashboard }],
  },
  {
    section: 'Analyzers',
    items: [
      { to: '/news',        label: 'News',        icon: Newspaper,  badge: '3' },
      { to: '/hotel',       label: 'Hotel',       icon: Hotel,      badge: '7' },
      { to: '/fashion',     label: 'Fashion',     icon: Shirt,      badge: '4' },
      { to: '/app-reviews', label: 'App Reviews', icon: Smartphone, badge: '6' },
      { to: '/ott',         label: 'OTT Content', icon: Tv2,        badge: '8' },
    ],
  },
  {
    section: 'Recommenders',
    items: [
      { to: '/app-recommender', label: 'App Recommender', icon: Bot      },
      { to: '/ott-recommender', label: 'OTT Recommender', icon: BookOpen },
    ],
  },
  {
    section: 'Analysis',
    items: [
      { to: '/eda-explorer',     label: 'EDA Explorer',     icon: BarChart3 },
      { to: '/model-comparison', label: 'Model Comparison', icon: BarChart3 },
      { to: '/batch-analyzer',   label: 'Batch Analyzer',   icon: Zap       },
    ],
  },
]

// ── Single nav item ───────────────────────────────────────────────────────────
function NavItem({ to, label, icon: Icon, badge, collapsed, onClick }) {
  const location = useLocation()
  const isActive =
    to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)

  return (
    <NavLink
      to={to}
      onClick={onClick}
      title={collapsed ? label : undefined}
      className={`nav-item${isActive ? ' active' : ''}`}
    >
      <Icon size={17} className="nav-icon" style={{ flexShrink: 0 }} />
      <span className="nav-label">{label}</span>
      {badge && <span className="nav-badge">{badge}</span>}
    </NavLink>
  )
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
// Props
//   collapsed    boolean  – desktop collapsed state
//   setCollapsed fn       – toggle desktop collapse
//   mobileOpen   boolean  – mobile drawer open state
//   setMobileOpen fn      – toggle mobile drawer
export default function Sidebar({ collapsed, setCollapsed, mobileOpen, setMobileOpen }) {
  const closeDrawer = () => mobileOpen && setMobileOpen(false)

  return (
    <>
      {/* Dim overlay – mobile only */}
      {mobileOpen && (
        <div className="sidebar-overlay visible" onClick={closeDrawer} />
      )}

      <aside
        className={[
          'sidebar',
          collapsed  ? 'collapsed'   : '',
          mobileOpen ? 'mobile-open' : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="logo-icon">S</div>
          <div className="logo-text">
            <div className="logo-name">SentimentIQ</div>
            <div className="logo-sub">Engine v2.0</div>
          </div>
        </div>

        {/* Nav sections */}
        {NAV.map((group) => (
          <div className="sidebar-section" key={group.section}>
            <div className="sidebar-section-label">{group.section}</div>
            {group.items.map((item) => (
              <NavItem
                key={item.to}
                {...item}
                collapsed={collapsed}
                onClick={closeDrawer}
              />
            ))}
          </div>
        ))}

        {/* Footer */}
        <div className="sidebar-footer">
          {/* Backend status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-4)',
              marginBottom: 10,
              justifyContent: collapsed ? 'center' : 'flex-start',
              overflow: 'hidden',
            }}
          >
            <span className="status-dot" />
            {!collapsed && 'FastAPI Backend'}
          </div>

          {/* Collapse toggle */}
          <button
            className="sidebar-collapse-btn"
            onClick={() => setCollapsed(!collapsed)}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
            {!collapsed && (
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                Collapse
              </span>
            )}
          </button>
        </div>
      </aside>
    </>
  )
}