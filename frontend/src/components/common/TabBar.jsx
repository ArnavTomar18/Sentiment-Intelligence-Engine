import React from 'react'

export default function TabBar({ tabs, active, onChange }) {
  return (
    <div className="task-tabs" style={{ marginBottom: 20 }}>
      {tabs.map(({ key, label, icon }) => (
        <button
          key={key}
          className={`task-tab${active === key ? ' active' : ''}`}
          onClick={() => onChange(key)}
        >
          {icon && <span style={{ marginRight: 5 }}>{icon}</span>}
          {label}
        </button>
      ))}
    </div>
  )
}
