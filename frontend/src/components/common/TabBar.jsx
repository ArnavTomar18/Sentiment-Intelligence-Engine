import React from 'react'
import { checkHealth } from '../../api/client'
const [status, setStatus] = useState("checking")

async function checkHealthWithRetry(retries = 3) {
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
