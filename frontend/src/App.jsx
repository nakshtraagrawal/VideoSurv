import { NavLink, Outlet, Route, Routes } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getStatus } from './api/client'
import ModelBadge from './components/ModelBadge'
import Dashboard from './pages/Dashboard'
import Alerts from './pages/Alerts'
import Benchmark from './pages/Benchmark'
import Training from './pages/Training'
import Settings from './pages/Settings'

const nav = [
  ['/', 'Dashboard'],
  ['/alerts', 'Alerts'],
  ['/benchmark', 'Benchmark'],
  ['/training', 'Training'],
  ['/settings', 'Settings'],
]

function Layout() {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    getStatus()
      .then((r) => setStatus(r.data))
      .catch(() => {})
    const id = setInterval(() => {
      getStatus()
        .then((r) => setStatus(r.data))
        .catch(() => {})
    }, 15000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="flex min-h-screen">
      <aside className="flex w-56 flex-col border-r border-tactical-border bg-tactical-panel">
        <div className="border-b border-tactical-border p-4">
          <div className="text-xs font-semibold tracking-[0.2em] text-tactical-muted">
            AUTOSURVEIL
          </div>
          <div className="mt-2 text-[10px] uppercase text-tactical-muted">
            Mode:{' '}
            <span className="text-slate-200">{status?.mode ?? '—'}</span>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-2">
          {nav.map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `rounded px-3 py-2 text-sm ${
                  isActive
                    ? 'bg-tactical-bg text-white'
                    : 'text-tactical-muted hover:bg-tactical-bg/60 hover:text-white'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-tactical-border p-3 text-xs">
          <div className="text-tactical-muted">Active</div>
          <div className="mt-1">
            <ModelBadge name={status?.active_model || 'hstforu'} />
          </div>
          <div className="mt-3 text-tactical-muted">Alerts</div>
          <div className="font-mono text-slate-200">
            {status?.total_alerts ?? 0} total · {status?.unreviewed_alerts ?? 0}{' '}
            open
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="benchmark" element={<Benchmark />} />
        <Route path="training" element={<Training />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
