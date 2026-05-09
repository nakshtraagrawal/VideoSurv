import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  getAlerts,
  getScenes,
  getScores,
  getStatus,
} from '../api/client'
import ModeSelector from '../components/ModeSelector'
import ModelBadge from '../components/ModelBadge'
import ScoreTimeline from '../components/ScoreTimeline'

export default function Dashboard() {
  const [status, setStatus] = useState(null)
  const [scenes, setScenes] = useState([])
  const [tab, setTab] = useState(0)
  const [scores, setScores] = useState([])
  const [recent, setRecent] = useState([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [st, sc, al] = await Promise.all([
          getStatus(),
          getScenes(),
          getAlerts({}),
        ])
        if (cancelled) return
        setStatus(st.data)
        setScenes(sc.data.scenes || [])
        setRecent((al.data || []).slice(0, 5))
        const sceneList = sc.data.scenes || []
        if (sceneList.length) {
          try {
            const r = await getScores(sceneList[0], '01')
            setScores(r.data.scores || [])
          } catch {
            setScores([])
          }
        }
      } catch (e) {
        console.error(e)
      }
    }
    load()
    const id = setInterval(load, 8000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const scene = scenes[tab] || scenes[0]
  const readiness =
    status?.harvest_hours != null
      ? Math.min(100, Math.round((status.harvest_hours / 10) * 100))
      : 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Surveillance</h1>
          <p className="text-sm text-tactical-muted">
            Live anomaly scores and recent alerts
          </p>
        </div>
        <ModeSelector
          current={status?.mode}
          onChange={(m) => setStatus((s) => ({ ...s, mode: m }))}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded border border-tactical-border bg-tactical-panel p-4 lg:col-span-2">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <h2 className="text-sm font-medium text-tactical-muted">Scenes</h2>
            <div className="flex flex-wrap gap-1">
              {scenes.map((s, i) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setTab(i)}
                  className={`rounded px-2 py-1 text-xs ${
                    tab === i
                      ? 'bg-tactical-accent/30 text-white'
                      : 'bg-tactical-bg text-tactical-muted hover:text-white'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <ScoreTimeline
            scores={scores}
            threshold={0.015}
            label={
              scene
                ? `${scene} · last frames (demo scores if available)`
                : 'Scores'
            }
          />
        </div>
        <div className="space-y-4">
          <div className="rounded border border-tactical-border bg-tactical-panel p-4">
            <h3 className="text-xs font-semibold uppercase text-tactical-muted">
              Model readiness
            </h3>
            <div className="mt-2 h-3 w-full overflow-hidden rounded bg-tactical-bg">
              <div
                className="h-full bg-tactical-accent transition-all"
                style={{ width: `${readiness}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-tactical-muted">
              ~{status?.harvest_hours ?? 0} h normal footage indexed (prototype
              estimate)
            </p>
          </div>
          <div className="rounded border border-tactical-border bg-tactical-panel p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase text-tactical-muted">
                Active model
              </h3>
              <ModelBadge name={status?.active_model || 'hstforu'} />
            </div>
            <p className="mt-3 text-xs text-tactical-muted">
              Pre-harvested normal footage: {scenes.length} scenes in{' '}
              <code className="text-slate-300">data/drone/</code>
            </p>
          </div>
          <div className="rounded border border-tactical-border bg-tactical-panel p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase text-tactical-muted">
                Alerts
              </h3>
              <Link to="/alerts" className="text-xs text-sky-300 hover:underline">
                Open
              </Link>
            </div>
            <ul className="mt-2 space-y-2 text-xs">
              {recent.length === 0 && (
                <li className="text-tactical-muted">No alerts yet</li>
              )}
              {recent.map((a) => (
                <li key={a.id} className="flex justify-between gap-2">
                  <span className="truncate text-slate-300">{a.scene}</span>
                  <span className="font-mono text-sky-200">
                    {a.anomaly_score?.toFixed?.(4)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
