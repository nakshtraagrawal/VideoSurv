import { useEffect, useState } from 'react'
import {
  getStatus,
  getTrainingHistory,
  getTrainingRun,
  retrainFromFeedback,
  startTraining,
} from '../api/client'

export default function Training() {
  const [status, setStatus] = useState(null)
  const [history, setHistory] = useState([])
  const [model, setModel] = useState('conv_ae')
  const [scene, setScene] = useState('bike')
  const [epochs, setEpochs] = useState(20)
  const [runId, setRunId] = useState(null)
  const [run, setRun] = useState(null)

  async function refresh() {
    const [st, hi] = await Promise.all([getStatus(), getTrainingHistory()])
    setStatus(st.data)
    setHistory(hi.data || [])
  }

  useEffect(() => {
    refresh().catch(console.error)
  }, [])

  useEffect(() => {
    if (!runId) return
    const t = setInterval(() => {
      getTrainingRun(runId)
        .then((r) => setRun(r.data))
        .catch(console.error)
    }, 2000)
    return () => clearInterval(t)
  }, [runId])

  async function onStart() {
    const { data } = await startTraining({
      model_type: model,
      scene,
      epochs,
    })
    setRunId(data.id)
    setRun(data)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold">Training & harvest</h1>
        <p className="text-sm text-tactical-muted">
          Queue training jobs and monitor progress
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded border border-tactical-border bg-tactical-panel p-4">
          <h2 className="text-sm font-semibold text-tactical-muted">
            Harvest status
          </h2>
          <p className="mt-2 text-sm">
            Indexed normal footage:{' '}
            <span className="font-mono">{status?.harvest_hours ?? 0}</span> h
            (estimate)
          </p>
          <p className="mt-2 text-xs text-tactical-muted">
            For the prototype, Drone-Anomaly training split stands in for harvest
            sessions.
          </p>
          <button
            type="button"
            className="mt-4 rounded border border-tactical-border px-3 py-2 text-xs text-tactical-muted"
            onClick={() =>
              alert('Harvest session logged locally (prototype placeholder).')
            }
          >
            Start harvest session
          </button>
        </div>

        <div className="rounded border border-tactical-border bg-tactical-panel p-4">
          <h2 className="text-sm font-semibold text-tactical-muted">
            New training job
          </h2>
          <div className="mt-3 grid gap-2 text-sm">
            <label className="block text-xs text-tactical-muted">Model</label>
            <select
              className="rounded border border-tactical-border bg-tactical-bg p-2"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="conv_ae">Conv AE</option>
              <option value="astnet">ASTNet</option>
              <option value="hstforu">HSTforU</option>
            </select>
            <label className="block text-xs text-tactical-muted">Scene</label>
            <input
              className="rounded border border-tactical-border bg-tactical-bg p-2"
              value={scene}
              onChange={(e) => setScene(e.target.value)}
            />
            <label className="block text-xs text-tactical-muted">Epochs</label>
            <input
              type="number"
              className="rounded border border-tactical-border bg-tactical-bg p-2"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
            <button
              type="button"
              onClick={onStart}
              className="mt-2 rounded bg-tactical-accent py-2 text-sm font-semibold text-white"
            >
              Start training
            </button>
          </div>
          {run && (
            <div className="mt-4 rounded border border-tactical-border bg-tactical-bg p-3 text-xs">
              <div className="font-mono">Run #{run.id}</div>
              <div>Status: {run.status}</div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded bg-slate-800">
                <div
                  className={`h-full ${
                    run.status === 'complete'
                      ? 'w-full bg-emerald-600'
                      : run.status === 'failed'
                        ? 'w-full bg-red-700'
                        : 'w-1/2 animate-pulse bg-sky-600'
                  }`}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="rounded border border-tactical-border bg-tactical-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-tactical-muted">History</h2>
          <button
            type="button"
            className="text-xs text-sky-300 hover:underline"
            onClick={() =>
              retrainFromFeedback().then((r) => alert(JSON.stringify(r.data)))
            }
          >
            Queue retrain from feedback
          </button>
        </div>
        <table className="mt-3 w-full text-left text-xs">
          <thead className="text-tactical-muted">
            <tr>
              <th className="py-1">ID</th>
              <th>Model</th>
              <th>Scene</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h) => (
              <tr key={h.id} className="border-t border-tactical-border">
                <td className="py-2 font-mono">{h.id}</td>
                <td>{h.model_type}</td>
                <td>{h.scene}</td>
                <td>{h.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
