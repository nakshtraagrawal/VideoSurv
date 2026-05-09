import { useEffect, useState } from 'react'
import {
  getScenes,
  getStatus,
  setActiveModel,
  setSensitivity,
  setThresholdApi,
} from '../api/client'

export default function Settings() {
  const [status, setStatus] = useState(null)
  const [scenes, setScenes] = useState([])
  const [model, setModel] = useState('hstforu')
  const [thresholds, setThresholds] = useState({})

  useEffect(() => {
    getStatus().then((r) => {
      setStatus(r.data)
      setModel(r.data.active_model || 'hstforu')
    })
    getScenes().then((r) => setScenes(r.data.scenes || []))
  }, [])

  async function saveModel() {
    await setActiveModel(model)
    const st = await getStatus()
    setStatus(st.data)
  }

  async function saveThreshold(scene) {
    const v = Number(thresholds[scene] ?? 0.015)
    await setThresholdApi(model, scene, v)
  }

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-tactical-muted">
          Active detector, thresholds, and alert sensitivity
        </p>
      </div>

      <section className="rounded border border-tactical-border bg-tactical-panel p-4">
        <h2 className="text-sm font-semibold text-tactical-muted">Active model</h2>
        <div className="mt-3 flex flex-wrap items-end gap-2">
          <select
            className="rounded border border-tactical-border bg-tactical-bg p-2 text-sm"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="conv_ae">Conv AE</option>
            <option value="astnet">ASTNet</option>
            <option value="hstforu">HSTforU</option>
          </select>
          <button
            type="button"
            onClick={saveModel}
            className="rounded bg-tactical-accent px-3 py-2 text-sm text-white"
          >
            Save
          </button>
        </div>
      </section>

      <section className="rounded border border-tactical-border bg-tactical-panel p-4">
        <h2 className="text-sm font-semibold text-tactical-muted">
          Alert sensitivity
        </h2>
        <p className="mt-1 text-xs text-tactical-muted">
          Maps to evaluation percentile used for thresholding in benchmarks.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {['low', 'medium', 'high'].map((lvl) => (
            <button
              key={lvl}
              type="button"
              className="rounded border border-tactical-border px-3 py-1 text-xs capitalize"
              onClick={() => setSensitivity(lvl)}
            >
              {lvl}
            </button>
          ))}
        </div>
      </section>

      <section className="rounded border border-tactical-border bg-tactical-panel p-4">
        <h2 className="text-sm font-semibold text-tactical-muted">
          Per-scene raw threshold
        </h2>
        <div className="mt-3 space-y-2">
          {(scenes.length ? scenes : ['bike']).map((sc) => (
            <div key={sc} className="flex flex-wrap items-center gap-2 text-sm">
              <span className="w-24 text-tactical-muted">{sc}</span>
              <input
                type="number"
                step="0.001"
                className="w-32 rounded border border-tactical-border bg-tactical-bg p-1 font-mono text-xs"
                placeholder="0.015"
                value={thresholds[sc] ?? ''}
                onChange={(e) =>
                  setThresholds({ ...thresholds, [sc]: e.target.value })
                }
              />
              <button
                type="button"
                className="text-xs text-sky-300 hover:underline"
                onClick={() => saveThreshold(sc)}
              >
                Save
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded border border-tactical-border bg-tactical-panel p-4 text-xs text-tactical-muted">
        Current mode:{' '}
        <span className="font-mono text-slate-200">{status?.mode}</span>
      </section>
    </div>
  )
}
