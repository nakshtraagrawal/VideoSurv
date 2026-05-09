import { useEffect, useState } from 'react'
import { getAlerts, submitFeedback, clipUrl } from '../api/client'
import AlertCard from '../components/AlertCard'
import HeatmapVideo from '../components/HeatmapVideo'

export default function Alerts() {
  const [filter, setFilter] = useState('all')
  const [scene, setScene] = useState('')
  const [rows, setRows] = useState([])
  const [selected, setSelected] = useState(null)
  const [note, setNote] = useState('')

  async function load() {
    const params = {}
    if (scene) params.scene = scene
    if (filter === 'unreviewed') params.reviewed = false
    if (filter === 'confirmed') params.confirmed = true
    if (filter === 'fp') params.reviewed = true
    const { data } = await getAlerts(params)
    let list = data || []
    if (filter === 'fp') {
      list = list.filter((a) => a.confirmed_anomaly === false)
    }
    setRows(list)
    setSelected((s) => list.find((x) => x.id === s?.id) || list[0] || null)
  }

  useEffect(() => {
    load().catch(console.error)
  }, [filter, scene])

  async function sendFeedback(confirmed) {
    if (!selected) return
    await submitFeedback(selected.id, {
      confirmed_anomaly: confirmed,
      feedback_note: note || null,
    })
    setNote('')
    await load()
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-3">
        <h1 className="text-xl font-semibold">Alerts</h1>
        <div className="flex flex-wrap gap-2 text-xs">
          {['all', 'unreviewed', 'confirmed', 'fp'].map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`rounded border px-2 py-1 capitalize ${
                filter === f
                  ? 'border-tactical-accent bg-tactical-accent/20'
                  : 'border-tactical-border'
              }`}
            >
              {f === 'fp' ? 'False positive' : f}
            </button>
          ))}
          <input
            className="ml-auto rounded border border-tactical-border bg-tactical-bg px-2 py-1 text-xs"
            placeholder="Scene filter"
            value={scene}
            onChange={(e) => setScene(e.target.value)}
          />
        </div>
        <div className="max-h-[70vh] space-y-2 overflow-y-auto pr-1">
          {rows.map((a) => (
            <AlertCard
              key={a.id}
              alert={a}
              selected={selected}
              onSelect={setSelected}
            />
          ))}
        </div>
      </div>
      <div className="space-y-3 rounded border border-tactical-border bg-tactical-panel p-4">
        <h2 className="text-sm font-semibold text-tactical-muted">Detail</h2>
        {!selected && <p className="text-sm text-tactical-muted">Select an alert</p>}
        {selected && (
          <>
            <HeatmapVideo
              src={selected.clip_path ? clipUrl(selected.id) : null}
              score={selected.anomaly_score}
            />
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-tactical-muted">Scene</span>
                <div className="font-mono">{selected.scene}</div>
              </div>
              <div>
                <span className="text-tactical-muted">Frame</span>
                <div className="font-mono">{selected.frame_idx}</div>
              </div>
              <div className="col-span-2">
                <span className="text-tactical-muted">Time</span>
                <div className="font-mono">{String(selected.timestamp)}</div>
              </div>
            </div>
            <textarea
              className="w-full rounded border border-tactical-border bg-tactical-bg p-2 text-sm"
              rows={3}
              placeholder="Notes"
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => sendFeedback(true)}
                className="rounded bg-emerald-900/60 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-800"
              >
                Confirm anomaly
              </button>
              <button
                type="button"
                onClick={() => sendFeedback(false)}
                className="rounded bg-slate-700 px-3 py-2 text-sm font-medium text-white hover:bg-slate-600"
              >
                Mark as normal
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
