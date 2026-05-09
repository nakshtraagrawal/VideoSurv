import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getBenchmarkResults,
  getBenchmarkSummary,
  runBenchmark,
} from '../api/client'
import MetricsTable from '../components/MetricsTable'

export default function Benchmark() {
  const [rows, setRows] = useState([])
  const [summary, setSummary] = useState(null)
  const [busy, setBusy] = useState(false)

  async function refresh() {
    const [r, s] = await Promise.all([
      getBenchmarkResults(),
      getBenchmarkSummary(),
    ])
    setRows(r.data || [])
    setSummary(s.data)
  }

  useEffect(() => {
    refresh().catch(console.error)
  }, [])

  async function onRun() {
    setBusy(true)
    try {
      await runBenchmark()
      await new Promise((res) => setTimeout(res, 2000))
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const chartData = []
  const scenes = [...new Set(rows.map((x) => x.scene))]
  for (const sc of scenes) {
    const row = { scene: sc }
    for (const m of ['conv_ae', 'astnet', 'hstforu']) {
      const hit = rows.find((r) => r.scene === sc && r.model_type === m)
      row[m] = hit ? hit.auc : null
    }
    chartData.push(row)
  }

  const best = summary?.per_scene_best?.[0]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Benchmark</h1>
          <p className="text-sm text-tactical-muted">
            Compare Conv AE, ASTNet, and HSTforU on Drone-Anomaly test splits
          </p>
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={onRun}
          className="rounded bg-tactical-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          {busy ? 'Running…' : 'Run benchmark'}
        </button>
      </div>

      {best && (
        <div className="rounded border border-tactical-border bg-tactical-panel p-4 text-sm">
          Recommendation: on scene <strong>{best.scene}</strong>,{' '}
          <strong>{best.best_model}</strong> leads with AUC{' '}
          <span className="font-mono">{best.auc}</span>.
        </div>
      )}

      <MetricsTable rows={rows} />

      <div className="h-80 rounded border border-tactical-border bg-tactical-panel p-2">
        <p className="mb-2 px-1 text-xs text-tactical-muted">AUC by scene</p>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3540" />
            <XAxis dataKey="scene" stroke="#8a9ba8" />
            <YAxis stroke="#8a9ba8" domain={[0, 1]} />
            <Tooltip
              contentStyle={{ background: '#141a1f', border: '1px solid #2a3540' }}
            />
            <Legend />
            <Bar dataKey="conv_ae" fill="#60a5fa" name="Conv AE" />
            <Bar dataKey="astnet" fill="#fbbf24" name="ASTNet" />
            <Bar dataKey="hstforu" fill="#34d399" name="HSTforU" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
