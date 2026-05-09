import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

export default function ScoreTimeline({ scores, threshold, label }) {
  const data = (scores || []).slice(-1000).map((y, i) => ({ i, y }))
  return (
    <div className="h-64 w-full rounded border border-tactical-border bg-tactical-panel p-2">
      <p className="mb-1 px-1 text-xs text-tactical-muted">{label}</p>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data}>
          <XAxis dataKey="i" hide />
          <YAxis stroke="#8a9ba8" fontSize={11} />
          <Tooltip
            contentStyle={{ background: '#141a1f', border: '1px solid #2a3540' }}
            labelFormatter={(v) => `Frame ${v}`}
          />
          {threshold != null && (
            <ReferenceLine
              y={threshold}
              stroke="#f97316"
              strokeDasharray="4 4"
            />
          )}
          <Line
            type="monotone"
            dataKey="y"
            stroke="#60a5fa"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
