export default function AlertCard({ alert, selected, onSelect }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(alert)}
      className={`flex w-full items-start gap-3 rounded border p-2 text-left transition ${
        selected?.id === alert.id
          ? 'border-tactical-accent bg-tactical-accent/10'
          : 'border-tactical-border bg-tactical-bg hover:border-slate-500'
      }`}
    >
      <div className="h-12 w-20 shrink-0 rounded bg-slate-800" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase text-slate-300">
            {alert.scene}
          </span>
          <span className="font-mono text-xs text-tactical-muted">
            {alert.video_name}
          </span>
        </div>
        <div className="mt-1 font-mono text-sm text-sky-200">
          score {alert.anomaly_score?.toFixed?.(5) ?? alert.anomaly_score}
        </div>
        <div className="text-[11px] text-tactical-muted">
          {alert.timestamp}
          {alert.reviewed ? ' · reviewed' : ' · pending'}
        </div>
      </div>
    </button>
  )
}
