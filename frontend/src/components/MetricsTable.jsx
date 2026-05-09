export default function MetricsTable({ rows }) {
  if (!rows?.length) {
    return (
      <p className="text-sm text-tactical-muted">No benchmark rows yet. Run a benchmark.</p>
    )
  }
  return (
    <div className="overflow-x-auto rounded border border-tactical-border">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-tactical-panel text-xs uppercase text-tactical-muted">
          <tr>
            <th className="px-3 py-2">Model</th>
            <th className="px-3 py-2">Scene</th>
            <th className="px-3 py-2">AUC</th>
            <th className="px-3 py-2">Precision</th>
            <th className="px-3 py-2">Recall</th>
            <th className="px-3 py-2">FPR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-tactical-border bg-tactical-bg/80">
              <td className="px-3 py-2 font-mono text-xs">{r.model_type}</td>
              <td className="px-3 py-2">{r.scene}</td>
              <td className="px-3 py-2 font-mono">{r.auc}</td>
              <td className="px-3 py-2 font-mono">{r.precision}</td>
              <td className="px-3 py-2 font-mono">{r.recall}</td>
              <td className="px-3 py-2 font-mono">{r.fpr}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
