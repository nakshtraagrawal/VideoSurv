import { Link } from 'react-router-dom'

const COLORS = {
  conv_ae: 'border-blue-500/60 text-blue-200',
  astnet: 'border-amber-500/60 text-amber-200',
  hstforu: 'border-emerald-500/60 text-emerald-200',
}

export default function ModelBadge({ name, auc }) {
  const c = COLORS[name] || 'border-slate-500 text-slate-200'
  const label =
    name === 'conv_ae' ? 'Conv AE' : name === 'hstforu' ? 'HSTforU' : 'ASTNet'
  return (
    <Link
      to="/settings"
      className={`inline-flex items-center gap-2 rounded border bg-tactical-panel px-2 py-1 text-xs ${c}`}
    >
      <span className="font-mono uppercase">{label}</span>
      {auc != null && (
        <span className="font-mono text-tactical-muted">AUC {auc}</span>
      )}
    </Link>
  )
}
