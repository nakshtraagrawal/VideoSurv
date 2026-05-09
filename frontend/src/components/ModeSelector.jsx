import { useState } from 'react'
import { setMode } from '../api/client'

const MODES = [
  { id: 'harvest', label: 'HARVEST', hint: 'Collect normal footage', color: 'bg-slate-600' },
  { id: 'surveillance', label: 'SURVEILLANCE', hint: 'Live detection', color: 'bg-red-900' },
  { id: 'review', label: 'REVIEW', hint: 'Triage alerts', color: 'bg-sky-900' },
]

export default function ModeSelector({ current, onChange }) {
  const [pending, setPending] = useState(null)

  async function select(m) {
    if (m === 'surveillance') {
      const ok = window.confirm(
        'Enable SURVEILLANCE mode? Alerts will fire when scores exceed the threshold.',
      )
      if (!ok) return
    }
    setPending(m)
    try {
      await setMode(m)
      onChange?.(m)
    } catch (e) {
      console.error(e)
      alert('Could not set mode')
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          disabled={pending !== null}
          onClick={() => select(m.id)}
          title={m.hint}
          className={`rounded border border-tactical-border px-3 py-1.5 text-xs font-semibold tracking-wide transition ${
            current === m.id
              ? `${m.color} text-white ring-2 ring-white/30`
              : 'bg-tactical-panel text-tactical-muted hover:bg-tactical-border/40'
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
