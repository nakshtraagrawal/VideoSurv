export default function HeatmapVideo({ src, score }) {
  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded border border-dashed border-tactical-border bg-tactical-panel text-sm text-tactical-muted">
        No clip
      </div>
    )
  }
  return (
    <div className="relative overflow-hidden rounded border border-tactical-border bg-black">
      <video
        key={src}
        className="h-full w-full"
        src={src}
        controls
        autoPlay
        loop
        playsInline
      />
      {score != null && (
        <div className="absolute right-2 top-2 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white">
          {typeof score === 'number' ? score.toFixed(5) : score}
        </div>
      )}
    </div>
  )
}
