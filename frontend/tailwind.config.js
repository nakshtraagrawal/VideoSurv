/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        tactical: {
          bg: '#0c0f12',
          panel: '#141a1f',
          border: '#2a3540',
          accent: '#3d7a5c',
          alert: '#c94c4c',
          muted: '#8a9ba8',
        },
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
