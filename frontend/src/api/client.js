import axios from 'axios'

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
})

export const getStatus = () => API.get('/api/system/status')
export const setMode = (mode) => API.post('/api/system/mode', { mode })
export const getScenes = () => API.get('/api/system/scenes')
export const setActiveModel = (model) =>
  API.post('/api/system/active-model', { model })
export const setThresholdApi = (model_type, scene, threshold) =>
  API.post('/api/system/threshold', { model_type, scene, threshold })
export const setSensitivity = (level) =>
  API.post('/api/system/sensitivity', { level })

export const getAlerts = (params) => API.get('/api/alerts', { params })
export const getAlert = (id) => API.get(`/api/alerts/${id}`)
export const submitFeedback = (id, data) =>
  API.post(`/api/alerts/${id}/feedback`, data)
export const deleteAlert = (id) => API.delete(`/api/alerts/${id}`)

export const startTraining = (data) => API.post('/api/training/start', data)
export const getTrainingRun = (id) => API.get(`/api/training/${id}`)
export const getTrainingHistory = () => API.get('/api/training/history')
export const retrainFromFeedback = () => API.post('/api/training/retrain')

export const runBenchmark = () => API.post('/api/benchmark/run')
export const getBenchmarkResults = () => API.get('/api/benchmark')
export const getBenchmarkSummary = () => API.get('/api/benchmark/summary')

export const runInference = (data) => API.post('/api/infer', data)
export const getInferJob = (id) => API.get(`/api/infer/${id}`)
export const getScores = (scene, video) =>
  API.get(`/api/scores/${scene}/${video}`)

export function clipUrl(alertId) {
  const base = import.meta.env.VITE_API_URL || ''
  return `${base}/api/alerts/${alertId}/clip`
}
