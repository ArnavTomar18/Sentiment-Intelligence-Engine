import axios from 'axios'

const api = axios.create({
  baseURL: 'https://sentiment-intelligence-engine.onrender.com/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
})

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.detail || err.response?.data?.message || err.message || 'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

export const checkHealth = () => api.get('/health')

export const predictHotelSentiment = (text, model) => api.post('/predict/hotel/sentiment', { review: text, model })
export const predictHotelRating    = (text, model) => api.post('/predict/hotel/rating',    { review: text, model })
export const predictHotelChurn     = (text, model) => api.post('/predict/hotel/churn',     { review: text, model })
export const compareHotelModels    = (text)        => api.post('/predict/hotel/compare',   { review: text })

export const predictAppFeedback  = (text, model) => api.post('/predict/app/feedback', { review: text, model })
export const predictAppRecommend = (text, model) => api.post('/predict/app/recommend',{ review: text, model })
export const compareAppModels    = (text)        => api.post('/predict/app/compare',  { review: text })

export const predictFashionSentiment = (text, model) => api.post('/predict/fashion/sentiment', { review: text, model })
export const predictFashionRating    = (text, model) => api.post('/predict/fashion/rating',    { review: text, model })
export const compareFashionModels    = (text)        => api.post('/predict/fashion/compare',   { review: text })

export const predictNews      = (text, model) => api.post('/predict/news',         { text, model })
export const compareNewsModels = (text)       => api.post('/predict/news/compare', { text })

export const predictOttSentiment = (text, model) => api.post('/predict/ott/sentiment', { review: text, model })
export const predictOttViral     = (text, model) => api.post('/predict/ott/viral',     { review: text, model })
export const predictOttRecommend = (text, model) => api.post('/predict/ott/recommend', { review: text, model })
export const compareOttModels    = (text)        => api.post('/predict/ott/compare',   { review: text })

export const getOttTitles       = (contentType)  => api.get('/ott/titles', { params: { content_type: contentType } })
export const getOttSimilar      = (title, contentType, n) => api.post('/ott/recommend/similar', { title, content_type: contentType, n })
export const getOttByPreference = (payload)      => api.post('/ott/recommend/preference', payload)

export const analyzeBatch       = (items)  => api.post('/analyze/batch', items)
export const getBestModels      = ()       => api.get('/reports/best-models')
export const getFullComparison  = ()       => api.get('/reports/full-comparison')

export default api
