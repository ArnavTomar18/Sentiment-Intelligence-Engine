import { useState, useCallback } from 'react'
import { predict } from '../api/client'

export function usePrediction(domain) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = useCallback(async (model, payload) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await predict(domain, model, payload)
      setResult(data)
      return data
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [domain])

  return { result, loading, error, run }
}
