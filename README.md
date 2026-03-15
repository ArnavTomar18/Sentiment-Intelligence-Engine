# Sentiment Intelligence Engine — React Frontend

## Setup

```bash
npm install
npm run dev   # → http://localhost:3000
```

Your FastAPI backend must be running at `localhost:8000`.

---

## FastAPI additions needed

### 1. Static file mounts (for EDA images and report charts)

Add to `src/api/app.py`:

```python
from fastapi.staticfiles import StaticFiles

# EDA images  →  /api/v1/static/eda/hotel/wordcloud.png
app.mount("/api/v1/static/eda",     StaticFiles(directory="notebooks/eda"), name="eda-static")

# Report charts  →  /api/v1/static/reports/best_models_heatmap.png
app.mount("/api/v1/static/reports", StaticFiles(directory="reports"),       name="reports-static")
```

### 2. New API endpoints needed

The React frontend calls these routes. Add them to `src/api/routes.py`:

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/v1/health` | Health check |
| `POST` | `/api/v1/predict/hotel/sentiment` | Hotel sentiment |
| `POST` | `/api/v1/predict/hotel/rating` | Hotel rating (Ridge) |
| `POST` | `/api/v1/predict/hotel/churn` | Hotel churn |
| `POST` | `/api/v1/predict/hotel/compare` | Hotel all models |
| `POST` | `/api/v1/predict/app/feedback` | App feedback type |
| `POST` | `/api/v1/predict/app/recommend` | App recommendation |
| `POST` | `/api/v1/predict/app/compare` | App all models |
| `POST` | `/api/v1/predict/fashion/sentiment` | Fashion sentiment |
| `POST` | `/api/v1/predict/fashion/rating` | Fashion rating |
| `POST` | `/api/v1/predict/fashion/compare` | Fashion all models |
| `POST` | `/api/v1/predict/news` | News fake/real |
| `POST` | `/api/v1/predict/news/compare` | News all models |
| `POST` | `/api/v1/predict/ott/sentiment` | OTT sentiment |
| `POST` | `/api/v1/predict/ott/viral` | OTT viral |
| `POST` | `/api/v1/predict/ott/recommend` | OTT recommend |
| `POST` | `/api/v1/predict/ott/compare` | OTT all models |
| `GET`  | `/api/v1/ott/titles` | OTT title list |
| `POST` | `/api/v1/ott/recommend/similar` | TF-IDF similar titles |
| `POST` | `/api/v1/ott/recommend/preference` | Filter by preference |
| `POST` | `/api/v1/analyze/batch` | Multi-domain batch |
| `GET`  | `/api/v1/reports/best-models` | Best models CSV as JSON |
| `GET`  | `/api/v1/reports/full-comparison` | Full comparison CSV as JSON |

### Compare endpoint response format

The `/compare` endpoints should return:
```json
[
  { "model": "Logistic Regression", "label": "Positive", "confidence": 0.91 },
  { "model": "XGBoost",             "label": "Positive", "confidence": 0.88 },
  { "model": "LightGBM",            "label": "Negative", "confidence": 0.63 }
]
```

---

## Project structure

```
src/
  api/          ← All FastAPI calls (client.js)
  components/
    layout/     ← Sidebar, Topbar
    common/     ← ResultPanel, ComparePanel, EDAGallery, TabBar
  hooks/        ← usePrediction
  pages/        ← 11 pages (one per Streamlit page)
```