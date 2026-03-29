<div align="center">

<!-- Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Sentiment%20Intelligence%20Engine&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Multi-Domain%20AI%2FML%20Sentiment%20Analysis%20Platform&descAlignY=58&descSize=18&animation=fadeIn" width="100%" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-02569B?style=for-the-badge)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

<br/>

> **A personal, production-ready full-stack AI/ML platform for intelligent sentiment analysis, fake news detection, and content recommendation — spanning 5 real-world domains with 18+ trained models. Built to run on your own device.**

<br/>

[🌐 Live Demo](https://sentiment-intelligence-engine.vercel.app/) · [📖 API Docs](#-api-reference) · [🧠 Models](#-ml-models--architecture) · [⚙️ Setup](#-getting-started) · [🗺️ Roadmap](#-roadmap)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🎯 Domains & Features](#-domains--features)
- [🧠 ML Models & Architecture](#-ml-models--architecture)
- [🗂️ Project Structure](#-project-structure)
- [⚙️ Getting Started](#-getting-started)
- [🔌 API Reference](#-api-reference)
- [💻 Frontend](#-frontend)
- [🚀 Deployment](#-deployment)
- [📊 Performance & Reports](#-performance--reports)
- [🗺️ Roadmap](#-roadmap)
- [👤 Author](#-author)

---

## ✨ Overview

The **Sentiment Intelligence Engine (SIE)** is a personal, full-stack AI/ML platform built to run **on your own device**. It combines natural language processing, ensemble machine learning, and real-time inference across five real-world domains — complete with EDA visualizations, model comparison reports, and a live React frontend.

🌐 **Live at:** [sentiment-intelligence-engine.vercel.app](https://sentiment-intelligence-engine.vercel.app/)

### Highlights

- 🧠 **18+ trained models** — Logistic Regression, SVC, XGBoost, LightGBM, SVR, Ridge Regression
- 🗳️ **Soft-voting ensembles** for maximum prediction robustness
- 📡 **TF-IDF cosine similarity** engine for OTT content recommendation
- 📊 **Rich EDA galleries** — wordclouds, distributions, heatmaps, trend charts per domain
- 📄 **Model comparison reports** — CSV + PNG artifacts in `/reports`
- 🌗 **Dual frontend** — React 18 + Vite (Vercel) and Streamlit dark-mode dashboard
- ⚡ **FastAPI backend** with domain-specific routers

---

## 🎯 Domains & Features

### 📰 1. News — Fake vs. Real Detection
> Binary classification of news articles

- Models: Logistic Regression, SVC, XGBoost, LightGBM + Soft Voting Ensemble
- EDA: `label_distribution`, `subject_distribution`, `article_length`, `wordcloud_fake`, `wordcloud_real`
- Output: `Fake` / `Real` + confidence score

---

### 🏨 2. Hotel Reviews — Sentiment Analysis
> Predict sentiment and star ratings from guest reviews

- Models: Logistic Regression, Ridge, SVR, SVC
- EDA: `rating_distribution`, `length_by_rating`, `review_length`, `sentiment_split`, `top_keywords`, `wordcloud`
- Output: Sentiment label + star rating prediction

---

### 👗 3. Fashion — Product Review Sentiment
> Understand customer sentiment for fashion & apparel

- Models: LightGBM, XGBoost, Logistic Regression
- EDA: `rating_distribution`, `aspect_counts`, `top_items`, `wordcloud`
- Output: Sentiment classification per review

---

### 📱 4. App Reviews — Mobile App Sentiment
> Classify user reviews for mobile applications

- Models: Logistic Regression, SVC, XGBoost, LightGBM + Ensemble
- EDA: `feedback_distribution`, `rating_distribution`, `top_apps`, `wordcloud`
- Output: Positive / Neutral / Negative

---

### 🎬 5. OTT — Streaming Content Recommendation
> TF-IDF based content discovery engine

- Method: Cosine similarity over a vectorized content corpus
- EDA: `content_type`, `platform_distribution`, `release_year_trend`, `top_genres`, `wordcloud`
- Output: Top-K recommended titles with similarity scores

---

## 🧠 ML Models & Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Sentiment Intelligence Engine                   │
│                                                                  │
│   Raw Text Input                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐     ┌──────────────┐     ┌────────────────┐  │
│   │  TF-IDF     │────▶│  Vectorizer  │────▶│ Label Encoder  │  │
│   │  Pipeline   │     │  (per domain)│     │  (per domain)  │  │
│   └─────────────┘     └──────────────┘     └────────────────┘  │
│                               │                                  │
│          ┌────────────────────┼───────────────────┐             │
│          ▼                    ▼                   ▼             │
│   ┌────────────┐     ┌──────────────┐     ┌────────────────┐   │
│   │ Logistic   │     │  SVC / SVR   │     │ XGBoost /      │   │
│   │ Regression │     │              │     │ LightGBM       │   │
│   └────────────┘     └──────────────┘     └────────────────┘   │
│          │                    │                   │             │
│          └────────────────────┴── Soft Voting ────┘             │
│                                    Ensemble                      │
│                                       │                         │
│                                       ▼                         │
│                              Final Prediction                    │
└──────────────────────────────────────────────────────────────────┘
```

| Domain      | Models Used                              | Task                    |
|-------------|------------------------------------------|-------------------------|
| News        | LR, SVC, XGBoost, LightGBM, Ensemble     | Binary Classification   |
| Hotel       | LR, Ridge, SVR, SVC                      | Multi-class / Regression|
| Fashion     | LR, XGBoost, LightGBM                    | Sentiment Classification|
| App Reviews | LR, SVC, XGBoost, LightGBM, Ensemble     | Multi-class Classification |
| OTT         | TF-IDF Cosine Similarity                 | Content Recommendation  |

---

## 🗂️ Project Structure

```
sentiment-intelligence-engine/
│
├── frontend/
│   ├── public/
│   │   └── eda/                          # EDA charts served to the React frontend
│   │       ├── app/
│   │       │   ├── feedback_distribution.png
│   │       │   ├── rating_distribution.png
│   │       │   ├── top_apps.png
│   │       │   └── wordcloud.png
│   │       ├── fashion/
│   │       │   ├── aspect_counts.png
│   │       │   ├── rating_distribution.png
│   │       │   ├── top_items.png
│   │       │   └── wordcloud.png
│   │       ├── hotel/
│   │       │   ├── length_by_rating.png
│   │       │   ├── rating_distribution.png
│   │       │   ├── review_length.png
│   │       │   ├── sentiment_split.png
│   │       │   ├── top_keywords.png
│   │       │   └── wordcloud.png
│   │       ├── news/
│   │       │   ├── article_length.png
│   │       │   ├── label_distribution.png
│   │       │   ├── subject_distribution.png
│   │       │   ├── wordcloud_fake.png
│   │       │   └── wordcloud_real.png
│   │       └── ott/
│   │           ├── content_type.png
│   │           ├── platform_distribution.png
│   │           ├── release_year_trend.png
│   │           ├── top_genres.png
│   │           └── wordcloud.png
│   └── src/
│       ├── index.html
│       ├── package.json
│       ├── package-lock.json
│       └── vite.config.js
│
├── models/                               # Trained .pkl artifacts (local only, gitignored)
│   ├── app_reviews/
│   ├── fashion/
│   ├── hotel/
│   ├── news/
│   ├── ott/
│   └── saved_models/
│
├── notebooks/
│   └── eda/                              # EDA notebooks per domain
│       ├── app/
│       ├── fashion/
│       └── hotel/
│   ├── eda_apps.ipynb
│   ├── eda_fashion.ipynb
│   ├── eda_hotel.ipynb
│   ├── eda_news.ipynb
│   └── eda_ott.ipynb
│
├── reports/                              # Model comparison artifacts
│   ├── best_models.csv
│   ├── best_models_heatmap.png
│   ├── model_comparison_chart.png
│   └── model_comparison_full.csv
│
├── .gitignore
└── README.md
```

> ⚠️ The `models/` directory (`.pkl` files) is **gitignored** and not committed to version control. Run the EDA + training notebooks locally to generate all model artifacts on your machine.

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- pip / virtualenv

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-intelligence-engine.git
cd sentiment-intelligence-engine
```

---

### 2. Set Up Python Environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

### 3. Generate Model Artifacts Locally

Run each EDA notebook to train and save models into `models/`:

```bash
cd notebooks
jupyter notebook
# Run: eda_apps.ipynb, eda_fashion.ipynb, eda_hotel.ipynb, eda_news.ipynb, eda_ott.ipynb
```

---

### 4. Start the FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

---

### 5. React Frontend (already live — optional local run)

```bash
cd frontend/src
npm install

# Set your backend URL
echo "VITE_API_URL=http://localhost:8000" > .env

npm run dev
# → http://localhost:3000
```

---

### 6. Streamlit Dashboard (optional)

```bash
streamlit run dashboard.py
# → http://localhost:8501
```

---

## 🔌 API Reference

Base URL (local): `http://localhost:8000`

### 📰 News

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/news/predict` | Classify a single article |
| `POST` | `/news/batch` | Batch news classification |

```json
// Request
{ "text": "Breaking: Scientists confirm discovery of..." }

// Response
{ "prediction": "Real", "confidence": 0.94, "model": "ensemble" }
```

---

### 🏨 Hotel

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/hotel/predict` | Predict review sentiment & rating |
| `POST` | `/hotel/batch` | Batch review analysis |

---

### 👗 Fashion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/fashion/predict` | Fashion review sentiment |
| `POST` | `/fashion/batch` | Batch fashion analysis |

---

### 📱 App Reviews

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/app/predict` | Classify a single app review |
| `POST` | `/app/batch` | Batch app review analysis |

---

### 🎬 OTT

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ott/recommend` | Get content recommendations |
| `GET`  | `/ott/titles` | List all available content titles |

```json
// Request
{ "query": "dark sci-fi thriller with time travel", "top_k": 5 }

// Response
{
  "recommendations": [
    { "title": "Dark", "score": 0.91 },
    { "title": "Interstellar", "score": 0.87 },
    { "title": "Coherence", "score": 0.82 }
  ]
}
```

---

## 💻 Frontend

### 🌐 React 18 + Vite — [Live on Vercel](https://sentiment-intelligence-engine.vercel.app/)

- Production SPA deployed on Vercel
- Domain-specific pages with live inference
- EDA gallery from `frontend/public/eda/` — all charts rendered in-browser
- Environment-aware routing via `VITE_API_URL`

### 📊 Streamlit Dark-Mode Dashboard — Run Locally

- Interactive EDA exploration with Plotly
- Live model inference UI per domain
- Model comparison visualizations from `/reports`

---

## 🚀 Deployment

### Frontend → Vercel ✅ Already Live

🌐 **[sentiment-intelligence-engine.vercel.app](https://sentiment-intelligence-engine.vercel.app/)**

To deploy your own fork:

1. Push to GitHub
2. Import on [Vercel](https://vercel.com) → set root to `frontend/src`
3. Add env var: `VITE_API_URL=https://your-render-backend.onrender.com`
4. Deploy ✅

---

### Backend → Render

1. Create a **Web Service** on [Render](https://render.com)
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add your environment variables

> ⚠️ Render free tier **sleeps after 15 min of inactivity**. Use a keep-alive cron or upgrade to paid for always-on access.

---

## 📊 Performance & Reports

Model evaluation reports are saved in `/reports/`:

| File | Description |
|------|-------------|
| `best_models.csv` | Best model per domain with metrics |
| `model_comparison_full.csv` | Full comparison across all models & domains |
| `best_models_heatmap.png` | Accuracy heatmap across models |
| `model_comparison_chart.png` | Bar chart comparison of top models |

### Accuracy Summary

| Domain      | Best Model           | Accuracy | F1 Score |
|-------------|----------------------|----------|----------|
| News        | Soft Voting Ensemble | ~96%     | ~0.96    |
| Hotel       | LightGBM             | ~88%     | ~0.87    |
| Fashion     | XGBoost              | ~85%     | ~0.84    |
| App Reviews | Soft Voting Ensemble | ~91%     | ~0.90    |
| OTT         | TF-IDF Cosine Sim    | —        | —        |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Python 3.10+ |
| **ML / NLP** | scikit-learn, XGBoost, LightGBM, Optuna, joblib |
| **Frontend** | React 18, Vite, Axios → Vercel |
| **Dashboard** | Streamlit, Plotly |
| **Vectorization** | TF-IDF (scikit-learn) |
| **Deployment** | Vercel (frontend), Render (backend) |
| **Dev Env** | Windows, PowerShell, VSCode |

---

## 🗺️ Roadmap

- [x] 5-domain FastAPI backend with `include_router` pattern
- [x] 18+ trained ML models across all domains
- [x] Soft-voting ensemble inference
- [x] TF-IDF OTT recommendation engine
- [x] React 18 + Vite frontend live on Vercel
- [x] Streamlit dark-mode dashboard
- [x] EDA charts per domain in `frontend/public/eda/`
- [x] Model comparison reports in `/reports`
- [x] Batch prediction endpoints
- [ ] BERT / DistilBERT fine-tuned models per domain
- [ ] API key authentication
- [ ] Docker & Docker Compose support
- [ ] Real-time data pipeline
- [ ] Model versioning & A/B testing framework

---

## 👤 Author

<div align="center">

**Arnav Tomar**

[![GitHub](https://img.shields.io/badge/GitHub-arnavtomar18-000?style=for-the-badge&logo=github)](https://github.com/Arnavtomar18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/arnavtomar1)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-000000?style=for-the-badge&logo=vercel)](https://sentiment-intelligence-engine.vercel.app/)

*Built with curiosity, coffee, and way too many model training runs.*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=100&section=footer" width="100%" />

⭐ **If SIE helped you, consider giving it a star!** ⭐

</div>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=100&section=footer" width="100%" />

⭐ **If SIE helped you, consider giving it a star!** ⭐

</div>
