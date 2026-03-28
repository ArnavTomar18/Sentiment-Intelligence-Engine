<div align="center">

<!-- Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Sentiment%20Intelligence%20Engine&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Multi-Domain%20AI%2FML%20Sentiment%20Analysis%20Platform&descAlignY=58&descSize=18&animation=fadeIn" width="100%" />

<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-02569B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **A production-ready, full-stack AI/ML platform for intelligent sentiment analysis, fake news detection, content recommendation, and NLP classification — spanning 5 real-world domains with 28+ trained models.**

<br/>

[🚀 Live Demo](#-deployment) · [📖 API Docs](#-api-reference) · [🧠 Models](#-ml-models--architecture) · [⚙️ Setup](#-getting-started) · [🗺️ Roadmap](#-roadmap)

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
- [📊 Performance](#-performance)
- [🗺️ Roadmap](#-roadmap)
- [👤 Author](#-author)

---

## ✨ Overview

The **Sentiment Intelligence Engine (SIE)** is a full-stack AI/ML platform that brings together **natural language processing**, **ensemble machine learning**, and **real-time inference** across five diverse domains. It goes beyond a simple sentiment classifier — SIE is an intelligent text understanding suite featuring:

- 🧠 **28+ trained models** across Logistic Regression, SVC, XGBoost, LightGBM, SVR, Ridge Regression
- 🗳️ **Soft-voting ensemble** strategies for maximum prediction accuracy
- 📡 **TF-IDF cosine similarity** engine for live OTT content recommendation
- 📦 **Batch prediction endpoints** for high-throughput inference
- 📊 **EDA galleries** with interactive Plotly visualizations
- 🌗 **Dual frontend** — React 18 (Vite) + Streamlit dark-mode dashboard
- ⚡ **FastAPI backend** with full REST API routing across all domains

---

## 🎯 Domains & Features

SIE operates across **5 specialized domains**, each with its own models, endpoints, and EDA pipelines:

### 📰 1. News — Fake vs. Real Detection
> Binary classification of news articles into **Fake** or **Real**

- Models: Logistic Regression, SVC, XGBoost, LightGBM + Soft Voting Ensemble
- Custom label remapping from encoded integers to human-readable outputs
- Batch endpoint for bulk news verification
- Input: Raw article text → Output: `Fake` / `Real` + confidence scores

---

### 🏨 2. Hotel Reviews — Sentiment Analysis
> Predict star ratings and classify sentiment from guest reviews

- Models: Logistic Regression, Ridge, SVR, SVC
- Multi-class rating prediction (1–5 stars)
- Sentiment polarity: Positive / Neutral / Negative
- Batch inference endpoint for bulk review processing

---

### 👗 3. Fashion — Product Review Sentiment
> Understand customer sentiment for fashion & apparel products

- Fine-tuned TF-IDF vectorizer on fashion-specific vocabulary
- Models: LightGBM, XGBoost, Logistic Regression
- Useful for: product scoring, trend analysis, recommendation filtering

---

### 📱 4. App Reviews — Mobile App Sentiment
> Classify user reviews for mobile applications

- Multi-class sentiment classification (Positive / Neutral / Negative)
- Ensemble of classifiers for robust output
- Batch support for large-scale app store review analysis

---

### 🎬 5. OTT — Streaming Content Recommendation
> A live, TF-IDF-based content discovery engine for streaming platforms

- **Cosine similarity** matching over a vectorized content corpus
- Real-time recommendations based on free-text queries
- Covers movies, series, documentaries, and more
- Fully integrated `/recommend` endpoint

---

## 🧠 ML Models & Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Sentiment Intelligence Engine                   │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │   TF-IDF     │   │  Vectorizers │   │  Label Encoders  │   │
│  │  Pipelines   │──▶│  (per domain)│──▶│  (per domain)    │   │
│  └──────────────┘   └──────────────┘   └──────────────────┘   │
│                              │                                  │
│            ┌─────────────────┼────────────────┐                │
│            ▼                 ▼                ▼                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│   │  Logistic    │  │  SVC / SVR   │  │ XGBoost /    │        │
│   │  Regression  │  │              │  │ LightGBM     │        │
│   └──────────────┘  └──────────────┘  └──────────────┘        │
│            │                 │                ▼                │
│            └─────────────────┴──── Soft Voting Ensemble ──▶   │
│                                          Final Prediction       │
└─────────────────────────────────────────────────────────────────┘
```

| Domain   | Models Used                                      | Task Type           |
|----------|--------------------------------------------------|---------------------|
| News     | LR, SVC, XGBoost, LightGBM, Ensemble             | Binary Classification |
| Hotel    | LR, Ridge, SVR, SVC                              | Multi-class / Regression |
| Fashion  | LR, XGBoost, LightGBM                            | Sentiment Classification |
| App      | LR, SVC, XGBoost, LightGBM, Ensemble             | Multi-class Classification |
| OTT      | TF-IDF Cosine Similarity                         | Content Recommendation |

---

## 🗂️ Project Structure

```
sentiment-intelligence-engine/
│
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── routers/
│   │   ├── news.py              # News domain router
│   │   ├── hotel.py             # Hotel domain router
│   │   ├── fashion.py           # Fashion domain router
│   │   ├── app_reviews.py       # App Reviews domain router
│   │   └── ott.py               # OTT recommendation router
│   ├── models/                  # Trained .pkl model files
│   ├── utils/
│   │   ├── preprocessing.py     # Text cleaning & TF-IDF helpers
│   │   └── label_maps.py        # Label encoder remapping logic
│   └── requirements.txt
│
├── frontend/
│   ├── dashboard.py             # Streamlit dark-mode dashboard
│   └── src/                     # Vite + React 18 frontend
│       ├── api/                 # Axios API layer
│       ├── components/          # Reusable UI components
│       ├── pages/               # Domain-specific pages
│       └── main.jsx
│
├── notebooks/
│   ├── news_eda.ipynb
│   ├── hotel_eda.ipynb
│   ├── fashion_eda.ipynb
│   ├── app_reviews_eda.ipynb
│   └── ott_eda.ipynb
│
├── data/                        # Raw & processed datasets (gitignored)
├── .env.example
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.10+**
- Node.js **18+** (for React frontend)
- pip / virtualenv

---

### 🔧 Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sentiment-intelligence-engine.git
cd sentiment-intelligence-engine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows (PowerShell)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be live at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

### 🖥️ Frontend Setup — React (Vite)

```bash
cd frontend/src

# Install dependencies
npm install

# Set environment variable
cp .env.example .env
# Edit .env → VITE_API_URL=http://localhost:8000

# Start dev server
npm run dev
```

React app will be live at: `http://localhost:3000`

---

### 📊 Frontend Setup — Streamlit Dashboard

```bash
cd frontend
streamlit run dashboard.py
```

Dashboard will be live at: `http://localhost:8501`

---

### 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# Backend
API_HOST=0.0.0.0
API_PORT=8000

# Frontend (Vite)
VITE_API_URL=http://localhost:8000
```

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

All endpoints accept `Content-Type: application/json`.

---

### 📰 News Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/news/predict` | Single news article classification |
| `POST` | `/news/batch` | Batch news classification |

**Request:**
```json
{
  "text": "Scientists discover a new planet in the solar system..."
}
```
**Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.94,
  "model": "ensemble"
}
```

---

### 🏨 Hotel Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/hotel/predict` | Predict review sentiment & rating |
| `POST` | `/hotel/batch` | Batch review analysis |

---

### 👗 Fashion Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/fashion/predict` | Fashion review sentiment |
| `POST` | `/fashion/batch` | Batch fashion analysis |

---

### 📱 App Review Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/app/predict` | App review sentiment classification |
| `POST` | `/app/batch` | Batch app review analysis |

---

### 🎬 OTT Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ott/recommend` | Get content recommendations |
| `GET`  | `/ott/titles` | List all available titles |

**Request:**
```json
{
  "query": "sci-fi thriller with time travel",
  "top_k": 5
}
```
**Response:**
```json
{
  "recommendations": [
    { "title": "Dark", "score": 0.91 },
    { "title": "Interstellar", "score": 0.87 },
    ...
  ]
}
```

---

## 💻 Frontend

SIE ships with **two frontend implementations** for different use cases:

### ⚛️ React 18 + Vite
- Modern SPA architecture
- Domain-specific pages with live API calls
- Responsive design, production-deployable to Netlify
- Environment-aware API routing via `VITE_API_URL`

### 📊 Streamlit Dashboard
- Dark-mode enabled analytics dashboard
- EDA galleries powered by Plotly
- Real-time model inference UI
- Ideal for internal/demo use

---

## 🚀 Deployment

### Backend → Render

1. Push your repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the Render dashboard

> ⚠️ **Note:** Render's free tier sleeps after ~15 minutes of inactivity. Use a keep-alive ping service or upgrade to a paid tier for production use.

---

### Frontend → Netlify

```bash
cd frontend/src
npm run build
```

1. Push to GitHub
2. Connect the repo on [Netlify](https://netlify.com)
3. Set build command: `npm run build`
4. Set publish directory: `dist`
5. Add environment variable: `VITE_API_URL=https://your-render-backend.onrender.com`

---

## 📊 Performance

> Model performance metrics across domains (test set evaluation):

| Domain   | Best Model          | Accuracy | F1 Score |
|----------|---------------------|----------|----------|
| News     | Soft Voting Ensemble | ~96%    | ~0.96    |
| Hotel    | LightGBM            | ~88%     | ~0.87    |
| Fashion  | XGBoost             | ~85%     | ~0.84    |
| App      | Soft Voting Ensemble | ~91%    | ~0.90    |
| OTT      | TF-IDF Cosine Sim   | N/A (Rec.)| N/A    |

---

## 🗺️ Roadmap

- [x] 5-domain FastAPI backend with full routing
- [x] 28+ trained scikit-learn models
- [x] Soft-voting ensemble inference
- [x] TF-IDF OTT recommender engine
- [x] React 18 + Vite frontend
- [x] Streamlit dark-mode dashboard
- [x] Batch prediction endpoints
- [x] EDA notebooks & galleries
- [ ] Transformer-based models (BERT / DistilBERT) per domain
- [ ] User authentication & API key management
- [ ] Model versioning & A/B testing framework
- [ ] Real-time data ingestion pipeline
- [ ] Docker & Docker Compose deployment
- [ ] Monitoring & alerting (Prometheus + Grafana)
- [ ] Admin analytics dashboard

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Python 3.10+ |
| **ML / NLP** | scikit-learn, XGBoost, LightGBM, Optuna, joblib |
| **Frontend (SPA)** | React 18, Vite, Axios |
| **Frontend (Dashboard)** | Streamlit, Plotly |
| **Vectorization** | TF-IDF (scikit-learn) |
| **Deployment** | Render (backend), Netlify (frontend) |
| **Dev Environment** | Windows, PowerShell, VSCode |

---

## 👤 Author

<div align="center">

**Arnav**

[![GitHub](https://img.shields.io/badge/GitHub-your--username-18171?style=for-the-badge&logo=github)](https://github.com/arnavtomar18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/arnavtomar18)

*Built with curiosity, coffee, and way too many model training runs.*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=100&section=footer" width="100%" />

⭐ **If SIE helped you, consider giving it a star!** ⭐

</div>
