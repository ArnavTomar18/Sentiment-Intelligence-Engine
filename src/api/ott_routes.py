"""
OTT Routes — FastAPI Router
────────────────────────────────────────────────
Matches exact URLs from client.js (baseURL = /api/v1):

  GET  /api/v1/ott/titles                  → getOttTitles(contentType)
  POST /api/v1/ott/recommend/similar       → getOttSimilar(title, contentType, n)
  POST /api/v1/ott/recommend/preference    → getOttByPreference(payload)
  POST /api/v1/predict/ott/recommend       → predictOttRecommend(text, model)

No .pkl needed for recommender routes — all call ott_recommender.py directly.
"""

import os, pickle, traceback
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.recommender.ott_recommender import (
    load_ott_data,
    recommend_show,
    recommend_movies,
    find_your_mind,
    search_content,
    get_content_summary,
    get_platform_stats,
    get_genre_stats,
    POS_WORDS,
    NEG_WORDS,
)

# ── Two routers matching the two URL prefixes ────────────
ott_router     = APIRouter(prefix="/ott",         tags=["OTT Recommender"])
predict_router = APIRouter(prefix="/predict/ott", tags=["OTT Predict"])


# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════

def df_to_records(result):
    """Convert DataFrame or error string to JSON-safe list."""
    if isinstance(result, str):
        return None, result
    if hasattr(result, "to_dict"):
        return result.to_dict(orient="records"), None
    return list(result), None


# ════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════

class SimilarRequest(BaseModel):
    title          : str
    content_type   : str            = "tv show"
    n              : int            = 6
    platform_filter: Optional[str]  = None

class PreferenceRequest(BaseModel):
    content_type    : str           = "movie"
    year_preference : str           = "recent"
    age             : int           = 25
    genres          : List[str]     = []
    platform        : str           = ""
    top_n           : int           = 10

class PredictRequest(BaseModel):
    review : str
    model  : Optional[str] = ""


# ════════════════════════════════════════════════════════
# GET /api/v1/ott/titles?content_type=tv+show
# client.js: getOttTitles(contentType)
# ════════════════════════════════════════════════════════

@ott_router.get("/titles")
def get_titles(content_type: str = "tv show", type: str = None):
    ct = (type or content_type or "tv show").lower()
    try:
        df = load_ott_data()
        if "movie" in ct:
            mask = df["type"].str.lower().str.contains("movie|film", na=False)
        else:
            mask = df["type"].str.lower().str.contains("tv show|show|series", na=False)
        titles = sorted(df[mask]["title"].dropna().unique().tolist())
        return { "titles": titles, "count": len(titles) }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════
# POST /api/v1/ott/recommend/similar
# client.js: getOttSimilar(title, contentType, n)
# ════════════════════════════════════════════════════════

@ott_router.post("/recommend/similar")
def get_similar(body: SimilarRequest):
    try:
        if "movie" in body.content_type.lower():
            result = recommend_movies(body.title, top_n=body.n, platform_filter=body.platform_filter)
        else:
            result = recommend_show(body.title, top_n=body.n, platform_filter=body.platform_filter)

        records, err = df_to_records(result)
        if err:
            return JSONResponse(status_code=404, content={"error": err})

        return { "results": records, "count": len(records), "based_on": body.title }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════
# POST /api/v1/ott/recommend/preference
# client.js: getOttByPreference(payload)
# ════════════════════════════════════════════════════════

@ott_router.post("/recommend/preference")
def get_by_preference(body: PreferenceRequest):
    if not body.genres:
        return JSONResponse(status_code=400, content={"error": "At least one genre is required"})

    try:
        result = find_your_mind(
            content_type = body.content_type,
            year_pref    = body.year_preference,
            age          = body.age,
            genres       = body.genres,
            platform     = body.platform,
            top_n        = body.top_n,
        )
        records, err = df_to_records(result)
        if err:
            return JSONResponse(status_code=404, content={"error": err})

        return { "results": records, "count": len(records) }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════
# POST /api/v1/predict/ott/recommend
# client.js: predictOttRecommend(text, model)
# ════════════════════════════════════════════════════════

@predict_router.post("/recommend")
def predict_recommend(body: PredictRequest):
    text       = body.review.strip()
    model_name = (body.model or "").strip().lower() or "ensemble"

    if not text:
        return JSONResponse(status_code=400, content={"error": "review is required"})

    candidates = (
        [model_name] if model_name not in ("", "ensemble")
        else ["ensemble", "lightgbm", "logistic", "xgboost"]
    )

    model_dir = "models/ott"
    pipeline  = None
    used_name = None

    for name in candidates:
        path = os.path.join(model_dir, f"ott_recommender_{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                pipeline = pickle.load(f)
            used_name = name
            break

    # ── No pkl yet → lexicon fallback ─────────────────
    if pipeline is None:
        words      = set(text.lower().split())
        pos_score  = len(words & POS_WORDS)
        neg_score  = len(words & NEG_WORDS)
        total      = pos_score + neg_score or 1
        confidence = round(pos_score / total, 3)
        label      = "Recommended" if pos_score >= neg_score else "Not Recommended"
        return {
            "label"     : label,
            "confidence": confidence,
            "model"     : "lexicon-fallback",
            "note"      : "Run train_ott to enable ML predictions",
        }

    # ── ML prediction ──────────────────────────────────
    try:
        proba = pipeline.predict_proba([text])[0]
        pred  = int(pipeline.predict([text])[0])
        return {
            "label"     : "Recommended" if pred == 1 else "Not Recommended",
            "prediction": pred,
            "confidence": round(float(proba[pred]), 3),
            "model"     : used_name,
        }
    except AttributeError:
        pred = int(pipeline.predict([text])[0])
        return {
            "label"     : "Recommended" if pred == 1 else "Not Recommended",
            "prediction": pred,
            "confidence": None,
            "model"     : used_name,
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})