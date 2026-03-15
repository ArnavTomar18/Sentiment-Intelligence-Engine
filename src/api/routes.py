from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.api.prediction_service import (
    predict_news, compare_news,
    predict_hotel_sentiment, predict_hotel_churn, predict_hotel_rating, compare_hotel,
    predict_fashion_sentiment, predict_fashion_rating, compare_fashion,
    predict_app_feedback, predict_app_recommend, compare_app,
    predict_ott_sentiment, predict_ott_viral, predict_ott_recommend, compare_ott,
    predict_batch,
)

router = APIRouter()

# ── Request schemas ───────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text   : Optional[str] = None
    review : Optional[str] = None   # some pages send "review" instead of "text"
    model  : Optional[str] = None   # e.g. "xgboost" | "svr" | "lightgbm" | ...

    def get_text(self):
        return (self.review or self.text or "").strip()

class BatchItem(BaseModel):
    domain: str
    text:   str

# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

# ── NEWS ──────────────────────────────────────────────────────────────────────

@router.post("/predict/news")
def route_news(req: TextRequest):
    try:
        return predict_news(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/news/compare")
def route_news_compare(req: TextRequest):
    try:
        return compare_news(req.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── HOTEL ─────────────────────────────────────────────────────────────────────

@router.post("/predict/hotel/sentiment")
def route_hotel_sentiment(req: TextRequest):
    try:
        return predict_hotel_sentiment(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hotel/churn")
def route_hotel_churn(req: TextRequest):
    try:
        return predict_hotel_churn(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hotel/rating")
def route_hotel_rating(req: TextRequest):
    try:
        return predict_hotel_rating(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/hotel/compare")
def route_hotel_compare(req: TextRequest):
    try:
        return compare_hotel(req.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── FASHION ───────────────────────────────────────────────────────────────────

@router.post("/predict/fashion/sentiment")
def route_fashion_sentiment(req: TextRequest):
    try:
        return predict_fashion_sentiment(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/fashion/rating")
def route_fashion_rating(req: TextRequest):
    """
    Fashion rating prediction.
    Pass model = "xgboost" or "svr" in the request body.
    Defaults to xgboost if not specified.
    """
    try:
        return predict_fashion_rating(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/fashion/compare")
def route_fashion_compare(req: TextRequest):
    try:
        return compare_fashion(req.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── APP REVIEWS ───────────────────────────────────────────────────────────────

@router.post("/predict/app/feedback")
def route_app_feedback(req: TextRequest):
    try:
        return predict_app_feedback(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/app/recommend")
def route_app_recommend(req: TextRequest):
    try:
        return predict_app_recommend(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/app/compare")
def route_app_compare(req: TextRequest):
    try:
        return compare_app(req.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── OTT ───────────────────────────────────────────────────────────────────────

@router.post("/predict/ott/sentiment")
def route_ott_sentiment(req: TextRequest):
    try:
        return predict_ott_sentiment(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/ott/viral")
def route_ott_viral(req: TextRequest):
    try:
        return predict_ott_viral(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/ott/recommend")
def route_ott_recommend(req: TextRequest):
    try:
        return predict_ott_recommend(req.get_text(), req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/ott/compare")
def route_ott_compare(req: TextRequest):
    try:
        return compare_ott(req.get_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── BATCH ─────────────────────────────────────────────────────────────────────

@router.post("/analyze/batch")
def route_batch(items: list[BatchItem]):
    try:
        return predict_batch([i.dict() for i in items])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── REPORTS ───────────────────────────────────────────────────────────────────

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

@router.get("/reports/best-models")
def report_best():
    path = ROOT / "reports" / "best_models.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

@router.get("/reports/full-comparison")
def report_full():
    path = ROOT / "reports" / "model_comparison_full.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))