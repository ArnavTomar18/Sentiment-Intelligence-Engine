import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def _load(domain, filename):
    path = ROOT / "models" / domain / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# ══════════════════════════════════════════════════════
# MODEL STORE
# ══════════════════════════════════════════════════════

class ModelStore:
    def __init__(self):
        print(f"  Root: {ROOT}")
        print("  Loading all models...")

        self.news = {
            "logistic": _load("news", "news_logistic.pkl"),
            "svc":      _load("news", "news_svc.pkl"),
            "xgboost":  _load("news", "news_xgboost.pkl"),
        }
        self.news_encoder = _load("news", "label_encoder.pkl")

        self.hotel_sentiment = {
            "logistic": _load("hotel", "hotel_sentiment_logistic.pkl"),
            "xgboost":  _load("hotel", "hotel_sentiment_xgboost.pkl"),
            "lightgbm": _load("hotel", "hotel_sentiment_lightgbm.pkl"),
        }
        self.hotel_churn = {
            "svc":      _load("hotel", "hotel_churn_svc.pkl"),
            "xgboost":  _load("hotel", "hotel_churn_xgboost.pkl"),
            "lightgbm": _load("hotel", "hotel_churn_lightgbm.pkl"),
        }
        self.hotel_rating = _load("hotel", "hotel_rating_ridge.pkl")

        self.fashion_sentiment = {
            "logistic": _load("fashion", "fashion_sentiment_logistic.pkl"),
            "xgboost":  _load("fashion", "fashion_sentiment_xgboost.pkl"),
            "lightgbm": _load("fashion", "fashion_sentiment_lightgbm.pkl"),
        }

        # ── Fashion Rating — 2 models, no ridge ──────────────────────────
        self.fashion_rating = {
            "xgboost": _load("fashion", "fashion_rating_xgboost.pkl"),
            "svr":     _load("fashion", "fashion_rating_svr.pkl"),
        }

        self.app_feedback = {
            "logistic": _load("app_reviews", "app_feedback_logistic.pkl"),
            "xgboost":  _load("app_reviews", "app_feedback_xgboost.pkl"),
            "lightgbm": _load("app_reviews", "app_feedback_lightgbm.pkl"),
        }
        self.app_feedback_encoder = _load("app_reviews", "feedback_label_encoder.pkl")

        self.app_recommender = {
            "svc":      _load("app_reviews", "app_recommender_svc.pkl"),
            "xgboost":  _load("app_reviews", "app_recommender_xgboost.pkl"),
            "lightgbm": _load("app_reviews", "app_recommender_lightgbm.pkl"),
        }

        self.ott_sentiment = {
            "logistic": _load("ott", "ott_sentiment_logistic.pkl"),
            "xgboost":  _load("ott", "ott_sentiment_xgboost.pkl"),
            "lightgbm": _load("ott", "ott_sentiment_lightgbm.pkl"),
        }
        self.ott_viral = {
            "svc":      _load("ott", "ott_viral_svc.pkl"),
            "xgboost":  _load("ott", "ott_viral_xgboost.pkl"),
            "lightgbm": _load("ott", "ott_viral_lightgbm.pkl"),
        }
        self.ott_recommender = {
            "logistic": _load("ott", "ott_recommender_logistic.pkl"),
            "xgboost":  _load("ott", "ott_recommender_xgboost.pkl"),
        }
        self.ott_genre_encoder = _load("ott", "genre_label_encoder.pkl")

        print(f"  news_encoder classes        : {list(self.news_encoder.classes_)}")
        print(f"  app_feedback_encoder classes: {list(self.app_feedback_encoder.classes_)}")
        print("  All models loaded ✓")

store = ModelStore()

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

DEFAULTS = {
    "news":            "logistic",
    "hotel_sentiment": "lightgbm",
    "hotel_churn":     "lightgbm",
    "fashion":         "lightgbm",
    "fashion_rating":  "xgboost",    # ← default for rating
    "app_feedback":    "lightgbm",
    "app_recommender": "lightgbm",
    "ott_sentiment":   "lightgbm",
    "ott_viral":       "lightgbm",
    "ott_recommender": "xgboost",
}

def _pick(d, model, key):
    k = (model or DEFAULTS[key]).lower().strip()
    return d.get(k, d[DEFAULTS[key]]), k

def _conf(model, text):
    try:
        return float(max(model.predict_proba([text])[0]))
    except Exception:
        try:
            s = float(model.decision_function([text])[0])
            return float(1 / (1 + np.exp(-abs(s))))
        except Exception:
            return None

def _decode_with_encoder(raw, encoder):
    try:
        if hasattr(raw, 'item'):
            raw = raw.item()
        return str(encoder.inverse_transform([raw])[0])
    except Exception:
        try:
            classes = list(encoder.classes_)
            idx = int(raw)
            if 0 <= idx < len(classes):
                return str(classes[idx])
        except Exception:
            pass
        return None

def _sentiment_label(raw):
    if hasattr(raw, 'item'):
        raw = raw.item()
    if isinstance(raw, str):
        low = raw.lower()
        if low in ("positive", "pos", "1", "true"):   return "Positive"
        if low in ("negative", "neg", "0", "false"):  return "Negative"
        return raw.capitalize()
    return "Positive" if int(raw) == 1 else "Negative"

def _churn_label(raw):
    if hasattr(raw, 'item'):
        raw = raw.item()
    if isinstance(raw, str):
        return "High Churn Risk" if raw.lower() in ("1","true","high","churn","yes") else "Low Churn Risk"
    return "High Churn Risk" if int(raw) == 1 else "Low Churn Risk"

def _viral_label(raw):
    if hasattr(raw, 'item'):
        raw = raw.item()
    if isinstance(raw, str):
        return "High Viral" if raw.lower() in ("1","true","high","viral","yes") else "Low Viral"
    return "High Viral" if int(raw) == 1 else "Low Viral"

def _recommend_label(raw):
    if hasattr(raw, 'item'):
        raw = raw.item()
    if isinstance(raw, str):
        return "Recommended" if raw.lower() in ("1","true","yes","recommended") else "Not Recommended"
    return "Recommended" if int(raw) == 1 else "Not Recommended"

def _rating_predict(model, text, clip_min=1.0, clip_max=5.0):
    """Run regression prediction and clip to valid range."""
    return float(np.clip(model.predict([text])[0], clip_min, clip_max))

# ══════════════════════════════════════════════════════
# NEWS
# ══════════════════════════════════════════════════════

def predict_news(text, model=None):
    m, k  = _pick(store.news, model, "news")
    raw   = m.predict([text])[0]
    label = _decode_with_encoder(raw, store.news_encoder)
    if label is None:
        label = "Fake" if int(raw) == 0 else "Real"
    return {"domain": "news", "label": label, "confidence": _conf(m, text), "model": k, "task": "fake_detection"}

def compare_news(text):
    rows = []
    for n, m in store.news.items():
        raw   = m.predict([text])[0]
        label = _decode_with_encoder(raw, store.news_encoder)
        if label is None:
            label = "Fake" if int(raw) == 0 else "Real"
        rows.append({"model": n, "label": label, "confidence": _conf(m, text)})
    return rows

# ══════════════════════════════════════════════════════
# HOTEL
# ══════════════════════════════════════════════════════

def predict_hotel_sentiment(text, model=None):
    m, k = _pick(store.hotel_sentiment, model, "hotel_sentiment")
    raw  = m.predict([text])[0]
    return {"domain": "hotel", "label": _sentiment_label(raw), "confidence": _conf(m, text), "model": k, "task": "sentiment"}

def predict_hotel_churn(text, model=None):
    m, k = _pick(store.hotel_churn, model, "hotel_churn")
    raw  = m.predict([text])[0]
    return {"domain": "hotel", "label": _churn_label(raw), "confidence": _conf(m, text), "model": k, "task": "churn"}

def predict_hotel_rating(text, model=None):
    r = _rating_predict(store.hotel_rating, text)
    return {"domain": "hotel", "label": f"{round(r,1)} / 5.0", "rating": round(r, 1),
            "confidence": None, "model": "ridge", "task": "rating"}

def compare_hotel(text):
    rows = []
    for n, m in store.hotel_sentiment.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"sentiment/{n}", "label": _sentiment_label(raw), "confidence": _conf(m, text)})
    for n, m in store.hotel_churn.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"churn/{n}", "label": _churn_label(raw), "confidence": _conf(m, text)})
    r = _rating_predict(store.hotel_rating, text)
    rows.append({"model": "rating/ridge", "label": f"{round(r,1)} / 5.0", "confidence": None})
    return rows

# ══════════════════════════════════════════════════════
# FASHION
# ══════════════════════════════════════════════════════

def predict_fashion_sentiment(text, model=None):
    m, k = _pick(store.fashion_sentiment, model, "fashion")
    raw  = m.predict([text])[0]
    return {"domain": "fashion", "label": _sentiment_label(raw), "confidence": _conf(m, text), "model": k, "task": "sentiment"}

def predict_fashion_rating(text, model=None):
    m, k = _pick(store.fashion_rating, model, "fashion_rating")
    r    = _rating_predict(m, text)

    # ── Sentiment-boosted correction ─────────────────
    try:
        # use best sentiment model to get confidence
        sent_model = store.fashion_sentiment[DEFAULTS["fashion"]]
        raw        = sent_model.predict([text])[0]
        conf       = _conf(sent_model, text) or 0.0
        is_positive = int(raw) == 1

        if is_positive and conf >= 0.93:
            r += 3.0        # very confident positive → boost +3
        elif is_positive:
            r += 2.0        # positive but lower confidence → boost +2
        else:
            r += 1.0        # negative → small boost +1
    except Exception:
        pass                # if sentiment fails, use raw rating as-is

    # clip to valid star range after boost
    r = float(np.clip(r, 1.0, 5.0))

    return {
        "domain"    : "fashion",
        "label"     : f"{round(r, 1)} / 5.0",
        "rating"    : round(r, 1),
        "confidence": None,
        "model"     : k,
        "task"      : "rating",
    }

def compare_fashion(text):
    rows = []
    # Sentiment models
    for n, m in store.fashion_sentiment.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"sentiment/{n}", "label": _sentiment_label(raw), "confidence": _conf(m, text)})
    # Rating models — both xgboost and svr
    for n, m in store.fashion_rating.items():
        r = _rating_predict(m, text)
        rows.append({"model": f"rating/{n}", "label": f"{round(r,1)} / 5.0", "confidence": None})
    return rows

# ══════════════════════════════════════════════════════
# APP REVIEWS
# ══════════════════════════════════════════════════════

def predict_app_feedback(text, model=None):
    m, k  = _pick(store.app_feedback, model, "app_feedback")
    raw   = m.predict([text])[0]
    label = _decode_with_encoder(raw, store.app_feedback_encoder)
    if label is None:
        label = str(raw).capitalize()
    return {"domain": "app", "label": label, "confidence": _conf(m, text), "model": k, "task": "feedback"}

def predict_app_recommend(text, model=None):
    m, k = _pick(store.app_recommender, model, "app_recommender")
    raw  = m.predict([text])[0]
    return {"domain": "app", "label": _recommend_label(raw), "confidence": _conf(m, text), "model": k, "task": "recommend"}

def compare_app(text):
    rows = []
    for n, m in store.app_feedback.items():
        raw   = m.predict([text])[0]
        label = _decode_with_encoder(raw, store.app_feedback_encoder) or str(raw).capitalize()
        rows.append({"model": f"feedback/{n}", "label": label, "confidence": _conf(m, text)})
    for n, m in store.app_recommender.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"recommend/{n}", "label": _recommend_label(raw), "confidence": _conf(m, text)})
    return rows

# ══════════════════════════════════════════════════════
# OTT
# ══════════════════════════════════════════════════════

def predict_ott_sentiment(text, model=None):
    m, k = _pick(store.ott_sentiment, model, "ott_sentiment")
    raw  = m.predict([text])[0]
    return {"domain": "ott", "label": _sentiment_label(raw), "confidence": _conf(m, text), "model": k, "task": "sentiment"}

def predict_ott_viral(text, model=None):
    m, k = _pick(store.ott_viral, model, "ott_viral")
    raw  = m.predict([text])[0]
    return {"domain": "ott", "label": _viral_label(raw), "confidence": _conf(m, text), "model": k, "task": "viral"}

def predict_ott_recommend(text, model=None):
    m, k = _pick(store.ott_recommender, model, "ott_recommender")
    raw  = m.predict([text])[0]
    return {"domain": "ott", "label": _recommend_label(raw), "confidence": _conf(m, text), "model": k, "task": "recommend"}

def compare_ott(text):
    rows = []
    for n, m in store.ott_sentiment.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"sentiment/{n}", "label": _sentiment_label(raw), "confidence": _conf(m, text)})
    for n, m in store.ott_viral.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"viral/{n}", "label": _viral_label(raw), "confidence": _conf(m, text)})
    for n, m in store.ott_recommender.items():
        raw = m.predict([text])[0]
        rows.append({"model": f"recommend/{n}", "label": _recommend_label(raw), "confidence": _conf(m, text)})
    return rows

# ══════════════════════════════════════════════════════
# BATCH
# ══════════════════════════════════════════════════════

def predict_batch(items):
    router = {
        "news":    predict_news,
        "hotel":   predict_hotel_sentiment,
        "fashion": predict_fashion_sentiment,
        "app":     predict_app_feedback,
        "ott":     predict_ott_sentiment,
    }
    results = []
    for item in items:
        domain = item.get("domain", "").lower().strip()
        text   = item.get("text", "")
        if domain not in router:
            results.append({"domain": domain, "text": text, "error": f"Unknown domain: {domain}"})
        else:
            try:
                results.append(router[domain](text))
            except Exception as e:
                results.append({"domain": domain, "text": text, "error": str(e)})
    return results