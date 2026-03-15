from pydantic import BaseModel
from typing import Optional

# ── Request Models ─────────────────────────────────

class NewsRequest(BaseModel):
    text: str

class HotelRequest(BaseModel):
    text: str

class FashionRequest(BaseModel):
    text: str

class AppRequest(BaseModel):
    text: str
    rating: Optional[float] = None

class OTTRequest(BaseModel):
    text: str
    genre: Optional[str] = None

class UniversalRequest(BaseModel):
    domain: str   # news / hotel / fashion / app / ott
    text: str
    rating: Optional[float] = None
    genre: Optional[str] = None

# ── Response Models ────────────────────────────────

class NewsResponse(BaseModel):
    domain: str = "news"
    text: str
    prediction: str          # Fake / Real
    confidence: str          # High / Medium / Low

class HotelResponse(BaseModel):
    domain: str = "hotel"
    text: str
    sentiment: str           # Positive / Negative
    churn_risk: str          # High / Low
    predicted_rating: float  # 1.0 – 5.0

class FashionResponse(BaseModel):
    domain: str = "fashion"
    text: str
    sentiment: str           # Positive / Negative
    predicted_rating: float  # 1.0 – 5.0

class AppResponse(BaseModel):
    domain: str = "app"
    text: str
    feedback_type: str       # bug / feature / praise
    recommend: str           # Recommended / Not Recommended

class OTTResponse(BaseModel):
    domain: str = "ott"
    text: str
    sentiment: str           # Positive / Negative
    genre_prediction: str    # predicted genre
    viral_probability: str   # High / Low