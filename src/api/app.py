from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from fastapi.staticfiles import StaticFiles
from src.api.ott_routes import ott_router, predict_router


app = FastAPI(
    title       = "Sentiment Intelligence Engine",
    description = """
    Multi-domain NLP sentiment analysis system.

    Supports 5 domains:
    - **News**    → Fake news detection
    - **Hotel**   → Sentiment + churn risk + rating prediction
    - **Fashion** → Aspect sentiment + rating prediction
    - **App**     → Feedback classification + recommendation
    - **OTT**     → Content sentiment + genre + viral probability

    Built with scikit-learn, XGBoost, LightGBM, FastAPI.
    """,
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.include_router(ott_router,     prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.mount("/api/v1/static/eda",     StaticFiles(directory="notebooks/eda"), name="eda")
app.mount("/api/v1/static/reports", StaticFiles(directory="reports"),       name="reports")

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message" : "Sentiment Intelligence Engine API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "domains" : ["news", "hotel", "fashion", "app", "ott"],
        "endpoints": {
            "universal"  : "/api/v1/analyze",
            "news"       : "/api/v1/analyze/news",
            "hotel"      : "/api/v1/analyze/hotel",
            "fashion"    : "/api/v1/analyze/fashion",
            "app"        : "/api/v1/analyze/app",
            "ott"        : "/api/v1/analyze/ott",
            "batch"      : "/api/v1/analyze/batch",
            "health"     : "/api/v1/health",
        }
    }