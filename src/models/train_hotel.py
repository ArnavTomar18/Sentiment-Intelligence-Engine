import pickle, os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.data_pipeline.data_loader import load_hotel
from src.evaluation.metrics import evaluate_model, evaluate_regression
import pandas as pd
def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("model", model)
    ])

def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")

def split(df, text_col, label_col):
    return train_test_split(
        df[text_col].fillna(""),
        df[label_col],
        test_size=0.2,
        random_state=42
    )

def train_hotel():
    print("\n" + "="*50)
    print("  HOTEL — Sentiment + Churn + Rating")
    print("="*50)

    df = load_hotel()
    df["clean_text"] = df["clean_text"].fillna("")

    # ── Task 1 — Sentiment ──────────────────────────
    print("\n  Task 1 → Sentiment")
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)
    X_train, X_test, y_train, y_test = split(df, "clean_text", "sentiment")

    for name, model in {
        "logistic" : LogisticRegression(max_iter=1000),
        "xgboost"  : XGBClassifier(n_estimators=200, max_depth=6,
                         learning_rate=0.1, eval_metric="logloss",
                         verbosity=0),
        "lightgbm" : LGBMClassifier(n_estimators=200, max_depth=6,
                         learning_rate=0.1, verbosity=-1),
    }.items():
        print(f"    [{name}]")
        p = make_pipeline(model)
        p.fit(X_train, y_train)
        evaluate_model(p, X_test, y_test, f"Hotel Sentiment — {name}")
        save(p, f"models/hotel/hotel_sentiment_{name}.pkl")

    # ── Task 2 — Churn Risk ─────────────────────────
    print("\n  Task 2 → Churn Risk")
    df["churn"] = df["rating"].apply(lambda x: 1 if x <= 2 else 0)
    X_train, X_test, y_train, y_test = split(df, "clean_text", "churn")

    for name, model in {
        "svc"      : LinearSVC(max_iter=1000),
        "xgboost"  : XGBClassifier(n_estimators=200, max_depth=6,
                         learning_rate=0.1, eval_metric="logloss",
                         verbosity=0),
        "lightgbm" : LGBMClassifier(n_estimators=200, max_depth=6,
                         learning_rate=0.1, verbosity=-1),
    }.items():
        print(f"    [{name}]")
        p = make_pipeline(model)
        p.fit(X_train, y_train)
        evaluate_model(p, X_test, y_test, f"Hotel Churn — {name}")
        save(p, f"models/hotel/hotel_churn_{name}.pkl")

    # Task 3 — Rating Predictor (regression)
    print("\n  Task 3 → Rating Predictor")
    df_rating = df.copy()
    df_rating["rating"] = pd.to_numeric(df_rating["rating"], errors="coerce")
    df_rating = df_rating.dropna(subset=["rating", "clean_text"])
    
    X_train, X_test, y_train, y_test = split(df_rating, "clean_text", "rating")
    p = make_pipeline(Ridge())
    p.fit(X_train, y_train)
    evaluate_regression(p, X_test, y_test, "Hotel — Rating Ridge")
    save(p, "models/hotel/hotel_rating_ridge.pkl")
    
    print("\n  HOTEL — all 3 tasks done ✓")
if __name__ == "__main__":
    train_hotel()