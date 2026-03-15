import pickle, os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_pipeline.data_loader import load_fashion
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

def train_fashion():
    print("\n" + "="*50)
    print("  FASHION — Aspect Sentiment + Rating")
    print("="*50)

    df = load_fashion()
    df["clean_text"] = df["clean_text"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "clean_text"])

    # Task 1 — Aspect Sentiment
    print("\n  Task 1 → Aspect Sentiment")
    print("  Rating distribution:")
    print(df["rating"].value_counts().sort_index())

    median_rating = df["rating"].median()
    df["sentiment"] = df["rating"].apply(
        lambda x: 1 if x >= median_rating else 0
    )
    print(f"  Threshold: {median_rating} | Classes: {df['sentiment'].value_counts().to_dict()}")

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
        evaluate_model(p, X_test, y_test, f"Fashion Sentiment — {name}")
        save(p, f"models/fashion/fashion_sentiment_{name}.pkl")

    from sklearn.svm import LinearSVR
    from xgboost import XGBRegressor

    print("\n  Task 2 → Rating Predictor")

    # load dataset
    df = load_fashion()

    df["clean_text"] = df["clean_text"].fillna("")

    # split data
    X_train, X_test, y_train, y_test = split(df, "clean_text", "rating")

    models = {

        "svr": LinearSVR(
            max_iter=4000
        ),

        "xgboost": XGBRegressor(
            n_estimators=700,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.5,
            eval_metric="rmse",
            random_state=42
        )
    }

    for name, model in models.items():

        print(f"\n  Fashion — Rating {name}")

        p = make_pipeline(model)

        p.fit(X_train, y_train)

        evaluate_regression(
            p,
            X_test,
            y_test,
            f"Fashion — Rating {name}"
        )

        save(
            p,
            f"models/fashion/fashion_rating_{name}.pkl"
        )

    print("\n  FASHION — all 2 tasks done ✓")

if __name__ == "__main__":
    train_fashion()