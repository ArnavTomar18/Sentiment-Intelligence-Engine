import pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_pipeline.data_loader import load_app
from src.evaluation.metrics import evaluate_model

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

def classify_feedback(text):
    text = str(text).lower()
    if any(w in text for w in ["crash","bug","error","fix","broken","freeze"]):
        return "bug"
    elif any(w in text for w in ["feature","add","wish","would","update","improve"]):
        return "feature"
    else:
        return "praise"

def train_app():
    print("\n" + "="*50)
    print("  APP REVIEWS — Feedback Classifier + Recommender")
    print("="*50)

    df = load_app()
    df["clean_text"] = df["clean_text"].fillna("")

    # ── Task 1 — Feedback Classifier ────────────────
    print("\n  Task 1 → Feedback Classifier")
    df["feedback_type"] = df["clean_text"].apply(classify_feedback)
    le = LabelEncoder()
    df["feedback_enc"] = le.fit_transform(df["feedback_type"])
    save(le, "models/app_reviews/feedback_label_encoder.pkl")

    X_train, X_test, y_train, y_test = split(df, "clean_text", "feedback_enc")

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
        evaluate_model(p, X_test, y_test, f"App Feedback — {name}")
        save(p, f"models/app_reviews/app_feedback_{name}.pkl")

    # ── Task 2 — Recommender ─────────────────────────
    print("\n  Task 2 → App Recommender")
    df["recommend"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    X_train, X_test, y_train, y_test = split(df, "clean_text", "recommend")

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
        evaluate_model(p, X_test, y_test, f"App Recommender — {name}")
        save(p, f"models/app_reviews/app_recommender_{name}.pkl")

    print("\n  APP REVIEWS — all 2 tasks done ✓")

if __name__ == "__main__":
    train_app()