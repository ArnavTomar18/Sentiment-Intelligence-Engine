import pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_pipeline.data_loader import load_ott
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

def ott_sentiment(text):
    text = str(text).lower()
    pos = ["brilliant","stunning","masterpiece","outstanding",
           "amazing","gripping","powerful","emotional"]
    neg = ["boring","disappointing","weak","poor",
           "terrible","awful","bad","slow"]
    p = sum(1 for w in pos if w in text)
    n = sum(1 for w in neg if w in text)
    return 1 if p >= n else 0

def train_ott():
    print("\n" + "="*50)
    print("  OTT — Sentiment + Recommender + Viral")
    print("="*50)

    df = load_ott()
    df["clean_text"] = df["clean_text"].fillna("")

    # ── Task 1 — Content Sentiment ───────────────────
    print("\n  Task 1 → Content Sentiment")
    df["sentiment"] = df["clean_text"].apply(ott_sentiment)
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
        evaluate_model(p, X_test, y_test, f"OTT Sentiment — {name}")
        save(p, f"models/ott/ott_sentiment_{name}.pkl")

# Task 2 — Genre Recommender
    print("\n  Task 2 → Genre Recommender")
    
    # Keep only top 10 genres — ignore rare ones
    top_genres = df["genre"].value_counts().nlargest(10).index
    df_genre = df[df["genre"].isin(top_genres)].copy()
    print(f"  Using top 10 genres: {list(top_genres)}")
    
    le = LabelEncoder()
    df_genre["genre_enc"] = le.fit_transform(df_genre["genre"])
    save(le, "models/ott/genre_label_encoder.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_genre["clean_text"].fillna(""),
        df_genre["genre_enc"],
        test_size=0.2,
        random_state=42,
        stratify=df_genre["genre_enc"]    # ← ensures all classes in both splits
    )
    
    for name, model in {
        "logistic" : LogisticRegression(max_iter=1000),
        "xgboost"  : XGBClassifier(
                         n_estimators=200,
                         max_depth=6,
                         learning_rate=0.1,
                         eval_metric="mlogloss",   # ← multiclass needs mlogloss
                         verbosity=0,
                         num_class=10
                     ),
    }.items():
        print(f"    [{name}]")
        p = make_pipeline(model)
        p.fit(X_train, y_train)
        evaluate_model(p, X_test, y_test, f"OTT Recommender — {name}")
        save(p, f"models/ott/ott_recommender_{name}.pkl")
    # ── Task 3 — Viral Probability ───────────────────
    print("\n  Task 3 → Viral Probability")
    df["viral"] = df["clean_text"].apply(
        lambda x: 1 if len(str(x).split()) > 30 else 0
    )
    X_train, X_test, y_train, y_test = split(df, "clean_text", "viral")

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
        evaluate_model(p, X_test, y_test, f"OTT Viral — {name}")
        save(p, f"models/ott/ott_viral_{name}.pkl")

    print("\n  OTT — all 3 tasks done ✓")

if __name__ == "__main__":
    train_ott()