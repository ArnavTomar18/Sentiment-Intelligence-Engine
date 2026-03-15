import pickle, os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data_pipeline.data_loader import load_news
from src.evaluation.metrics import evaluate_model


def make_pipeline(model):

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1,3),
            stop_words="english",
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("model", model)
    ])


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"Saved → {path}")


def train_news():

    print("\n" + "="*50)
    print("NEWS — Fake News Detection")
    print("="*50)

    df = load_news()

    df["clean_text"] = df["clean_text"].fillna("")

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(
        df["label"].fillna("unknown").astype(str)
    )

    save(le, "models/news/label_encoder.pkl")

    X = df["clean_text"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {

        "logistic": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ),

        "svc": LinearSVC(
            max_iter=3000,
            class_weight="balanced"
        ),

        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    }

    for name, model in models.items():

        print(f"\n[{name}]")

        pipeline = make_pipeline(model)

        pipeline.fit(X_train, y_train)

        evaluate_model(
            pipeline,
            X_test,
            y_test,
            f"News — {name}"
        )

        save(
            pipeline,
            f"models/news/news_{name}.pkl"
        )

    print("\nNEWS — all models trained ✓")


if __name__ == "__main__":
    train_news()