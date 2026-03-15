import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, mean_squared_error, mean_absolute_error
)

from src.data_pipeline.data_loader import (
    load_news, load_hotel, load_fashion, load_app, load_ott
)

os.makedirs("reports", exist_ok=True)


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_encoder(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clf_scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "f1"        : round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision" : round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

def reg_scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse" : round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae"  : round(mean_absolute_error(y_test, y_pred), 4),
    }

def split(X, y, stratify=None):
    return train_test_split(
        X.fillna(""), y,
        test_size=0.2, random_state=42,
        stratify=stratify
    )


# ══════════════════════════════════════════════════════
# DOMAIN EVALUATIONS
# ══════════════════════════════════════════════════════

def evaluate_news():
    print("\n[NEWS] Loading data...")
    df  = load_news()
    le  = load_encoder("models/news/label_encoder.pkl")
    df["label_enc"] = le.transform(df["label"].fillna("unknown").astype(str))
    _, X_test, _, y_test = split(df["clean_text"], df["label_enc"])

    results = []
    for name in ["logistic", "svc", "xgboost"]:
        path = f"models/news/news_{name}.pkl"
        if not os.path.exists(path):
            continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "News", "task": "Fake Detection", "model": name})
        results.append(s)
        print(f"  {name:12} → acc={s['accuracy']}  f1={s['f1']}")
    return results


def evaluate_hotel():
    print("\n[HOTEL] Loading data...")
    df = load_hotel()
    df["clean_text"] = df["clean_text"].fillna("")
    df["rating"]     = pd.to_numeric(df["rating"], errors="coerce")

    results = []

    # Sentiment
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)
    _, X_test, _, y_test = split(df["clean_text"], df["sentiment"])
    for name in ["logistic", "xgboost", "lightgbm"]:
        path = f"models/hotel/hotel_sentiment_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "Hotel", "task": "Sentiment", "model": name})
        results.append(s)
        print(f"  sentiment/{name:10} → acc={s['accuracy']}  f1={s['f1']}")

    # Churn
    df["churn"] = df["rating"].apply(lambda x: 1 if x <= 2 else 0)
    _, X_test, _, y_test = split(df["clean_text"], df["churn"])
    for name in ["svc", "xgboost", "lightgbm"]:
        path = f"models/hotel/hotel_churn_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "Hotel", "task": "Churn Risk", "model": name})
        results.append(s)
        print(f"  churn/{name:12} → acc={s['accuracy']}  f1={s['f1']}")

    # Hotel rating regression (ridge still exists for hotel)
    df_r = df.dropna(subset=["rating"])
    _, X_test, _, y_test = split(df_r["clean_text"], df_r["rating"])
    path = "models/hotel/hotel_rating_ridge.pkl"
    if os.path.exists(path):
        m = load_model(path)
        s = reg_scores(m, X_test, y_test)
        s.update({"domain": "Hotel", "task": "Rating Prediction",
                  "model": "ridge", "accuracy": None, "f1": None,
                  "precision": None, "recall": None})
        results.append(s)
        print(f"  rating/ridge        → rmse={s['rmse']}  mae={s['mae']}")

    return results


def evaluate_fashion():
    print("\n[FASHION] Loading data...")
    df = load_fashion()
    df["clean_text"] = df["clean_text"].fillna("")
    df["rating"]     = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    results = []
    median  = df["rating"].median()
    df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= median else 0)

    # Sentiment models
    _, X_test, _, y_test = split(df["clean_text"], df["sentiment"])
    for name in ["logistic", "xgboost", "lightgbm"]:
        path = f"models/fashion/fashion_sentiment_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "Fashion", "task": "Sentiment", "model": name})
        results.append(s)
        print(f"  sentiment/{name:10} → acc={s['accuracy']}  f1={s['f1']}")

    # ── Fashion Rating — xgboost + svr (ridge removed) ───────────────
    _, X_test, _, y_test = split(df["clean_text"], df["rating"])
    for name in ["xgboost", "svr"]:
        path = f"models/fashion/fashion_rating_{name}.pkl"
        if not os.path.exists(path):
            print(f"  rating/{name} → MISSING, skipping")
            continue
        m = load_model(path)
        s = reg_scores(m, X_test, y_test)
        s.update({"domain": "Fashion", "task": "Rating Prediction",
                  "model": name, "accuracy": None, "f1": None,
                  "precision": None, "recall": None})
        results.append(s)
        print(f"  rating/{name:12} → rmse={s['rmse']}  mae={s['mae']}")

    return results


def evaluate_app():
    print("\n[APP] Loading data...")
    df = load_app()
    df["clean_text"] = df["clean_text"].fillna("")

    results = []

    def classify_feedback(text):
        text = str(text).lower()
        if any(w in text for w in ["crash","bug","error","fix","broken","freeze"]):
            return "bug"
        elif any(w in text for w in ["feature","add","wish","would","update","improve"]):
            return "feature"
        else:
            return "praise"

    le = load_encoder("models/app_reviews/feedback_label_encoder.pkl")
    df["feedback_type"] = df["clean_text"].apply(classify_feedback)
    df["feedback_enc"]  = le.transform(df["feedback_type"])

    _, X_test, _, y_test = split(df["clean_text"], df["feedback_enc"])
    for name in ["logistic", "xgboost", "lightgbm"]:
        path = f"models/app_reviews/app_feedback_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "App Reviews", "task": "Feedback Classifier", "model": name})
        results.append(s)
        print(f"  feedback/{name:10} → acc={s['accuracy']}  f1={s['f1']}")

    df["recommend"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    _, X_test, _, y_test = split(df["clean_text"], df["recommend"])
    for name in ["svc", "xgboost", "lightgbm"]:
        path = f"models/app_reviews/app_recommender_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "App Reviews", "task": "Recommender", "model": name})
        results.append(s)
        print(f"  recommender/{name:8} → acc={s['accuracy']}  f1={s['f1']}")

    return results


def evaluate_ott():
    print("\n[OTT] Loading data...")
    df = load_ott()
    df["clean_text"] = df["clean_text"].fillna("")

    results = []

    def ott_sentiment(text):
        pos = ["brilliant","stunning","masterpiece","outstanding",
               "amazing","gripping","powerful","emotional"]
        neg = ["boring","disappointing","weak","poor",
               "terrible","awful","bad","slow"]
        t = str(text).lower()
        return 1 if sum(w in t for w in pos) >= sum(w in t for w in neg) else 0

    df["sentiment"] = df["clean_text"].apply(ott_sentiment)
    _, X_test, _, y_test = split(df["clean_text"], df["sentiment"])
    for name in ["logistic", "xgboost", "lightgbm"]:
        path = f"models/ott/ott_sentiment_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "OTT", "task": "Content Sentiment", "model": name})
        results.append(s)
        print(f"  sentiment/{name:10} → acc={s['accuracy']}  f1={s['f1']}")

    top_genres = df["genre"].value_counts().nlargest(10).index
    df_g = df[df["genre"].isin(top_genres)].copy()
    le   = load_encoder("models/ott/genre_label_encoder.pkl")
    known = set(le.classes_)
    df_g  = df_g[df_g["genre"].isin(known)]
    df_g["genre_enc"] = le.transform(df_g["genre"])
    _, X_test, _, y_test = split(
        df_g["clean_text"], df_g["genre_enc"], stratify=df_g["genre_enc"]
    )
    for name in ["logistic", "xgboost"]:
        path = f"models/ott/ott_recommender_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "OTT", "task": "Genre Recommender", "model": name})
        results.append(s)
        print(f"  recommender/{name:8} → acc={s['accuracy']}  f1={s['f1']}")

    df["viral"] = df["clean_text"].apply(lambda x: 1 if len(str(x).split()) > 30 else 0)
    _, X_test, _, y_test = split(df["clean_text"], df["viral"])
    for name in ["svc", "xgboost", "lightgbm"]:
        path = f"models/ott/ott_viral_{name}.pkl"
        if not os.path.exists(path): continue
        m = load_model(path)
        s = clf_scores(m, X_test, y_test)
        s.update({"domain": "OTT", "task": "Viral Probability", "model": name})
        results.append(s)
        print(f"  viral/{name:12} → acc={s['accuracy']}  f1={s['f1']}")

    return results


# ══════════════════════════════════════════════════════
# REPORT BUILDER
# ══════════════════════════════════════════════════════

def build_report(all_results):
    df = pd.DataFrame(all_results)
    df.to_csv("reports/model_comparison_full.csv", index=False)
    print("\n  Saved → reports/model_comparison_full.csv")

    clf_df = df[df["f1"].notna()].copy()
    reg_df = df[df["f1"].isna()].copy()

    best_clf = (
        clf_df.sort_values("f1", ascending=False)
              .groupby(["domain", "task"])
              .first()
              .reset_index()
    )[["domain", "task", "model", "accuracy", "f1", "precision", "recall"]]

    print("\n" + "="*65)
    print("  BEST MODEL PER TASK (by F1 score)")
    print("="*65)
    print(best_clf.to_string(index=False))

    if not reg_df.empty:
        print("\n" + "="*65)
        print("  REGRESSION TASKS (lower RMSE = better)")
        print("="*65)
        # Best per task by lowest RMSE
        best_reg = (
            reg_df.sort_values("rmse", ascending=True)
                  .groupby(["domain", "task"])
                  .first()
                  .reset_index()
        )
        print(best_reg[["domain","task","model","rmse","mae"]].to_string(index=False))

    best_clf.to_csv("reports/best_models.csv", index=False)
    print("\n  Saved → reports/best_models.csv")

    return df, best_clf


# ══════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════

def plot_report(df):
    clf_df  = df[df["f1"].notna()].copy()
    reg_df  = df[df["f1"].isna()].copy()
    domains = clf_df["domain"].unique()
    n       = len(domains)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Comparison — Sentiment Intelligence Engine",
                 fontsize=16, fontweight="bold", y=1.02)
    axes = axes.flatten()

    colors = {
        "logistic": "#4C72B0",
        "svc"     : "#DD8452",
        "xgboost" : "#55A868",
        "lightgbm": "#C44E52",
        "svr"     : "#8172B2",      # ← svr colour added
    }

    for i, domain in enumerate(domains):
        ax    = axes[i]
        data  = clf_df[clf_df["domain"] == domain]
        tasks = data["task"].unique()

        x      = np.arange(len(tasks))
        width  = 0.2
        models = data["model"].unique()

        for j, model in enumerate(models):
            vals = []
            for task in tasks:
                row = data[(data["task"] == task) & (data["model"] == model)]
                vals.append(row["f1"].values[0] if len(row) > 0 else 0)
            ax.bar(x + j * width, vals, width,
                   label=model, color=colors.get(model, "#999"), alpha=0.85)

        ax.set_title(domain, fontsize=13, fontweight="bold")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x + width)
        ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # ── Regression subplot (last panel) ──────────────
    if not reg_df.empty:
        ax = axes[n] if n < len(axes) else axes[-1]
        for domain_task, grp in reg_df.groupby(["domain", "task"]):
            label = f"{domain_task[0]} — {domain_task[1]}"
            ax.bar(
                [f"{r['model']}\n({domain_task[0]})" for _, r in grp.iterrows()],
                grp["rmse"].values,
                color=[colors.get(r["model"], "#999") for _, r in grp.iterrows()],
                alpha=0.85,
            )
        ax.set_title("Regression Tasks (RMSE — lower is better)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.3)

    for i in range(n + (1 if not reg_df.empty else 0), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("reports/model_comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → reports/model_comparison_chart.png")


def plot_best_models_heatmap(best_clf):
    pivot = best_clf.pivot_table(
        index="domain", columns="task", values="f1", aggfunc="max"
    )
    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.5, annot_kws={"size": 11})
    plt.title("Best F1 Score per Domain × Task", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/best_models_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → reports/best_models_heatmap.png")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  SENTIMENT INTELLIGENCE ENGINE — MODEL COMPARISON REPORT")
    print("="*65)

    all_results = []
    all_results += evaluate_news()
    all_results += evaluate_hotel()
    all_results += evaluate_fashion()
    all_results += evaluate_app()
    all_results += evaluate_ott()

    df, best_clf = build_report(all_results)
    plot_report(df)
    plot_best_models_heatmap(best_clf)

    print("\n" + "="*65)
    print("  REPORT COMPLETE")
    print("  reports/model_comparison_full.csv  ← all scores")
    print("  reports/best_models.csv            ← winner per task")
    print("  reports/model_comparison_chart.png ← bar charts")
    print("  reports/best_models_heatmap.png    ← heatmap")
    print("="*65)