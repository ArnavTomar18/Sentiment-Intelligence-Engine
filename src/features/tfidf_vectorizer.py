from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd

def build_tfidf(texts, max_features=20000, ngram_range=(1, 2)):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )
    X = tfidf.fit_transform(texts)
    return tfidf, X

def save_tfidf(tfidf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"Saved TF-IDF → {path}")

def load_tfidf(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def prepare_data(df, text_col, label_col, test_size=0.2):
    """
    Full prep: TF-IDF + train/test split in one call.
    Returns X_train, X_test, y_train, y_test, tfidf
    """
    tfidf, X = build_tfidf(df[text_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test, tfidf


if __name__ == "__main__":
    from src.data_pipeline.data_loader import load_hotel
    df = load_hotel()
    df["label"] = df["rating"].apply(lambda x: 1 if x >= 3 else 0)

    X_train, X_test, y_train, y_test, tfidf = prepare_data(
        df, text_col="clean_text", label_col="label"
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("TF-IDF vocab size:", len(tfidf.vocabulary_))