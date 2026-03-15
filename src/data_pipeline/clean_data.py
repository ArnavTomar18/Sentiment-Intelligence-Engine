import pandas as pd
from src.data_pipeline.preprocess import preprocess_series

def clean_fashion(df):
    df["clean_text"] = preprocess_series(df["review_text"])
    return df[["review_id", "item_name", "clean_text", "rating"]]

def clean_hotel(df):
    df["clean_text"] = preprocess_series(df["review_text"])
    return df[["review_id", "clean_text", "rating", "city", "country"]]

def clean_app(df):
    df["clean_text"] = preprocess_series(df["review_text"])
    return df[["review_id", "app_name", "clean_text", "rating", "thumbs_up_count"]]

def clean_news(df):
    df["combined"] = df["title"] + " " + df["text"]
    df["clean_text"] = preprocess_series(df["combined"])
    return df[["id", "clean_text", "subject", "label"]]

def clean_ott(df):
    df["clean_text"] = preprocess_series(df["description"])
    return df[["content_id", "title", "clean_text", "genre", "platform"]]