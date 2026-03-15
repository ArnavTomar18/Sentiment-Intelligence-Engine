import pandas as pd
from src.data_pipeline.clean_data import (
    clean_fashion, clean_hotel, clean_app, clean_news, clean_ott
)

def load_fashion():
    df = pd.read_csv("data/processed/fashion_reviews.csv")
    return clean_fashion(df)

def load_hotel():
    df = pd.read_csv("data/processed/hotel_reviews.csv")
    return clean_hotel(df)

def load_app():
    df = pd.read_csv("data/processed/app_reviews_dataset.csv")
    return clean_app(df)

def load_news():
    df = pd.read_csv("data/processed/news_dataset.csv")
    return clean_news(df)

def load_ott():
    df = pd.read_csv("data/processed/ott_content_dataset.csv")
    return clean_ott(df)