from src.models.train_news    import train_news
from src.models.train_hotel   import train_hotel
from src.models.train_fashion import train_fashion
from src.models.train_app     import train_app
from src.models.train_ott     import train_ott

if __name__ == "__main__":
    train_news()
    train_hotel()
    train_fashion()
    train_app()
    train_ott()

    print("\n" + "="*50)
    print("  ALL MODELS TRAINED AND SAVED ✓")
    print("="*50)