import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("All libraries loaded successfully")

# ─── One-time downloads (run once) ────────────────────────────────────────────
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

nlp = spacy.load("en_core_web_sm")          # python -m spacy download en_core_web_sm


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www links."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


def strip_html(text: str) -> str:
    """Strip HTML tags like <br>, <p>, <strong>, etc."""
    return re.sub(r"<.*?>", "", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return re.sub(r"[^\w\s]", "", text)


def remove_numbers(text: str) -> str:
    """Remove standalone numbers (keep alphanumeric words)."""
    return re.sub(r"\b\d+\b", "", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces into one and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pass — runs all cleaning steps in order.
    Returns a single cleaned string (not yet tokenized).

    Pipeline:
        Raw text
          → lowercase
          → remove URLs
          → strip HTML
          → remove punctuation
          → remove numbers
          → remove extra spaces
    """
    text = lowercase(text)
    text = remove_urls(text)
    text = strip_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tokenization
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """
    Split cleaned string into a list of word tokens using NLTK.
    Example: "great hotel nice staff" → ["great", "hotel", "nice", "staff"]
    """
    return word_tokenize(text)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Stopword Removal
# ══════════════════════════════════════════════════════════════════════════════

STOP_WORDS = set(stopwords.words("english"))

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stopwords.
    Keeps meaningful content words (nouns, adjectives, verbs).
    Example: ["the", "room", "was", "clean"] → ["room", "clean"]
    """
    return [token for token in tokens if token not in STOP_WORDS]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Lemmatization
# ══════════════════════════════════════════════════════════════════════════════

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(token) for token in tokens]


def lemmatize_with_pos(text: str) -> list[str]:
    doc = nlp(text)
    return [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE (NLTK-based)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(text: str) -> list[str]:

    text   = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens


def preprocess_spacy(text: str) -> list[str]:
    text   = clean_text(text)
    tokens = lemmatize_with_pos(text)
    return tokens


def preprocess_to_string(text: str) -> str:
    return " ".join(preprocess(text))


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING  (for DataFrames)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_series(series, use_spacy: bool = False) -> object:
    """
    Apply the pipeline to a pandas Series (a text column).

    Args:
        series     : pd.Series of raw text strings
        use_spacy  : if True, uses spaCy pipeline (slower, more accurate)

    Returns:
        pd.Series of cleaned joined strings, ready for TF-IDF

    Usage:
        df["clean_text"] = preprocess_series(df["review_text"])
    """
    fn = preprocess_spacy if use_spacy else preprocess
    return series.fillna("").apply(lambda x: " ".join(fn(x)))


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_review(text: str) -> str:
    """For fashion, hotel, app review columns (review_text)."""
    return preprocess_to_string(text)


def preprocess_news(text: str) -> str:
    """
    For news dataset — concatenates title + body before cleaning.
    Handles the news_dataset.csv schema: title + text fields.
    """
    combined = f"{text}"
    return preprocess_to_string(combined)


def preprocess_news_full(title: str, body: str) -> str:
    """For news_dataset.csv where title and text are separate columns."""
    combined = f"{title} {body}"
    return preprocess_to_string(combined)


def preprocess_ott(description: str) -> str:
    """For ott_content_dataset.csv description column."""
    return preprocess_to_string(description)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    samples = {
        "hotel"  : "The rooms were VERY clean!! Staff was rude. Visit http://tripadvisor.com",
        "fashion": "Size runs small <b>quality</b> is amazing, stitching 10/10 would buy again!",
        "app"    : "App crashes on startup. Lots of bugs!!! Please fix ASAP. Version 2.3.1",
        "news"   : "<p>BREAKING: Scientists discover new treatment — visit www.health.org</p>",
        "ott"    : "A gripping thriller set in 2045. The cast delivers outstanding performances.",
    }

    print("=" * 60)
    print("NLP PREPROCESSING PIPELINE — OUTPUT")
    print("=" * 60)

    for domain, text in samples.items():
        tokens = preprocess(text)
        joined = " ".join(tokens)
        print(f"\n[{domain.upper()}]")
        print(f"  Raw    : {text}")
        print(f"  Tokens : {tokens}")
        print(f"  Joined : {joined}")

    print("\n" + "=" * 60)
    print("All steps: lowercase → remove URLs → strip HTML →")
    print("remove punct/numbers → tokenize → stopwords → lemmatize")
    print("=" * 60)