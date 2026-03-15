import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════

_cache = {
    "df"           : None,
    "shows_matrix" : None,
    "shows_df"     : None,
    "movies_matrix": None,
    "movies_df"    : None,
    "tfidf_shows"  : None,
    "tfidf_movies" : None,
}

DATA_PATH = r"C:\Users\arnav\Downloads\Sentiment-Intelligence-Engine\data\processed\ott_content_dataset.csv"

# ══════════════════════════════════════════════════════
# SENTIMENT LEXICONS  (used by predict fallback)
# ══════════════════════════════════════════════════════

POS_WORDS = {
    "brilliant", "stunning", "masterpiece", "outstanding", "amazing",
    "gripping", "powerful", "emotional", "captivating", "thrilling",
    "excellent", "superb", "fantastic", "incredible", "inspiring",
    "heartwarming", "riveting", "compelling", "breathtaking", "genius",
    "flawless", "mesmerizing", "unforgettable", "moving", "spectacular",
    "entertaining", "enjoyable", "engaging", "addictive", "great", "good",
}

NEG_WORDS = {
    "boring", "disappointing", "weak", "poor", "terrible", "awful",
    "bad", "slow", "dull", "predictable", "forgettable", "mediocre",
    "bland", "tedious", "overrated", "waste", "horrible", "frustrating",
    "confusing", "pointless", "ridiculous", "pathetic", "annoying",
    "dreadful", "unbearable", "unwatchable", "worse", "worst",
}

# ══════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════

def load_ott_data(path: str = DATA_PATH) -> pd.DataFrame:
    if _cache["df"] is not None:
        return _cache["df"]

    df = pd.read_csv(path)

    for col in ["genre", "description", "cast", "title",
                "platform", "content_type", "age_rating", "director"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "content_type" in df.columns and "type" not in df.columns:
        df["type"] = df["content_type"]
    if "age_rating" in df.columns and "rating" not in df.columns:
        df["rating"] = df["age_rating"]
    if "release_year" not in df.columns:
        df["release_year"] = 0

    df["release_year"] = (
        pd.to_numeric(df["release_year"], errors="coerce")
        .fillna(0).astype(int)
    )

    # Weighted combined text — title + genre boosted 2×
    df["combined"] = (
        df["title"]       + " " +
        df["title"]       + " " +
        df["genre"]       + " " +
        df["genre"]       + " " +
        df.get("director", pd.Series([""] * len(df))).fillna("") + " " +
        df.get("cast",     pd.Series([""] * len(df))).fillna("") + " " +
        df["description"]
    )

    _cache["df"] = df
    return df


# ══════════════════════════════════════════════════════
# MATRIX BUILDER  (cached per content type)
# ══════════════════════════════════════════════════════

def _get_matrix(content_type: str):
    key_df     = f"{content_type}_df"
    key_matrix = f"{content_type}_matrix"
    key_tfidf  = f"tfidf_{content_type}"

    if _cache[key_matrix] is not None:
        return _cache[key_df], _cache[key_matrix], _cache[key_tfidf]

    df = load_ott_data()

    if content_type == "shows":
        mask = df["type"].str.lower().str.contains("tv show|show|series", na=False)
    else:
        mask = df["type"].str.lower().str.contains("movie|film", na=False)

    sub_df = df[mask].copy().reset_index(drop=True)

    tfidf = TfidfVectorizer(
        stop_words   = "english",
        max_features = 25_000,
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 2,
    )
    matrix = tfidf.fit_transform(sub_df["combined"])

    _cache[key_df]     = sub_df
    _cache[key_matrix] = matrix
    _cache[key_tfidf]  = tfidf
    return sub_df, matrix, tfidf


# ══════════════════════════════════════════════════════
# SIMILARITY BOOSTERS
# ══════════════════════════════════════════════════════

def _boosted_scores(
    base_sim : np.ndarray,
    sub_df   : pd.DataFrame,
    ref_idx  : int,
    genre_w  : float = 0.15,
    year_w   : float = 0.05,
) -> np.ndarray:
    scores = base_sim.copy()

    # Genre overlap (Jaccard)
    ref_genres = {g.strip() for g in sub_df.loc[ref_idx, "genre"].lower().split(",") if g.strip()}

    def genre_overlap(g_str):
        cand = {g.strip() for g in g_str.lower().split(",") if g.strip()}
        if not ref_genres or not cand:
            return 0.0
        return len(ref_genres & cand) / len(ref_genres | cand)

    genre_bonus = sub_df["genre"].apply(genre_overlap).values
    scores += genre_w * genre_bonus

    # Year proximity
    ref_year = sub_df.loc[ref_idx, "release_year"]
    if ref_year > 0:
        year_diff  = np.abs(sub_df["release_year"].values - ref_year)
        year_bonus = np.where(year_diff == 0, 1.0, 1.0 / (1.0 + year_diff / 5))
        scores    += year_w * year_bonus

    return np.clip(scores, 0, 1)


# ══════════════════════════════════════════════════════
# PLATFORM DIVERSIFIER
# ══════════════════════════════════════════════════════

def _diversify_by_platform(
    df          : pd.DataFrame,
    scores      : np.ndarray,
    top_n       : int,
    per_platform: int = 2,
) -> list:
    """
    Pick top_n results ensuring at most `per_platform` results
    from the same platform. Falls back to best remaining if needed.
    Returns list of (index, score) tuples.
    """
    # Sort all candidates by score descending (excluding self = score -1)
    ranked = sorted(
        [(i, float(scores[i])) for i in range(len(scores)) if scores[i] >= 0],
        key=lambda x: x[1], reverse=True
    )

    platform_counts = {}
    selected        = []

    # Pass 1 — fill with platform diversity
    for idx, score in ranked:
        if len(selected) >= top_n:
            break
        plat = str(df.loc[idx, "platform"]).lower().strip()
        count = platform_counts.get(plat, 0)
        if count < per_platform:
            selected.append((idx, score))
            platform_counts[plat] = count + 1

    # Pass 2 — if still not enough, fill with best remaining
    if len(selected) < top_n:
        selected_idx = {i for i, _ in selected}
        for idx, score in ranked:
            if len(selected) >= top_n:
                break
            if idx not in selected_idx:
                selected.append((idx, score))

    return selected[:top_n]


# ══════════════════════════════════════════════════════
# TITLE RESOLVER
# ══════════════════════════════════════════════════════

def _resolve_title(title: str, df: pd.DataFrame) -> str | None:
    if title in df["title"].values:
        return title
    partial = df[df["title"].str.contains(title, case=False, na=False, regex=False)]
    if not partial.empty:
        best = partial.loc[partial["title"].str.len().idxmin(), "title"]
        print(f"  Matched '{title}' → '{best}'")
        return best
    return None


# ══════════════════════════════════════════════════════
# TV SHOW RECOMMENDER
# ══════════════════════════════════════════════════════

def recommend_show(
    title           : str,
    top_n           : int = 5,
    platform_filter : str = None,
) -> pd.DataFrame | str:
    shows_df, matrix, _ = _get_matrix("shows")

    working_df  = shows_df
    working_mat = matrix
    if platform_filter and platform_filter.lower() not in ("", "all"):
        mask        = shows_df["platform"].str.lower() == platform_filter.lower()
        working_df  = shows_df[mask].reset_index(drop=True)
        working_mat = matrix[mask.values]

    matched = _resolve_title(title, working_df)
    if matched is None:
        return f"'{title}' not found in TV Shows."

    idx      = working_df[working_df["title"] == matched].index[0]
    raw_sim  = cosine_similarity(working_mat[idx], working_mat)[0]
    boosted  = _boosted_scores(raw_sim, working_df, idx)
    boosted[idx] = -1   # exclude self

    # ── Platform-diverse selection ────────────────
    selected   = _diversify_by_platform(working_df, boosted, top_n, per_platform=2)
    top_idx    = [i for i, _ in selected]
    top_scores = [round(s, 3) for _, s in selected]

    result = working_df.loc[
        top_idx, ["title", "platform", "genre", "type", "release_year", "description"]
    ].copy()
    result["similarity"] = top_scores
    return result.sort_values("similarity", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════
# MOVIE RECOMMENDER
# ══════════════════════════════════════════════════════

def recommend_movies(
    title           : str,
    top_n           : int = 5,
    platform_filter : str = None,
) -> pd.DataFrame | str:
    movies_df, matrix, _ = _get_matrix("movies")

    working_df  = movies_df
    working_mat = matrix
    if platform_filter and platform_filter.lower() not in ("", "all"):
        mask        = movies_df["platform"].str.lower() == platform_filter.lower()
        working_df  = movies_df[mask].reset_index(drop=True)
        working_mat = matrix[mask.values]

    matched = _resolve_title(title, working_df)
    if matched is None:
        return f"'{title}' not found in Movies."

    idx      = working_df[working_df["title"] == matched].index[0]
    raw_sim  = cosine_similarity(working_mat[idx], working_mat)[0]
    boosted  = _boosted_scores(raw_sim, movies_df, idx)
    boosted[idx] = -1

    selected   = _diversify_by_platform(working_df, boosted, top_n, per_platform=2)
    top_idx    = [i for i, _ in selected]
    top_scores = [round(s, 3) for _, s in selected]

    result = working_df.loc[
        top_idx, ["title", "platform", "genre", "type", "release_year", "description"]
    ].copy()
    result["similarity"] = top_scores
    return result.sort_values("similarity", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════
# SMART FILTER  (find_your_mind)
# ══════════════════════════════════════════════════════

AGE_RATINGS_KIDS  = ["G", "TV-G", "PG", "TV-Y", "TV-Y7", "U", "ALL"]
AGE_RATINGS_TEENS = ["G", "PG", "PG-13", "TV-PG", "TV-14", "U", "12", "12A"]

def find_your_mind(
    content_type : str,
    year_pref    : str,
    age          : int,
    genres       : list,
    platform     : str = "",
    top_n        : int = 10,
) -> pd.DataFrame | str:
    df = load_ott_data()

    # Content type
    ct = content_type.lower().strip()
    if "movie" in ct:
        type_mask = df["type"].str.lower().str.contains("movie|film", na=False)
    else:
        type_mask = df["type"].str.lower().str.contains("tv show|show|series", na=False)

    # Year
    if year_pref == "recent":
        year_mask = df["release_year"] >= 2018
    elif year_pref == "old":
        year_mask = df["release_year"].between(1, 2017)
    else:
        year_mask = pd.Series([True] * len(df))

    # Age rating
    if age < 13:
        allowed = AGE_RATINGS_KIDS
    elif age < 18:
        allowed = AGE_RATINGS_TEENS
    else:
        allowed = list(df["rating"].unique())

    rating_mask = df["rating"].isin(allowed) | (df["rating"] == "")
    base        = df[type_mask & year_mask & rating_mask].copy()

    # Platform
    if platform and platform.lower() not in ("", "all", "any"):
        plat_filtered = base[base["platform"].str.lower() == platform.lower()]
    else:
        plat_filtered = base

    # Genre filter helper
    def apply_genre(frame, genres):
        if not genres:
            return frame
        genres_clean = [g.strip().lower() for g in genres if g.strip()]
        mask = frame["genre"].str.lower().apply(
            lambda g: any(gen in g for gen in genres_clean)
        )
        return frame[mask]

    # Progressive fallback
    result = apply_genre(plat_filtered, genres)
    if result.empty and genres:
        print("  Relaxing platform filter...")
        result = apply_genre(base, genres)
    if result.empty and genres:
        print("  Relaxing year filter...")
        base_no_year = df[type_mask & rating_mask].copy()
        result = apply_genre(base_no_year, genres)
    if result.empty:
        result = plat_filtered if not plat_filtered.empty else base
    if result.empty:
        return "No recommendations found. Try different filters."

    # ── Platform-diverse output ───────────────────
    result = result.sort_values("release_year", ascending=False)

    # Pick at most 3 per platform
    per_platform  = max(2, top_n // 3)
    platform_counts = {}
    selected_rows   = []

    for _, row in result.iterrows():
        if len(selected_rows) >= top_n:
            break
        plat  = str(row.get("platform", "")).lower().strip()
        count = platform_counts.get(plat, 0)
        if count < per_platform:
            selected_rows.append(row)
            platform_counts[plat] = count + 1

    # Fill remaining if needed
    if len(selected_rows) < top_n:
        selected_titles = {r["title"] for r in selected_rows}
        for _, row in result.iterrows():
            if len(selected_rows) >= top_n:
                break
            if row["title"] not in selected_titles:
                selected_rows.append(row)

    cols = [c for c in ["title", "genre", "release_year", "platform",
                        "rating", "description"] if c in result.columns]
    return pd.DataFrame(selected_rows)[cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════
# FULL-TEXT SEARCH
# ══════════════════════════════════════════════════════

def search_content(
    query         : str,
    content_type  : str   = "all",
    top_n         : int   = 10,
    min_relevance : float = 0.05,
) -> pd.DataFrame | str:
    df = load_ott_data()

    if content_type.lower() != "all":
        if "movie" in content_type.lower():
            df = df[df["type"].str.lower().str.contains("movie|film", na=False)]
        else:
            df = df[df["type"].str.lower().str.contains("show|series", na=False)]

    if df.empty:
        return "No content found."

    corpus = df["combined"].tolist() + [query]
    tfidf  = TfidfVectorizer(
        stop_words   = "english",
        max_features = 25_000,
        ngram_range  = (1, 2),
        sublinear_tf = True,
    )
    matrix  = tfidf.fit_transform(corpus)
    sims    = cosine_similarity(matrix[-1], matrix[:-1])[0]
    top_idx = sims.argsort()[::-1][:top_n]

    result = df.iloc[top_idx][
        ["title", "platform", "genre", "type", "release_year"]
    ].copy()
    result["relevance"] = [round(float(sims[i]), 3) for i in top_idx]
    result = result[result["relevance"] >= min_relevance]

    return result.reset_index(drop=True) if not result.empty else f"No results for '{query}'."


# ══════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════

def get_platform_stats() -> pd.DataFrame:
    df = load_ott_data()
    return df["platform"].value_counts().reset_index().rename(
        columns={"index": "platform", "platform": "count"}
    )

def get_genre_stats(top_n: int = 15) -> pd.Series:
    return load_ott_data()["genre"].value_counts().head(top_n)

def get_yearly_stats() -> pd.DataFrame:
    df = load_ott_data()
    return df[df["release_year"] > 1950].groupby(
        ["release_year", "platform"]
    ).size().unstack(fill_value=0)

def get_content_summary() -> dict:
    df = load_ott_data()
    return {
        "total"     : len(df),
        "movies"    : int(df["type"].str.lower().str.contains("movie|film", na=False).sum()),
        "shows"     : int(df["type"].str.lower().str.contains("show|series", na=False).sum()),
        "platforms" : df["platform"].nunique(),
        "genres"    : df["genre"].nunique(),
        "year_range": f"{df['release_year'][df['release_year']>0].min()}–{df['release_year'].max()}",
    }

def reset_cache():
    for k in _cache:
        _cache[k] = None
    print("Cache cleared.")