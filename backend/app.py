"""
Movie Recommendation System - Flask Backend
==========================================
Pipeline intégré :
  1. Page d'accueil → top films par popularité/vote (cold start)
  2. Utilisateur connecté → filtrage collaboratif (historique)
  3. Recherche film → content-based (TF-IDF + cosine similarity)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import ast, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# ─── Chemins des datasets ──────────────────────────────────────────────
TMDB_PATH   = os.getenv("TMDB_PATH", "data/tmdb_5000_movies.csv")
MOVIES_PATH = os.getenv("MOVIES_PATH", "data/movies.dat")
RATINGS_PATH= os.getenv("RATINGS_PATH", "data/ratings.dat")
USERS_PATH  = os.getenv("USERS_PATH",  "data/users.dat")

# ─── Chargement & preprocessing ───────────────────────────────────────

def extraire_noms(cellule):
    """Parse une colonne JSON-like et extrait les 'name'."""
    if isinstance(cellule, str):
        try:
            lst = ast.literal_eval(cellule)
            if isinstance(lst, list):
                return [d.get("name", "") for d in lst if isinstance(d, dict)]
        except Exception:
            pass
    return []

def clean_text(s):
    s = s.lower()
    return re.sub(r'[^\w\s]', '', s)

def load_tmdb():
    df = pd.read_csv(TMDB_PATH)
    df["Genre"]        = df["genres"].apply(extraire_noms)
    df["key_names"]    = df["keywords"].apply(extraire_noms)
    df["company_name"] = df["production_companies"].apply(extraire_noms)
    df["country_prod"] = df["production_countries"].apply(extraire_noms)

    df["tagline"]  = df["tagline"].fillna("")
    df["overview"] = df["overview"].fillna("")

    df["tag_clean"] = (
        df["tagline"]
        + " " + df["key_names"].apply(lambda x: ' '.join(x))
        + " " + df["overview"]
    ).apply(clean_text)

    df["genre_clean"] = df["Genre"].apply(
        lambda x: clean_text(' '.join(x)) if isinstance(x, list) else ""
    )
    df["country_company"] = (
        df["country_prod"].apply(lambda x: ' '.join(x))
        + " " + df["company_name"].apply(lambda x: ' '.join(x))
    ).apply(clean_text)

    # Score popularité (cold start)
    scaler = MinMaxScaler()
    df["norm_vote"] = scaler.fit_transform(df[["vote_average"]].fillna(0))
    df["norm_pop"]  = scaler.fit_transform(df[["popularity"]].fillna(0))
    df["default_score"] = 0.6 * df["norm_pop"] + 0.4 * df["norm_vote"]

    df = df.reset_index(drop=True)
    return df

def build_similarity_matrix(df):
    """Construit la matrice de similarité TF-IDF combinée."""
    from scipy.sparse import hstack

    tfidf_tag  = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
    tfidf_prod = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
    tfidf_genre= TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)

    mat_tag   = tfidf_tag.fit_transform(df["tag_clean"])
    mat_prod  = tfidf_prod.fit_transform(df["country_company"])
    mat_genre = tfidf_genre.fit_transform(df["genre_clean"])

    final_mat = hstack([mat_tag, mat_prod, mat_genre])
    sim_matrix = cosine_similarity(final_mat)
    return sim_matrix

def load_movielens():
    """Charge MovieLens et calcule un score moyen par film."""
    try:
        movies_ml = pd.read_csv(
            MOVIES_PATH, sep="::", engine="python",
            names=["movie_id", "title", "genres"], encoding="latin-1"
        )
        ratings = pd.read_csv(
            RATINGS_PATH, sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        avg_ratings = ratings.groupby("movie_id")["rating"].agg(
            avg_rating="mean", n_ratings="count"
        ).reset_index()
        ml_data = movies_ml.merge(avg_ratings, on="movie_id", how="left")
        return ratings, ml_data
    except FileNotFoundError:
        return None, None

# ─── Initialisation globale ───────────────────────────────────────────
print("⏳ Chargement des données...")
tmdb       = load_tmdb()
sim_matrix = build_similarity_matrix(tmdb)
ml_ratings, ml_movies = load_movielens()
print("✅ Données chargées !")

# ─── Helpers ──────────────────────────────────────────────────────────

def movie_to_dict(row):
    genres = row.get("Genre", [])
    if isinstance(genres, list):
        genres_str = ", ".join(genres)
    else:
        genres_str = str(genres)
    return {
        "id":           int(row.name),
        "title":        row.get("title", ""),
        "vote_average": round(float(row.get("vote_average", 0) or 0), 1),
        "popularity":   round(float(row.get("popularity", 0) or 0), 1),
        "genres":       genres_str,
        "overview":     row.get("overview", ""),
        "score":        round(float(row.get("default_score", 0) or 0), 4),
    }

# ─── Routes ──────────────────────────────────────────────────────────

@app.route("/api/home", methods=["GET"])
def home_recommendations():
    """
    Recommandations page d'accueil.
    - Sans user_id : top films par popularité (cold start)
    - Avec user_id  : films bien notés par des utilisateurs similaires (collaborative)
    """
    user_id = request.args.get("user_id", type=int)
    top_n   = request.args.get("top_n", 20, type=int)

    if user_id is None or ml_ratings is None:
        # ── Cold start ─────────────────────────────────────────────
        top = tmdb.sort_values("default_score", ascending=False).head(top_n)
        results = [movie_to_dict(row) for _, row in top.iterrows()]
        return jsonify({"mode": "popularity", "movies": results})

    # ── Collaborative (user-based) ──────────────────────────────────
    user_ratings = ml_ratings[ml_ratings["user_id"] == user_id]
    if user_ratings.empty:
        top = tmdb.sort_values("default_score", ascending=False).head(top_n)
        results = [movie_to_dict(row) for _, row in top.iterrows()]
        return jsonify({"mode": "popularity_fallback", "movies": results})

    # Utilisateurs qui ont noté les mêmes films
    liked_movies = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()
    similar_users = ml_ratings[
        (ml_ratings["movie_id"].isin(liked_movies)) &
        (ml_ratings["user_id"] != user_id) &
        (ml_ratings["rating"] >= 4)
    ]["user_id"].value_counts().head(20).index.tolist()

    # Films recommandés par ces utilisateurs similaires (non vus par user)
    seen = set(user_ratings["movie_id"].tolist())
    recs = ml_ratings[
        (ml_ratings["user_id"].isin(similar_users)) &
        (~ml_ratings["movie_id"].isin(seen)) &
        (ml_ratings["rating"] >= 4)
    ].groupby("movie_id")["rating"].mean().sort_values(ascending=False).head(top_n)

    rec_ids   = recs.index.tolist()
    rec_titles= ml_movies[ml_movies["movie_id"].isin(rec_ids)]["title"].tolist()

    # Chercher ces titres dans TMDB pour avoir les métadonnées
    matched = tmdb[tmdb["title"].isin(rec_titles)]
    results = [movie_to_dict(row) for _, row in matched.iterrows()]

    # Compléter avec popularité si pas assez de résultats
    if len(results) < top_n:
        extra = tmdb.sort_values("default_score", ascending=False).head(top_n)
        seen_titles = {r["title"] for r in results}
        for _, row in extra.iterrows():
            if row["title"] not in seen_titles:
                results.append(movie_to_dict(row))
            if len(results) >= top_n:
                break

    return jsonify({"mode": "collaborative", "movies": results})


@app.route("/api/search", methods=["GET"])
def search():
    """
    Recherche + recommandations similaires (content-based).
    ?q=Avatar&top_n=10
    """
    query  = request.args.get("q", "").strip()
    top_n  = request.args.get("top_n", 10, type=int)

    if not query:
        return jsonify({"error": "Paramètre 'q' requis"}), 400

    # Recherche exacte puis partielle
    exact = tmdb[tmdb["title"].str.lower() == query.lower()]
    if exact.empty:
        partial = tmdb[tmdb["title"].str.lower().str.contains(query.lower(), na=False)]
    else:
        partial = exact

    if partial.empty:
        return jsonify({"query": query, "found": False, "movies": []})

    # Film de référence = 1er résultat
    ref_idx = partial.index[0]
    ref_row = tmdb.loc[ref_idx]

    sim_scores = list(enumerate(sim_matrix[ref_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != ref_idx][:top_n]

    similar = []
    for idx, score in sim_scores:
        d = movie_to_dict(tmdb.loc[idx])
        d["similarity"] = round(float(score), 4)
        similar.append(d)

    return jsonify({
        "query":     query,
        "found":     True,
        "reference": movie_to_dict(ref_row),
        "similar":   similar,
    })


@app.route("/api/movies", methods=["GET"])
def list_movies():
    """Liste paginée de films pour l'autocomplete."""
    q     = request.args.get("q", "").lower()
    limit = request.args.get("limit", 10, type=int)
    if q:
        mask = tmdb["title"].str.lower().str.contains(q, na=False)
        subset = tmdb[mask].head(limit)
    else:
        subset = tmdb.head(limit)
    titles = subset["title"].dropna().tolist()
    return jsonify({"titles": titles})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "total_tmdb_movies": len(tmdb),
        "ml_ratings_loaded": ml_ratings is not None,
        "total_ml_ratings":  len(ml_ratings) if ml_ratings is not None else 0,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
