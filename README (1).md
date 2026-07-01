# 🎬 Movie Recommender System

A hybrid movie recommendation web application combining **Content-Based Filtering**, **Collaborative Filtering**, and **Popularity-Based** recommendations — powered by a Flask REST API and a vanilla HTML/JS frontend.

---

## 📁 Project Structure

```
movie_recommender/
├── backend/
│   ├── app.py               # Flask API — all recommendation logic
│   └── requirements.txt     # Python dependencies
│
├── data/
│   ├── movies.dat           # MovieLens movie metadata
│   ├── ratings.dat          # MovieLens user ratings
│   ├── users.dat            # MovieLens user profiles
│   └── tmdb_5000_movies.csv # TMDB movie dataset (genres, keywords, overview…)
│
└── frontend/
    └── index.html           # Single-page web interface
```

---

## 🧠 Recommendation Pipeline

The system implements **three complementary strategies** depending on the context:

### 1. 🔥 Popularity-Based (Cold Start)
> Used when no user identity is provided.

- Scores each film with a weighted formula:
  ```
  score = 0.6 × normalized_popularity + 0.4 × normalized_vote_average
  ```
- Returns the globally top-ranked movies — ideal for new/anonymous users.

---

### 2. 🤝 Collaborative Filtering (User History)
> Used when a `user_id` is passed to `/api/home`.

- Identifies movies the user rated ≥ 4 stars.
- Finds the **20 most similar users** (those who liked the same films).
- Recommends movies those similar users rated highly that the current user hasn't seen yet.
- Falls back to popularity if the user has no rating history.

---

### 3. 🔍 Content-Based Filtering (Search)
> Used when a user searches for a specific movie via `/api/search`.

- Builds a **TF-IDF matrix** combining three text fields per film:
  - `tagline + keywords + overview` → semantic content
  - `production_countries + companies` → production context
  - `genres` → genre fingerprint
- Computes **cosine similarity** between all films.
- Given a reference film, returns the top-N most similar movies.

---

## 🗂️ Datasets

| File | Source | Description |
|---|---|---|
| `tmdb_5000_movies.csv` | [TMDB / Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | 5 000 movies with genres, keywords, overview, popularity, vote |
| `movies.dat` | [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | Movie IDs and titles |
| `ratings.dat` | MovieLens 1M | User–movie ratings (1–5 stars) |
| `users.dat` | MovieLens 1M | User demographic profiles |

> **Note:** The `data/` folder is not included in this repository due to file size. Download each dataset from the links above and place the files in `data/` before running.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/movie_recommender.git
cd movie_recommender
```

### 2. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Add the datasets
Download and place the following files in `data/`:
- `tmdb_5000_movies.csv`
- `movies.dat`
- `ratings.dat`
- `users.dat`

### 4. Run the Flask API
```bash
python app.py
```
The API starts at `http://localhost:5000`.

### 5. Open the frontend
Open `frontend/index.html` directly in your browser — no build step required.

---

## 🔌 API Endpoints

### `GET /api/home`
Returns movie recommendations for the home page.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_id` | int | — | If provided, uses collaborative filtering |
| `top_n` | int | 20 | Number of movies to return |

**Response:**
```json
{
  "mode": "popularity | collaborative | popularity_fallback",
  "movies": [
    {
      "id": 0,
      "title": "The Dark Knight",
      "vote_average": 8.5,
      "popularity": 123.4,
      "genres": "Action, Crime, Drama",
      "overview": "...",
      "score": 0.9123
    }
  ]
}
```

---

### `GET /api/search`
Searches for a movie and returns content-based similar films.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | **required** | Movie title (exact or partial) |
| `top_n` | int | 10 | Number of similar movies to return |

**Response:**
```json
{
  "query": "Avatar",
  "found": true,
  "reference": { "title": "Avatar", "genres": "Action, Adventure, ...", ... },
  "similar": [
    { "title": "Interstellar", "similarity": 0.8741, ... }
  ]
}
```

---

### `GET /api/movies`
Autocomplete endpoint — returns matching movie titles.

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Partial title to search |
| `limit` | int | Max results (default: 10) |

---

### `GET /api/stats`
Returns dataset statistics.

```json
{
  "total_tmdb_movies": 4803,
  "ml_ratings_loaded": true,
  "total_ml_ratings": 1000209
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, Flask-CORS |
| **ML / NLP** | scikit-learn (TF-IDF, cosine similarity, MinMaxScaler), pandas, numpy, scipy |
| **Data** | TMDB 5000, MovieLens 1M |
| **Frontend** | HTML, CSS, JavaScript (vanilla) |

---

## 📦 requirements.txt

```
flask
flask-cors
pandas
numpy
scikit-learn
scipy
```

---

## 🚀 Environment Variables

You can override default dataset paths using environment variables:

| Variable | Default |
|---|---|
| `TMDB_PATH` | `data/tmdb_5000_movies.csv` |
| `MOVIES_PATH` | `data/movies.dat` |
| `RATINGS_PATH` | `data/ratings.dat` |
| `USERS_PATH` | `data/users.dat` |

Example:
```bash
export TMDB_PATH=/custom/path/tmdb.csv
python app.py
```

---

## 📌 Notes

- The similarity matrix is computed **once at startup** and held in memory — startup takes a few seconds but all search queries are then instant.
- If MovieLens files are missing, the system gracefully falls back to **popularity-only** mode without crashing.
- The collaborative filter is **user-based** (not item-based): it finds users with similar taste, not similar items.

---

## 👥 Authors

> Project developed as part of a data science / recommender systems course.

---

## 📄 License

This project is for educational purposes. Datasets are subject to their respective licenses ([TMDB Terms](https://www.themoviedb.org/documentation/api/terms-of-use), [MovieLens Terms](https://grouplens.org/datasets/movielens/)).
