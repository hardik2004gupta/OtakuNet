# 🎬 OtakuNet --- ML-Powered Anime Recommendation System

OtakuNet is a **production-style, full-stack Machine Learning system**
that delivers intelligent anime recommendations using **NLP,
multi-source feature engineering, and cosine similarity modeling**.

Unlike simple genre filters, OtakuNet models:

-   Semantic similarity (synopsis embeddings)
-   Multi-label categorical signals (genres, themes, demographics)
-   Engagement analytics (completion/drop ratios)
-   Character popularity influence
-   Voice actor participation patterns

This project demonstrates a **complete end-to-end ML pipeline** --- from
data engineering to deployment.

------------------------------------------------------------------------

# 🚀 Key Highlights

-   ✅ Multi-source relational dataset integration (28,955 anime)
-   ✅ TF-IDF NLP modeling (5,000-dimensional semantic space)
-   ✅ MultiLabelBinarizer encoding for categorical features
-   ✅ Structured engagement feature engineering
-   ✅ Sparse matrix optimization (memory-efficient design)
-   ✅ FastAPI production-ready backend
-   ✅ Dynamic frontend with autocomplete search
-   ✅ Cloud-deployable architecture

------------------------------------------------------------------------

# 🧠 Problem Statement

Anime discovery is complex due to:

-   Thousands of titles
-   Overlapping genres & themes
-   Character and voice actor networks
-   Engagement metrics that influence popularity

Traditional filtering fails to capture semantic similarity.

**OtakuNet solves this by modeling anime as high-dimensional feature
vectors and ranking similarity using cosine similarity.**

------------------------------------------------------------------------

# 📊 Dataset Architecture

## Core Dataset

-   `details.csv` --- Anime metadata

## Supporting Datasets

-   `stats.csv` --- Engagement statistics
-   `characters.csv` --- Character-level popularity
-   `character_anime_works.csv` --- Character-anime mapping
-   `person_voice_works.csv` --- Voice actor participation

### Relational Graph Structure

Anime ↔ Characters ↔ Voice Actors\
Anime ↔ Engagement Stats\
Anime ↔ Metadata

Total Processed Entries: **28,955 anime**

------------------------------------------------------------------------

# ⚙️ Feature Engineering Pipeline

## 1️⃣ Textual Features (NLP)

-   Column: `synopsis`
-   Technique: TF-IDF Vectorization
-   Max Features: 5000
-   Stopwords Removed: English
-   Output: Sparse high-dimensional semantic matrix

## 2️⃣ Multi-Label Encoding

Columns: - Genres - Themes - Demographics

Technique: - MultiLabelBinarizer (multi-hot encoding)

## 3️⃣ Numeric Feature Scaling

Features: - Score - Members - Favorites - Completion Ratio - Drop
Ratio - Character Counts - Voice Actor Count

Technique: - StandardScaler (Z-score normalization)

------------------------------------------------------------------------

# 🧮 Final Feature Matrix

Shape: (28955, \~5287)

  Component     Feature Count
  ------------- ---------------
  TF-IDF        5000
  Categorical   \~278
  Numeric       9

Stored as **SciPy sparse matrix** for efficiency.

------------------------------------------------------------------------

# 🧠 Recommendation Algorithm

Model Type: Content-Based Filtering

Similarity Metric: Cosine Similarity

Formula:

similarity(A, B) = (A · B) / (\|\|A\|\| × \|\|B\|\|)

### Inference Steps

1.  Extract selected anime vector
2.  Compute cosine similarity against matrix
3.  Rank descending
4.  Return top-N recommendations

### Why This Approach?

-   No retraining required
-   Fast inference
-   Scalable
-   No user history needed

------------------------------------------------------------------------

# 🏗 Backend Architecture

Framework: FastAPI

## Endpoints

### `/`

Serves the HTML frontend.

### `/recommend?title=...`

Returns: - Title - Score - Genres - Image URL

### `/search?query=...`

Provides autocomplete suggestions.

Artifacts loaded at startup: - anime_df.pkl - feature_matrix.pkl -
vectorizer.pkl

------------------------------------------------------------------------

# 🎨 Frontend

-   Pure HTML + CSS (Dark Theme)
-   Vanilla JavaScript
-   Autocomplete dropdown
-   Loading spinner
-   Responsive card grid
-   MAL CDN poster integration

Lightweight. Fast. No heavy frameworks.

------------------------------------------------------------------------

# 🏛 System Architecture

Offline Training\
→ Feature Engineering\
→ Model Artifacts (.pkl)\
→ FastAPI Inference Layer\
→ User Request\
→ Cosine Similarity\
→ JSON Response\
→ Dynamic Rendering

------------------------------------------------------------------------

# ☁️ Deployment

Supported Platforms: - Render - Fly.io - Google Cloud Run

Start Command:

uvicorn app.main:app --host 0.0.0.0 --port 10000

------------------------------------------------------------------------

# 🔥 Technical Strengths

-   Multi-entity relational data modeling
-   NLP-based semantic similarity
-   Sparse matrix optimization
-   API-first architecture
-   Cloud-ready deployment design
-   Clean modular separation (training / model / app)

------------------------------------------------------------------------

# ⚠️ Current Limitations

-   No collaborative filtering
-   No user personalization
-   Exact-match search
-   No caching layer
-   Full similarity computed per request

------------------------------------------------------------------------

# 🚀 Future Roadmap

-   Hybrid Recommender (Collaborative + Content)
-   Sentence-Transformer embeddings
-   FAISS Approximate Nearest Neighbor search
-   Redis caching
-   User accounts & ratings ingestion
-   Dockerized production deployment
-   Kubernetes auto-scaling

------------------------------------------------------------------------

# 🎯 Resume Summary

Built **OtakuNet**, a production-style ML-powered anime recommendation
system integrating NLP (TF-IDF), multi-label encoding, engagement
analytics, and cosine similarity. Developed and deployed a FastAPI
backend with dynamic frontend rendering and cloud-ready architecture.

------------------------------------------------------------------------

# 👤 Author

Hardik Gupta\
Machine Learning & AI Engineer

------------------------------------------------------------------------

⭐ If you found this project interesting, consider starring the
repository!
