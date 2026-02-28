# OtakuNet 🎬

OtakuNet is an AI-powered Anime Recommendation System built using
FastAPI and content-based similarity modeling.

It integrates anime metadata, character influence, engagement
statistics, and NLP-based text similarity to deliver intelligent
recommendations.

------------------------------------------------------------------------

## 🚀 Features

-   Content-based recommendation engine
-   TF-IDF vectorization on anime synopsis
-   Multi-feature engineering (genres, themes, demographics, engagement
    metrics)
-   Character popularity and voice actor influence modeling
-   Cosine similarity ranking
-   Autocomplete search
-   Loading spinner & dynamic UI rendering
-   FastAPI backend + HTML frontend

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   FastAPI
-   Scikit-learn
-   Pandas & NumPy
-   HTML + JavaScript
-   Uvicorn

------------------------------------------------------------------------

## 📦 Architecture

Offline Training Pipeline → Feature Engineering → Model Artifacts\
→ FastAPI Inference Layer → Web UI

------------------------------------------------------------------------

## 📁 Project Structure

anime-recommender-mvp/ │ ├── data/ ├── training/ ├── model/ ├── app/ ├──
requirements.txt └── README.md

------------------------------------------------------------------------

## 🧠 How It Works

1.  Multiple datasets are merged and engineered into structured
    features.
2.  Text features are vectorized using TF-IDF.
3.  Numeric and categorical features are normalized and encoded.
4.  A combined feature matrix is created.
5.  Cosine similarity is computed at inference time.
6.  Top similar anime are returned via API.

------------------------------------------------------------------------

## 🌐 Deployment

OtakuNet can be deployed using:

-   Render
-   Fly.io
-   Google Cloud Run

Start command:

uvicorn app.main:app --host 0.0.0.0 --port 10000

------------------------------------------------------------------------

## 📈 Future Improvements

-   Hybrid collaborative filtering
-   User accounts & personalization
-   Graph-based recommendation modeling
-   Caching layer for faster inference
-   UI redesign with modern framework

------------------------------------------------------------------------

## 👤 Author

Hardik Gupta\
Machine Learning & AI Enthusiast

------------------------------------------------------------------------

⭐ If you found this project interesting, consider starring the
repository!
