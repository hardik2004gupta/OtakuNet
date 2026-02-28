import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Artifacts (once)
# -----------------------------

with open("model/anime_df.pkl", "rb") as f:
    anime_df = pickle.load(f)

with open("model/feature_matrix.pkl", "rb") as f:
    feature_matrix = pickle.load(f)

print("Model artifacts loaded.")


# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_anime(title: str, top_n: int = 10):
    """
    Returns top_n similar anime based on cosine similarity.
    """

    # Check if title exists
    if title not in anime_df["title"].values:
        return {"error": "Anime not found"}

    # Get index
    idx = anime_df[anime_df["title"] == title].index[0]

    # Compute cosine similarity
    similarity_scores = cosine_similarity(
        feature_matrix[idx],
        feature_matrix
    )[0]

    # Sort indices (highest similarity first)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Skip first one (itself)
    top_indices = sorted_indices[1:top_n+1]

    recommendations = []

    for i in top_indices:
        recommendations.append({
            "title": anime_df.iloc[i]["title"],
            "image_url": anime_df.iloc[i]["image_url"],
            "score": float(anime_df.iloc[i]["score"]),
            "genres": ", ".join(
                eval(anime_df.iloc[i]["genres"])
                if isinstance(anime_df.iloc[i]["genres"], str)
                else []
                )
        })

    return {"recommendations": recommendations}