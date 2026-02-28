import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

# -----------------------------
# Load Processed Data
# -----------------------------

df = pd.read_csv("data/processed/merged_anime.csv")

print("Merged dataset loaded:", df.shape)

# -----------------------------
# Prepare Text Features
# -----------------------------

df["synopsis"] = df["synopsis"].fillna("")
df["synopsis"] = df["synopsis"].astype(str)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

synopsis_matrix = tfidf.fit_transform(df["synopsis"])

print("TF-IDF created.")
# -----------------------------
# Prepare Multi-Label Features
# -----------------------------

def split_text_column(col):
    return col.fillna("").apply(lambda x: [i.strip() for i in str(x).split(",") if i.strip() != ""])

df["genres_list"] = split_text_column(df["genres"])
df["themes_list"] = split_text_column(df["themes"])
df["demographics_list"] = split_text_column(df["demographics"])

mlb_genres = MultiLabelBinarizer()
mlb_themes = MultiLabelBinarizer()
mlb_demo = MultiLabelBinarizer()

genres_matrix = mlb_genres.fit_transform(df["genres_list"])
themes_matrix = mlb_themes.fit_transform(df["themes_list"])
demo_matrix = mlb_demo.fit_transform(df["demographics_list"])

print("Multi-label encoding done.")

# -----------------------------
# Numeric Features
# -----------------------------

numeric_cols = [
    "score",
    "members",
    "favorites",
    "completion_ratio",
    "drop_ratio",
    "main_character_count",
    "supporting_character_count",
    "total_character_favorites",
    "voice_actor_count"
]

scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(df[numeric_cols])

print("Numeric features scaled.")

# -----------------------------
# Combine All Features
# -----------------------------

from scipy.sparse import csr_matrix

combined_features = hstack([
    synopsis_matrix,
    csr_matrix(genres_matrix),
    csr_matrix(themes_matrix),
    csr_matrix(demo_matrix),
    csr_matrix(numeric_matrix)
])

print("Final feature matrix shape:", combined_features.shape)

# -----------------------------
# Save Artifacts
# -----------------------------

# Save dataframe (for inference lookup)
pickle.dump(df, open("model/anime_df.pkl", "wb"))

# Save feature matrix
pickle.dump(combined_features, open("model/feature_matrix.pkl", "wb"))

# Save vectorizer (optional future use)
pickle.dump(tfidf, open("model/vectorizer.pkl", "wb"))

print("Artifacts saved successfully.")