import pandas as pd
import numpy as np

# -----------------------------
# Load Data
# -----------------------------

details = pd.read_csv("data/raw/details.csv")
stats = pd.read_csv("data/raw/stats.csv")
characters = pd.read_csv("data/raw/characters.csv")
char_works = pd.read_csv("data/raw/character_anime_works.csv")
voice_works = pd.read_csv("data/raw/person_voice_works.csv")

print("Datasets loaded successfully.")

# -----------------------------
# Basic Cleaning (details)
# -----------------------------

# Keep only necessary columns
details = details[[
    "mal_id",
    "title",
    "image_url",
    "genres",
    "themes",
    "demographics",
    "synopsis",
    "score",
    "members",
    "favorites"
]]

details.dropna(subset=["title"], inplace=True)
details.reset_index(drop=True, inplace=True)

print("Details cleaned.")

# -----------------------------
# Merge Stats
# -----------------------------

stats = stats[[
    "mal_id",
    "completed",
    "dropped",
    "total"
]]

# Avoid division by zero
stats["completion_ratio"] = stats["completed"] / stats["total"].replace(0, 1)
stats["drop_ratio"] = stats["dropped"] / stats["total"].replace(0, 1)

stats = stats[[
    "mal_id",
    "completion_ratio",
    "drop_ratio"
]]

details = details.merge(stats, on="mal_id", how="left")

print("Stats merged.")

# -----------------------------
# Character Features
# -----------------------------

# Count main/supporting characters per anime
char_counts = char_works.groupby(
    ["anime_mal_id", "role"]
).size().unstack(fill_value=0).reset_index()

char_counts.rename(columns={
    "anime_mal_id": "mal_id",
    "Main": "main_character_count",
    "Supporting": "supporting_character_count"
}, inplace=True)

details = details.merge(char_counts, on="mal_id", how="left")

# Fill missing values
details["main_character_count"] = details["main_character_count"].fillna(0)
details["supporting_character_count"] = details["supporting_character_count"].fillna(0)

print("Character counts merged.")

# -----------------------------
# Character Popularity Feature
# -----------------------------

# Merge characters to anime
char_favs = char_works.merge(
    characters[["character_mal_id", "favorites"]],
    on="character_mal_id",
    how="left"
)

char_favs_grouped = char_favs.groupby("anime_mal_id")["favorites"].sum().reset_index()
char_favs_grouped.rename(columns={
    "anime_mal_id": "mal_id",
    "favorites": "total_character_favorites"
}, inplace=True)

details = details.merge(char_favs_grouped, on="mal_id", how="left")
details["total_character_favorites"] = details["total_character_favorites"].fillna(0)

print("Character popularity feature added.")

# -----------------------------
# Voice Actor Feature
# -----------------------------

voice_counts = voice_works.groupby("anime_mal_id").size().reset_index(name="voice_actor_count")
voice_counts.rename(columns={"anime_mal_id": "mal_id"}, inplace=True)

details = details.merge(voice_counts, on="mal_id", how="left")
details["voice_actor_count"] = details["voice_actor_count"].fillna(0)

print("Voice actor feature added.")

# -----------------------------
# Final Cleaning
# -----------------------------

# -----------------------------
# Final Cleaning (Proper Type Handling)
# -----------------------------

# Fill numeric columns with 0
numeric_cols = details.select_dtypes(include=["float64", "int64"]).columns
details[numeric_cols] = details[numeric_cols].fillna(0)

# Fill text columns with empty string
text_cols = details.select_dtypes(include=["object"]).columns
details[text_cols] = details[text_cols].fillna("")

print("Missing values handled correctly.")

print("Final dataset shape:", details.shape)

# -----------------------------
# Save Processed Data
# -----------------------------

details.to_csv("data/processed/merged_anime.csv", index=False)

print("Merged dataset saved successfully.")