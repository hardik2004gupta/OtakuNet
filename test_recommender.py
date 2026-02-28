from app.recommender import recommend_anime

result = recommend_anime("Naruto", top_n=5)
print(result)