from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.recommender import recommend_anime, anime_df

app = FastAPI()

# Allow frontend requests (safe for MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")


# -----------------------------
# Home Route (HTML)
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# Recommendation API
# -----------------------------

@app.get("/recommend")
def recommend(title: str, top_n: int = 10):
    return recommend_anime(title, top_n)

@app.get("/search")
def search(query: str):
    matches = anime_df[
        anime_df["title"].str.contains(query, case=False, na=False)
    ]["title"].head(10)

    return {"results": matches.tolist()}