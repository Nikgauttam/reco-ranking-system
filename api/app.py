import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.encoder import encode_ids
from src.data.loader import load_interactions
from src.features.implicit import convert_to_implicit
from src.inference.recommender import Recommender

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = load_interactions()
df = convert_to_implicit(df)          # must mirror the training pipeline
df_encoded, _, _ = encode_ids(df)

recommender = Recommender.from_artifacts(
    artifacts_dir="artifacts",
    config=config,
    train_df=df_encoded,
    device=str(device),
)

app = FastAPI(
    title="Recommendation Ranking API",
    description="BPR-trained Matrix Factorisation — serves personalised top-k item lists.",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="Encoded user index")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    if request.user_id >= recommender.model.user_embedding.num_embeddings:
        raise HTTPException(status_code=400, detail="user_id out of range")

    recs = recommender.recommend(user_id=request.user_id, top_k=request.top_k)
    return RecommendationResponse(user_id=request.user_id, recommendations=recs)
