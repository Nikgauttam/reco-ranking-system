import pandas as pd
import torch
import pytest

from src.models.neural_cf import NeuralCF
from src.inference.ann_recommender import AnnRecommender


@pytest.fixture
def model() -> NeuralCF:
    m = NeuralCF(num_users=10, num_items=20, gmf_dim=8, mlp_dims=[16, 8, 4])
    m.eval()
    return m


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 2]})


def test_recommend_returns_top_k(model, train_df):
    rec = AnnRecommender(model, train_df, device="cpu", retrieval_k=15)
    recs = rec.recommend(user_id=0, top_k=5)
    assert len(recs) == 5


def test_recommend_excludes_seen_items(model, train_df):
    rec = AnnRecommender(model, train_df, device="cpu", retrieval_k=15)
    recs = rec.recommend(user_id=0, top_k=10)
    seen = {0, 1}
    assert not seen.intersection(recs), "seen items must be filtered"


def test_recommend_items_in_valid_range(model, train_df):
    rec = AnnRecommender(model, train_df, device="cpu", retrieval_k=15)
    recs = rec.recommend(user_id=0, top_k=5)
    assert all(0 <= r < 20 for r in recs)


def test_faiss_index_built(model, train_df):
    rec = AnnRecommender(model, train_df, device="cpu")
    assert rec._index.ntotal == 20
