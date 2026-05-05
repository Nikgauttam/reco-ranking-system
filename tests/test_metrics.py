import pandas as pd
import torch
import pytest

from src.evaluation.metrics import ndcg_at_k, mrr_at_k, recall_at_k, sampled_hr_at_k, sampled_ndcg_at_k
from src.models.matrix_factorization import MatrixFactorization


@pytest.fixture
def perfect_model() -> MatrixFactorization:
    """A model whose embeddings are hand-crafted so item 0 scores highest for user 0."""
    model = MatrixFactorization(num_users=3, num_items=5, embedding_dim=4)
    with torch.no_grad():
        model.user_embedding.weight.zero_()
        model.item_embedding.weight.zero_()
        model.user_bias.weight.zero_()
        model.item_bias.weight.zero_()
        # Make item 0 rank first for user 0
        model.item_bias.weight[0] = 10.0
    return model


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame({"user_idx": [0], "item_idx": [4]})  # seen: item 4


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame({"user_idx": [0], "item_idx": [0]})  # relevant: item 0


def test_recall_perfect(perfect_model, train_df, test_df):
    r = recall_at_k(perfect_model, train_df, test_df, num_items=5, k=3)
    assert r == pytest.approx(1.0)


def test_mrr_perfect(perfect_model, train_df, test_df):
    mrr = mrr_at_k(perfect_model, train_df, test_df, num_items=5, k=3)
    assert mrr == pytest.approx(1.0)


def test_ndcg_perfect(perfect_model, train_df, test_df):
    ndcg = ndcg_at_k(perfect_model, train_df, test_df, num_items=5, k=3)
    assert ndcg == pytest.approx(1.0)


def test_recall_zero_when_item_not_in_top_k(perfect_model, train_df):
    # relevant item (item 3) will not rank in top-1
    test = pd.DataFrame({"user_idx": [0], "item_idx": [3]})
    r = recall_at_k(perfect_model, train_df, test, num_items=5, k=1)
    assert r == pytest.approx(0.0)


def test_sampled_hr_perfect(perfect_model, train_df, test_df):
    # item 0 has bias=10, so it always wins — HR should be 1.0
    hr = sampled_hr_at_k(perfect_model, train_df, test_df, num_items=5, k=3, n_neg=3)
    assert hr == pytest.approx(1.0)


def test_sampled_ndcg_perfect(perfect_model, train_df, test_df):
    # item 0 ranks 1st → NDCG = 1/log2(2) = 1.0
    ndcg = sampled_ndcg_at_k(perfect_model, train_df, test_df, num_items=5, k=3, n_neg=3)
    assert ndcg == pytest.approx(1.0)


def test_sampled_hr_miss(perfect_model, train_df):
    # item 3 will rank behind item 0 (bias 0 vs 10) — HR@1 = 0
    test = pd.DataFrame({"user_idx": [0], "item_idx": [3]})
    hr = sampled_hr_at_k(perfect_model, train_df, test, num_items=5, k=1, n_neg=3)
    assert hr == pytest.approx(0.0)
