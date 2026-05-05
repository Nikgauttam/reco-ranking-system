import numpy as np
import torch
import pytest

from src.models.content_ncf import ContentNeuMF


@pytest.fixture
def model() -> ContentNeuMF:
    m = ContentNeuMF(
        num_users=10, num_items=20, num_categories=5,
        gmf_dim=8, mlp_dims=[16, 8, 4], category_emb_dim=4,
    )
    # register dummy item features
    cat_idx = np.random.randint(0, 5, size=20)
    cont_feat = np.random.rand(20, 2).astype(np.float32)
    m.set_item_features(cat_idx, cont_feat, torch.device("cpu"))
    m.eval()
    return m


def test_forward_shape(model):
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    out = model(users, items)
    assert out.shape == (3,)


def test_score_user_all_items_shape(model):
    scores = model.score_user_all_items(0, torch.device("cpu"))
    assert scores.shape == (20,)


def test_score_matches_forward(model):
    """Vectorised scoring must match forward() element-wise."""
    device = torch.device("cpu")
    vectorised = model.score_user_all_items(0, device).detach()

    users = torch.tensor([0] * 20)
    items = torch.arange(20)
    forward_scores = model(users, items).detach()

    assert torch.allclose(vectorised, forward_scores, atol=1e-5)


def test_no_grad_in_score_all(model):
    scores = model.score_user_all_items(0, torch.device("cpu"))
    assert not scores.requires_grad


def test_different_content_changes_scores():
    """Two models with different item features should produce different scores."""
    def make_model(seed):
        np.random.seed(seed)
        m = ContentNeuMF(num_users=5, num_items=10, num_categories=3,
                         gmf_dim=4, mlp_dims=[8, 4, 2])
        m.set_item_features(
            np.random.randint(0, 3, 10),
            np.random.rand(10, 2).astype(np.float32),
            torch.device("cpu"),
        )
        return m

    m1, m2 = make_model(0), make_model(42)
    # same user embeddings (both freshly initialised with same arch),
    # different content → scores should differ
    with torch.no_grad():
        m1.user_gmf.weight.copy_(m2.user_gmf.weight)
        m1.item_gmf.weight.copy_(m2.item_gmf.weight)
    s1 = m1.score_user_all_items(0, torch.device("cpu"))
    s2 = m2.score_user_all_items(0, torch.device("cpu"))
    assert not torch.allclose(s1, s2)
