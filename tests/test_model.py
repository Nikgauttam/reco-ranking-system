import torch
import pytest

from src.models.matrix_factorization import MatrixFactorization


@pytest.fixture
def model() -> MatrixFactorization:
    return MatrixFactorization(num_users=10, num_items=20, embedding_dim=8)


def test_forward_shape(model):
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    scores = model(users, items)
    assert scores.shape == (3,)


def test_score_user_all_items_shape(model):
    scores = model.score_user_all_items(user_idx=0, device=torch.device("cpu"))
    assert scores.shape == (20,)


def test_score_user_all_items_matches_forward(model):
    """Vectorised scoring must match the standard forward pass element-wise."""
    model.eval()
    user_idx = 3
    device = torch.device("cpu")

    vectorised = model.score_user_all_items(user_idx, device).detach()

    users = torch.tensor([user_idx] * 20)
    items = torch.arange(20)
    forward_scores = model(users, items).detach()

    assert torch.allclose(vectorised, forward_scores, atol=1e-5)


def test_no_gradient_in_score_all(model):
    scores = model.score_user_all_items(0, torch.device("cpu"))
    assert not scores.requires_grad
