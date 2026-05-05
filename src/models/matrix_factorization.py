import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Biased matrix factorisation model trained with BPR loss."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(
        self, user_indices: torch.Tensor, item_indices: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_embedding(user_indices)
        v = self.item_embedding(item_indices)
        dot = (u * v).sum(dim=1)
        b_u = self.user_bias(user_indices).squeeze(-1)
        b_v = self.item_bias(item_indices).squeeze(-1)
        return dot + b_u + b_v

    @torch.no_grad()
    def score_user_all_items(self, user_idx: int, device: torch.device) -> torch.Tensor:
        """Return scores for a single user against every item in one matrix multiply."""
        u_idx = torch.tensor([user_idx], device=device)
        u_emb = self.user_embedding(u_idx)           # [1, d]
        u_bias = self.user_bias(u_idx)               # [1, 1]
        v_embs = self.item_embedding.weight          # [N, d]
        v_biases = self.item_bias.weight.squeeze(1)  # [N]
        scores = (u_emb @ v_embs.T).squeeze(0) + u_bias.squeeze() + v_biases
        return scores
