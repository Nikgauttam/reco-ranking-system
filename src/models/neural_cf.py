"""
Neural Collaborative Filtering — NeuMF (He et al., 2017)
https://arxiv.org/abs/1708.05031

Combines two branches:
  GMF  : element-wise product of dedicated user/item embeddings  (captures linear interactions)
  MLP  : MLP over concatenated embeddings                        (captures non-linear interactions)
Final score = linear(concat(gmf_out, mlp_out))
"""

import torch
import torch.nn as nn


class NeuralCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 32,
        mlp_dims: list[int] | None = None,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [64, 32, 16]

        # GMF branch
        self.user_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_gmf = nn.Embedding(num_items, gmf_dim)

        # MLP branch — input size is 2 × first hidden dim
        mlp_input_dim = mlp_dims[0]
        self.user_mlp = nn.Embedding(num_users, mlp_input_dim // 2)
        self.item_mlp = nn.Embedding(num_items, mlp_input_dim // 2)

        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for out_dim in mlp_dims[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(gmf_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, user_indices: torch.Tensor, item_indices: torch.Tensor
    ) -> torch.Tensor:
        # GMF
        u_g = self.user_gmf(user_indices)
        v_g = self.item_gmf(item_indices)
        gmf_out = u_g * v_g                                # [B, gmf_dim]

        # MLP
        u_m = self.user_mlp(user_indices)
        v_m = self.item_mlp(item_indices)
        mlp_out = self.mlp(torch.cat([u_m, v_m], dim=1))  # [B, mlp_out_dim]

        out = self.output(torch.cat([gmf_out, mlp_out], dim=1))
        return out.squeeze(-1)

    @torch.no_grad()
    def score_user_all_items(self, user_idx: int, device: torch.device) -> torch.Tensor:
        """Score every item for one user in a single batched pass — no Python loop."""
        N = self.item_gmf.num_embeddings
        u_tensor = torch.tensor([user_idx], device=device)

        # GMF
        u_g = self.user_gmf(u_tensor)            # [1, d]
        v_g = self.item_gmf.weight               # [N, d]
        gmf_out = u_g * v_g                      # [N, d]  (broadcast)

        # MLP — expand user embedding across all items
        u_m = self.user_mlp(u_tensor).expand(N, -1)  # [N, d]
        v_m = self.item_mlp.weight                    # [N, d]
        mlp_out = self.mlp(torch.cat([u_m, v_m], dim=1))  # [N, mlp_out]

        return self.output(torch.cat([gmf_out, mlp_out], dim=1)).squeeze(-1)  # [N]
