"""
Content-Aware Neural Collaborative Filtering.

Extends NeuMF (He et al., 2017) by injecting item content features
(category embedding + continuous features) into the MLP branch.

This is the standard hybrid approach used in production recommenders:
  collaborative signal  (user/item interaction embeddings)  +
  content signal        (item metadata features)
  = better cold-start handling and improved generalisation.

Architecture:
  GMF branch  : user_gmf ⊙ item_gmf                              (unchanged)
  MLP branch  : FC(concat(user_mlp, item_mlp, cat_emb, cont_feat))
  Output      : linear(concat(gmf_out, mlp_out))
"""

import numpy as np
import torch
import torch.nn as nn


class ContentNeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        gmf_dim: int = 32,
        mlp_dims: list[int] | None = None,
        cont_feature_dim: int = 2,   # price_norm + avg_rating_norm
        category_emb_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [64, 32, 16]

        # ── GMF branch (unchanged from NeuMF) ─────────────────────
        self.user_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_gmf = nn.Embedding(num_items, gmf_dim)

        # ── MLP branch ─────────────────────────────────────────────
        mlp_input_dim = mlp_dims[0]
        self.user_mlp = nn.Embedding(num_users, mlp_input_dim // 2)
        self.item_mlp = nn.Embedding(num_items, mlp_input_dim // 2)

        # ── Content features ───────────────────────────────────────
        self.category_emb = nn.Embedding(num_categories + 1, category_emb_dim, padding_idx=0)
        content_dim = category_emb_dim + cont_feature_dim  # 8 + 2 = 10

        # MLP input: user_mlp + item_mlp + content
        in_dim = mlp_input_dim + content_dim
        layers: list[nn.Module] = []
        for out_dim in mlp_dims[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(gmf_dim + mlp_dims[-1], 1)

        # Buffers — set after training via set_item_features()
        self.register_buffer("_cat_idx", torch.zeros(num_items, dtype=torch.long))
        self.register_buffer("_cont_feat", torch.zeros(num_items, cont_feature_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.normal_(self.category_emb.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_item_features(
        self,
        cat_idx: np.ndarray,
        cont_feat: np.ndarray,
        device: torch.device,
    ) -> None:
        """Register item content features as model buffers (called once after loading metadata)."""
        self._cat_idx = torch.tensor(cat_idx, dtype=torch.long, device=device)
        self._cont_feat = torch.tensor(cont_feat, dtype=torch.float32, device=device)

    def _content(self, item_indices: torch.Tensor) -> torch.Tensor:
        cat_emb = self.category_emb(self._cat_idx[item_indices])  # [B, cat_emb_dim]
        cont = self._cont_feat[item_indices]                       # [B, cont_feat_dim]
        return torch.cat([cat_emb, cont], dim=1)                   # [B, content_dim]

    def forward(
        self, user_indices: torch.Tensor, item_indices: torch.Tensor
    ) -> torch.Tensor:
        # GMF
        u_g = self.user_gmf(user_indices)
        v_g = self.item_gmf(item_indices)
        gmf_out = u_g * v_g                                          # [B, gmf_dim]

        # MLP with content
        u_m = self.user_mlp(user_indices)
        v_m = self.item_mlp(item_indices)
        content = self._content(item_indices)
        mlp_out = self.mlp(torch.cat([u_m, v_m, content], dim=1))  # [B, mlp_out]

        return self.output(torch.cat([gmf_out, mlp_out], dim=1)).squeeze(-1)

    @torch.no_grad()
    def score_user_all_items(self, user_idx: int, device: torch.device) -> torch.Tensor:
        """Batched scoring for all items — single forward pass, no Python loop."""
        N = self.item_gmf.num_embeddings
        u_tensor = torch.tensor([user_idx], device=device)

        # GMF
        u_g = self.user_gmf(u_tensor)        # [1, d]
        v_g = self.item_gmf.weight           # [N, d]
        gmf_out = u_g * v_g                  # [N, d]

        # MLP
        u_m = self.user_mlp(u_tensor).expand(N, -1)   # [N, d]
        v_m = self.item_mlp.weight                     # [N, d]

        all_items = torch.arange(N, device=device)
        content = self._content(all_items)             # [N, content_dim]

        mlp_out = self.mlp(torch.cat([u_m, v_m, content], dim=1))  # [N, mlp_out]
        return self.output(torch.cat([gmf_out, mlp_out], dim=1)).squeeze(-1)  # [N]
