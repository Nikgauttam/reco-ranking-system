"""
Two-stage recommendation pipeline — the architecture used at YouTube, Google, Spotify.

Stage 1  RETRIEVAL  — FAISS ANN on GMF item embeddings → top-K candidates  O(log N)
Stage 2  RANKING    — Full NeuMF MLP reranks candidates → final top-k       O(K·d)

This scales to millions of items. Stage 1 with brute-force is O(N·d); with IVF index
it drops to O(√N·d) — critical when N = 10M+.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _to_faiss(t: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a C-contiguous float32 numpy array for FAISS."""
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


class AnnRecommender:
    """
    Wraps a NeuralCF (or any model with .user_gmf / .item_gmf embeddings)
    and serves recommendations via a two-stage pipeline.
    """

    def __init__(
        self,
        model,
        train_df: pd.DataFrame,
        device: str | torch.device = "cpu",
        retrieval_k: int = 200,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required for AnnRecommender. "
                "Install it with: pip install faiss-cpu"
            )

        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.retrieval_k = retrieval_k

        self._seen: dict[int, set[int]] = (
            train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        self._index = self._build_faiss_index()

    def _build_faiss_index(self) -> "faiss.IndexFlatIP":
        """Build a FAISS inner-product index from the GMF item embeddings."""
        with torch.no_grad():
            item_embs = _to_faiss(self.model.item_gmf.weight)  # [N, d], c-contiguous f32

        faiss.normalize_L2(item_embs)
        index = faiss.IndexFlatIP(item_embs.shape[1])
        index.add(item_embs)
        return index

    def recommend(self, user_id: int, top_k: int = 10) -> list[int]:
        seen = self._seen.get(user_id, set())

        # ── Stage 1: fast ANN retrieval ──────────────────────────────────────
        with torch.no_grad():
            u_tensor = torch.tensor([user_id], device=self.device)
            u_gmf = _to_faiss(self.model.user_gmf(u_tensor))  # [1, d]

        faiss.normalize_L2(u_gmf)
        n_retrieve = min(self.retrieval_k + len(seen), self.model.item_gmf.num_embeddings)
        _, candidate_ids = self._index.search(u_gmf, n_retrieve)
        candidates = [int(i) for i in candidate_ids[0] if i not in seen]

        # ── Stage 2: NeuMF reranking ─────────────────────────────────────────
        if not candidates:
            return []

        cand_tensor = torch.tensor(candidates, device=self.device)
        u_rep = torch.tensor([user_id] * len(candidates), device=self.device)
        with torch.no_grad():
            scores = self.model(u_rep, cand_tensor).cpu().numpy()

        ranked = np.argsort(-scores)[:top_k]
        return [candidates[i] for i in ranked]
