"""
Evaluation metrics for recommendation ranking.

Two protocols are provided:

Full-ranking (MovieLens / small catalogues)
  recall_at_k, mrr_at_k, ndcg_at_k
  — score every item, rank them all.  O(N) per user.

Sampled ranking (Amazon / large catalogues)  ← standard in NeuMF paper
  sampled_hr_at_k, sampled_ndcg_at_k
  — rank the positive item against n_neg random negatives.
  — 100-200x faster; numbers are higher but directly comparable across papers.
"""

import numpy as np
import pandas as pd
import torch


def _rank_items(
    model,
    user: int,
    seen_items: set[int],
    num_items: int,
    k: int,
    device: torch.device,
) -> np.ndarray:
    scores = model.score_user_all_items(user, device).cpu().numpy()
    if seen_items:
        scores[list(seen_items)] = -np.inf
    return np.argsort(-scores)[:k]


def _build_lookup(df: pd.DataFrame) -> dict[int, set[int]]:
    return df.groupby("user_idx")["item_idx"].apply(set).to_dict()


# ── Full-ranking metrics ───────────────────────────────────────────────────────

def recall_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    train_lookup = _build_lookup(train_df)
    test_lookup  = _build_lookup(test_df)
    recalls = []
    for user, relevant in test_lookup.items():
        top_k = set(_rank_items(model, user, train_lookup.get(user, set()), num_items, k, device))
        recalls.append(len(top_k & relevant) / len(relevant))
    return float(np.mean(recalls))


def mrr_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    train_lookup = _build_lookup(train_df)
    test_lookup  = _build_lookup(test_df)
    mrrs = []
    for user, relevant in test_lookup.items():
        ranked = _rank_items(model, user, train_lookup.get(user, set()), num_items, k, device)
        rr = next((1.0 / r for r, item in enumerate(ranked, 1) if item in relevant), 0.0)
        mrrs.append(rr)
    return float(np.mean(mrrs))


def ndcg_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    train_lookup = _build_lookup(train_df)
    test_lookup  = _build_lookup(test_df)
    ndcgs = []
    for user, relevant in test_lookup.items():
        ranked = _rank_items(model, user, train_lookup.get(user, set()), num_items, k, device)
        dcg  = sum(1.0 / np.log2(r + 2) for r, item in enumerate(ranked) if item in relevant)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


# ── Sampled metrics (NeuMF protocol — leave-one-out + 99 random negatives) ────

def _sampled_rank(
    model,
    user: int,
    pos_item: int,
    all_items: set[int],
    seen_items: set[int],
    n_neg: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    """Return the rank (1-based) of pos_item among pos + n_neg random negatives."""
    pool = list(all_items - seen_items - {pos_item})
    neg_items = rng.choice(pool, size=min(n_neg, len(pool)), replace=False).tolist()
    candidates = [pos_item] + neg_items

    cand_t = torch.tensor(candidates, device=device)
    u_t    = torch.tensor([user] * len(candidates), device=device)
    with torch.no_grad():
        scores = model(u_t, cand_t).cpu().numpy()

    order = np.argsort(-scores)
    rank  = int(np.where(order == 0)[0][0]) + 1  # pos_item is index 0
    return rank


def sampled_hr_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    n_neg: int = 99,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> float:
    """
    Hit Rate @ k using sampled negatives (standard NeuMF evaluation).
    HR@k = fraction of users whose positive test item ranks in the top k
           when competing against n_neg random negatives.
    """
    model.eval()
    rng          = np.random.default_rng(seed)
    all_items    = set(range(num_items))
    train_lookup = _build_lookup(train_df)
    test_lookup  = _build_lookup(test_df)

    hits = []
    for user, relevant in test_lookup.items():
        seen = train_lookup.get(user, set()) | relevant
        for pos in relevant:
            rank = _sampled_rank(model, user, pos, all_items, seen, n_neg, device, rng)
            hits.append(1.0 if rank <= k else 0.0)
    return float(np.mean(hits))


def sampled_ndcg_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    n_neg: int = 99,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> float:
    """NDCG @ k using sampled negatives (standard NeuMF evaluation)."""
    model.eval()
    rng          = np.random.default_rng(seed)
    all_items    = set(range(num_items))
    train_lookup = _build_lookup(train_df)
    test_lookup  = _build_lookup(test_df)

    ndcgs = []
    for user, relevant in test_lookup.items():
        seen = train_lookup.get(user, set()) | relevant
        for pos in relevant:
            rank = _sampled_rank(model, user, pos, all_items, seen, n_neg, device, rng)
            ndcgs.append(1.0 / np.log2(rank + 1) if rank <= k else 0.0)
    return float(np.mean(ndcgs))
