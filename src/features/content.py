"""
Item content feature engineering for Amazon Reviews.

Extracts two features per item:
  price_bucket  — integer [0–4] based on price range
  category_idx  — integer category index (for embedding lookup)

These are later embedded and concatenated into the ContentNeuMF MLP branch,
turning pure collaborative filtering into a hybrid CF + content model.
"""

import numpy as np
import pandas as pd


PRICE_BINS = [0.0, 10.0, 30.0, 100.0, float("inf")]
PRICE_LABELS = ["free_or_unknown", "under_10", "10_to_30", "30_to_100", "over_100"]


def build_content_features(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    item_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Build an item feature matrix aligned with item_map.

    Returns:
        cont_features — float32 array of shape [num_items, 2]
                        columns: [price_bucket_normalised, avg_rating_normalised]
        cat_idx       — int64 array of shape [num_items] mapping items to category indices
        cat_map       — {category_string: integer_index}
    """
    num_items = len(item_map)

    # Align metadata to item_map indices
    meta = metadata_df.set_index("item_id").reindex(
        pd.Series(item_map).sort_values().index
    )

    # ── Price bucket ─────────────────────────────────────────────
    price_bucket = pd.cut(
        meta["price"].fillna(-1),
        bins=[-float("inf")] + PRICE_BINS[1:],
        labels=False,
        right=False,
    ).fillna(0).astype(int)

    # Normalise to [0, 1]
    price_norm = (price_bucket / (len(PRICE_BINS) - 2)).values.astype(np.float32)

    # ── Average rating ────────────────────────────────────────────
    avg_rating = meta["avg_rating"].fillna(3.0).clip(1.0, 5.0)
    rating_norm = ((avg_rating - 1.0) / 4.0).values.astype(np.float32)

    # ── Category index ────────────────────────────────────────────
    categories = meta["main_category"].fillna("Unknown").tolist()
    unique_cats = sorted(set(categories))
    cat_map = {c: i for i, c in enumerate(unique_cats)}
    cat_idx = np.array([cat_map[c] for c in categories], dtype=np.int64)

    # Continuous features: [price_norm, rating_norm]  shape [N, 2]
    cont_features = np.stack([price_norm, rating_norm], axis=1)  # [N, 2]

    return cont_features, cat_idx, cat_map


def get_item_feature_dim() -> int:
    """Number of continuous item features (excluding category embedding)."""
    return 2  # price_bucket_norm + avg_rating_norm
