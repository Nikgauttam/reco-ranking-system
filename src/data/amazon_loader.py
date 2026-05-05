"""
Amazon Reviews 2023 data loader.

Dataset: https://amazon-reviews-2023.github.io/ (McAuley Lab, UCSD)

Review fields used:  user_id, parent_asin, rating, timestamp
Metadata fields used: parent_asin, main_category, price, average_rating
"""

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/amazon")


def _iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_amazon_reviews(category: str = "Video_Games") -> pd.DataFrame:
    """Load raw reviews and return a DataFrame with [user_id, item_id, rating, timestamp]."""
    path = DATA_DIR / f"{category}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run:  bash scripts/download_amazon.sh {category}"
        )

    logger.info("Loading Amazon Reviews: %s", path)
    records = []
    for row in _iter_jsonl_gz(path):
        uid = row.get("user_id")
        iid = row.get("parent_asin") or row.get("asin")
        rating = row.get("rating")
        ts = row.get("timestamp")
        if uid and iid and rating is not None and ts is not None:
            records.append((uid, iid, float(rating), int(ts)))

    df = pd.DataFrame(records, columns=["user_id", "item_id", "rating", "timestamp"])
    logger.info("Loaded %d raw reviews", len(df))
    return df


def load_amazon_metadata(category: str = "Video_Games") -> pd.DataFrame:
    """
    Load item metadata and return a DataFrame with:
      [item_id, main_category, price, avg_rating]
    """
    path = DATA_DIR / f"meta_{category}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run:  bash scripts/download_amazon.sh {category}"
        )

    logger.info("Loading Amazon Metadata: %s", path)
    records = []
    for row in _iter_jsonl_gz(path):
        iid = row.get("parent_asin")
        if not iid:
            continue
        price = _parse_price(row.get("price"))
        records.append({
            "item_id": iid,
            "main_category": row.get("main_category", "Unknown"),
            "price": price,
            "avg_rating": float(row.get("average_rating") or 0.0),
        })

    df = pd.DataFrame(records).drop_duplicates("item_id")
    logger.info("Loaded metadata for %d items", len(df))
    return df


def _parse_price(raw) -> float:
    """Extract a float from price strings like '$29.99' or None."""
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def apply_kcore_filter(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Iteratively remove users and items with fewer than k interactions.
    A k=10 core ensures sufficient signal for all users and items.
    Standard preprocessing step in RecSys research.
    """
    logger.info("Applying %d-core filter...", k)
    while True:
        before = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]
        if len(df) == before:
            break
    logger.info("After %d-core: %d interactions remain", k, len(df))
    return df.reset_index(drop=True)
