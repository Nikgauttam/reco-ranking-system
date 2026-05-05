import pandas as pd
from pathlib import Path


def load_interactions(variant: str = "100k") -> pd.DataFrame:
    """
    Load MovieLens interactions.

    variant:
        "100k" — 100K dataset  (data/raw/ml-100k/u.data,  tab-separated)
        "1m"   — 1M  dataset   (data/raw/ml-1m/ratings.dat, "::" separated)
    """
    if variant == "1m":
        path = Path("data/raw/ml-1m/ratings.dat")
        return pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

    path = Path("data/raw/ml-100k/u.data")
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


if __name__ == "__main__":
    df = load_interactions()
    print(df.head())
    print(f"Total interactions: {len(df)}")
