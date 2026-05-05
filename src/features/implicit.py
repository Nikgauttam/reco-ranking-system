import pandas as pd


def convert_to_implicit(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Keep only interactions with rating >= threshold and binarise to 1."""
    df = df[df["rating"] >= threshold].copy()
    df["interaction"] = 1
    return df[["user_id", "item_id", "interaction", "timestamp"]]
