import pandas as pd


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map raw user/item IDs to contiguous integer indices."""
    user_map = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df["item_id"].unique())}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map
