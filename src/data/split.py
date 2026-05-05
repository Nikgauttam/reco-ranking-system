import pandas as pd


def time_based_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global chronological split — last test_ratio% of interactions by timestamp."""
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def leave_one_out_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user chronological split — each user's single latest interaction is held
    out for testing; everything else is training.

    This is the standard evaluation protocol used in the NeuMF paper (He et al.,
    2017) and most Amazon RecSys benchmarks. It ensures every user has exactly
    one test item regardless of how many interactions they have, making recall
    comparable across users.
    """
    df = df.sort_values("timestamp")
    test_idx = df.groupby("user_idx")["timestamp"].transform("max") == df["timestamp"]

    # Break ties — keep only one row per user in test
    test_df = (
        df[test_idx]
        .drop_duplicates(subset="user_idx", keep="last")
    )
    train_df = df.drop(test_df.index)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
