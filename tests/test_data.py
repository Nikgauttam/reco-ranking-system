import pandas as pd
import pytest

from src.data.encoder import encode_ids
from src.data.split import leave_one_out_split, time_based_split
from src.features.implicit import convert_to_implicit


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 30, 20],
            "rating": [5, 3, 4, 5, 2],
            "timestamp": [100, 200, 150, 250, 300],
        }
    )


def test_convert_to_implicit_threshold(raw_df):
    result = convert_to_implicit(raw_df, threshold=4)
    assert set(result.columns) == {"user_id", "item_id", "interaction", "timestamp"}
    assert (result["interaction"] == 1).all()
    assert len(result) == 3  # ratings 5, 4, 5


def test_encode_ids_contiguous(raw_df):
    df_enc, user_map, item_map = encode_ids(raw_df)
    assert set(df_enc["user_idx"]) == set(range(len(user_map)))
    assert set(df_enc["item_idx"]) == set(range(len(item_map)))
    assert len(user_map) == 3
    assert len(item_map) == 3


def test_time_based_split_chronological(raw_df):
    df_enc, _, _ = encode_ids(raw_df)
    train, test = time_based_split(df_enc, test_ratio=0.4)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(df_enc)


def test_time_based_split_ratio(raw_df):
    df_enc, _, _ = encode_ids(raw_df)
    train, test = time_based_split(df_enc, test_ratio=0.4)
    assert len(test) == 2
    assert len(train) == 3


def test_leave_one_out_each_user_has_one_test_row(raw_df):
    df_enc, _, _ = encode_ids(raw_df)
    train, test = leave_one_out_split(df_enc)
    # Every user appears exactly once in test
    assert test["user_idx"].nunique() == df_enc["user_idx"].nunique()
    assert len(test) == df_enc["user_idx"].nunique()


def test_leave_one_out_latest_interaction_in_test(raw_df):
    df_enc, _, _ = encode_ids(raw_df)
    train, test = leave_one_out_split(df_enc)
    # Each test row must be the maximum timestamp for that user
    for _, row in test.iterrows():
        user_max_ts = df_enc[df_enc["user_idx"] == row["user_idx"]]["timestamp"].max()
        assert row["timestamp"] == user_max_ts


def test_leave_one_out_no_overlap(raw_df):
    df_enc, _, _ = encode_ids(raw_df)
    train, test = leave_one_out_split(df_enc)
    assert len(train) + len(test) == len(df_enc)
    # No (user, item, timestamp) triple should appear in both splits
    train_keys = set(zip(train["user_idx"], train["item_idx"], train["timestamp"]))
    test_keys  = set(zip(test["user_idx"],  test["item_idx"],  test["timestamp"]))
    assert len(train_keys & test_keys) == 0
