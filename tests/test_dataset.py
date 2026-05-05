import pandas as pd
import pytest

from src.models.dataset import BPRDataset


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"user_idx": [0, 0, 1, 1, 2], "item_idx": [0, 1, 1, 2, 0]})


def test_bpr_dataset_length(small_df):
    ds = BPRDataset(small_df, num_items=5)
    assert len(ds) == 5


def test_bpr_negative_not_positive(small_df):
    ds = BPRDataset(small_df, num_items=10)
    for i in range(len(ds)):
        user, pos, neg = ds[i]
        assert pos.item() != neg.item()
        assert neg.item() not in ds.user_item_set[user.item()]


def test_bpr_negative_in_range(small_df):
    num_items = 10
    ds = BPRDataset(small_df, num_items=num_items)
    for i in range(len(ds)):
        _, _, neg = ds[i]
        assert 0 <= neg.item() < num_items
