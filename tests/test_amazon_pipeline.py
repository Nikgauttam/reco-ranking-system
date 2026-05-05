"""
Unit tests for the Amazon data pipeline.
These run without the actual dataset files — they test the logic only.
"""

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.amazon_loader import _parse_price, apply_kcore_filter
from src.features.content import build_content_features


# ── amazon_loader tests ───────────────────────────────────────────────────────

def test_parse_price_string():
    assert _parse_price("$29.99") == pytest.approx(29.99)

def test_parse_price_float():
    assert _parse_price(9.99) == pytest.approx(9.99)

def test_parse_price_none():
    assert np.isnan(_parse_price(None))

def test_parse_price_garbage():
    assert np.isnan(_parse_price("N/A"))


def test_kcore_removes_sparse_users():
    df = pd.DataFrame({
        "user_id": ["u1"] * 10 + ["u2"] * 2,
        "item_id": [f"i{i}" for i in range(10)] + ["i0", "i1"],
        "rating": [5] * 12,
        "timestamp": list(range(12)),
    })
    result = apply_kcore_filter(df, k=5)
    assert "u2" not in result["user_id"].values


def test_kcore_stable_on_dense_data():
    df = pd.DataFrame({
        "user_id": [f"u{i % 5}" for i in range(50)],
        "item_id": [f"i{i % 10}" for i in range(50)],
        "rating": [5] * 50,
        "timestamp": list(range(50)),
    })
    before = len(df)
    result = apply_kcore_filter(df, k=2)
    assert len(result) == before


# ── content features tests ────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    interactions = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "item_id": ["A", "B", "A"],
        "rating": [5, 4, 5],
        "timestamp": [1, 2, 3],
        "user_idx": [0, 0, 1],
        "item_idx": [0, 1, 0],
    })
    metadata = pd.DataFrame({
        "item_id": ["A", "B"],
        "main_category": ["Video Games", "Electronics"],
        "price": [49.99, 9.99],
        "avg_rating": [4.5, 3.8],
    })
    item_map = {"A": 0, "B": 1}
    return interactions, metadata, item_map


def test_content_features_shape(sample_data):
    interactions, metadata, item_map = sample_data
    cont, cat_idx, cat_map = build_content_features(interactions, metadata, item_map)
    assert cont.shape == (2, 2)
    assert cat_idx.shape == (2,)


def test_content_features_price_normalised(sample_data):
    interactions, metadata, item_map = sample_data
    cont, _, _ = build_content_features(interactions, metadata, item_map)
    assert (cont[:, 0] >= 0).all() and (cont[:, 0] <= 1).all()


def test_content_features_rating_normalised(sample_data):
    interactions, metadata, item_map = sample_data
    cont, _, _ = build_content_features(interactions, metadata, item_map)
    assert (cont[:, 1] >= 0).all() and (cont[:, 1] <= 1).all()


def test_category_map_complete(sample_data):
    interactions, metadata, item_map = sample_data
    _, cat_idx, cat_map = build_content_features(interactions, metadata, item_map)
    assert "Video Games" in cat_map
    assert "Electronics" in cat_map
    assert len(cat_idx) == 2
