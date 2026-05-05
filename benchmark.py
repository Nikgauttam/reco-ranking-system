"""
Latency benchmark for the recommendation model.

Usage:
    python benchmark.py
"""

import time

import numpy as np
import torch
import yaml

from src.data.encoder import encode_ids
from src.data.loader import load_interactions
from src.features.implicit import convert_to_implicit
from src.inference.recommender import Recommender

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = load_interactions()
df = convert_to_implicit(df)          # must mirror the training pipeline
df_encoded, _, _ = encode_ids(df)

recommender = Recommender.from_artifacts(
    artifacts_dir="artifacts",
    config=config,
    train_df=df_encoded,
    device=str(device),
)

num_users = recommender.model.user_embedding.num_embeddings
N_RUNS = 200

latencies = []
for _ in range(N_RUNS):
    user = int(np.random.randint(0, num_users))
    t0 = time.perf_counter()
    recommender.recommend(user_id=user, top_k=10)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"Runs          : {N_RUNS}")
print(f"Mean latency  : {np.mean(latencies):.2f} ms")
print(f"P50 latency   : {np.percentile(latencies, 50):.2f} ms")
print(f"P95 latency   : {np.percentile(latencies, 95):.2f} ms")
print(f"P99 latency   : {np.percentile(latencies, 99):.2f} ms")
print(f"Max latency   : {np.max(latencies):.2f} ms")
