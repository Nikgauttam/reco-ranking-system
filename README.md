# Recommendation Ranking System

A production-style personalised recommendation engine with **three models** — BPR Matrix Factorisation, Neural CF (NeuMF), and Content-Aware NeuMF — trained on real-world Amazon Reviews data and served via a FastAPI endpoint with a FAISS two-stage retrieval pipeline.

![CI](https://github.com/Nikgauttam/reco-ranking-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Results — Amazon Reviews 2023 (Video Games, 5-core)

Evaluation protocol: leave-one-out split + sampled ranking (1 positive vs 99 random negatives), the standard from He et al. 2017.

| Model | HR@10 | NDCG@10 | vs BPR-MF |
|---|---|---|---|
| BPR-MF | 0.5047 | 0.3046 | baseline |
| NeuMF | 0.6023 | 0.3782 | +19.3% HR |
| **Content-Aware NeuMF** | **0.6078** | **0.3819** | **+20.4% HR** |

> NeuMF vs Content-Aware NeuMF: content features (category + price + avg rating) add a further +0.9% HR and +1.0% NDCG — a small but consistent gain from injecting item metadata into the MLP branch.

**MovieLens 100K results** (pre-trained, for reference):

| Model | Recall@10 | MRR@10 | NDCG@10 | vs Baseline |
|---|---|---|---|---|
| Popularity baseline | 0.0581 | — | — | — |
| BPR-MF | **0.0954** | **0.4657** | **0.2787** | **+64.1%** |
| NeuMF | 0.0723 | 0.3776 | 0.2272 | +24.4% |

> BPR-MF outperforms NeuMF on sparse 100K data — consistent with the original NeuMF paper. Neural models benefit from denser interaction matrices (1M+).

---

## Architecture

```
Amazon Reviews 2023 (4.6M interactions, 137K items)
        │
        ▼
┌────────────────────────────────────────────────────┐
│  Data Pipeline                                     │
│  implicit filter → 5-core filter → encode → split  │
└────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────┐
        │                             │
        ▼                             ▼
┌────────────────┐     ┌──────────────────────────────────────────┐
│  BPR-MF        │     │  Content-Aware NeuMF  ← flagship model   │
│                │     │                                          │
│  s = u·v+bias  │     │  GMF: user_g ⊙ item_g                   │
│  BPR loss      │     │  MLP: FC(user_m ‖ item_m ‖ cat_emb       │
│                │     │             ‖ price_norm ‖ avg_rating)   │
│                │     │  out: linear(GMF ‖ MLP)                  │
└───────┬────────┘     └──────────────────┬───────────────────────┘
        │                                 │
        └────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Two-Stage Serving  (AnnRecommender + FAISS)                    │
│                                                                 │
│  Stage 1 — RETRIEVAL                                            │
│    FAISS IndexFlatIP on GMF item embeddings                     │
│    → top-200 candidates                       O(log N)          │
│                                                                 │
│  Stage 2 — RANKING                                              │
│    Full ContentNeuMF MLP reranks candidates                     │
│    → final top-k                              O(K·d)            │
│                                                                 │
│  Swap FlatIP → IVF/HNSW for approximate search at 10M+ items   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
   FastAPI  /recommend  +  MLflow experiment tracking
```

### Why Content-Aware NeuMF?

Pure collaborative filtering only uses interaction history — it can't recommend cold-start items (no interactions yet). By injecting item metadata (category embedding + price bucket + average rating) directly into the MLP branch, the model can score items it has never seen in training, using content signals as a fallback. This is how production recommenders at Google, Amazon, and Netflix handle the cold-start problem.

### Key design decisions

| Decision | Alternative | Why |
|---|---|---|
| BPR pairwise loss | MSE rating prediction | We rank, not predict scores |
| Leave-one-out split (Amazon) | Random split | Standard NeuMF eval protocol; avoids future data leakage |
| 5-core filter | 10-core | More training data; 10-core collapses model toward popularity |
| Content features in MLP only | Features everywhere | GMF captures interaction patterns; MLP captures feature interactions |
| Matrix multiply inference | N forward passes | Single `u @ V.T` is ~10× faster |
| Two-stage FAISS pipeline | Brute-force top-k | O(log N) vs O(N·d) — required at 100K+ items |

---

## Project Structure

```
reco-ranking-system/
├── api/
│   └── app.py                      # FastAPI — /health + /recommend
├── configs/
│   ├── config.yaml                 # MovieLens hyperparameters
│   └── amazon_config.yaml          # Amazon hyperparameters
├── scripts/
│   └── download_amazon.sh          # One-command data download
├── src/
│   ├── data/
│   │   ├── amazon_loader.py        # Amazon JSONL loader + k-core filter
│   │   ├── encoder.py              # ID → contiguous integer mapping
│   │   ├── loader.py               # MovieLens 100K / 1M loader
│   │   └── split.py                # Time-based train/test split
│   ├── features/
│   │   ├── content.py              # Category + price feature extraction
│   │   └── implicit.py             # Rating → binary implicit feedback
│   ├── models/
│   │   ├── content_ncf.py          # Content-Aware NeuMF ← flagship
│   │   ├── dataset.py              # BPR triplet sampler
│   │   ├── matrix_factorization.py # BPR-MF
│   │   └── neural_cf.py            # NeuMF (He et al., 2017)
│   ├── training/
│   │   └── trainer.py              # BPR training loop
│   ├── inference/
│   │   ├── ann_recommender.py      # FAISS two-stage pipeline
│   │   └── recommender.py          # Vectorised MF inference
│   └── evaluation/
│       └── metrics.py              # Recall, MRR, NDCG, HR, sampled NDCG @ k
├── tests/                          # 49 pytest unit tests
├── .github/workflows/ci.yml        # GitHub Actions CI
├── main.py                         # Training CLI
├── benchmark.py                    # Latency benchmark
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Nikgauttam/reco-ranking-system.git
cd reco-ranking-system
pip install -r requirements.txt
```

### 2. Download Amazon data

```bash
# Video Games — 4.6M reviews, 137K items (recommended)
bash scripts/download_amazon.sh Video_Games

# Smaller options
bash scripts/download_amazon.sh Beauty     # ~200K reviews
bash scripts/download_amazon.sh Software   # ~67K reviews
```

### 3. Train all three models

```bash
# BPR-MF baseline
python main.py --dataset amazon --model mf

# NeuMF
python main.py --dataset amazon --model ncf

# Content-Aware NeuMF (flagship)
python main.py --dataset amazon --model content_ncf

# View and compare all runs
mlflow ui   # → http://localhost:5000
```

### 4. Serve

```bash
uvicorn api.app:app --reload
# Swagger UI → http://localhost:8000/docs
```

### 5. Call the API

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_k": 10}'
```

---

## MLflow Experiment Tracking

All runs are automatically logged — hyperparameters, per-epoch BPR loss, and final ranking metrics.

```bash
mlflow ui
```

Open `http://localhost:5000` to compare BPR-MF vs NeuMF vs ContentNeuMF side-by-side.

---

## FAISS Two-Stage Pipeline

```python
from src.inference.ann_recommender import AnnRecommender

rec = AnnRecommender(content_ncf_model, train_df, device="cpu", retrieval_k=200)
rec.recommend(user_id=42, top_k=10)
# Stage 1: FAISS retrieves 200 candidates from GMF inner product  O(log N)
# Stage 2: ContentNeuMF MLP reranks with content features         O(K·d)
```

To scale to millions of items swap `IndexFlatIP` (exact) → `IndexIVFFlat` or `IndexHNSWFlat`. The interface is unchanged.

---

## Tests

```bash
pytest tests/ -v
# 49 passed
```

Covers: Amazon loader (price parsing, k-core filter), content feature extraction, ContentNeuMF forward pass + vectorised scoring equivalence, FAISS pipeline + seen-item filtering, BPR-MF, NeuMF, leave-one-out split correctness, sampled HR/NDCG with known ground truth, API endpoints.

---

## Docker

```bash
docker compose up
# API → http://localhost:8000  |  Swagger → http://localhost:8000/docs
```

---

## Configuration

### `configs/amazon_config.yaml`

| Parameter | Default | |
|---|---|---|
| `kcore` | 5 | Min interactions per user/item |
| `implicit_threshold` | 3 | Min star rating for positive signal |
| `epochs` | 50 | Training epochs |
| `batch_size` | 2048 | Training batch size |
| `gmf_dim` | 32 | GMF embedding dim |
| `mlp_dims` | [128, 64, 32] | MLP hidden layer sizes |
| `category_emb_dim` | 8 | Category embedding dimension |
| `dropout` | 0.2 | MLP dropout rate |

---

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) — Video Games category: 4.6M reviews across 137K products. Real-world e-commerce data with rich item metadata (category, price, description, ratings).

---

## Potential Extensions

- **Larger catalogue** — plug in Movies (1.7M), Books (>10M), or All_Beauty for cross-domain experiments
- **Text embeddings** — encode item descriptions with a sentence transformer as additional content features
- **IVF/HNSW index** — approximate nearest neighbours to scale retrieval beyond 1M items
- **Session-based** — replace static embeddings with a GRU/Transformer for sequential recommendation
- **A/B testing framework** — shadow traffic between models, measure online metrics (CTR, dwell time)

---

## License

MIT
