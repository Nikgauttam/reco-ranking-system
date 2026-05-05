"""
Training entry-point.

Usage:
    # MovieLens (default)
    python main.py --model mf
    python main.py --model ncf
    python main.py --model mf --epochs 50

    # Amazon Reviews 2023
    python main.py --dataset amazon --model mf
    python main.py --dataset amazon --model content_ncf
    python main.py --dataset amazon --model content_ncf --category Beauty

    # View all runs
    mlflow ui
"""

import argparse
import json
import logging
import os
import pickle
import time
from collections import Counter

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.encoder import encode_ids
from src.data.loader import load_interactions
from src.data.split import leave_one_out_split, time_based_split
from src.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    sampled_hr_at_k,
    sampled_ndcg_at_k,
)
from src.features.implicit import convert_to_implicit
from src.models.content_ncf import ContentNeuMF
from src.models.dataset import BPRDataset
from src.models.matrix_factorization import MatrixFactorization
from src.models.neural_cf import NeuralCF
from src.training.trainer import train_bpr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a recommendation ranking model")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML. Auto-selected if omitted.")
    parser.add_argument(
        "--dataset", choices=["movielens", "amazon"], default="movielens",
    )
    parser.add_argument(
        "--model", choices=["mf", "ncf", "content_ncf"], default="mf",
        help="mf=BPR-MF  ncf=NeuMF  content_ncf=Content-Aware NeuMF (amazon only)",
    )
    parser.add_argument("--category", default="Video_Games",
                        help="Amazon category (ignored for movielens)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_movielens(config: dict):
    df = load_interactions()
    df = convert_to_implicit(df)
    df, user_map, item_map = encode_ids(df)
    train_df, test_df = time_based_split(df)
    return train_df, test_df, user_map, item_map, None, None, None


def load_amazon(config: dict, category: str):
    from src.data.amazon_loader import (
        apply_kcore_filter,
        load_amazon_metadata,
        load_amazon_reviews,
    )
    from src.features.content import build_content_features

    df = load_amazon_reviews(category)
    # Threshold=3 keeps positive + neutral-positive signals (more training data)
    df = convert_to_implicit(df, threshold=config.get("implicit_threshold", 3))
    df = apply_kcore_filter(df, k=config.get("kcore", 5))
    df, user_map, item_map = encode_ids(df)
    # Leave-one-out: each user's latest interaction held out for test
    train_df, test_df = leave_one_out_split(df)

    metadata_df = load_amazon_metadata(category)
    cont_feat, cat_idx, cat_map = build_content_features(df, metadata_df, item_map)

    return train_df, test_df, user_map, item_map, cont_feat, cat_idx, cat_map


# ── Model construction ─────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    num_users: int,
    num_items: int,
    config: dict,
    cat_map=None,
):
    if model_type == "content_ncf":
        if cat_map is None:
            raise ValueError("content_ncf requires Amazon dataset (--dataset amazon)")
        return ContentNeuMF(
            num_users=num_users,
            num_items=num_items,
            num_categories=len(cat_map),
            gmf_dim=config.get("gmf_dim", 32),
            mlp_dims=config.get("mlp_dims", [128, 64, 32]),
            category_emb_dim=config.get("category_emb_dim", 8),
            dropout=config.get("dropout", 0.2),
        )
    if model_type == "ncf":
        return NeuralCF(
            num_users=num_users,
            num_items=num_items,
            gmf_dim=config.get("gmf_dim", 32),
            mlp_dims=config.get("mlp_dims", [64, 32, 16]),
        )
    return MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def popularity_recall_at_k(train_df, test_df, k: int = 10) -> float:
    popular = {item for item, _ in Counter(train_df["item_idx"]).most_common(k)}
    test_groups = test_df.groupby("user_idx")["item_idx"].apply(set)
    return float(np.mean([len(popular & rel) / len(rel) for rel in test_groups]))


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Auto-select config
    if args.config:
        cfg_path = args.config
    elif args.dataset == "amazon":
        cfg_path = "configs/amazon_config.yaml"
    else:
        cfg_path = "configs/config.yaml"

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Dataset: %s  |  Model: %s", args.dataset.upper(), args.model.upper())

    if args.dataset == "amazon":
        train_df, test_df, user_map, item_map, cont_feat, cat_idx, cat_map = \
            load_amazon(config, args.category)
    else:
        train_df, test_df, user_map, item_map, cont_feat, cat_idx, cat_map = \
            load_movielens(config)

    num_users, num_items = len(user_map), len(item_map)
    logger.info(
        "Train: %d  |  Test: %d  |  Users: %d  |  Items: %d",
        len(train_df), len(test_df), num_users, num_items,
    )

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(args.model, num_users, num_items, config, cat_map).to(device)

    if args.model == "content_ncf" and cont_feat is not None:
        model.set_item_features(cat_idx, cont_feat, device)
        logger.info("Item content features loaded (%d categories)", len(cat_map))

    dataset = BPRDataset(train_df, num_items)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    # ── MLflow ────────────────────────────────────────────────────
    use_mlflow = not args.no_mlflow
    mlflow.set_experiment(f"reco-{args.dataset}")
    run_name = f"{args.model}-{args.category if args.dataset == 'amazon' else 'ml'}"
    ctx = mlflow.start_run(run_name=run_name) if use_mlflow else _noop_ctx()

    with ctx:
        if use_mlflow:
            mlflow.log_params({
                "dataset": args.dataset,
                "model": args.model,
                "category": args.category if args.dataset == "amazon" else "n/a",
                "num_users": num_users,
                "num_items": num_items,
                "epochs": config["epochs"],
                "lr": config["learning_rate"],
                "batch_size": config["batch_size"],
            })

        # ── Train ──────────────────────────────────────────────────
        logger.info("Training for %d epochs...", config["epochs"])
        t0 = time.time()
        losses = train_bpr(model, dataloader, optimizer, device, config["epochs"])
        train_time = time.time() - t0
        logger.info("Done in %.1fs", train_time)

        if use_mlflow:
            for epoch, loss in enumerate(losses, 1):
                mlflow.log_metric("bpr_loss", loss, step=epoch)

        # ── Save ───────────────────────────────────────────────────
        tag = f"{args.dataset}_{args.model}"
        torch.save(model.state_dict(), f"{args.output_dir}/{tag}.pt")
        with open(f"{args.output_dir}/user_map_{tag}.pkl", "wb") as f:
            pickle.dump(user_map, f)
        with open(f"{args.output_dir}/item_map_{tag}.pkl", "wb") as f:
            pickle.dump(item_map, f)
        if cat_map is not None:
            with open(f"{args.output_dir}/cat_map_{tag}.pkl", "wb") as f:
                pickle.dump(cat_map, f)
        logger.info("Saved → artifacts/%s.*", tag)

        # ── Evaluate ───────────────────────────────────────────────
        logger.info("Evaluating...")
        k = config.get("top_k", 10)

        if args.dataset == "amazon":
            # Sampled protocol: rank positive vs 99 random negatives (NeuMF paper standard)
            logger.info("Using sampled evaluation (1 positive + 99 negatives per user)")
            hr   = sampled_hr_at_k  (model, train_df, test_df, num_items, k=k, device=device)
            ndcg = sampled_ndcg_at_k(model, train_df, test_df, num_items, k=k, device=device)

            logger.info("─" * 44)
            logger.info("Dataset    : amazon (%s)", args.category)
            logger.info("Model      : %s", args.model.upper())
            logger.info("HR@%d      : %.4f", k, hr)
            logger.info("NDCG@%d    : %.4f", k, ndcg)
            logger.info("Train time : %.1fs", train_time)
            logger.info("─" * 44)

            if use_mlflow:
                mlflow.log_metrics({
                    f"hr_at_{k}": hr, f"ndcg_at_{k}": ndcg,
                    "train_time_s": train_time,
                })

            results = {
                "dataset": args.dataset, "model": args.model,
                "eval_protocol": "sampled_99neg",
                f"HR@{k}": hr, f"NDCG@{k}": ndcg,
            }
        else:
            # Full-ranking protocol for MovieLens
            recall = recall_at_k(model, train_df, test_df, num_items, k=k, device=device)
            mrr    = mrr_at_k   (model, train_df, test_df, num_items, k=k, device=device)
            ndcg   = ndcg_at_k  (model, train_df, test_df, num_items, k=k, device=device)
            baseline    = popularity_recall_at_k(train_df, test_df, k=k)
            improvement = (recall - baseline) / baseline * 100

            logger.info("─" * 44)
            logger.info("Dataset    : MovieLens")
            logger.info("Model      : %s", args.model.upper())
            logger.info("Recall@%d  : %.4f", k, recall)
            logger.info("MRR@%d     : %.4f", k, mrr)
            logger.info("NDCG@%d    : %.4f", k, ndcg)
            logger.info("Baseline Recall@%d (popularity): %.4f", k, baseline)
            logger.info("Improvement: %.2f%%", improvement)
            logger.info("Train time : %.1fs", train_time)
            logger.info("─" * 44)

            if use_mlflow:
                mlflow.log_metrics({
                    f"recall_at_{k}": recall, f"mrr_at_{k}": mrr, f"ndcg_at_{k}": ndcg,
                    f"baseline_recall_at_{k}": baseline,
                    "improvement_pct": improvement, "train_time_s": train_time,
                })

            results = {
                "dataset": args.dataset, "model": args.model,
                "eval_protocol": "full_ranking",
                f"recall@{k}": recall, f"mrr@{k}": mrr, f"ndcg@{k}": ndcg,
                f"baseline_recall@{k}": baseline, "improvement_percent": improvement,
            }
        with open(f"{args.output_dir}/results_{tag}.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Results → artifacts/results_%s.json", tag)


if __name__ == "__main__":
    main()
