import logging

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.matrix_factorization import MatrixFactorization

logger = logging.getLogger(__name__)


def train_bpr(
    model: MatrixFactorization,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
) -> list[float]:
    """Train the model with BPR loss. Returns per-epoch average losses."""
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for users, pos_items, neg_items in dataloader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info("Epoch %d/%d  BPR loss: %.4f", epoch, epochs, avg_loss)

    return losses
