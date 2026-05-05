import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """Generates (user, positive_item, negative_item) triplets for BPR training."""

    def __init__(self, df: pd.DataFrame, num_items: int):
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.num_items = num_items

        self.user_item_set: dict[int, set[int]] = {}
        for u, i in zip(self.users, self.items):
            self.user_item_set.setdefault(int(u), set()).add(int(i))

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = int(self.users[idx])
        pos_item = int(self.items[idx])

        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_item_set[user]:
            neg_item = random.randint(0, self.num_items - 1)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )
