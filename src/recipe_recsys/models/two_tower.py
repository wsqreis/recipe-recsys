"""Two-tower retrieval model with content-aware item embeddings.

User tower: the user is represented by the recipes they cooked (mean of
history embeddings + MLP). A user with a few interactions gets a vector at
inference time without retraining, unlike matrix factorization where every
user needs a trained embedding.

Item tower: recipe id embedding + mean ingredient embedding + mean tag
embedding + nutrition features, through an MLP. The content part is what
will let phase 3 serve recipes that never appeared in training.

Training: each training interaction (u, i) becomes one example whose input is
the user's previous `max_history` recipes (strictly earlier in time, as in
serving) and whose target is i. Loss is softmax over in-batch negatives with
logQ correction: popular recipes appear as in-batch negatives far more often
than rare ones, and without correction the model is pushed to under-score
them (Yi et al., 2019, "Sampling-bias-corrected neural modeling").
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from recipe_recsys.data import NUTRITION_COLUMNS
from recipe_recsys.dataset import InteractionData
from recipe_recsys.models.base import Recommender


def _vocab(lists: pd.Series, min_count: int) -> dict[str, int]:
    counts = Counter(token for tokens in lists for token in tokens)
    # Index 0 is reserved for padding / unknown tokens.
    return {tok: i + 1 for i, (tok, n) in enumerate(counts.most_common()) if n >= min_count}


def _padded(lists: pd.Series, vocab: dict[str, int]) -> torch.Tensor:
    """Token lists -> (n, max_len) index matrix; 0 = padding / unknown (ignored by the mean)."""
    rows = [[vocab[t] for t in tokens if t in vocab] for tokens in lists]
    out = torch.zeros((len(rows), max(map(len, rows), default=1) or 1), dtype=torch.long)
    for i, ids in enumerate(rows):
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


class _ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_ingredients: int,
        n_tags: int,
        n_numeric: int,
        dim: int,
        use_content: bool = True,
    ):
        super().__init__()
        self.use_content = use_content
        self.ids = nn.Embedding(n_items, dim)
        if use_content:
            self.ingredients = nn.EmbeddingBag(n_ingredients + 1, dim, mode="mean", padding_idx=0)
            self.tags = nn.EmbeddingBag(n_tags + 1, dim, mode="mean", padding_idx=0)
            self.numeric = nn.Linear(n_numeric, dim)
        n_parts = 4 if use_content else 1
        self.mlp = nn.Sequential(
            nn.Linear(n_parts * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, items, ingredients, tags, numeric):
        parts = [self.ids(items)]
        if self.use_content:
            parts += [self.ingredients(ingredients), self.tags(tags), self.numeric(numeric)]
        return F.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


class _UserTower(nn.Module):
    def __init__(self, n_items: int, dim: int):
        super().__init__()
        # Index n_items is padding for users with shorter histories.
        self.history = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.pad = n_items

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        mask = (history != self.pad).unsqueeze(-1).float()
        pooled = (self.history(history) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.mlp(pooled), dim=1)


class TwoTowerRecommender(Recommender):
    name = "two_tower"

    def __init__(
        self,
        dim: int = 64,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 0.001,
        temperature: float = 0.05,
        max_history: int = 30,
        min_token_count: int = 10,
        seed: int = 42,
        device: str = "auto",
        # Ablation switches (1 = on, 0 = off).
        content: int = 1,
        logq: int = 1,
    ):
        self.dim = dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.max_history = max_history
        self.min_token_count = min_token_count
        self.seed = seed
        self.device = device
        self.content = content
        self.logq = logq

    # ---- features -------------------------------------------------------------------------

    def _item_features(self, content: pd.DataFrame) -> None:
        ingredient_vocab = _vocab(content["ingredients"], self.min_token_count)
        tag_vocab = _vocab(content["tags"], self.min_token_count)
        self._ingredients = _padded(content["ingredients"], ingredient_vocab)
        self._tags = _padded(content["tags"], tag_vocab)
        self._n_vocab = (len(ingredient_vocab), len(tag_vocab))

        numeric = content[NUTRITION_COLUMNS + ["minutes", "n_steps", "n_ingredients"]]
        # Heavy tails (one recipe takes 2 billion minutes): log, then standardize.
        numeric = np.log1p(numeric.clip(lower=0, upper=1e5).to_numpy(dtype=np.float32))
        numeric = (numeric - numeric.mean(0)) / (numeric.std(0) + 1e-6)
        self._numeric = torch.from_numpy(numeric.astype(np.float32))

    def _encode_items(self, rows: torch.Tensor) -> torch.Tensor:
        return self._item_tower(
            rows,
            self._ingredients[rows],
            self._tags[rows],
            self._numeric[rows],
        )

    # ---- training examples -----------------------------------------------------------------

    def _histories(self, data: InteractionData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each training interaction: previous `max_history` items and the target item.

        Returns (histories, targets, last_history_per_user).
        """
        H, pad = self.max_history, data.n_items
        df = data.interactions.sort_values(["user_idx", "date"], kind="stable")
        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()
        starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
        ends = np.r_[starts[1:], len(users)]

        histories = np.full((len(items), H), pad, dtype=np.int64)
        has_history = np.zeros(len(items), dtype=bool)
        last = np.full((data.n_users, H), pad, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True):
            seq = items[start:end]
            for pos in range(1, len(seq)):
                prev = seq[max(0, pos - H) : pos]
                histories[start + pos, : len(prev)] = prev
                has_history[start + pos] = True
            tail = seq[-H:]
            last[users[start], : len(tail)] = tail
        return histories[has_history], items[has_history], last

    # ---- Recommender API -------------------------------------------------------------------

    def fit(self, data: InteractionData, verbose: bool = True) -> TwoTowerRecommender:
        if data.item_content is None:
            raise ValueError("two_tower needs recipe content: call data.with_content(recipes)")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        dev = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )
        self._item_features(data.item_content)
        # Item features are small (~40k rows): keep them on the training device.
        self._ingredients, self._tags, self._numeric = (
            t.to(dev) for t in (self._ingredients, self._tags, self._numeric)
        )
        histories, targets, self._last_history = self._histories(data)

        self._item_tower = _ItemTower(
            data.n_items, *self._n_vocab, self._numeric.shape[1], self.dim, bool(self.content)
        ).to(dev)
        self._user_tower = _UserTower(data.n_items, self.dim).to(dev)
        params = [*self._item_tower.parameters(), *self._user_tower.parameters()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        item_freq = np.bincount(targets, minlength=data.n_items) / len(targets)
        log_q = torch.from_numpy(np.log(item_freq + 1e-12).astype(np.float32)).to(dev)
        if not self.logq:
            log_q = torch.zeros_like(log_q)

        histories_t, targets_t = torch.from_numpy(histories), torch.from_numpy(targets)
        for epoch in range(self.epochs):
            start, total, steps = time.perf_counter(), 0.0, 0
            order = torch.from_numpy(rng.permutation(len(targets)))
            for b in range(0, len(order), self.batch_size):
                idx = order[b : b + self.batch_size]
                target = targets_t[idx].to(dev)
                user_vec = self._user_tower(histories_t[idx].to(dev))
                item_vec = self._encode_items(target)

                logits = user_vec @ item_vec.T / self.temperature - log_q[target]
                # The same recipe twice in a batch is not a negative for itself.
                same = target.unsqueeze(0) == target.unsqueeze(1)
                eye = torch.eye(len(idx), dtype=torch.bool, device=dev)
                logits = logits.masked_fill(same & ~eye, -1e9)
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, steps = total + loss.item(), steps + 1
            if verbose:
                print(
                    f"  two_tower epoch {epoch + 1}: loss={total / steps:.3f} "
                    f"({time.perf_counter() - start:.0f}s on {dev})"
                )

        self._precompute(data, dev)
        return self

    @torch.no_grad()
    def _precompute(self, data: InteractionData, dev: torch.device, chunk: int = 4096) -> None:
        self._item_tower.eval()
        self._user_tower.eval()
        self._item_vectors = (
            torch.cat(
                [
                    self._encode_items(torch.arange(s, min(s + chunk, data.n_items), device=dev))
                    for s in range(0, data.n_items, chunk)
                ]
            )
            .cpu()
            .numpy()
        )
        last = torch.from_numpy(self._last_history)
        self._user_vectors = (
            torch.cat(
                [self._user_tower(last[s : s + chunk].to(dev)) for s in range(0, len(last), chunk)]
            )
            .cpu()
            .numpy()
        )

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self._user_vectors[user_rows] @ self._item_vectors.T
