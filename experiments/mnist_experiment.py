"""MNIST data preparation and PyTorch training/evaluation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from evaluation.metrics import classification_metrics
from models.pytorch_model import MNISTANN


@dataclass
class MNISTData:
    search_train: Dataset
    validation: Dataset
    full_train: Dataset
    test: Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(
    data_dir: str | Path, subset_size: int | None, seed: int
) -> MNISTData:
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    if subset_size is None or subset_size >= len(full_train):
        selected_indices = list(range(len(full_train)))
    elif subset_size < 10:
        raise ValueError("subset_size must be at least 10 to include all MNIST classes.")
    else:
        selected_indices = _stratified_indices(full_train.targets, subset_size, seed)

    selected = Subset(full_train, selected_indices)
    validation_size = max(1, round(len(selected) * 0.1))
    search_size = len(selected) - validation_size
    search_train, validation = random_split(
        selected,
        [search_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return MNISTData(search_train, validation, full_train, test)


def _stratified_indices(
    targets: torch.Tensor, subset_size: int, seed: int
) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(targets).tolist()
    base, remainder = divmod(subset_size, len(classes))
    selected = []
    for class_index, label in enumerate(classes):
        indices = torch.where(targets == label)[0]
        count = base + (1 if class_index < remainder else 0)
        permutation = torch.randperm(len(indices), generator=generator)[:count]
        selected.extend(indices[permutation].tolist())
    random.Random(seed).shuffle(selected)
    return selected


class MNISTObjective:
    """Train each candidate from scratch, except Hyperband's promoted trials."""

    def __init__(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        device: torch.device,
        seed: int,
        verbose: bool = True,
    ) -> None:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.evaluation_count = 0
        self.states: dict[str, dict[str, Any]] = {}

    def __call__(
        self, config: dict[str, int | float], epochs: int, trial_id: str | None = None
    ) -> dict[str, float]:
        started = time.perf_counter()
        state = self.states.get(trial_id) if trial_id is not None else None
        if state is None:
            trial_seed = self.seed + self.evaluation_count
            self.evaluation_count += 1
            state = self._new_state(config, trial_seed)
            if trial_id is not None:
                self.states[trial_id] = state
        elif state["config"] != config:
            raise ValueError(f"Hyperband trial {trial_id!r} changed configuration.")
        if epochs < state["epochs_trained"]:
            raise ValueError("A promoted Hyperband trial cannot request fewer epochs.")

        torch.set_rng_state(state["torch_rng_state"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        while state["epochs_trained"] < epochs:
            _train_epoch(
                state["model"],
                state["optimizer"],
                state["train_loader"],
                state["criterion"],
                self.device,
            )
            state["epochs_trained"] += 1
        state["torch_rng_state"] = torch.get_rng_state()
        if self.device.type == "cuda":
            state["cuda_rng_states"] = torch.cuda.get_rng_state_all()

        metrics = _evaluate_model(
            state["model"], self.validation_dataset, self.device, prefix="val"
        )
        metrics["duration_seconds"] = time.perf_counter() - started
        if self.verbose:
            print(
                f"    trial {self.evaluation_count:>3} | "
                f"epochs={epochs:<3} val_accuracy={metrics['val_accuracy']:.4f} "
                f"val_loss={metrics['val_loss']:.4f}",
                flush=True,
            )
        return metrics

    def release_trial(self, trial_id: str) -> None:
        self.states.pop(trial_id, None)

    def _new_state(self, config: dict[str, int | float], seed: int) -> dict[str, Any]:
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model = MNISTANN(
            hidden_neurons=int(config["hidden_neurons"]),
            dropout=float(config["dropout"]),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        return {
            "config": dict(config),
            "model": model,
            "optimizer": optimizer,
            "train_loader": train_loader,
            "criterion": nn.CrossEntropyLoss(),
            "epochs_trained": 0,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
            ),
        }


def fit_and_test(
    config: dict[str, int | float],
    train_dataset: Dataset,
    test_dataset: Dataset,
    epochs: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = MNISTANN(
        hidden_neurons=int(config["hidden_neurons"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        _train_epoch(model, optimizer, train_loader, criterion, device)
    return _evaluate_model(model, test_dataset, device, prefix="test")


def _train_epoch(model, optimizer, loader, criterion, device) -> None:
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def _evaluate_model(
    model: nn.Module, dataset: Dataset, device: torch.device, prefix: str
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets = []
    predictions = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels)) * labels.size(0)
            total_items += labels.size(0)
            targets.append(labels.cpu())
            predictions.append(logits.argmax(dim=1).cpu())

    scores = classification_metrics(torch.cat(targets), torch.cat(predictions))
    return {
        f"{prefix}_loss": total_loss / total_items,
        **{f"{prefix}_{name}": value for name, value in scores.items()},
    }
