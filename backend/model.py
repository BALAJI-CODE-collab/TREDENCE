"""Prunable neural network components and training helpers for PruneVision."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_CHANNELS = 3
DEFAULT_HIDDEN_1 = 32
DEFAULT_HIDDEN_2 = 64
DEFAULT_HIDDEN_3 = 128
DEFAULT_LINEAR_1 = 512
DEFAULT_LINEAR_2 = 256
DEFAULT_INPUT_SPATIAL_SIZE = 4
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_SPARSITY_THRESHOLD = 0.01
CHECKPOINT_VERSION = 1


@dataclass(slots=True)
class TrainingHistoryEntry:
    """Represents one epoch of training history."""

    epoch: int
    train_loss: float
    train_acc: float
    sparsity: float


class PrunableLinear(nn.Module):
    """Linear layer with learnable multiplicative gates for structured pruning."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Create a prunable linear layer."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.gate_scores)
        if self.bias is not None:
            bound = 1 / self.in_features ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the linear transformation with sigmoid gates applied to weights."""

        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(input_tensor, pruned_weights, self.bias)

    def get_sparsity(self) -> float:
        """Return the percentage of gate values below the pruning threshold."""

        gates = torch.sigmoid(self.gate_scores)
        sparsity = (gates < DEFAULT_SPARSITY_THRESHOLD).float().mean() * 100.0
        return float(sparsity.item())

    def get_active_gates(self) -> torch.Tensor:
        """Return the current gate values."""

        return torch.sigmoid(self.gate_scores).detach()


class PrunableCNN(nn.Module):
    """CNN backbone with prunable fully connected layers for plant disease detection."""

    def __init__(self, num_classes: int) -> None:
        """Construct the convolutional and prunable linear stack."""

        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv2d(DEFAULT_INPUT_CHANNELS, DEFAULT_HIDDEN_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(DEFAULT_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(DEFAULT_HIDDEN_1, DEFAULT_HIDDEN_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(DEFAULT_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(DEFAULT_HIDDEN_2, DEFAULT_HIDDEN_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(DEFAULT_HIDDEN_3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((DEFAULT_INPUT_SPATIAL_SIZE, DEFAULT_INPUT_SPATIAL_SIZE)),
        )
        flattened_features = DEFAULT_HIDDEN_3 * DEFAULT_INPUT_SPATIAL_SIZE * DEFAULT_INPUT_SPATIAL_SIZE
        self.classifier = nn.Sequential(
            PrunableLinear(flattened_features, DEFAULT_LINEAR_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            PrunableLinear(DEFAULT_LINEAR_1, DEFAULT_LINEAR_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            PrunableLinear(DEFAULT_LINEAR_2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        features = self.conv_features(input_tensor)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)

    def _prunable_layers(self) -> list[PrunableLinear]:
        """Collect all prunable linear layers in the model."""

        return [module for module in self.modules() if isinstance(module, PrunableLinear)]

    def get_total_sparsity(self) -> float:
        """Return the average sparsity across all prunable layers."""

        layers = self._prunable_layers()
        if not layers:
            return 0.0
        return float(sum(layer.get_sparsity() for layer in layers) / len(layers))

    def get_model_size_mb(self) -> float:
        """Estimate the model size in megabytes from parameter storage."""

        total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())
        return float(total_bytes / (1024.0 * 1024.0))

    def get_all_gate_values(self) -> torch.Tensor:
        """Return all gate values flattened into a single tensor."""

        gate_values = [layer.get_active_gates().flatten() for layer in self._prunable_layers()]
        if not gate_values:
            return torch.empty(0)
        return torch.cat(gate_values, dim=0)

    def get_total_parameters(self) -> int:
        """Count all trainable and non-trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def get_pruned_parameters(self) -> int:
        """Estimate the number of effectively pruned parameters."""

        total_pruned = 0
        for layer in self._prunable_layers():
            total_pruned += int((torch.sigmoid(layer.gate_scores) < DEFAULT_SPARSITY_THRESHOLD).sum().item())
        return total_pruned

    def get_active_parameters(self) -> int:
        """Estimate the number of active parameters."""

        return self.get_total_parameters() - self.get_pruned_parameters()


def build_sparse_regularization(model: PrunableCNN) -> torch.Tensor:
    """Compute the L1 penalty over all gate values."""

    gate_values = model.get_all_gate_values()
    if gate_values.numel() == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return gate_values.abs().mean()


def train_prunable_model(
    model: PrunableCNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    lambda_sparse: float,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    checkpoint_path: Optional[Path] = None,
) -> tuple[list[TrainingHistoryEntry], dict[str, Any]]:
    """Train a prunable CNN and optionally save the best checkpoint."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[TrainingHistoryEntry] = []
    best_state: dict[str, Any] = {}
    best_score = float("-inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            ce_loss = criterion(logits, batch_targets)
            sparse_loss = build_sparse_regularization(model)
            loss = ce_loss + lambda_sparse * sparse_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_inputs.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch_targets).sum().item())
            total += int(batch_targets.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        sparsity = model.get_total_sparsity()
        history.append(TrainingHistoryEntry(epoch=epoch, train_loss=train_loss, train_acc=train_acc, sparsity=sparsity))

        val_score = evaluate_accuracy(model, val_loader, device) if val_loader is not None else train_acc
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_score=%.4f sparsity=%.2f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_score,
            sparsity,
        )

        if val_score > best_score:
            best_score = val_score
            best_state = {
                "version": CHECKPOINT_VERSION,
                "model_state_dict": model.state_dict(),
                "lambda_sparse": lambda_sparse,
                "epoch": epoch,
                "train_history": [asdict(entry) for entry in history],
            }
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

    return history, best_state


def evaluate_accuracy(
    model: PrunableCNN,
    data_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
) -> float:
    """Evaluate accuracy for a loader."""

    if data_loader is None:
        return 0.0

    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == batch_targets).sum().item())
            total_samples += int(batch_targets.size(0))

    return total_correct / max(total_samples, 1)


def evaluate_loss(
    model: PrunableCNN,
    data_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
) -> float:
    """Evaluate cross-entropy loss for a loader."""

    if data_loader is None:
        return 0.0

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            total_loss += float(loss.item()) * batch_inputs.size(0)
            total_samples += int(batch_targets.size(0))

    return total_loss / max(total_samples, 1)
